"""Behavioral window features for Zeek conn.log-style flow records."""

from collections import Counter, deque


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_port(value):
    if value in (None, "", "-", "?", "None", "nan", "NaN"):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip() or None


def _flow_size(row):
    orig = _to_float(row.get("orig_bytes"))
    resp = _to_float(row.get("resp_bytes"))
    return (orig or 0.0) + (resp or 0.0)


def _trim_window(dq, now_ts, window_s):
    cutoff = now_ts - window_s
    while dq and dq[0].get("_ts_f") is not None and dq[0]["_ts_f"] < cutoff:
        dq.popleft()


def _periodic_label(gaps):
    if len(gaps) < 3:
        return "Low"
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 0:
        return "Low"
    variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    coeff_var = (variance ** 0.5) / mean_gap
    if coeff_var <= 0.10:
        return "High"
    if coeff_var <= 0.25:
        return "Medium"
    return "Low"


def build_behavior_contexts(rows, short_window=60.0, long_window=300.0):
    """Return a behavior context dict for each row.

    Uses only prior flows in the same capture, never future rows.
    Expected row keys:
      ts, orig_h, resp_h, resp_p, conn_state, orig_bytes, resp_bytes
    """
    if not rows:
        return []

    ordered = []
    for idx, row in enumerate(rows):
        ts_f = _to_float(row.get("ts"))
        ordered.append((idx, {**row, "_ts_f": ts_f}))

    ordered.sort(key=lambda item: (
        float("inf") if item[1]["_ts_f"] is None else item[1]["_ts_f"],
        item[0],
    ))

    src_short = {}
    src_long = {}
    pair_long = {}
    contexts = [None] * len(rows)

    for idx, row in ordered:
        ts_f = row.get("_ts_f")
        if ts_f is None:
            continue

        src = row.get("orig_h") or "?"
        pair = (
            row.get("orig_h") or "?",
            row.get("resp_h") or "?",
            _clean_port(row.get("resp_p")) or "?",
            (row.get("proto") or "?").lower(),
        )

        dq_src_short = src_short.setdefault(src, deque())
        dq_src_long = src_long.setdefault(src, deque())
        dq_pair_long = pair_long.setdefault(pair, deque())

        _trim_window(dq_src_short, ts_f, short_window)
        _trim_window(dq_src_long, ts_f, long_window)
        _trim_window(dq_pair_long, ts_f, long_window)

        state_counts = {"S0": 0, "RSTO": 0, "SF": 0, "S1": 0}
        dst_ips = set()
        dst_ports = set()
        same_port_60s = 0
        for prev in dq_src_short:
            dst = prev.get("resp_h")
            port = _clean_port(prev.get("resp_p"))
            state = prev.get("conn_state")
            if dst:
                dst_ips.add(dst)
            if port:
                dst_ports.add(port)
            if state in state_counts:
                state_counts[state] += 1
            if port == _clean_port(row.get("resp_p")):
                same_port_60s += 1

        pair_gaps = []
        prior_sizes = []
        last_ts = None
        for prev in dq_pair_long:
            prev_ts = prev.get("_ts_f")
            if prev_ts is not None:
                if last_ts is not None:
                    pair_gaps.append(prev_ts - last_ts)
                last_ts = prev_ts
            prior_sizes.append(_flow_size(prev))
        if last_ts is not None:
            pair_gaps.append(ts_f - last_ts)

        this_size = _flow_size(row)
        size_repeats = sum(1 for size in prior_sizes if abs(size - this_size) <= 16.0)
        mean_gap = (sum(pair_gaps) / len(pair_gaps)) if pair_gaps else None

        contexts[idx] = {
            "src_conn_60s": len(dq_src_short),
            "src_conn_300s": len(dq_src_long),
            "src_unique_dst_60s": len(dst_ips),
            "src_unique_ports_60s": len(dst_ports),
            "src_s0_60s": state_counts["S0"],
            "src_rsto_60s": state_counts["RSTO"],
            "src_sf_60s": state_counts["SF"] + state_counts["S1"],
            "pair_conn_300s": len(dq_pair_long),
            "pair_mean_gap_s": mean_gap,
            "pair_periodic_score": _periodic_label(pair_gaps),
            "same_resp_port_60s": same_port_60s,
            "same_flow_size_repeats_300s": size_repeats,
        }

        dq_src_short.append(row)
        dq_src_long.append(row)
        dq_pair_long.append(row)

    return contexts


def _top_labels(counter, n=3):
    items = [f"{label}({count})" for label, count in counter.most_common(n) if label]
    return ", ".join(items) if items else "N/A"


def build_host_summaries(rows, flow_results, behavior_ctxs=None,
                         uid_http=None, uid_dns=None, uid_ssl=None):
    """Aggregate pass-1 flow outputs into host-level summaries keyed by orig_h."""
    behavior_ctxs = behavior_ctxs or [None] * len(rows)
    uid_http = uid_http or {}
    uid_dns = uid_dns or {}
    uid_ssl = uid_ssl or {}

    hosts = {}
    for i, row in enumerate(rows):
        host = row.get("host_key") or row.get("orig_h") or "?"
        host_display = row.get("orig_h") or host
        verdict = flow_results[i][0] if i < len(flow_results) else "UNKNOWN"
        behavior = behavior_ctxs[i] if i < len(behavior_ctxs) else None
        entry = hosts.setdefault(host, {
            "host": host_display,
            "host_key": host,
            "total_flows": 0,
            "pred_attack": 0,
            "pred_benign": 0,
            "pred_unknown": 0,
            "unique_dst_ips": set(),
            "unique_dst_ports": set(),
            "state_counts": Counter(),
            "port_counts": Counter(),
            "service_counts": Counter(),
            "http_flows": 0,
            "dns_flows": 0,
            "ssl_flows": 0,
            "periodic_high": 0,
            "periodic_medium": 0,
            "bursty_flows": 0,
            "fanout_flows": 0,
            "same_size_repeat_flows": 0,
            "top_attack_ports": Counter(),
            "top_attack_dsts": Counter(),
        })

        entry["total_flows"] += 1
        if verdict == "ATTACK":
            entry["pred_attack"] += 1
        elif verdict == "FALSE POSITIVE":
            entry["pred_benign"] += 1
        else:
            entry["pred_unknown"] += 1

        dst = row.get("resp_h")
        port = _clean_port(row.get("resp_p"))
        state = row.get("conn_state") or "?"
        service = row.get("service") or "-"
        uid = row.get("uid")

        if dst:
            entry["unique_dst_ips"].add(dst)
        if port:
            entry["unique_dst_ports"].add(port)
            entry["port_counts"][port] += 1
        if service and service != "-":
            entry["service_counts"][service] += 1
        entry["state_counts"][state] += 1

        if uid in uid_http:
            entry["http_flows"] += 1
        if uid in uid_dns:
            entry["dns_flows"] += 1
        if uid in uid_ssl:
            entry["ssl_flows"] += 1

        if behavior:
            periodic = str(behavior.get("pair_periodic_score") or "").lower()
            if periodic == "high":
                entry["periodic_high"] += 1
            elif periodic == "medium":
                entry["periodic_medium"] += 1
            if int(behavior.get("src_conn_60s") or 0) >= 10:
                entry["bursty_flows"] += 1
            if (int(behavior.get("src_unique_dst_60s") or 0) >= 5 or
                    int(behavior.get("src_unique_ports_60s") or 0) >= 5):
                entry["fanout_flows"] += 1
            if int(behavior.get("same_flow_size_repeats_300s") or 0) >= 3:
                entry["same_size_repeat_flows"] += 1

        if verdict == "ATTACK":
            if port:
                entry["top_attack_ports"][port] += 1
            if dst:
                entry["top_attack_dsts"][dst] += 1

    summaries = []
    for host, entry in sorted(hosts.items()):
        total = entry["total_flows"]
        summaries.append({
            "host": entry["host"],
            "host_key": host,
            "total_flows": total,
            "pred_attack": entry["pred_attack"],
            "pred_benign": entry["pred_benign"],
            "pred_unknown": entry["pred_unknown"],
            "attack_ratio": (entry["pred_attack"] / total) if total else 0.0,
            "unique_dst_ips": len(entry["unique_dst_ips"]),
            "unique_dst_ports": len(entry["unique_dst_ports"]),
            "state_s0": entry["state_counts"].get("S0", 0),
            "state_rsto": entry["state_counts"].get("RSTO", 0),
            "state_sf": entry["state_counts"].get("SF", 0) + entry["state_counts"].get("S1", 0),
            "http_flows": entry["http_flows"],
            "dns_flows": entry["dns_flows"],
            "ssl_flows": entry["ssl_flows"],
            "periodic_high": entry["periodic_high"],
            "periodic_medium": entry["periodic_medium"],
            "bursty_flows": entry["bursty_flows"],
            "fanout_flows": entry["fanout_flows"],
            "same_size_repeat_flows": entry["same_size_repeat_flows"],
            "top_ports": _top_labels(entry["port_counts"]),
            "top_services": _top_labels(entry["service_counts"]),
            "top_attack_ports": _top_labels(entry["top_attack_ports"]),
            "top_attack_dsts": _top_labels(entry["top_attack_dsts"]),
        })
    return summaries
