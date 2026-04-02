"""
preprocess_sample.py — Sample-building helpers for preprocess_zeek.

Provides:
  pick_reason(verdict)        — random reason string from the appropriate pool
  score_hard_benign(...)      — score a benign flow by how attack-like it looks
  make_sample(...)            — build a training JSONL record from raw flow fields
"""

import random

from preprocess_config import (
    CONN_STATE_MASK_PROB,
    CONTEXT_MASK_PROB,
    HARD_BENIGN_MIN_SCORE,
    ATTACK_REASONS,
    BENIGN_REASONS,
)
from prompt_utils import SYSTEM_PROMPT, build_prompt


def pick_reason(verdict):
    pool = ATTACK_REASONS if verdict == "ATTACK" else BENIGN_REASONS
    return random.choice(pool)


def score_hard_benign(proto, conn_state, service="-", resp_port="-", orig_port="-",
                      http_ctx=None, dns_ctx=None, ssl_ctx=None, behavior_ctx=None):
    """Return (score, flags) for benign samples that look attack-like.

    Higher scores mean the benign flow is more useful as a hard negative.
    """
    score = 0
    flags = []

    proto_s    = str(proto      or "").strip().lower()
    state_s    = str(conn_state or "").strip().upper()
    service_s  = str(service    or "").strip().lower()
    resp_port_s = str(resp_port or "").strip()

    try:
        resp_port_i = int(float(resp_port_s))
    except (ValueError, TypeError):
        resp_port_i = None

    if http_ctx is not None:
        score += 2
        flags.append("http_ctx")
        method = str(http_ctx.get("method") or "").upper()
        host   = str(http_ctx.get("host")   or "")
        uri    = str(http_ctx.get("uri")    or "")
        if method in ("GET", "POST"):
            score += 1
            flags.append(f"http_{method.lower()}")
        if host or uri:
            score += 1
            flags.append("http_named_endpoint")

    if dns_ctx is not None:
        score += 2
        flags.append("dns_ctx")
        rcode = str(dns_ctx.get("rcode_name") or "").upper()
        ttl   = str(dns_ctx.get("ttl")        or "").strip()
        if rcode in ("NXDOMAIN", "NXERROR"):
            score += 2
            flags.append("dns_nxdomain")
        try:
            if ttl and float(ttl) <= 60:
                score += 1
                flags.append("dns_short_ttl")
        except (ValueError, TypeError):
            pass

    if ssl_ctx is not None:
        score += 2
        flags.append("ssl_ctx")
        version    = str(ssl_ctx.get("version")           or "").upper()
        cipher     = str(ssl_ctx.get("cipher")            or "").upper()
        validation = str(ssl_ctx.get("validation_status") or "").upper()
        issuer     = str(ssl_ctx.get("issuer")            or "")
        if "FAILED" in validation or "SELF-SIGNED" in issuer.upper():
            score += 2
            flags.append("ssl_untrusted")
        if version in ("SSLV3", "TLSV1", "TLSV1.0") or "RC4" in cipher:
            score += 1
            flags.append("ssl_legacy")

    if resp_port_i in {22, 23, 25, 53, 80, 123, 443, 445, 502, 6667, 8080, 8443, 3128, 3389}:
        score += 1
        flags.append(f"port_{resp_port_i}")
    if resp_port_i in {23, 445, 6667, 3389, 4848}:
        score += 1
        flags.append("high_risk_service_port")

    if state_s in {"S0", "RSTO", "S1"}:
        score += 1
        flags.append(f"state_{state_s.lower()}")
    if resp_port_i == 443 and service_s in {"", "-", "unknown"}:
        score += 1
        flags.append("port443_no_service")
    if proto_s == "udp" and resp_port_i == 53:
        score += 1
        flags.append("udp_dns")

    if behavior_ctx:
        score += 1
        flags.append("behavior_ctx")
        src_conn_60s    = int(behavior_ctx.get("src_conn_60s")            or 0)
        unique_dst_60s  = int(behavior_ctx.get("src_unique_dst_60s")      or 0)
        unique_ports_60s = int(behavior_ctx.get("src_unique_ports_60s")   or 0)
        same_port_60s   = int(behavior_ctx.get("same_resp_port_60s")      or 0)
        repeats_300s    = int(behavior_ctx.get("same_flow_size_repeats_300s") or 0)
        periodic        = str(behavior_ctx.get("pair_periodic_score")     or "").lower()
        if src_conn_60s >= 10:
            score += 2
            flags.append("burst_60s")
        if unique_dst_60s >= 5 or unique_ports_60s >= 5:
            score += 2
            flags.append("fanout_60s")
        if same_port_60s >= 5:
            score += 1
            flags.append("same_port_repeat")
        if repeats_300s >= 3:
            score += 1
            flags.append("same_size_repeat")
        if periodic in {"high", "medium"}:
            score += 2 if periodic == "high" else 1
            flags.append(f"periodic_{periodic}")

    return score, flags


def make_sample(proto, duration, orig_pkts, resp_pkts,
                orig_bytes, resp_bytes, conn_state, verdict, source,
                service="-", resp_port="-", orig_port="-",
                http_ctx=None, dns_ctx=None, ssl_ctx=None, behavior_ctx=None):
    hard_benign_score = 0
    hard_benign_flags = []
    if verdict == "FALSE POSITIVE":
        hard_benign_score, hard_benign_flags = score_hard_benign(
            proto, conn_state, service=service, resp_port=resp_port, orig_port=orig_port,
            http_ctx=http_ctx, dns_ctx=dns_ctx, ssl_ctx=ssl_ctx, behavior_ctx=behavior_ctx,
        )

    # Mask conn_state with CONN_STATE_MASK_PROB — forces model to use numeric
    # features (bytes, packets, port) when state is unavailable or ambiguous.
    prompt_conn_state = "-" if random.random() < CONN_STATE_MASK_PROB else conn_state

    # Per-section context masking (50% chance each section is dropped).
    # Prevents "has http section → ATTACK" shortcut; forces conn.log-only correctness.
    prompt_behavior_ctx = behavior_ctx
    if http_ctx is not None and random.random() < CONTEXT_MASK_PROB:
        http_ctx = None
    if dns_ctx is not None and random.random() < CONTEXT_MASK_PROB:
        dns_ctx = None
    if ssl_ctx is not None and random.random() < CONTEXT_MASK_PROB:
        ssl_ctx = None
    if prompt_behavior_ctx is not None and random.random() < CONTEXT_MASK_PROB:
        prompt_behavior_ctx = None

    prompt = build_prompt(
        proto, duration, orig_pkts, resp_pkts,
        orig_bytes, resp_bytes, prompt_conn_state, service,
        resp_port=resp_port, orig_port=orig_port,
        http_ctx=http_ctx, dns_ctx=dns_ctx, ssl_ctx=ssl_ctx,
        behavior_ctx=prompt_behavior_ctx,
    )
    reason = pick_reason(verdict)
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": f"VERDICT: {verdict}\nREASON: {reason}"},
        ],
        "source":            source,
        "verdict":           verdict,
        "conn_state":        conn_state,  # original (pre-mask) — used for SF oversampling
        "hard_benign_score": hard_benign_score,
        "hard_benign_flags": hard_benign_flags,
        "is_hard_benign":    hard_benign_score >= HARD_BENIGN_MIN_SCORE,
    }
