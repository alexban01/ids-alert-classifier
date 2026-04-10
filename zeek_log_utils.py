"""
zeek_log_utils.py — Shared utilities for parsing Zeek TSV logs and
CTU-Malware-Capture download/labelling helpers.

Used by loader_ctu_malware.py; available to any loader that needs raw
Zeek log parsing.
"""

import csv
import os
import re
import urllib.request

from preprocess_config import CTU_MALWARE_DIR


# ── Generic Zeek log parser ────────────────────────────────────────────────────

def parse_zeek_log(path):
    """Parse a Zeek TSV log (#fields header), return list of row dicts."""
    rows   = []
    fields = None
    sep    = "\t"
    unset  = {"-", "(empty)"}
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#fields"):
                fields = line[len("#fields"):].strip().split(sep)
            elif line.startswith("#separator"):
                raw = line.split()[-1]
                sep = (bytes(raw, "utf-8").decode("unicode_escape")
                       if "\\x" in raw else raw)
            elif line.startswith("#empty_field"):
                unset.add(line.split(sep)[-1])
            elif line.startswith("#unset_field"):
                unset.add(line.split(sep)[-1])
            elif line.startswith("#") or not line.strip():
                continue
            elif fields is not None:
                parts = line.split(sep)
                row = {k: (None if (parts[i] if i < len(parts) else "-") in unset
                           else (parts[i] if i < len(parts) else None))
                       for i, k in enumerate(fields)}
                rows.append(row)
    return rows


# ── CTU-Malware-Capture download helpers ──────────────────────────────────────

def norm_key(proto, ip_a, port_a, ip_b, port_b):
    """Normalise 5-tuple so (A→B) and (B→A) produce the same lookup key."""
    pair_a = (str(ip_a).strip(), str(port_a).strip())
    pair_b = (str(ip_b).strip(), str(port_b).strip())
    lo, hi = (pair_a, pair_b) if pair_a <= pair_b else (pair_b, pair_a)
    return (str(proto).strip().lower(), lo[0], lo[1], hi[0], hi[1])


def ctu_download(url, scenario_id, filename=None, optional=False):
    """Download url to CTU_MALWARE_DIR/{scenario_id}_{filename}. Returns path or None."""
    if filename is None:
        filename = url.rstrip("/").split("/")[-1]
    local = os.path.join(CTU_MALWARE_DIR, f"{scenario_id}_{filename}")
    if os.path.isfile(local):
        print(f"    [cache] {os.path.basename(local)}")
        return local
    try:
        os.makedirs(CTU_MALWARE_DIR, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(local, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        print(f"    Downloaded {os.path.basename(local)} "
              f"({os.path.getsize(local) // 1024} KB)")
        return local
    except Exception as e:
        if optional:
            print(f"    [SKIP] {filename}: {e}")
            if os.path.isfile(local):
                os.remove(local)
            return None
        raise


def find_binetflow_url(base_url):
    """Fetch the Stratosphere capture directory index and return the binetflow URL.

    Priority order:
    1. .binetflow.labeled in root (e.g. Botnet-78-2/Zeus)
    2. .binetflow in detailed-bidirectional-flow-labels/ subdir (e.g. Botnet-42/44/52/54)
    3. Plain .binetflow in root (fallback — may be unlabeled, yields 0 samples)
    """
    def _list_dir(url):
        req = urllib.request.Request(
            url.rstrip("/") + "/", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        return re.findall(r'href="([^"/][^"]*?)"', html)

    try:
        links = _list_dir(base_url)

        labeled = [l for l in links if l.endswith(".binetflow.labeled")]
        if labeled:
            return base_url.rstrip("/") + "/" + labeled[0].lstrip("/")

        if any("detailed-bidirectional-flow-labels" in l for l in links):
            try:
                sub = _list_dir(
                    base_url.rstrip("/") + "/detailed-bidirectional-flow-labels/")
                bf = [l for l in sub if l.endswith(".binetflow")]
                if bf:
                    return (base_url.rstrip("/")
                            + "/detailed-bidirectional-flow-labels/"
                            + bf[0].lstrip("/"))
            except Exception:
                pass

        plain = [l for l in links if l.endswith(".binetflow")]
        if plain:
            return base_url.rstrip("/") + "/" + plain[0].lstrip("/")

    except Exception as e:
        print(f"    [WARN] Could not fetch index for {base_url}: {e}")
    return None


def build_binetflow_lookup(path):
    """Parse binetflow CSV → {norm_5tuple: label} dict.

    Label mapping: Botnet → ATTACK, Normal → FALSE POSITIVE, Background → skip.
    """
    lookup = {}
    with open(path, newline="", errors="replace") as f:
        reader = csv.reader(f)
        header = None
        for row in reader:
            if header is None:
                header = [h.strip() for h in row]
                idx    = {h: i for i, h in enumerate(header)}
                # Some captures (e.g. Botnet-25-1) use "Label(Normal:CC:Background)"
                # instead of plain "Label" — treat any header starting with "Label" as
                # the label column.
                if "Label" not in idx:
                    for h in header:
                        if h.startswith("Label"):
                            idx["Label"] = idx[h]
                            break
                needed = {"Proto", "SrcAddr", "Sport", "DstAddr", "Dport", "Label"}
                missing = needed - set(idx)
                if missing:
                    raise ValueError(f"binetflow missing columns: {missing}\nGot: {header}")
                continue
            if len(row) <= max(idx["Label"], idx["Proto"],
                               idx["SrcAddr"], idx["Sport"],
                               idx["DstAddr"], idx["Dport"]):
                continue
            raw_label = row[idx["Label"]].strip().lower()
            if "botnet" in raw_label or "malware" in raw_label:
                label = "ATTACK"
            elif "normal" in raw_label:
                label = "FALSE POSITIVE"
            else:
                continue  # Background → skip
            key = norm_key(
                row[idx["Proto"]],
                row[idx["SrcAddr"]], row[idx["Sport"]],
                row[idx["DstAddr"]], row[idx["Dport"]],
            )
            if key not in lookup or label == "ATTACK":  # ATTACK wins on conflict
                lookup[key] = label
    return lookup


# ── Auxiliary log context builders ────────────────────────────────────────────

def build_http_lookup(path):
    """Build uid → http_ctx dict from Zeek http.log (first request per uid)."""
    lookup = {}
    for row in parse_zeek_log(path):
        uid = row.get("uid")
        if not uid or uid in lookup:
            continue
        lookup[uid] = {
            "method":        row.get("method"),
            "host":          row.get("host"),
            "uri":           row.get("uri"),
            "user_agent":    row.get("user_agent"),
            "status_code":   row.get("status_code"),
            "resp_body_len": row.get("response_body_len"),
        }
    return lookup


def build_dns_lookup(path):
    """Build uid → dns_ctx dict from Zeek dns.log (first response per uid)."""
    lookup = {}
    for row in parse_zeek_log(path):
        uid = row.get("uid")
        if not uid or uid in lookup:
            continue
        answers_raw = row.get("answers") or ""
        ttls_raw    = row.get("TTLs") or row.get("ttls") or ""
        lookup[uid] = {
            "query":      row.get("query"),
            "answers":    answers_raw.split(",")[0].strip() or None,
            "qtype_name": row.get("qtype_name"),
            "ttl":        ttls_raw.split(",")[0].strip() or None,
            "rcode_name": row.get("rcode_name"),
        }
    return lookup


def build_ssl_lookup(path):
    """Build uid → ssl_ctx dict from Zeek ssl.log (first session per uid)."""
    lookup = {}
    for row in parse_zeek_log(path):
        uid = row.get("uid")
        if not uid or uid in lookup:
            continue
        issuer_raw  = row.get("issuer")  or ""
        subject_raw = row.get("subject") or ""
        if not issuer_raw or issuer_raw == subject_raw:
            issuer = "Self-Signed"
        else:
            cn = next((p.replace("CN=", "").strip()
                       for p in issuer_raw.split(",")
                       if p.strip().startswith("CN=")), issuer_raw)
            issuer = cn[:48]
        lookup[uid] = {
            "version":           row.get("version"),
            "cipher":            row.get("cipher"),
            "issuer":            issuer,
            "validation_status": row.get("validation_status"),
        }
    return lookup
