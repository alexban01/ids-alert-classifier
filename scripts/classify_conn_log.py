"""
classify_conn_log.py — Classify a Zeek conn.log using the v9.0 IDS classifier.

Two inference modes:
  --ollama   Use the Ollama-served ids-classifier (no GPU/transformers needed)
  (default)  Load v9.0 LoRA adapter via HuggingFace + PEFT (requires GPU)

Optionally enrich prompts with application-layer context from auxiliary Zeek logs:
  --http-log PATH    Zeek http.log (adds [HTTP] section to each matched flow)
  --dns-log  PATH    Zeek dns.log  (adds [DNS]  section to each matched flow)
  --ssl-log  PATH    Zeek ssl.log  (adds [SSL]  section to each matched flow)
  --host-pass2       Run an optional second pass on aggregated source-host behavior

Usage:
    .venv/bin/python classify_conn_log.py [CONN_LOG] [--ollama] [--limit N]
    .venv/bin/python classify_conn_log.py conn.log --ollama
    .venv/bin/python classify_conn_log.py conn.log --limit 500
    .venv/bin/python classify_conn_log.py conn.log --http-log http.log --ssl-log ssl.log
"""

import os
import sys
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from behavior_features import build_behavior_contexts, build_host_summaries
from prompt_utils import SYSTEM_PROMPT, build_prompt, build_host_prompt, extract_verdict

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR      = "./v9.0-ids-lora-adapter"
OLLAMA_MODEL     = "ids-classifier"
OLLAMA_URL       = "http://localhost:11434/api/generate"
CONN_LOG         = "real_conn/conn.log.5"
MAX_NEW_TOKENS   = 80
BATCH_SIZE       = 8

# ── Zeek conn.log field indices (tab-separated, 21-field standard format) ─────
#  0:ts  1:uid  2:id.orig_h  3:id.orig_p  4:id.resp_h  5:id.resp_p
#  6:proto  7:service  8:duration  9:orig_bytes  10:resp_bytes
# 11:conn_state  12:local_orig  13:local_resp  14:missed_bytes  15:history
# 16:orig_pkts  17:orig_ip_bytes  18:resp_pkts  19:resp_ip_bytes
# 20:tunnel_parents
F_PROTO      = 6
F_SERVICE    = 7
F_DURATION   = 8
F_ORIG_BYTES = 9
F_RESP_BYTES = 10
F_CONN_STATE = 11
F_ORIG_PKTS  = 16
F_RESP_PKTS  = 18


# ── Parse conn.log ─────────────────────────────────────────────────────────────
def parse_conn_log(path, limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 19:
                continue
            rows.append({
                "ts":         parts[0],
                "uid":        parts[1],
                "orig_h":     parts[2],
                "orig_p":     parts[3],
                "resp_h":     parts[4],
                "resp_p":     parts[5],
                "proto":      parts[F_PROTO],
                "service":    parts[F_SERVICE],
                "duration":   parts[F_DURATION],
                "orig_bytes": parts[F_ORIG_BYTES],
                "resp_bytes": parts[F_RESP_BYTES],
                "conn_state": parts[F_CONN_STATE],
                "orig_pkts":  parts[F_ORIG_PKTS],
                "resp_pkts":  parts[F_RESP_PKTS],
            })
            if limit and len(rows) >= limit:
                break
    return rows


# ── Auxiliary Zeek log parsers ─────────────────────────────────────────────────

def _parse_zeek_log(path):
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


def build_uid_http(http_log_path):
    """Build uid → http_ctx dict from Zeek http.log."""
    lookup = {}
    for row in _parse_zeek_log(http_log_path):
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
    print(f"  http.log: {len(lookup)} uid entries")
    return lookup


def build_uid_dns(dns_log_path):
    """Build uid → dns_ctx dict from Zeek dns.log."""
    lookup = {}
    for row in _parse_zeek_log(dns_log_path):
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
    print(f"  dns.log:  {len(lookup)} uid entries")
    return lookup


def build_uid_ssl(ssl_log_path):
    """Build uid → ssl_ctx dict from Zeek ssl.log."""
    lookup = {}
    for row in _parse_zeek_log(ssl_log_path):
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
    print(f"  ssl.log:  {len(lookup)} uid entries")
    return lookup


def build_prompts(rows, uid_http=None, uid_dns=None, uid_ssl=None, behavior_ctxs=None):
    """Build prompt strings for each row, optionally enriched with context logs."""
    result = []
    for i, r in enumerate(rows):
        uid      = r.get("uid", "")
        http_ctx = (uid_http or {}).get(uid)
        dns_ctx  = (uid_dns  or {}).get(uid)
        ssl_ctx  = (uid_ssl  or {}).get(uid)
        behavior_ctx = behavior_ctxs[i] if behavior_ctxs and i < len(behavior_ctxs) else None
        result.append(build_prompt(
            r["proto"], r["duration"], r["orig_pkts"], r["resp_pkts"],
            r["orig_bytes"], r["resp_bytes"], r["conn_state"], r["service"],
            resp_port=r["resp_p"], orig_port=r["orig_p"],
            http_ctx=http_ctx, dns_ctx=dns_ctx, ssl_ctx=ssl_ctx,
            behavior_ctx=behavior_ctx,
        ))
    return result


# ── Ollama inference ───────────────────────────────────────────────────────────
def _qwen_prompt(user_text):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def ollama_classify_one(prompt_text):
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": _qwen_prompt(prompt_text),
        "stream": False,
        "raw":    True,
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["response"]


def classify_ollama(prompts):
    results = []
    for i, p in enumerate(prompts):
        raw = ollama_classify_one(p)
        results.append((extract_verdict(raw), raw))
        done = i + 1
        if done % 50 == 0 or done == len(prompts):
            print(f"  {done}/{len(prompts)} classified ...", end="\r")
    return results


# ── HuggingFace inference ──────────────────────────────────────────────────────
def load_hf_model():
    print(f"Loading {ADAPTER_DIR} ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base  = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def classify_hf_batch(model, tokenizer, prompts):
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user",   "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]
    inputs    = tokenizer(texts, return_tensors="pt", padding=True,
                          truncation=True, max_length=1024)
    inputs    = {k: v.to("cuda") for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [
        (extract_verdict(t := tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()), t)
        for seq in out
    ]


def classify_hf(model, tokenizer, prompts):
    results = []
    total   = len(prompts)
    for i in range(0, total, BATCH_SIZE):
        batch   = prompts[i:i + BATCH_SIZE]
        results += classify_hf_batch(model, tokenizer, batch)
        done     = min(i + BATCH_SIZE, total)
        print(f"  {done}/{total} classified ...", end="\r")
    return results


# ── Output ────────────────────────────────────────────────────────────────────
def print_results(rows, results):
    attacks  = [(i, rows[i], raw) for i, (v, raw) in enumerate(results) if v == "ATTACK"]
    unknowns = [(i, rows[i], raw) for i, (v, raw) in enumerate(results) if v == "UNKNOWN"]
    benign_n = len(rows) - len(attacks) - len(unknowns)

    print(f"\n\n{'='*70}")
    print(f"  RESULTS: {len(rows)} connections classified")
    print(f"  Attacks:  {len(attacks)}")
    print(f"  Benign:   {benign_n}")
    print(f"  Unknown:  {len(unknowns)} (format failures)")
    print(f"{'='*70}\n")

    if attacks:
        print("── ATTACK verdicts ──────────────────────────────────────────────────────")
        for idx, r, raw_text in attacks:
            reason = next(
                (ln[7:].strip() for ln in raw_text.splitlines() if ln.upper().startswith("REASON:")),
                ""
            )
            print(
                f"  [{idx:4d}] {r['orig_h']:>40s}:{r['orig_p']:<6s} → "
                f"{r['resp_h']}:{r['resp_p']:<6s}  "
                f"{r['proto']:<5s} {r['conn_state']:<6s} "
                f"dur={r['duration']:<10s} "
                f"orig={r['orig_bytes']:<10s} resp={r['resp_bytes']}"
            )
            if reason:
                print(f"         Reason: {reason}")
        print()

    if unknowns:
        print("── UNKNOWN (format failure) ─────────────────────────────────────────────")
        for idx, r, raw_text in unknowns:
            print(
                f"  [{idx:4d}] {r['orig_h']:>40s}:{r['orig_p']:<6s} → "
                f"{r['resp_h']}:{r['resp_p']:<6s}  "
                f"{r['proto']:<5s} {r['conn_state']}"
            )
            print(f"         Raw: {raw_text[:120]}")
        print()


def print_host_results(host_summaries, results):
    attacks = [(host_summaries[i], raw) for i, (v, raw) in enumerate(results) if v == "ATTACK"]
    unknowns = [(host_summaries[i], raw) for i, (v, raw) in enumerate(results) if v == "UNKNOWN"]
    benign_n = len(host_summaries) - len(attacks) - len(unknowns)

    print(f"\n{'='*70}")
    print(f"  HOST PASS-2: {len(host_summaries)} source hosts classified")
    print(f"  Attacks:  {len(attacks)}")
    print(f"  Benign:   {benign_n}")
    print(f"  Unknown:  {len(unknowns)}")
    print(f"{'='*70}\n")

    if attacks:
        print("── HOST ATTACK verdicts ────────────────────────────────────────────────")
        for summary, raw_text in attacks:
            reason = next(
                (ln[7:].strip() for ln in raw_text.splitlines() if ln.upper().startswith("REASON:")),
                ""
            )
            print(
                f"  {summary['host']:>40s}  "
                f"flows={summary['total_flows']:<5d} "
                f"pass1_attack={summary['pred_attack']:<5d} "
                f"uniq_dst={summary['unique_dst_ips']:<4d} "
                f"ports={summary['top_ports']}"
            )
            if reason:
                print(f"         Reason: {reason}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    args       = sys.argv[1:]
    use_ollama = "--ollama" in args
    args       = [a for a in args if a != "--ollama"]
    host_pass2 = "--host-pass2" in args
    args       = [a for a in args if a != "--host-pass2"]

    limit = None
    if "--limit" in args:
        idx   = args.index("--limit")
        limit = int(args[idx + 1])
        args  = args[:idx] + args[idx + 2:]

    def _pop_arg(flag, lst):
        if flag in lst:
            idx = lst.index(flag)
            val = lst[idx + 1]
            del lst[idx:idx + 2]
            return val
        return None

    http_log_path = _pop_arg("--http-log", args)
    dns_log_path  = _pop_arg("--dns-log",  args)
    ssl_log_path  = _pop_arg("--ssl-log",  args)

    log_path = args[0] if args else CONN_LOG

    print(f"Parsing {log_path} ..." + (f" (limit {limit})" if limit else ""))
    rows = parse_conn_log(log_path, limit=limit)
    print(f"  {len(rows)} connections")
    behavior_ctxs = build_behavior_contexts(rows)
    behavior_n = sum(1 for ctx in behavior_ctxs if ctx is not None)
    print(f"  {behavior_n}/{len(rows)} flows have behavioral window context")

    uid_http = build_uid_http(http_log_path) if http_log_path else None
    uid_dns  = build_uid_dns(dns_log_path)   if dns_log_path  else None
    uid_ssl  = build_uid_ssl(ssl_log_path)   if ssl_log_path  else None

    if any(x is not None for x in (uid_http, uid_dns, uid_ssl)):
        ctx_n = sum(1 for r in rows
                    if any(d.get(r.get("uid","")) is not None
                           for d in (uid_http or {}, uid_dns or {}, uid_ssl or {})))
        print(f"  {ctx_n}/{len(rows)} flows have application-layer context")
    print()

    prompts = build_prompts(
        rows,
        uid_http=uid_http,
        uid_dns=uid_dns,
        uid_ssl=uid_ssl,
        behavior_ctxs=behavior_ctxs,
    )

    if use_ollama:
        print(f"Mode: Ollama ({OLLAMA_MODEL})")
        results = classify_ollama(prompts)
    else:
        print(f"Mode: HuggingFace ({ADAPTER_DIR})")
        model, tokenizer = load_hf_model()
        results = classify_hf(model, tokenizer, prompts)

    print_results(rows, results)

    if host_pass2 and rows:
        host_summaries = build_host_summaries(
            rows,
            results,
            behavior_ctxs=behavior_ctxs,
            uid_http=uid_http,
            uid_dns=uid_dns,
            uid_ssl=uid_ssl,
        )
        host_prompts = [build_host_prompt(s["host"], s) for s in host_summaries]

        print(f"Running host pass-2 on {len(host_prompts)} source hosts ...")
        if use_ollama:
            host_results = classify_ollama(host_prompts)
        else:
            host_results = classify_hf(model, tokenizer, host_prompts)

        print_host_results(host_summaries, host_results)
