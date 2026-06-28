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

from ids.behavior_features import build_behavior_contexts, build_host_summaries
from ids.infer_utils import (BASE_MODEL, chat_text, load_lora_model, load_tokenizer,
                             resolve_system_prompt)
from ids.prompt_utils import (SYSTEM_PROMPT, SYSTEM_PROMPT_VERDICT_ONLY,
                              build_prompt, build_host_prompt, extract_verdict)
from ids.zeek_log_utils import (
    build_dns_lookup,
    build_http_lookup,
    build_ssl_lookup,
    conn_row_from_parts,
    parse_zeek_log,
)

# ── Config ────────────────────────────────────────────────────────────────────
ADAPTER_DIR      = "./models/v9.1-ids-lora-adapter"
OLLAMA_MODEL     = "ids-classifier"
OLLAMA_URL       = "http://localhost:11434/api/generate"
CONN_LOG         = "real_conn/conn.log.5"
MAX_NEW_TOKENS   = 80
BATCH_SIZE       = 8

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
            rows.append(conn_row_from_parts(parts, with_uid=True))
            if limit and len(rows) >= limit:
                break
    return rows


# ── Auxiliary Zeek log parsers (shared builders + a count print) ───────────────

def build_uid_http(http_log_path):
    lookup = build_http_lookup(http_log_path)
    print(f"  http.log: {len(lookup)} uid entries")
    return lookup


def build_uid_dns(dns_log_path):
    lookup = build_dns_lookup(dns_log_path)
    print(f"  dns.log:  {len(lookup)} uid entries")
    return lookup


def build_uid_ssl(ssl_log_path):
    lookup = build_ssl_lookup(ssl_log_path)
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
def _qwen_prompt(user_text, system_prompt=SYSTEM_PROMPT):
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def ollama_classify_one(prompt_text, system_prompt=SYSTEM_PROMPT):
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": _qwen_prompt(prompt_text, system_prompt),
        "stream": False,
        "raw":    True,
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["response"]


def classify_ollama(prompts, system_prompt=SYSTEM_PROMPT):
    results = []
    for i, p in enumerate(prompts):
        raw = ollama_classify_one(p, system_prompt)
        results.append((extract_verdict(raw), raw))
        done = i + 1
        if done % 50 == 0 or done == len(prompts):
            print(f"  {done}/{len(prompts)} classified ...", end="\r")
    return results


# ── HuggingFace inference ──────────────────────────────────────────────────────
def load_hf_model():
    print(f"Loading {ADAPTER_DIR} ...")
    model     = load_lora_model(ADAPTER_DIR)
    tokenizer = load_tokenizer()
    return model, tokenizer


def classify_hf_batch(model, tokenizer, prompts, system_prompt=SYSTEM_PROMPT):
    texts = [chat_text(tokenizer, p, system_prompt) for p in prompts]
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


def classify_hf(model, tokenizer, prompts, system_prompt=SYSTEM_PROMPT):
    results = []
    total   = len(prompts)
    for i in range(0, total, BATCH_SIZE):
        batch   = prompts[i:i + BATCH_SIZE]
        results += classify_hf_batch(model, tokenizer, batch, system_prompt)
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

    args         = sys.argv[1:]
    use_ollama   = "--ollama" in args
    args         = [a for a in args if a != "--ollama"]
    host_pass2   = "--host-pass2" in args
    args         = [a for a in args if a != "--host-pass2"]
    # Ollama models have no run.json to auto-detect from, so the verdict-only
    # prompt is opt-in there; the HF path auto-detects from the adapter's run.json.
    verdict_only = "--verdict-only" in args
    args         = [a for a in args if a != "--verdict-only"]

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
        system_prompt = SYSTEM_PROMPT_VERDICT_ONLY if verdict_only else SYSTEM_PROMPT
        kind = "verdict-only" if verdict_only else "VERDICT+REASON"
        print(f"Mode: Ollama ({OLLAMA_MODEL}) — {kind} system prompt"
              + ("" if verdict_only else "  (pass --verdict-only for a --no-reason model)"))
        results = classify_ollama(prompts, system_prompt)
    else:
        # Auto-detect the matching prompt from the adapter's run.json (verdict-only
        # for a --no-reason model); falls back to default if there's no manifest.
        system_prompt, run_info = resolve_system_prompt(ADAPTER_DIR)
        kind = "verdict-only" if "REASON" not in system_prompt else "VERDICT+REASON"
        src  = "run.json" if run_info else "default (no run.json)"
        print(f"Mode: HuggingFace ({ADAPTER_DIR}) — {kind} system prompt [{src}]")
        model, tokenizer = load_hf_model()
        results = classify_hf(model, tokenizer, prompts, system_prompt)

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
            host_results = classify_ollama(host_prompts, system_prompt)
        else:
            host_results = classify_hf(model, tokenizer, host_prompts, system_prompt)

        print_host_results(host_summaries, host_results)
