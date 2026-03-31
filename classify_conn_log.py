"""
classify_conn_log.py — Classify a Zeek conn.log using the v7.1 IDS classifier.

Two inference modes:
  --ollama   Use the Ollama-served ids-classifier (no GPU/transformers needed)
  (default)  Load v7.1 LoRA adapter via HuggingFace + PEFT (requires GPU)

Usage:
    .venv/bin/python classify_conn_log.py [CONN_LOG] [--ollama] [--limit N]
    .venv/bin/python classify_conn_log.py conn.log --ollama
    .venv/bin/python classify_conn_log.py conn.log --limit 500
"""

import sys
import json
import urllib.request
import urllib.error
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from prompt_utils import SYSTEM_PROMPT, build_prompt, extract_verdict

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL       = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR      = "./v8.1-ids-lora-adapter"
OLLAMA_MODEL     = "ids-classifier"
OLLAMA_URL       = "http://localhost:11434/api/generate"
CONN_LOG         = "real_conn/conn.log"
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


def build_prompts(rows):
    return [
        build_prompt(
            r["proto"], r["duration"], r["orig_pkts"], r["resp_pkts"],
            r["orig_bytes"], r["resp_bytes"], r["conn_state"], r["service"],
            resp_port=r["resp_p"], orig_port=r["orig_p"],
        )
        for r in rows
    ]


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
                          truncation=True, max_length=512)
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


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    args      = sys.argv[1:]
    use_ollama = "--ollama" in args
    args       = [a for a in args if a != "--ollama"]

    limit = None
    if "--limit" in args:
        idx   = args.index("--limit")
        limit = int(args[idx + 1])
        args  = args[:idx] + args[idx + 2:]

    log_path = args[0] if args else CONN_LOG

    print(f"Parsing {log_path} ..." + (f" (limit {limit})" if limit else ""))
    rows = parse_conn_log(log_path, limit=limit)
    print(f"  {len(rows)} connections\n")

    prompts = build_prompts(rows)

    if use_ollama:
        print(f"Mode: Ollama ({OLLAMA_MODEL})")
        results = classify_ollama(prompts)
    else:
        print(f"Mode: HuggingFace ({ADAPTER_DIR})")
        model, tokenizer = load_hf_model()
        results = classify_hf(model, tokenizer, prompts)

    print_results(rows, results)
