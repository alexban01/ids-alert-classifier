"""Classify a Zeek conn.log using the fine-tuned LoRA adapter."""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from prompt_utils import SYSTEM_PROMPT, build_prompt, extract_verdict

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "./v4-ids-lora-adapter"
CONN_LOG    = "conn.log"
MAX_NEW_TOKENS = 80
BATCH_SIZE  = 8

# ── Zeek conn.log field indices (tab-separated) ──────────────────────────────
#  0:ts  1:uid  2:id.orig_h  3:id.orig_p  4:id.resp_h  5:id.resp_p
#  6:proto  7:service  8:duration  9:orig_bytes  10:resp_bytes
# 11:conn_state  12:local_orig  13:local_resp  14:missed_bytes  15:history
# 16:orig_pkts  17:orig_ip_bytes  18:resp_pkts  19:resp_ip_bytes
# 20:tunnel_parents  21:ip_proto
F_PROTO      = 6
F_DURATION   = 8
F_ORIG_BYTES = 9
F_RESP_BYTES = 10
F_CONN_STATE = 11
F_ORIG_PKTS  = 16
F_RESP_PKTS  = 18

# ── Parse conn.log ────────────────────────────────────────────────────────────
def parse_conn_log(path):
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
                "service":    parts[7],
                "duration":   parts[F_DURATION],
                "orig_bytes": parts[F_ORIG_BYTES],
                "resp_bytes": parts[F_RESP_BYTES],
                "conn_state": parts[F_CONN_STATE],
                "orig_pkts":  parts[F_ORIG_PKTS],
                "resp_pkts":  parts[F_RESP_PKTS],
            })
    return rows

# ── Inference ─────────────────────────────────────────────────────────────────
def classify_batch(model, tokenizer, prompts):
    texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": p},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for seq in out:
        text = tokenizer.decode(seq[input_len:], skip_special_tokens=True).strip()
        results.append((extract_verdict(text), text))
    return results

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True

    log_path    = sys.argv[1] if len(sys.argv) > 1 else CONN_LOG
    adapter_dir = sys.argv[2] if len(sys.argv) > 2 else ADAPTER_DIR

    print(f"Parsing {log_path} ...")
    rows = parse_conn_log(log_path)
    print(f"  {len(rows)} connections\n")

    # Build prompts
    prompts = []
    for r in rows:
        prompts.append(build_prompt(
            r["proto"], r["duration"], r["orig_pkts"], r["resp_pkts"],
            r["orig_bytes"], r["resp_bytes"], r["conn_state"], r["service"],
        ))

    # Load model
    print(f"Loading base model + LoRA adapter from {adapter_dir} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="cuda",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Classify in batches
    attacks  = []
    unknowns = []
    total    = len(rows)

    for i in range(0, total, BATCH_SIZE):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        results = classify_batch(model, tokenizer, batch_prompts)

        for j, (verdict, raw_text) in enumerate(results):
            idx = i + j
            r   = rows[idx]
            if verdict == "ATTACK":
                attacks.append((idx, r, raw_text))
            elif verdict == "UNKNOWN":
                unknowns.append((idx, r, raw_text))

        done = min(i + BATCH_SIZE, total)
        print(f"  {done}/{total} classified ...", end="\r")

    print(f"\n\n{'='*70}")
    print(f"  RESULTS: {len(rows)} connections classified")
    print(f"  Attacks:  {len(attacks)}")
    print(f"  Benign:   {total - len(attacks) - len(unknowns)}")
    print(f"  Unknown:  {len(unknowns)} (format failures)")
    print(f"{'='*70}\n")

    if attacks:
        print("── ATTACK verdicts ─────────────────────────────────────────────────────")
        for idx, r, raw_text in attacks:
            reason = ""
            for line in raw_text.splitlines():
                if line.upper().startswith("REASON:"):
                    reason = line[7:].strip()
                    break
            print(
                f"  [{idx:4d}] {r['orig_h']:>40s}:{r['orig_p']:<6s} → "
                f"{r['resp_h']}:{r['resp_p']:<6s} "
                f"{r['proto']:<5s} {r['conn_state']:<4s} "
                f"dur={r['duration']:<12s} "
                f"orig_b={r['orig_bytes']:<10s} resp_b={r['resp_bytes']:<10s}"
            )
            if reason:
                print(f"         Reason: {reason}")
        print()

    if unknowns:
        print("── UNKNOWN (format failure) ────────────────────────────────────────────")
        for idx, r, raw_text in unknowns:
            print(
                f"  [{idx:4d}] {r['orig_h']:>40s}:{r['orig_p']:<6s} → "
                f"{r['resp_h']}:{r['resp_p']:<6s} "
                f"{r['proto']:<5s} {r['conn_state']:<4s}"
            )
            print(f"         Raw: {raw_text[:120]}")
        print()
