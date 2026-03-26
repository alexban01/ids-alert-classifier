"""Classify Zeek weird.log entries by cross-referencing conn.log for flow stats."""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "./v4-ids-lora-adapter"
WEIRD_LOG   = "weird.log"
CONN_LOG    = "conn.log"
MAX_NEW_TOKENS = 80
BATCH_SIZE  = 8

SYSTEM_PROMPT = (
    "You are a network security analyst. "
    "Always respond with VERDICT: <ATTACK or FALSE POSITIVE> on the first line, "
    "followed by REASON: <brief explanation>."
)

# ── Prompt builder (same as preprocess_zeek.py / benchmark.py) ───────────────
def _safe(v, fmt=".1f"):
    try:
        return format(float(v), fmt) if v not in (None, "", "-", "?") else "N/A"
    except (ValueError, TypeError):
        return "N/A"

def build_prompt(proto, duration, orig_pkts, resp_pkts,
                 orig_bytes, resp_bytes, conn_state):
    try:
        dur_f = float(duration)
        ob_f  = float(orig_bytes)
        rb_f  = float(resp_bytes)
        op_f  = float(orig_pkts)
        rp_f  = float(resp_pkts)
        bps   = (ob_f + rb_f) / dur_f if dur_f > 0 else 0.0
        op_sz = ob_f / op_f if op_f > 0 else 0.0
        rp_sz = rb_f / rp_f if rp_f > 0 else 0.0
    except (ValueError, TypeError, ZeroDivisionError):
        bps = op_sz = rp_sz = 0.0

    lines = [
        "Analyze this network connection and classify it as ATTACK or FALSE POSITIVE.\n",
        f"  Proto:              {proto}",
        f"  Duration (s):       {_safe(duration, '.6f')}",
        f"  Orig Packets:       {_safe(orig_pkts, '.0f')}",
        f"  Resp Packets:       {_safe(resp_pkts, '.0f')}",
        f"  Orig Bytes:         {_safe(orig_bytes, '.0f')}",
        f"  Resp Bytes:         {_safe(resp_bytes, '.0f')}",
        f"  Conn State:         {conn_state}",
        f"  Bytes/sec:          {_safe(bps, '.1f')}",
        f"  Orig Bytes/Pkt:     {_safe(op_sz, '.1f')}",
        f"  Resp Bytes/Pkt:     {_safe(rp_sz, '.1f')}",
    ]
    return "\n".join(lines)

# ── Parse conn.log into UID lookup ────────────────────────────────────────────
def parse_conn_log(path):
    """Returns dict mapping UID -> flow record."""
    flows = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 19:
                continue
            flows[parts[1]] = {
                "proto":      parts[6],
                "duration":   parts[8],
                "orig_bytes": parts[9],
                "resp_bytes": parts[10],
                "conn_state": parts[11],
                "orig_pkts":  parts[16],
                "resp_pkts":  parts[18],
                "orig_h":     parts[2],
                "orig_p":     parts[3],
                "resp_h":     parts[4],
                "resp_p":     parts[5],
                "service":    parts[7],
            }
    return flows

# ── Parse weird.log ──────────────────────────────────────────────────────────
def parse_weird_log(path):
    # fields: ts uid id.orig_h id.orig_p id.resp_h id.resp_p name addl notice peer source
    entries = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 11:
                continue
            entries.append({
                "ts":     parts[0],
                "uid":    parts[1],
                "orig_h": parts[2],
                "orig_p": parts[3],
                "resp_h": parts[4],
                "resp_p": parts[5],
                "name":   parts[6],
                "addl":   parts[7],
                "notice": parts[8],
                "source": parts[10],
            })
    return entries

# ── Inference ─────────────────────────────────────────────────────────────────
def extract_verdict(output):
    for line in output.upper().splitlines():
        if "VERDICT:" in line:
            if "FALSE POSITIVE" in line:
                return "FALSE POSITIVE"
            if "ATTACK" in line:
                return "ATTACK"
    return "UNKNOWN"

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

    weird_path  = sys.argv[1] if len(sys.argv) > 1 else WEIRD_LOG
    conn_path   = sys.argv[2] if len(sys.argv) > 2 else CONN_LOG
    adapter_dir = sys.argv[3] if len(sys.argv) > 3 else ADAPTER_DIR

    print(f"Parsing {conn_path} ...")
    flows = parse_conn_log(conn_path)
    print(f"  {len(flows)} flows indexed by UID")

    print(f"Parsing {weird_path} ...")
    weirds = parse_weird_log(weird_path)
    print(f"  {len(weirds)} weird entries\n")

    # Match weird entries to conn.log flows
    matched   = []  # (weird_entry, flow_record)
    unmatched = []  # weird entries with no UID or no matching conn.log flow

    seen_uids = set()
    for w in weirds:
        uid = w["uid"]
        if uid == "-" or uid not in flows:
            unmatched.append(w)
            continue
        if uid in seen_uids:
            continue  # dedupe — same flow can trigger multiple weirds
        seen_uids.add(uid)
        matched.append((w, flows[uid]))

    print(f"  {len(matched)} unique flows matched (to classify)")
    print(f"  {len(unmatched)} entries without conn.log flow (skipped)\n")

    if not matched:
        print("Nothing to classify.")
        sys.exit(0)

    # Build prompts
    prompts = []
    for w, flow in matched:
        prompts.append(build_prompt(
            flow["proto"], flow["duration"], flow["orig_pkts"], flow["resp_pkts"],
            flow["orig_bytes"], flow["resp_bytes"], flow["conn_state"],
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
    verdicts = []
    total = len(matched)

    for i in range(0, total, BATCH_SIZE):
        batch_prompts = prompts[i:i + BATCH_SIZE]
        results = classify_batch(model, tokenizer, batch_prompts)
        verdicts.extend(results)
        done = min(i + BATCH_SIZE, total)
        print(f"  {done}/{total} classified ...", end="\r")

    print(f"\n\n{'='*70}")

    attacks  = []
    benign   = []
    unknowns = []
    for (w, flow), (verdict, raw_text) in zip(matched, verdicts):
        entry = (w, flow, verdict, raw_text)
        if verdict == "ATTACK":
            attacks.append(entry)
        elif verdict == "UNKNOWN":
            unknowns.append(entry)
        else:
            benign.append(entry)

    print(f"  RESULTS: {len(matched)} weird-flagged flows classified")
    print(f"  Attacks:  {len(attacks)}")
    print(f"  Benign:   {len(benign)}")
    print(f"  Unknown:  {len(unknowns)} (format failures)")
    print(f"{'='*70}\n")

    # Print all entries grouped by verdict
    if attacks:
        print("── ATTACK ──────────────────────────────────────────────────────────────")
        for w, flow, verdict, raw_text in attacks:
            reason = ""
            for line in raw_text.splitlines():
                if line.upper().startswith("REASON:"):
                    reason = line[7:].strip()
                    break
            # Collect all weird names for this UID
            weird_names = [e["name"] for e in weirds if e["uid"] == w["uid"]]
            print(
                f"  {w['orig_h']:>40s}:{w['orig_p']:<6s} → "
                f"{w['resp_h']}:{w['resp_p']:<6s} "
                f"{flow['proto']:<5s} {flow['conn_state']:<4s} "
                f"dur={flow['duration']:<12s} "
                f"orig_b={flow['orig_bytes']:<10s} resp_b={flow['resp_bytes']:<10s}"
            )
            print(f"         Weird:  {', '.join(weird_names)}")
            if reason:
                print(f"         Reason: {reason}")
            print()

    if benign:
        print("── FALSE POSITIVE ──────────────────────────────────────────────────────")
        for w, flow, verdict, raw_text in benign:
            weird_names = [e["name"] for e in weirds if e["uid"] == w["uid"]]
            print(
                f"  {w['orig_h']:>40s}:{w['orig_p']:<6s} → "
                f"{w['resp_h']}:{w['resp_p']:<6s} "
                f"{flow['proto']:<5s} {flow['conn_state']:<4s} "
                f"weird: {', '.join(weird_names)}"
            )

    if unknowns:
        print("\n── UNKNOWN (format failure) ────────────────────────────────────────────")
        for w, flow, verdict, raw_text in unknowns:
            weird_names = [e["name"] for e in weirds if e["uid"] == w["uid"]]
            print(
                f"  {w['orig_h']:>40s}:{w['orig_p']:<6s} → "
                f"{w['resp_h']}:{w['resp_p']:<6s} "
                f"{flow['proto']:<5s} weird: {', '.join(weird_names)}"
            )
            print(f"         Raw: {raw_text[:120]}")

    if unmatched:
        print(f"\n── SKIPPED ({len(unmatched)} entries — no conn.log flow) ─────────────")
        for w in unmatched:
            src = f"{w['orig_h']}:{w['orig_p']} → {w['resp_h']}:{w['resp_p']}" if w["orig_h"] != "-" else "(no endpoint)"
            print(f"  {w['name']:45s} {src}")
