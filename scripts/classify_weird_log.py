"""Classify Zeek weird.log entries by cross-referencing conn.log for flow stats."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from infer_utils import chat_text, load_lora_model, load_tokenizer
from prompt_utils import build_prompt, extract_verdict
from zeek_log_utils import conn_row_from_parts

# ── Config ────────────────────────────────────────────────────────────────────
ADAPTER_DIR = "./v9.1-ids-lora-adapter"
WEIRD_LOG   = "weird.log"
CONN_LOG    = "conn.log"
MAX_NEW_TOKENS = 80
BATCH_SIZE  = 8

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
            flows[parts[1]] = conn_row_from_parts(parts)
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
def classify_batch(model, tokenizer, prompts):
    texts = [chat_text(tokenizer, p) for p in prompts]
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
            flow["orig_bytes"], flow["resp_bytes"], flow["conn_state"], flow["service"],
            resp_port=flow["resp_p"], orig_port=flow["orig_p"],
        ))

    # Load model
    print(f"Loading base model + LoRA adapter from {adapter_dir} ...")
    model     = load_lora_model(adapter_dir)
    tokenizer = load_tokenizer()

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
