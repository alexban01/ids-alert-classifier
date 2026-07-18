"""
calibrate_threshold.py — v14b logit-threshold calibration (inference-only).

Scores eval-split samples with a single forward pass per sample
(score = logit(' ATTACK') − logit(' FALSE') after a forced "VERDICT:" prefix)
and picks the threshold τ that maximizes MCC on the EVAL split — never the
benchmark (tuning on the benchmark is leakage).

Writes the result into <adapter>/run.json under "calibration"; the benchmark
picks it up via --logits.

Usage:
    .venv/bin/python scripts/calibrate_threshold.py <adapter-dir> [<adapter-dir> ...]
        [--eval zeek_dataset_eval.jsonl] [--n 6000] [--batch 24]
    .venv/bin/python scripts/calibrate_threshold.py --self-test
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def best_threshold(scores, labels):
    """Sweep all cut points; return (tau, mcc_at_tau, mcc_at_0).

    labels: 1 = ATTACK, 0 = FALSE POSITIVE. Vectorized cumulative-count MCC —
    predicting the top-k scores as ATTACK for every k, τ = midpoint between
    the k-th and (k+1)-th sorted score.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    order  = np.argsort(-scores)
    s, y   = scores[order], labels[order]
    P, N   = y.sum(), len(y) - y.sum()

    tp = np.cumsum(y)          # tp[k-1] = attacks among top-k
    fp = np.cumsum(1 - y)
    fn, tn = P - tp, N - fp
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    with np.errstate(invalid="ignore"):
        mcc = np.where(denom > 0, (tp * tn - fp * fn) / denom, 0.0)

    k = int(np.argmax(mcc))
    tau = (s[k] + s[k + 1]) / 2 if k + 1 < len(s) else s[k] - 1.0

    pred0 = (scores > 0).astype(np.float64)
    tp0 = (pred0 * labels).sum(); fp0 = (pred0 * (1 - labels)).sum()
    fn0 = P - tp0; tn0 = N - fp0
    d0 = np.sqrt((tp0 + fp0) * (tp0 + fn0) * (tn0 + fp0) * (tn0 + fn0))
    mcc0 = (tp0 * tn0 - fp0 * fn0) / d0 if d0 > 0 else 0.0
    return float(tau), float(mcc[k]), float(mcc0)


def self_test():
    from sklearn.metrics import matthews_corrcoef
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = 500
        y = rng.integers(0, 2, n)
        s = y * 1.5 + rng.normal(0, 1, n)
        tau, mcc_tau, mcc0 = best_threshold(s, y)
        assert abs(mcc0 - matthews_corrcoef(y, (s > 0).astype(int))) < 1e-9
        assert abs(mcc_tau - matthews_corrcoef(y, (s > tau).astype(int))) < 1e-9
        # tau must be at least as good as any random threshold
        for t in rng.normal(0, 1, 10):
            assert mcc_tau >= matthews_corrcoef(y, (s > t).astype(int)) - 1e-9
    print("self-test OK")


def load_eval_samples(path, n, seed=42):
    with open(path) as f:
        lines = f.readlines()
    random.seed(seed)
    if n and n < len(lines):
        lines = random.sample(lines, n)
    prompts, labels = [], []
    from ids.prompt_utils import extract_verdict
    for line in lines:
        msgs = {m["role"]: m["content"] for m in json.loads(line)["messages"]}
        v = extract_verdict(msgs["assistant"])
        if v == "UNKNOWN":
            continue
        prompts.append(msgs["user"])
        labels.append(1 if v == "ATTACK" else 0)
    return prompts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapters", nargs="*", help="adapter dir(s) with run.json")
    ap.add_argument("--eval", default="zeek_dataset_eval.jsonl")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.adapters:
        ap.error("no adapter dirs given")

    from ids.infer_utils import (chat_text, load_lora_model, load_tokenizer,
                                 resolve_system_prompt, verdict_token_ids,
                                 verdict_logit_scores, VERDICT_PREFIX)
    import torch

    prompts, labels = load_eval_samples(args.eval, args.n)
    print(f"Eval samples: {len(prompts)} ({sum(labels)} attack / "
          f"{len(labels) - sum(labels)} benign) from {args.eval}")

    tokenizer = load_tokenizer()
    atk_id, fp_id = verdict_token_ids(tokenizer)

    for adapter in args.adapters:
        system_prompt, run = resolve_system_prompt(adapter)
        model = load_lora_model(adapter)
        texts = [chat_text(tokenizer, p, system_prompt) + VERDICT_PREFIX
                 for p in prompts]
        scores = []
        for i in range(0, len(texts), args.batch):
            batch = tokenizer(texts[i:i + args.batch], return_tensors="pt",
                              padding=True, truncation=True, max_length=512).to("cuda")
            scores += verdict_logit_scores(model, batch, atk_id, fp_id)
            print(f"  {min(i + args.batch, len(texts))}/{len(texts)}...", end="\r")
        print()

        tau, mcc_tau, mcc0 = best_threshold(scores, labels)
        print(f"{adapter}: τ={tau:+.4f}  eval MCC {mcc0:+.4f} → {mcc_tau:+.4f} "
              f"(Δ{mcc_tau - mcc0:+.4f})")

        run_path = os.path.join(adapter, "run.json")
        # No run.json (pre-manifest adapter) → create a calibration-only one;
        # resolve_system_prompt still serves the default prompt for it.
        run = run or {}
        run["calibration"] = {
            "tau":             round(tau, 6),
            "eval_mcc_at_tau": round(mcc_tau, 4),
            "eval_mcc_at_0":   round(mcc0, 4),
            "n":               len(scores),
            "eval_file":       args.eval,
        }
        with open(run_path, "w") as f:
            json.dump(run, f, indent=2)
        print(f"  → written to {run_path}")

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
