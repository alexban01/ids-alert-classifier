"""
preprocess_downsample.py — Stratified (by class) random downsample of an already-built
training JSONL, for fast data-volume ablations.

Deliberately does NOT re-run the loader/composition pipeline (preprocess_zeek.py):
regenerating with a lower TRAINING_FACTOR changes per-source pool caps, not just volume,
and risks re-skewing source composition (the exact bug that caused V11's Win7AD-1
regression — see STATE.md). Downsampling the final messages-only JSONL keeps composition
intact by construction (uniform random draw within each class preserves source mix with
negligible noise at these bucket sizes) at the cost of not being able to print/verify an
exact post-downsample per-source breakdown (source labels aren't retained in the final
JSONL — only "messages" is written by preprocess_zeek.py).

Only downsamples the train file. Eval stays full-size (already subsampled at train time
via --eval-subset, and using the same eval set for both runs keeps eval_loss comparable).
"""
import argparse
import json
import random


def verdict_of(line: str) -> str:
    obj = json.loads(line)
    content = obj["messages"][-1]["content"]
    return "ATTACK" if content.startswith("VERDICT: ATTACK") else "FALSE POSITIVE"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="zeek_dataset.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frac", type=float, required=True,
                         help="Fraction of each class to keep, e.g. 0.5 for 50%%.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input) as f:
        lines = f.readlines()

    by_class = {"ATTACK": [], "FALSE POSITIVE": []}
    for line in lines:
        by_class[verdict_of(line)].append(line)

    rng = random.Random(args.seed)
    kept = []
    for cls, cls_lines in by_class.items():
        rng.shuffle(cls_lines)
        n_keep = round(len(cls_lines) * args.frac)
        kept.extend(cls_lines[:n_keep])
        print(f"{cls}: {len(cls_lines)} -> {n_keep}")

    rng.shuffle(kept)  # re-mix classes so packing/batches aren't class-sorted
    with open(args.output, "w") as f:
        f.writelines(kept)
    print(f"Wrote {len(kept)} samples ({100*args.frac:.0f}% of {len(lines)}) to {args.output}")


if __name__ == "__main__":
    main()
