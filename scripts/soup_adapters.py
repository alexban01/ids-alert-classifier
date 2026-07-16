#!/usr/bin/env python
"""Model soup: weighted average of two LoRA adapters.

Usage:
    .venv/bin/python scripts/soup_adapters.py <adapter_a> <adapter_b> -o <out_dir> [-w 0.5]

-w is the weight of the FIRST adapter (default 0.5 = plain average).
Config/tokenizer files are copied from the first adapter; run.json (needed by
resolve_system_prompt for --no-reason runs) is copied from whichever dir has one.
"""
import argparse
import json
import os
import shutil

from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser()
parser.add_argument("adapters", nargs=2, help="two LoRA adapter dirs")
parser.add_argument("-o", "--out", required=True, help="output adapter dir")
parser.add_argument("-w", "--weight", type=float, default=0.5,
                    help="weight of the first adapter (default 0.5)")
args = parser.parse_args()

a_dir, b_dir = args.adapters
wa, wb = args.weight, 1.0 - args.weight

ta = load_file(os.path.join(a_dir, "adapter_model.safetensors"))
tb = load_file(os.path.join(b_dir, "adapter_model.safetensors"))
assert ta.keys() == tb.keys(), "adapter key mismatch — different LoRA configs?"
# ponytail: averages A/B factors directly (not the BA product) — standard practice, worked for v11
souped = {k: (wa * ta[k].float() + wb * tb[k].float()).to(ta[k].dtype) for k in ta}

os.makedirs(args.out, exist_ok=True)
save_file(souped, os.path.join(args.out, "adapter_model.safetensors"))

# Sidecar files: configs/tokenizer from A; run.json from whichever has one.
for f in ("adapter_config.json", "tokenizer.json", "tokenizer_config.json",
          "chat_template.jinja", "README.md"):
    src = os.path.join(a_dir, f)
    if os.path.exists(src):
        shutil.copy(src, args.out)
for d in (a_dir, b_dir):
    src = os.path.join(d, "run.json")
    if os.path.exists(src):
        shutil.copy(src, args.out)
        break

with open(os.path.join(args.out, "soup.json"), "w") as fh:
    json.dump({"ingredients": [a_dir, b_dir], "weights": [wa, wb]}, fh, indent=2)

print(f"✅ soup saved to {args.out}  ({wa:g}×{a_dir} + {wb:g}×{b_dir})")
