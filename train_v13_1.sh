#!/usr/bin/env bash
# v13.1 unpacked arm (r=16, --no-pack, 1 epoch, 50% data) — resumable: re-run with --resume after an interrupt
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python train.py --dataset zeek_dataset_50pct.jsonl --tag v13.1 --epochs 1 --no-pack --resume --save-steps 750 "$@"
