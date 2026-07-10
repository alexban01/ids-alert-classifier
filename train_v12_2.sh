#!/usr/bin/env bash
# v12.2 data-volume check (50% downsample) — resumable: re-run with --resume after an interrupt
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python train.py --dataset zeek_dataset_50pct.jsonl --tag v12.2 --epochs 1 --resume --save-steps 750 "$@"
