#!/usr/bin/env bash
# v13.2: 2-epoch no-pack run on 50% data (STATE.md Next steps #4).
# --resume picks up the latest checkpoint in models/v13.2-ids-model;
# on a first run it just warns "no checkpoint found" and starts fresh.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python train.py --tag v13.2 --epochs 2 --no-pack --save-steps 1000 \
    --dataset zeek_dataset_50pct.jsonl --resume "$@"
