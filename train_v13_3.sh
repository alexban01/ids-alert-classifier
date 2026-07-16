#!/usr/bin/env bash
# v13.3: packed + FlashAttention-2, 2 epochs, 50% data (STATE.md Next steps #4).
# Rerun of the v12.2 config with correct packing. 2 epochs because train.py now
# snapshots the adapter at every epoch end (models/v13.3-ids-model/epoch-N/):
# epoch-1 = the v12.2/v13.1 leak A/B arm (LR schedule is one cosine cycle per
# epoch, so epoch-1 is comparable to a standalone 1-epoch run), epoch-2 retests
# epoch-2-hurts under correct packing, and epoch-1+epoch-2 is the clean soup
# test v13.2 couldn't do. Soup after: scripts/soup_adapters.py epoch-1 epoch-2.
# --resume picks up the latest checkpoint in models/v13.3-ids-model;
# on a first run it just warns "no checkpoint found" and starts fresh.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python train.py --tag v13.3 --epochs 2 --flash-attn --save-steps 1000 \
    --dataset zeek_dataset_50pct.jsonl --resume "$@"
