#!/bin/bash
# Run this on the RunPod pod after SSH/terminal access.
# Assumes the Runpod Pytorch 2.8.0 template (CUDA 12.8, Ubuntu 24.04).
# Target GPU: RTX 5090 (32 GB VRAM)
#
# Usage:
#   1. Upload zeek_dataset.jsonl + zeek_dataset_eval.jsonl + train.py to /workspace/
#   2. Run: bash setup_runpod.sh [--torch29]
#   3. Run: python train.py --runpod
#   4. Download v12-ids-lora-adapter/ when done
#
# Flags:
#   --torch29   Force-reinstall torch 2.9.0+cu128 instead of using the template's 2.8.0.
#               Use if the template torch causes OOM or kernel errors. torch 2.10/2.11
#               have incomplete SM_120 (Blackwell) support and fall back to unoptimized
#               kernel paths on RTX 5090, causing OOM at batch=12+.

set -e

TORCH29=0
for arg in "$@"; do
    case $arg in
        --torch29) TORCH29=1 ;;
    esac
done

if [ "$TORCH29" = "1" ]; then
    echo "── Installing PyTorch 2.9.0+cu128 (--torch29) ──"
    pip install torch==2.9.0+cu128 torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 \
        --force-reinstall
    echo ""
else
    echo "── Using template PyTorch 2.8.0 (pass --torch29 to override) ──"
    echo ""
fi

echo "── Installing training dependencies (pinned to local versions) ──"
pip install \
    transformers==5.3.0 \
    peft==0.18.1 \
    trl==0.29.1 \
    bitsandbytes==0.49.2 \
    datasets==4.8.4 \
    accelerate==1.13.0 \
    scikit-learn

echo ""
echo "── Checking GPU ──"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB'); print(f'CUDA: {torch.version.cuda}')"

echo ""
echo "── Checking dataset ──"
if [ -f /workspace/zeek_dataset.jsonl ]; then
    LINES=$(wc -l < /workspace/zeek_dataset.jsonl)
    echo "zeek_dataset.jsonl: $LINES samples"
else
    echo "ERROR: zeek_dataset.jsonl not found in /workspace/"
    echo "Upload it before running train.py"
    exit 1
fi
if [ -f /workspace/zeek_dataset_eval.jsonl ]; then
    LINES=$(wc -l < /workspace/zeek_dataset_eval.jsonl)
    echo "zeek_dataset_eval.jsonl: $LINES samples"
else
    echo "ERROR: zeek_dataset_eval.jsonl not found in /workspace/"
    exit 1
fi

echo ""
echo "── Ready ──"
echo "Run:  cd /workspace && python train.py --runpod"
echo "Then: scp -P <port> -i ~/.ssh/runpod root@<host>:/workspace/v12-ids-lora-adapter/ ."
