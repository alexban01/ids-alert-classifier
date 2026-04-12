#!/bin/bash
# Run this on the RunPod pod after SSH/terminal access.
# Assumes the Runpod Pytorch 2.8.0 template (CUDA 12.8, Ubuntu 24.04).
# Target GPU: RTX 5090 (32 GB VRAM)
#
# Usage:
#   1. Upload zeek_dataset.jsonl + zeek_dataset_eval.jsonl + train.py to /workspace/
#   2. Run: bash setup_runpod.sh
#   3. Run: python train.py --runpod
#   4. Download v12-ids-lora-adapter/ when done

set -e

echo "── Installing PyTorch (cu128, force-reinstall to override any mismatched build) ──"
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall

echo ""
echo "── Installing training dependencies ──"
pip install --upgrade \
    transformers \
    peft \
    trl \
    bitsandbytes \
    datasets \
    accelerate \
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
