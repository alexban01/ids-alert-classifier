#!/usr/bin/env bash
# setup_ibm.sh — bootstrap an IBM Cloud gx3 L40S VSI (stock Ubuntu 24.04) and
# launch the v13.3 run. Run from the project dir after uploading it (see rsync
# line below). Idempotent: safe to rerun, e.g. after the driver-install reboot.
#
#   ./scripts/setup_ibm.sh                # full: driver + env + pre-flight + LAUNCH v13.3
#   ./scripts/setup_ibm.sh --setup-only   # stop after pre-flight, launch manually
#
# Upload from the local machine (excludes the big regenerable dirs):
#   rsync -av --exclude .venv --exclude models --exclude datasets --exclude llama.cpp \
#       ~/fine_tunning/ ubuntu@<VM_IP>:~/fine_tunning/
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_ARGS="--ibm --flash-attn --epochs 2 --dataset zeek_dataset_50pct.jsonl --tag v13.3"
# cu12+torch2.9+cp312 prebuilt wheel — the exact stack validated locally 2026-07-16
# after the cu130 build caused illegal-memory-access crashes. Ubuntu 24.04's system
# python IS 3.12, which is why no uv/pyenv is needed. L40S is SM89 (Ada): runs the
# wheel's SM80 cubins (same-major binary compat), like the local 3070 (SM86) does.
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

# ── 0. Fail on missing files BEFORE any GPU-hour is spent installing ─────────
for f in train.py ids zeek_dataset_50pct.jsonl zeek_dataset_eval.jsonl; do
    [ -e "$f" ] || { echo "MISSING: $f — rsync the project + JSONLs first (see header)"; exit 1; }
done

# ── 1. NVIDIA driver (stock Ubuntu ships none; skipped when nvidia-smi works) ─
if ! nvidia-smi >/dev/null 2>&1; then
    echo "── Installing NVIDIA driver ──"
    $SUDO apt-get update -y
    $SUDO apt-get install -y nvidia-driver-570-server \
        || $SUDO apt-get install -y nvidia-driver-550-server
    if ! nvidia-smi >/dev/null 2>&1; then
        echo ">>> Driver installed but not loaded (nouveau still owns the GPU)."
        echo ">>> Run: sudo reboot   — then rerun this script; it resumes here."
        exit 2
    fi
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── 2. Python 3.12 venv ───────────────────────────────────────────────────────
$SUDO apt-get install -y python3.12-venv
[ -d .venv ] || python3.12 -m venv .venv

# ── 3. Base-model prefetch, parallel with the big installs ───────────────────
.venv/bin/pip install -q --upgrade pip huggingface_hub
.venv/bin/python -c "from huggingface_hub import snapshot_download; \
print('model cached ->', snapshot_download('Qwen/Qwen2.5-1.5B-Instruct'))" &
HF_PID=$!

# ── 4. Pinned training stack (kept in sync with the local venv) ──────────────
.venv/bin/pip install "torch==2.9.0+cu128" --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install transformers==5.3.0 peft==0.18.1 trl==0.29.1 \
    bitsandbytes==0.49.2 datasets==4.8.4 accelerate==1.13.0 scikit-learn pandas
.venv/bin/pip install "$FA_WHEEL"
wait "$HF_PID"

# ── 5. Pre-flight: imports + real FA2 varlen fwd+bwd on THIS GPU. Abort if bad ─
.venv/bin/python - <<'EOF'
import torch, flash_attn, transformers, trl, peft, bitsandbytes, datasets, sklearn, pandas
assert torch.cuda.is_available(), "CUDA not available"
print("torch", torch.__version__, "| flash_attn", flash_attn.__version__,
      "| trl", trl.__version__, "|", torch.cuda.get_device_name(0))
from flash_attn import flash_attn_varlen_func
q  = torch.randn(2048, 12, 128, dtype=torch.bfloat16, device="cuda", requires_grad=True)
k  = torch.randn(2048,  2, 128, dtype=torch.bfloat16, device="cuda")
v  = torch.randn(2048,  2, 128, dtype=torch.bfloat16, device="cuda")
cu = torch.tensor([0, 300, 812, 1536, 2048], dtype=torch.int32, device="cuda")
flash_attn_varlen_func(q, k, v, cu, cu, 1024, 1024, causal=True).sum().backward()
torch.cuda.synchronize()
print("FA2 varlen fwd+bwd: OK")
EOF
echo "── Pre-flight passed ──"

if [ "${1:-}" = "--setup-only" ]; then
    echo "Setup complete. Launch manually with:"
    echo "  nohup .venv/bin/python train.py $TRAIN_ARGS > train.log 2>&1 &"
    exit 0
fi

# ── 6. Launch v13.3 (nohup: survives SSH disconnects) ────────────────────────
echo "── Launching v13.3 ──"
nohup .venv/bin/python train.py $TRAIN_ARGS > train.log 2>&1 &
echo "PID $! — monitor with: tail -f train.log"
echo "When done, download: models/v13.3-ids-lora-adapter/ + models/v13.3-ids-model/epoch-*/"
