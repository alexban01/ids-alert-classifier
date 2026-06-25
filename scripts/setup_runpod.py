#!/usr/bin/env python3
"""
setup_runpod.py — single-file parallel bootstrap for V12 training on RunPod.

Replaces setup_runpod.sh. The two slow startup costs are independent:
  (A) installing the pinned training deps, and
  (B) downloading the ~3 GB base model from the HF Hub.
This runs them CONCURRENTLY and uses `uv` (parallel resolver/installer) for (A)
instead of pip, so the GPU starts training as soon as possible.

Usage on the pod (Runpod Pytorch 2.8.0 template; assumes the two JSONLs are in
/workspace):
    python setup_runpod.py                 # set up only
    python setup_runpod.py --train         # set up, then run train.py --runpod
    python setup_runpod.py --train --torch29   # also force torch 2.9.0+cu128

Why --torch29: torch 2.10/2.11 have incomplete SM_120 (Blackwell) support and fall
back to unoptimized kernels on the RTX 5090 → OOM at batch=12+. Use 2.9.0+cu128.
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time

# Pinned to the local versions (kept in sync with requirements.txt).
DEPS = [
    "transformers==5.3.0",
    "peft==0.18.1",
    "trl==0.29.1",
    "bitsandbytes==0.49.2",
    "datasets==4.8.4",
    "accelerate==1.13.0",
    "scikit-learn",
]
TORCH = ["torch==2.9.0+cu128", "torchvision", "torchaudio"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

ROOT = "/workspace" if os.path.isdir("/workspace") else os.getcwd()
DATASETS = ["zeek_dataset.jsonl", "zeek_dataset_eval.jsonl"]


def _stream(tag, cmd, env=None):
    """Run cmd, prefix every output line with [tag elapsed]. Return (rc, secs)."""
    t0 = time.time()
    print(f"[{tag}    0s] $ {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1, env=env)
    for line in p.stdout:
        print(f"[{tag} {time.time() - t0:4.0f}s] {line}", end="", flush=True)
    p.wait()
    return p.returncode, time.time() - t0


def _bootstrap():
    """Serially install the tiny tools the two parallel tasks need (uv + hub).

    Done up front so the concurrent tasks never write site-packages at the same
    time: after this, (A) only touches site-packages and (B) only touches the HF
    cache — disjoint, so they can't clash.
    """
    print("── Bootstrapping uv + huggingface_hub (fast) ──", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv", "huggingface_hub"],
                   check=True)


def _install_deps(torch29):
    uv = shutil.which("uv")
    if uv:
        env = {**os.environ, "UV_SYSTEM_PYTHON": "1"}
        base = [uv, "pip", "install", "--python", sys.executable]
        reinstall = ["--reinstall"]
    else:  # uv bootstrap somehow failed — fall back to pip
        env = os.environ.copy()
        base = [sys.executable, "-m", "pip", "install"]
        reinstall = ["--force-reinstall"]
    if torch29:
        rc, _ = _stream("deps", base + reinstall + TORCH + ["--index-url", TORCH_INDEX], env)
        if rc != 0:
            return rc
    return _stream("deps", base + DEPS, env)[0]


def _prefetch_model():
    # Pure download into the HF cache; no pip, so it can't clash with the deps task.
    code = (f"from huggingface_hub import snapshot_download; "
            f"p = snapshot_download('{BASE_MODEL}'); print('cached ->', p)")
    return _stream("model", [sys.executable, "-c", code])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true",
                    help="After setup, launch train.py --runpod immediately.")
    ap.add_argument("--torch29", action="store_true",
                    help="Force-reinstall torch 2.9.0+cu128 (RTX 5090 / Blackwell).")
    args = ap.parse_args()

    t0 = time.time()
    _bootstrap()

    # ── Run the two slow halves in parallel ───────────────────────────────────
    results = {}
    tasks = {
        "deps":  lambda: _install_deps(args.torch29),
        "model": _prefetch_model,
    }
    threads = {name: threading.Thread(target=lambda n=name, f=fn: results.__setitem__(n, f()))
               for name, fn in tasks.items()}
    for th in threads.values():
        th.start()
    for th in threads.values():
        th.join()

    failed = [n for n, rc in results.items() if rc != 0]
    if failed:
        sys.exit(f"\n✗ Setup failed: {', '.join(failed)} (see [tag] output above)")

    # ── Verify GPU + datasets ─────────────────────────────────────────────────
    print("\n── Verifying GPU ──", flush=True)
    subprocess.run([sys.executable, "-c",
                    "import torch;print('GPU:',torch.cuda.get_device_name(0));"
                    "print('VRAM: %.1f GB'%(torch.cuda.get_device_properties(0).total_memory/1024**3));"
                    "print('CUDA:',torch.version.cuda)"], check=True)

    print("\n── Verifying datasets ──", flush=True)
    missing = [d for d in DATASETS if not os.path.isfile(os.path.join(ROOT, d))]
    if missing:
        sys.exit(f"✗ Missing in {ROOT}: {', '.join(missing)} — upload them first.")
    for d in DATASETS:
        path = os.path.join(ROOT, d)
        with open(path) as f:
            n = sum(1 for _ in f)
        print(f"  {d}: {n:,} samples")

    print(f"\n✓ Ready in {time.time() - t0:.0f}s (deps + model fetched in parallel).", flush=True)

    if args.train:
        print("\n── Launching training ──", flush=True)
        sys.exit(subprocess.run([sys.executable, "train.py", "--runpod"], cwd=ROOT).returncode)
    else:
        print(f"Run:  cd {ROOT} && python train.py --runpod")


if __name__ == "__main__":
    main()
