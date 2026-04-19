"""
merge_adapter.py — Merge a LoRA adapter into the base model for GGUF conversion.

LoRA adapters can't be converted to GGUF directly — llama.cpp needs a full
model directory with config.json. This script merges adapter weights into the
base model and saves the result, ready for convert_hf_to_gguf.py.

Note: loads in fp16 (no 4-bit), requires ~3 GB VRAM or runs on CPU.

Usage:
    .venv/bin/python merge_adapter.py [ADAPTER_DIR] [--out OUT_DIR] [--cpu]

    ADAPTER_DIR   LoRA adapter directory (default: v6-ids-lora-adapter)
    --out OUT_DIR Output directory for merged model (default: <adapter>-merged)
    --cpu         Force CPU (if VRAM is tight)

Examples:
    .venv/bin/python merge_adapter.py
    .venv/bin/python merge_adapter.py v4-ids-lora-adapter
    .venv/bin/python merge_adapter.py v6-ids-lora-adapter --out v6-ids-merged

Then convert:
    llama.cpp/.venv/bin/python llama.cpp/convert_hf_to_gguf.py v6-ids-merged/ --outfile v6-ids.gguf
    ollama create ids-classifier -f Modelfile
"""

import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# ── Parse args ─────────────────────────────────────────────────────────────────
args        = sys.argv[1:]
adapter_dir = "v6-ids-lora-adapter"
out_dir     = None
use_cpu     = False
i = 0
while i < len(args):
    if args[i] == "--out" and i + 1 < len(args):
        out_dir = args[i + 1]; i += 2
    elif args[i] == "--cpu":
        use_cpu = True; i += 1
    elif not args[i].startswith("--"):
        adapter_dir = args[i]; i += 1
    else:
        i += 1

if out_dir is None:
    out_dir = adapter_dir.rstrip("/") + "-merged"

# ── Validate ───────────────────────────────────────────────────────────────────
if not os.path.isdir(adapter_dir):
    print(f"[ERROR] Adapter directory not found: {adapter_dir}")
    sys.exit(1)

if not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
    print(f"[ERROR] No adapter_config.json in {adapter_dir} — is this a LoRA adapter?")
    sys.exit(1)

device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Adapter : {adapter_dir}")
print(f"Output  : {out_dir}")
print(f"Device  : {device}")
print(f"Base    : {BASE_MODEL}")
print()

# ── Load base model in fp16 (NOT 4-bit — can't merge quantized weights) ────────
print("Loading base model in fp16 ...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map=device,
    low_cpu_mem_usage=True,
)

# ── Load adapter ───────────────────────────────────────────────────────────────
print(f"Loading adapter from {adapter_dir} ...")
model = PeftModel.from_pretrained(model, adapter_dir)

# ── Merge and unload ───────────────────────────────────────────────────────────
print("Merging LoRA weights into base model ...")
model = model.merge_and_unload()
model.eval()

# ── Save merged model ──────────────────────────────────────────────────────────
print(f"Saving merged model to {out_dir} ...")
os.makedirs(out_dir, exist_ok=True)
model.save_pretrained(out_dir)

# Save tokenizer (convert_hf_to_gguf.py needs it in the same dir)
print("Saving tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(out_dir)

print(f"\n✅ Done — merged model saved to {out_dir}/")
print(f"\nNext steps:")
print(f"  llama.cpp/.venv/bin/python llama.cpp/convert_hf_to_gguf.py {out_dir}/ --outfile v6-ids.gguf")
print(f"  ollama create ids-classifier -f Modelfile")
print(f"  ollama run ids-classifier")
