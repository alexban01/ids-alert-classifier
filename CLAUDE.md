# IDS Alert Classifier — Fine-Tuning Project

Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` via QLoRA to classify network flows as
**ATTACK** or **FALSE POSITIVE**, targeting deployment against Zeek conn.log / PCAP captures.

**Local hardware:** Ryzen 7 3700X, 32 GB RAM, RTX 3070 (8 GB VRAM)
**Training:** RunPod RTX 5090 (32 GB VRAM), ~$0.44/hr on-demand

## Keeping project state current

The authoritative project snapshot lives in the `ids-project.skill` skill at
`references/current-state.md`. **Whenever a big change happens — a training run
started/finished, a new benchmark result, a dataset added/removed/restored, an
architecture or objective decision — update that file in the same session.** Keep it
concise (update in place, no changelog; git history is the changelog).

## Python Environment

Arch Linux managed environment — system `python3`/`pip3` refuse to install packages.

- Project: `.venv/bin/python` and `.venv/bin/pip`
- llama.cpp: `llama.cpp/.venv/bin/python` (separate venv)

## Project Structure

```
├── train.py                  # QLoRA fine-tuning via SFTTrainer
├── preprocess_zeek.py        # Build training JSONL from all sources
├── preprocess_config.py      # Caps, ratio targets, masking probs, reason pools
├── preprocess_sample.py      # score_hard_benign(), make_sample(), pick_reason()
├── prompt_utils.py           # Shared build_prompt, SYSTEM_PROMPT, extract_verdict
├── infer_utils.py            # Shared 4-bit+LoRA model load, tokenizer, chat templating
├── behavior_features.py      # Behavioral context features for enriched prompts
├── zeek_log_utils.py         # Zeek TSV parser + conn.log row helper + CTU-Malware helpers
├── Modelfile                 # Ollama config — Qwen2.5 chat template, temperature 0
├── requirements.txt
│
├── loaders/                  # Dataset loaders (imported by preprocess_zeek.py)
│   ├── loader_iot23.py       #   IoT-23 conn.log.labeled (tar.gz)
│   ├── loader_ctu13.py       #   CTU-13 binetflow (tar.bz2)
│   ├── loader_unsw.py        #   UNSW-NB15 parquet / CSV
│   ├── loader_cicids.py      #   CICIDS2017 CICFlowMeter CSVs (disabled in v7+)
│   ├── loader_uwf.py         #   UWF-ZeekData24 Spark CSV
│   ├── loader_ctu_normal.py  #   CTU-Normal benign Zeek conn.log
│   └── loader_ctu_malware.py #   CTU-Malware-Capture multi-log enriched samples
│
├── benchmarks/               # Benchmark scripts
│   ├── benchmark_realworld.py  # Primary: real Zeek sources (IoT-23, CTU-13, UWF, CTU-Normal)
│   ├── benchmark_v6.py         # Historical: CICIDS2017 comparison
│   ├── benchmark_ollama.py     # Ollama HTTP API benchmark (no GPU needed)
│   └── bench_loaders.py        # Data loaders for benchmark_realworld.py
│
├── scripts/                  # Utility & deployment scripts
│   ├── classify_conn_log.py  #   Classify a real Zeek conn.log
│   ├── classify_weird_log.py #   Classify weird.log via conn.log cross-reference
│   ├── compare_binetflow.py  #   Cross-reference conn.log vs binetflow ground truth
│   ├── merge_adapter.py      #   Merge LoRA adapter into base model for GGUF
│   ├── analyze_gap.py        #   Distribution gap analysis
│   ├── baseline_ml.py        #   Random Forest / Logistic Regression baseline
│   ├── setup_runpod.py       #   RunPod pod setup (5090) — parallel deps+model, single-file
│   ├── setup_runpod.sh       #   RunPod pod setup (5090) — legacy bash, superseded by .py
│   └── setup_runpod_4090.sh  #   RunPod pod setup (4090)
│
├── tests/                    # Test files
│   ├── test_behavior_features.py
│   └── test_novel_cases.py
│
├── notes/                    # Research notes & thesis notes
├── results/                  # Benchmark reports & result JSONs
├── real_conn/                # Real Zeek conn.log files for testing
├── datasets/                 # Raw datasets (gitignored)
└── llama.cpp/                # External tool (gitignored)
```

**Gitignored (regenerable):**
- `datasets/` — IoT-23, CTU-13, UNSW-NB15, UWF-ZeekData24, CTU-Normal
- `*.jsonl` — train/eval splits from preprocess_zeek.py
- `v*-ids-model/`, `v*-ids-lora-adapter/` — training checkpoints & adapters
- `*-ids-lora-adapter-merged/`, `*.gguf` — merged models for Ollama
- `llama.cpp/`, `test_captures/`

## Model Architecture

- **Base:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Quantization:** 4-bit NF4 (BitsAndBytes), bf16 compute dtype
- **LoRA:** r=32, lora_alpha=64, dropout=0.05, bias=none
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (7 modules — all attention + MLP)

## Training

Run on RunPod pod (single command — installs deps + prefetches base model in
parallel via `uv`, then trains):

```bash
python scripts/setup_runpod.py --train            # add --torch29 on RTX 5090/Blackwell
```

`setup_runpod.py` supersedes `setup_runpod.sh`. Use `--train` to chain straight into
`train.py --runpod`; omit it to set up only. Upload `zeek_dataset.jsonl` +
`zeek_dataset_eval.jsonl` to `/workspace` first.

Train and eval datasets are pre-split by `preprocess_zeek.py` using source-stratified
sampling (10% per source/class bucket) — `zeek_dataset.jsonl` + `zeek_dataset_eval.jsonl`.

**RTX 5090 (RunPod, 24 GB VRAM) — recommended:**
```python
per_device_train_batch_size = 24
per_device_eval_batch_size  = 24
gradient_accumulation_steps = 1     # effective batch = 24
```

**RTX 3070 (local, 8 GB VRAM) — set TRAINING_FACTOR in preprocess_zeek.py first:**
```python
per_device_train_batch_size = 4
per_device_eval_batch_size  = 4
gradient_accumulation_steps = 6     # effective batch = 24
```

**Shared settings:**
```python
optim = "paged_adamw_8bit"
num_train_epochs = 2                 # --epochs (default 2); load_best keeps best eval_loss
packing = True                       # --no-pack to disable; ~20% cheaper/epoch, objective unchanged
learning_rate = 2e-4
lr_scheduler_type = "cosine_with_restarts"
lr_scheduler_kwargs = {"num_cycles": EPOCHS}   # one cosine restart per epoch
warmup_ratio = 0.03
weight_decay = 0.01
bf16 = True
gradient_checkpointing = True        # RunPod path sets this False (32 GB headroom)
eval_strategy = "epoch"
load_best_model_at_end = True
metric_for_best_model = "eval_loss"
save_strategy = "epoch"
save_total_limit = 2
max_length = 512
logging_steps = 100
dataloader_num_workers = 0
dataloader_pin_memory = True
```

**Cost knobs** (RunPod RTX 5090, ~$0.44/hr — cost = GPU-hours): training set is
~241k samples, token lengths mean 296 / p99 500 (so `max_length=512` is right and
not a lever). To train cheaper: `packing=True` (default) cuts opt-steps to 0.69×
and ~20% of tokens/epoch, `--epochs 2` saves ~33% vs 3, `--eval-subset 6000` trims
eval forward work to ~0.19×, and `TRAINING_FACTOR` in `preprocess_config.py` scales
the dataset (and cost) linearly at the expense of coverage. Measured (real tokenizer
+ bin-packing, no training): combined default run (2 epochs + packing) ≈ **0.53×**
the train compute of the old 3-epoch unpacked run — roughly **half** the GPU-hours.

**Inference runs locally on RTX 3070** — adapter is hardware-agnostic.

## Preprocessing

Run: `.venv/bin/python preprocess_zeek.py`

Produces `zeek_dataset.jsonl` (train) and `zeek_dataset_eval.jsonl` (eval) using a
source-stratified 90/10 split — 10% held out per (source, class) bucket.

**TRAINING_FACTOR** (default 1.0): set to 0.1 in preprocess_zeek.py for fast local
validation runs.

### Dataset sources

| Source | Format | Path | Label logic |
|---|---|---|---|
| IoT-23 | Zeek `conn.log.labeled` in tar.gz | `datasets/iot-23/` | `"Malicious"` → ATTACK, `"Benign"` → FP |
| CTU-13 | Binetflow CSV in tar.bz2 | `datasets/ctu-13/` | `"Botnet"` → ATTACK, `"Normal"` → FP |
| UNSW-NB15 | Parquet (HuggingFace) | `datasets/unsw-nb15/` | `binary_label=1` → ATTACK, `0` → FP |
| UWF-ZeekData24 | CSV (Spark output) | `datasets/uwf-zeekdata24/` | benign only |
| CTU-Normal | Zeek conn.log TSV | `datasets/ctu-normal/` | All entries → FP |

### Source-specific notes

**IoT-23:** Zeek conn.log has 21 tab-separated fields. The last field bundles
`tunnel_parents label detailed-label` as space-separated sub-tokens (IoT-23 specific).
Skip lines starting with `#` (Zeek header comments). Dash placeholders (`"-"`) are
passed through as-is — `build_prompt()` / `_safe()` converts them to `"N/A"`.

**CTU-13:** Binetflow only has `TotPkts` (no per-direction split), so `orig_pkts` and
`resp_pkts` are both set to `TotPkts // 2`. Argus states mapped to Zeek equivalents.

**UNSW-NB15:** Column names differ from Zeek — `protocol` (not `proto`),
`binary_label` (not `label`). The HuggingFace repo also contains `Packet-Bytes/` and
`Payload-Bytes/` (~95 GB each) — always exclude with `ignore_patterns`.

**UWF-ZeekData24:** Real Zeek conn.log from University of West Florida. Benign only.

**CTU-Normal:** Benign-only Zeek conn.log captures (CTU-Normal-20 through 32).

## Prompt Format

All sources are normalized to this 10-field Zeek-native prompt (7 base fields +
3 derived: Bytes/sec, Orig Bytes/Pkt, Resp Bytes/Pkt):

```
Analyze this network connection and classify it as ATTACK or FALSE POSITIVE.

  Proto:              tcp
  Duration (s):       1.234567
  Orig Packets:       10
  Resp Packets:       8
  Orig Bytes:         1024
  Resp Bytes:         512
  Conn State:         SF
  Bytes/sec:          1250.0
  Orig Bytes/Pkt:     102.4
  Resp Bytes/Pkt:     64.0
```

System prompt (shared by training/benchmark/Modelfile):
> You are a network security analyst. Always respond with VERDICT: <ATTACK or FALSE POSITIVE> on the first line, followed by REASON: <brief explanation>.

## Benchmark

**Primary (real-world):** `.venv/bin/python benchmarks/benchmark_realworld.py [--regen]`
- Sources: IoT-23, CTU-13, UWF-ZeekData24, CTU-Normal (native Zeek)
- 300 samples per (source, class), cache: `results/benchmark_realworld_cache.json`
- Per-source breakdown + per-attack-type breakdown

**CICIDS2017 comparison:** `.venv/bin/python benchmarks/benchmark_v6.py`
- 736 samples from 8 CICIDS2017 CSVs, cache: `results/benchmark_samples_v4.json`

**Ollama deployment:** `.venv/bin/python benchmarks/benchmark_ollama.py [MODEL] [--cache FILE]`
- No GPU/transformers needed — calls Ollama HTTP API

## Ollama Deployment

```bash
# 1. Merge adapter into base model
.venv/bin/python scripts/merge_adapter.py <adapter-dir>

# 2. Convert merged model to GGUF
llama.cpp/.venv/bin/python llama.cpp/convert_hf_to_gguf.py \
    <adapter-dir>-merged/ --outfile model.gguf

# 3. Update Modelfile FROM line, create Ollama model
ollama create ids-classifier -f Modelfile

# 4. Benchmark
.venv/bin/python benchmarks/benchmark_ollama.py
```

**Modelfile must include explicit TEMPLATE block** — GGUF conversion does not embed
the Qwen2.5 chat template in metadata.
