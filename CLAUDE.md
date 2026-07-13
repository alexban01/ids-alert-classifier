# IDS Alert Classifier — Fine-Tuning Project

Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` via QLoRA to classify network flows as
**ATTACK** or **FALSE POSITIVE**, targeting deployment against Zeek conn.log / PCAP captures.

**Local hardware:** Ryzen 7 3700X, 32 GB RAM, RTX 3070 (8 GB VRAM)
**Training:** RunPod RTX 5090 (32 GB VRAM), ~$1/hr on-demand

## Git conventions

- **Never add Claude (or any AI) as a commit co-author.** Do NOT append
  `Co-Authored-By: Claude …`, `🤖 Generated with Claude Code`, or any similar
  AI-attribution trailer to commit messages or PR bodies. Commits are authored
  solely by the repository owner. (This overrides any default tooling behavior.)
- Never `git push` — the owner pushes manually.

## Keeping project state current

The authoritative project snapshot is **`STATE.md`** at the repo root — the single source
of truth. (The `ids-project.skill` skill's `references/current-state.md` is a symlink to it,
and the auto-memory `project_ids_classifier.md` only points here, so there is exactly one
file to maintain.) **Whenever a big change happens — a training run started/finished, a new
benchmark result, a dataset added/removed/restored, an architecture or objective decision —
update `STATE.md` in the same session.** Keep it concise (update in place, no changelog; git
history is the changelog).

## Python Environment

Arch Linux managed environment — system `python3`/`pip3` refuse to install packages.

- Project: `.venv/bin/python` and `.venv/bin/pip`
- llama.cpp: `llama.cpp/.venv/bin/python` (separate venv)

## Project Structure

```
├── train.py                  # QLoRA fine-tuning via SFTTrainer (entry point)
├── preprocess_zeek.py        # Build training JSONL from all sources (entry point)
├── Modelfile                 # Ollama config — Qwen2.5 chat template, temperature 0
├── requirements.txt
│
├── ids/                      # Shared library package (import as `ids.<module>`)
│   ├── preprocess_config.py  #   Caps, ratio targets, masking probs, reason pools
│   ├── preprocess_sample.py  #   score_hard_benign(), make_sample(), pick_reason()
│   ├── prompt_utils.py       #   Shared build_prompt, SYSTEM_PROMPT, extract_verdict
│   ├── infer_utils.py        #   Shared 4-bit+LoRA model load, tokenizer, chat templating
│   ├── behavior_features.py  #   Behavioral context features for enriched prompts
│   ├── zeek_log_utils.py     #   Zeek TSV parser + conn.log row helper + CTU-Malware helpers
│   ├── run_manifest.py       #   Run provenance: dataset meta + run.json sidecars + EXPERIMENTS.md
│   └── loaders/              #   Dataset loaders (imported by preprocess_zeek.py)
│       ├── loader_iot23.py       #   IoT-23 conn.log.labeled (tar.gz)
│       ├── loader_ctu13.py       #   CTU-13 binetflow (tar.bz2)
│       ├── loader_unsw.py        #   UNSW-NB15 parquet / CSV
│       ├── loader_cicids.py      #   CICIDS2017 CICFlowMeter CSVs (disabled in v7+)
│       ├── loader_uwf.py         #   UWF-ZeekData24 Spark CSV
│       ├── loader_ctu_normal.py  #   CTU-Normal benign Zeek conn.log
│       └── loader_ctu_malware.py #   CTU-Malware-Capture multi-log enriched samples
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
│   ├── experiments.py        #   Rebuild EXPERIMENTS.md ledger from models/*/run.json
│   ├── setup_runpod.py       #   RunPod pod setup (5090) — parallel deps+model, single-file
│   ├── setup_runpod.sh       #   RunPod pod setup (5090) — legacy bash, superseded by .py
│   └── setup_runpod_4090.sh  #   RunPod pod setup (4090)
│
├── tests/                    # Test files
│   ├── test_behavior_features.py
│   └── test_novel_cases.py
│
├── notes/                    # Research notes & thesis notes
├── thesis/                   # Thesis drafts
├── results/                  # Benchmark reports & result JSONs
├── real_conn/                # Real Zeek conn.log files for testing
├── models/                   # Training checkpoints, adapters, merged (gitignored)
├── datasets/                 # Raw datasets (gitignored)
└── llama.cpp/                # External tool (gitignored)
```

**Import convention:** entry-point scripts run from the project root
(`.venv/bin/python preprocess_zeek.py`, `.venv/bin/python benchmarks/benchmark_realworld.py`).
Shared code lives in the `ids/` package and is imported as `from ids.<module> import …`
(e.g. `from ids.prompt_utils import build_prompt`). Scripts under `benchmarks/`,
`scripts/`, and `tests/` add the project root to `sys.path` so `import ids` resolves.

**Gitignored (regenerable):**
- `datasets/` — IoT-23, CTU-13, UNSW-NB15, UWF-ZeekData24, CTU-Normal
- `*.jsonl` — train/eval splits from preprocess_zeek.py (kept at root: train CWD + RunPod upload)
- `models/` — all training checkpoints, LoRA adapters, merged models, and `*.gguf`
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
save_strategy = "epoch"              # --save-steps N switches save+eval to every N steps
save_total_limit = 2                 # keep best + last (step-based saving still keeps just 2)
max_length = 512
logging_steps = 100
dataloader_num_workers = 0
dataloader_pin_memory = True
```

**Cost knobs** (RunPod RTX 5090, ~$1/hr — cost = GPU-hours): training set is
~241k samples, token lengths mean 296 / p99 500 (so `max_length=512` is right and
not a lever). To train cheaper: `packing=True` (default) cuts opt-steps to 0.69×
and ~20% of tokens/epoch, `--epochs 2` saves ~33% vs 3, `--eval-subset 6000` trims
eval forward work to ~0.19×, and `TRAINING_FACTOR` in `preprocess_config.py` scales
the dataset (and cost) linearly at the expense of coverage. `preprocess_zeek.py
--no-reason` drops the REASON line from targets (~14% fewer tokens → ~14% cheaper;
see below). Measured (real tokenizer + bin-packing, no training): combined default
run (2 epochs + packing) ≈ **0.53×** the train compute of the old 3-epoch unpacked
run — roughly **half** the GPU-hours.

**Stop & resume** (local RTX 3070 path — RunPod runs are short enough to not need it):
- `--save-steps N` switches save+eval from per-epoch to every `N` optimizer steps, so a
  run can be interrupted mid-epoch and resumed. `eval_steps` is pinned equal to
  `save_steps` (load_best requires matching strategies), so a too-small `N` adds eval
  overhead — at effective batch 24 a packed epoch is ~7k steps, so `--save-steps 500`
  ≈ 14 checkpoints+evals/epoch; bump to 1000 if eval cost bites. `0` (default) keeps
  per-epoch behaviour. Don't pass it on RunPod.
- `--resume` continues from the latest checkpoint in `OUTPUT_DIR`; `--resume <path>`
  uses a specific checkpoint dir. Restores model + optimizer + LR scheduler + step
  counter (true continuation, not restart). Warns and starts fresh if no checkpoint
  exists. The dataset/batch/packing/epochs must be unchanged from the interrupted run,
  since HF fast-forwards the dataloader assuming the same sample ordering.
- Checkpoints (incl. optimizer state) live in `OUTPUT_DIR`. `save_total_limit=2` keeps
  best + latest, so the most recent is always resumable. On RunPod they'd need to be on
  a persistent volume to survive a pod stop; locally this is a non-issue.

```bash
.venv/bin/python train.py --save-steps 500            # interruptible local run
.venv/bin/python train.py --save-steps 500 --resume   # pick up where it stopped
```

**Inference runs locally on RTX 3070** — adapter is hardware-agnostic.

## Experiment tracking & provenance

Every run self-describes via stdlib sidecars (`ids/run_manifest.py`):
- `preprocess_zeek.py` → `zeek_dataset.meta.json` (git SHA, CLI args, resolved knobs,
  counts, content hash).
- `train.py` → `models/<adapter>/run.json` (hyperparams + dataset link by content hash
  + best `eval_loss`). Travels with the adapter; download it back with the adapter.
- `benchmark_realworld.py` writes MCC/recalls back into `run.json` (FULL mode only).
- **`EXPERIMENTS.md`** (repo root, committed) — generated leaderboard. Rebuild manually:
  `.venv/bin/python scripts/experiments.py`.

**`--no-reason` ablation:** drops the (randomly-picked, non-grounded) REASON line so
targets become bare `VERDICT: <X>` under `SYSTEM_PROMPT_VERDICT_ONLY`. ~11–14% fewer
training tokens (mean seq 303→260) ⇒ proportional savings with packing. Comparable
cross-run only via **MCC**, not eval_loss (different target token counts).

**Prompt matching is automatic on every HF inference path** via `resolve_system_prompt()`
(`ids/infer_utils.py`), which reads the adapter's `run.json` and serves the verdict-only
prompt when `dataset.reason == False` (else default; no `run.json` ⇒ default). Wired into
`benchmark_realworld.py` (prints a provenance banner), `benchmark_v6.py`,
`scripts/classify_conn_log.py`, and `scripts/classify_weird_log.py`. `train.py` records
`dataset.reason` by **sniffing the dataset itself** (`detect_reason_from_dataset`), so it's
correct on RunPod where `zeek_dataset.meta.json` isn't uploaded. **Ollama/GGUF can't
auto-detect** (no `run.json`): pass `--verdict-only` to `benchmark_ollama.py` /
`classify_conn_log.py --ollama`, and use the verdict-only `SYSTEM` line documented in `Modelfile`.

## Datasets

Raw datasets are **not** stored in git (`datasets/` is gitignored). Fetch them from
their public sources with:

```bash
.venv/bin/python scripts/download_datasets.py --list        # show keys
.venv/bin/python scripts/download_datasets.py --all         # fetch everything
.venv/bin/python scripts/download_datasets.py --only unsw,ctu_normal
```

Sources are also documented in `datasets/SOURCES.txt`. CTU-Malware-Capture is fetched
on-demand per scenario by `preprocess_zeek.py`; CICIDS2017 was dropped in v7+.

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
