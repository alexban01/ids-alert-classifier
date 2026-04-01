# IDS Alert Classifier — Fine-Tuning Project

<!-- @AGENTS.md -->

## Claude-specific reinforcement

- Default mode in this repository is **review-only**.
- Unless the user explicitly tells Claude to write code, Claude should restrict itself to:
  - reviewing `master...v9.1` (or whatever the current implementation branch is)
  - updating `REVIEW_TASKS.md`
  - summarizing risks and missing tests
- If implementation is requested, keep the change as small as possible and state clearly that this is an explicit override of the default contract.


Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` via QLoRA to classify network flows as
**ATTACK** or **FALSE POSITIVE**, targeting deployment against Zeek conn.log / PCAP captures.

**Local hardware:** Ryzen 7 3700X, 32 GB RAM, RTX 3070 (8 GB VRAM)
**Training (v7):** RunPod RTX 5090 (32 GB VRAM), ~$0.44/hr on-demand

## Python Environment

Arch Linux managed environment — system `python3`/`pip3` refuse to install packages.

- Project: `.venv/bin/python` and `.venv/bin/pip`
- llama.cpp: `llama.cpp/.venv/bin/python` (separate venv)

## Files

| File | Purpose |
|---|---|
| `preprocess_zeek.py` | Builds `zeek_dataset.jsonl` + `zeek_dataset_eval.jsonl` (v7: 360k samples, 5 sources) |
| `prompt_utils.py` | Shared `build_prompt`, `_safe`, `SYSTEM_PROMPT`, `extract_verdict` — single source of truth |
| `train.py` | QLoRA fine-tuning via SFTTrainer (v7: RunPod 5090 or local 3070), saves adapter to `v7-ids-lora-adapter/` |
| `merge_adapter.py` | Merges LoRA adapter into base model (fp16) for GGUF conversion — required before llama.cpp |
| `benchmark.py` | Fine-tuned model benchmark on CICIDS2017 samples (Zeek-native prompt) |
| `benchmark_v6.py` | v4 vs v6 vs v7 comparison on CICIDS2017, batched HuggingFace inference |
| `benchmark_realworld.py` | v4 vs v6 vs v7 on real Zeek sources (IoT-23, CTU-13, UWF, CTU-Normal) — primary benchmark |
| `benchmark_ollama.py` | Benchmark any Ollama-served model — no GPU/transformers needed, uses HTTP API |
| `benchmark_vanilla.py` | Standalone vanilla Qwen benchmark (same Zeek-native prompt) |
| `classify_conn_log.py` | Classify a real Zeek conn.log using the fine-tuned adapter |
| `classify_weird_log.py` | Classify weird.log entries by cross-referencing conn.log for flow stats |
| `setup_runpod.sh` | RunPod pod setup: pip installs + dataset check |
| `Modelfile` | Ollama config — `FROM ./v6-ids.gguf`, Qwen2.5 chat template, temperature 0, num_predict 80 |
| `.gitignore` | Excludes datasets, models, checkpoints, generated JSONL, venvs |
| `datasets/SOURCES.txt` | Download URLs for all dataset sources |

**Gitignored (regenerable):**
- `datasets/` — IoT-23 (8.7 GB .tar.gz), CTU-13 (1.9 GB .tar.bz2), UNSW-NB15 (175 MB .parquet), UWF-ZeekData24 (21 MB), CTU-Normal (49 MB)
- `*.pcap_ISCX.csv` — 8 CICIDS2017 CSVs in project root (unused in v7)
- `zeek_dataset.jsonl`, `zeek_dataset_eval.jsonl` — train/eval splits from preprocess_zeek.py
- `v4-ids-model/`, `v6-ids-model/`, `v7-ids-model/` — training checkpoints
- `v4-ids-lora-adapter/`, `v6-ids-lora-adapter/`, `v7-ids-lora-adapter/` — final LoRA adapters
- `*-ids-merged/` — merged model dirs (output of merge_adapter.py, input to convert_hf_to_gguf.py)
- `*.gguf`, `llama.cpp/`, `benchmark_samples*.json`, `benchmark_realworld_cache.json`

## Model Architecture

- **Base:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Quantization:** 4-bit NF4 (BitsAndBytes), bf16 compute dtype
- **LoRA (v6):** r=16, lora_alpha=32, dropout=0.05, bias=none (v4 was r=8, lora_alpha=16)
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (7 modules — all attention + MLP)

## Training (v7)

Run on RunPod pod: `python train.py` (after `bash setup_runpod.sh`)

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
num_train_epochs = 3
learning_rate = 2e-4
lr_scheduler_type = "cosine_with_restarts"
lr_scheduler_kwargs = {"num_cycles": 3}
warmup_ratio = 0.03
weight_decay = 0.01
bf16 = True
gradient_checkpointing = True
eval_strategy = "epoch"
load_best_model_at_end = True
metric_for_best_model = "eval_loss"
save_strategy = "epoch"
max_length = 512
logging_steps = 250
dataloader_num_workers = 4
dataloader_pin_memory = True
```

**Time estimate (full dataset, 360k samples):** ~4-5 hours on RTX 5090 (~$2).
~15-20 hours on RTX 3070 (local, free but slow — use `TRAINING_FACTOR=0.1` for fast validation runs).

After training, the best adapter (by eval_loss) is saved to `v7-ids-lora-adapter/`.
Download to local machine for inference/GGUF conversion.

**Inference still runs locally on RTX 3070** — adapter is hardware-agnostic.

## Preprocessing

Run: `.venv/bin/python preprocess_zeek.py`

Produces `zeek_dataset.jsonl` (train) and `zeek_dataset_eval.jsonl` (eval) using a
source-stratified 90/10 split — 10% held out per (source, class) bucket.

v7 dataset: ~360k total (120k ATTACK / 240k benign). **2:1 benign:attack ratio** —
real networks are overwhelmingly benign; 1:1 training made the model trigger-happy.

Each training sample includes a randomly selected reason from pools of 10 varied
ATTACK_REASONS and 10 BENIGN_REASONS to add diversity.

**TRAINING_FACTOR** (default 1.0): set to 0.1 in preprocess_zeek.py for fast local
validation runs (~36k samples instead of 360k, proportionally scaled).

### Dataset sources (v7)

| Source | Format | Path | Label logic | v7 notes |
|---|---|---|---|---|
| IoT-23 | Zeek `conn.log.labeled` in tar.gz | `datasets/iot-23/` | `"Malicious"` → ATTACK, `"Benign"` → FP | benign capped at 20k (was 80k) |
| CTU-13 | Binetflow CSV in tar.bz2 | `datasets/ctu-13/` | `"Botnet"` → ATTACK, `"Normal"` → FP | Argus states mapped to Zeek |
| UNSW-NB15 | Parquet (HuggingFace) | `datasets/unsw-nb15/` | `binary_label=1` → ATTACK, `0` → FP | unchanged |
| UWF-ZeekData24 | CSV (Spark output) | `datasets/uwf-zeekdata24/` | `label_binary == "True"` → ATTACK | **attacks dropped** — benign only |
| CTU-Normal | Zeek conn.log TSV | `datasets/ctu-normal/` | All entries → FP | cap increased to 100k |
| ~~CICIDS2017~~ | ~~CICFlowMeter CSVs~~ | ~~`*.pcap_ISCX.csv`~~ | ~~dropped in v7~~ | conn_state=`-` + proto=`unknown` on every row |

### Caps (v7)

- `MAX_PER_SOURCE_CLASS = 80,000` — default cap per (source, class)
- `IOT23_BENIGN_CAP = 20,000` — IoT-23 benign only (89% S0 UDP — reduced to prevent S0-bias)
- `CTU_NORMAL_CAP = 100,000` — only significant SF benign source, needs more weight
- `FINAL_ATTACK = 120,000` / `FINAL_BENIGN = 240,000` — 2:1 final ratio

### Why CICIDS2017 was dropped (v7)

Inspection of the training data revealed that **100% of CICIDS2017 samples have
`conn_state = "-"` and `proto = "unknown"`** — CICFlowMeter produces no conn_state
equivalent and uses numeric protocol codes that don't map to Zeek text names. These
are the two most discriminative Zeek features; 80k samples with both missing adds
noise. Additionally, CICIDS2017 has documented label accuracy issues (~10-15%
mislabeled flows). UNSW-NB15 (Bro/Zeek-generated) covers the same attack diversity.

### Why UWF attacks were dropped (v7)

All UWF attacks are "Credential Access" (100% SF TCP, ~0.02s duration), which are
indistinguishable from normal short web connections at the flow level — benchmark
recall was 2%. Training on them taught "short SF TCP = ATTACK", causing false
positives on legitimate HTTPS, API calls, and DB queries.

### CTU-13 state mapping (v7)

Binetflow (Argus) uses different state notation from Zeek. v7 maps them in `load_ctu13()`:

| Argus state | Zeek equivalent | Meaning |
|---|---|---|
| `INT` | `S1` | mid-flow, established, no FIN |
| `CON` | `SF` | completed connection |
| `FSPA_FSPA`, `FSA_FSA`, `SPA_FSPA`, `FIN` | `SF` | completed with FIN |
| `PA_PA`, `EST` | `OTH` / `S1` | no SYN seen / established |
| `S_`, `REQ` | `S0` | SYN only, no response |
| `SRPA_SPA`, `SRST` | `RSTO` | RST from originator |

### Source-specific notes

**IoT-23:** Zeek conn.log has 21 tab-separated fields. The last field bundles
`tunnel_parents label detailed-label` as space-separated sub-tokens (IoT-23 specific).
Use `len(parts) < 21` (not 22) and search for `"Malicious"`/`"Benign"` in `parts[-1]`.
Skip lines starting with `#` (Zeek header comments). Dash placeholders (`"-"`) are
passed through as-is — `build_prompt()` / `_safe()` converts them to `"N/A"` in the
prompt, matching real inference behavior.

**CTU-13:** Binetflow only has `TotPkts` (no per-direction split), so `orig_pkts` and
`resp_pkts` are both set to `TotPkts // 2`. `dst_bytes` is derived as
`TotBytes - SrcBytes`. Argus states mapped to Zeek equivalents (see table above).

**UNSW-NB15:** Column names differ from Zeek — `protocol` (not `proto`),
`binary_label` (not `label`). Handled via fallback lists in `load_unsw()`.
The HuggingFace repo (`rdpahalavan/UNSW-NB15`) also contains `Packet-Bytes/` and
`Payload-Bytes/` (~95 GB each) — always exclude with `ignore_patterns` when downloading.

**UWF-ZeekData24:** Real Zeek conn.log from University of West Florida cyber range.
Columns match Zeek naming (`proto`, `conn_state`, `orig_bytes`, etc.). Empty strings
for missing values in S0 connections (e.g. duration, bytes) — passed through as-is,
rendered as `N/A` by `_safe()`. v7: attack rows skipped, benign only.

**CTU-Normal:** Benign-only Zeek conn.log captures (CTU-Normal-20 through 32) from
Stratosphere Lab. Standard 21-field TSV format, identical to IoT-23. Uses `-` for
unset fields — passed through as `N/A`. All entries labeled as FALSE POSITIVE.

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

Expected response format:
```
VERDICT: ATTACK
REASON: Traffic pattern matches known malicious behavior with anomalous packet ratios.
```

## Benchmark

Three benchmark scripts, each targeting a different use case:

**Primary (real-world):** `.venv/bin/python benchmark_realworld.py [--regen]`
- Sources: IoT-23, CTU-13, UWF-ZeekData24, CTU-Normal (native Zeek, no synthetic mapping)
- 300 samples per (source, class), cache: `benchmark_realworld_cache.json`
- Runs all adapters in `MODELS` list (currently v4, v6, v7)
- Per-source breakdown + per-attack-type breakdown + v4 vs v6 vs v7 delta table
- This is the honest benchmark — CICIDS2017 results are partially circular

**CICIDS2017 comparison:** `.venv/bin/python benchmark_v6.py`
- 736 samples from 8 CICIDS2017 CSVs, cache: `benchmark_samples_v4.json`
- Runs all adapters in `MODELS` list (v4, v6, v7)
- Useful for historical comparison with v3/v4 results

**Ollama deployment:** `.venv/bin/python benchmark_ollama.py [MODEL] [--cache FILE]`
- No GPU/transformers needed — calls Ollama HTTP API
- Works with any model served by Ollama (default: `ids-classifier`)
- Default cache: `benchmark_realworld_cache.json`; pass `--cache benchmark_samples_v4.json` for CICIDS2017
- Uses raw `/api/generate` with manually formatted Qwen2.5 chat template (bypasses
  Ollama template handling — required because GGUF conversion doesn't embed the template)

All benchmarks use batched or sequential inference with `max_new_tokens=80`, parse
`VERDICT:` line via `extract_verdict()`, and report MCC, classification report,
confusion matrix, format failure rate, and per-source/per-type breakdown.

## Ollama Deployment

`convert_hf_to_gguf.py` requires a full model directory (`config.json` + weights).
A LoRA adapter dir only has delta weights — merge first with `merge_adapter.py`.

```bash
# 1. Merge adapter into base model (loads fp16, ~3 GB VRAM or use --cpu)
.venv/bin/python merge_adapter.py v7-ids-lora-adapter
# → saves to v7-ids-lora-adapter-merged/

# 2. Convert merged model to GGUF
llama.cpp/.venv/bin/python llama.cpp/convert_hf_to_gguf.py \
    v7-ids-lora-adapter-merged/ --outfile v7-ids.gguf

# 3. Update Modelfile FROM line: FROM ./v7-ids.gguf

# 4. Create Ollama model (Modelfile includes Qwen2.5 TEMPLATE + stop tokens)
ollama create ids-classifier -f Modelfile

# 5. Test
ollama run ids-classifier

# 6. Benchmark
.venv/bin/python benchmark_ollama.py
```

**Modelfile must include explicit TEMPLATE block** — GGUF conversion does not embed
the Qwen2.5 `<|im_start|>/<|im_end|>` chat template in metadata, so Ollama defaults
to raw text completion (completing the prompt instead of classifying). The TEMPLATE
block and `PARAMETER stop "<|im_end|>"` in Modelfile fix this for interactive use.
`benchmark_ollama.py` additionally uses raw `/api/generate` with a manually formatted
prompt for maximum reliability regardless of GGUF metadata state.

## Version History

| Version | Status | Notes |
|---|---|---|
| v3 | Done | CICIDS2017 only, CICFlowMeter 15-feature prompt, ~89% accuracy |
| v4 | Done | 4-source Zeek-native prompt, 300k samples, r=8, 1 epoch — 82% on CICIDS2017 but ~95% FP on real Zeek logs |
| v5 | Skipped | Would have had same real-world FP rate as v4 |
| v6 | Done | 6-source dataset (~330k), UWF + CTU-Normal added — 86.5% on CICIDS2017 but **worse** on real Zeek (67%, CTU-Normal FP recall 18%) |
| v7 | Training ready | 5-source, 360k samples (2:1 benign:attack), CICIDS2017 dropped, UWF attacks dropped, IoT-23 benign capped 80k→20k, CTU-13 states mapped, source-stratified eval |
