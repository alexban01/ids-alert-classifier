# IDS Alert Classifier — Fine-Tuning Project

Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` via QLoRA to classify network flows as
**ATTACK** or **FALSE POSITIVE**, targeting deployment against Zeek conn.log / PCAP captures.

**Hardware:** Ryzen 7 3700X, 32 GB RAM, RTX 3070 (8 GB VRAM)

## Python Environment

Arch Linux managed environment — system `python3`/`pip3` refuse to install packages.

- Project: `.venv/bin/python` and `.venv/bin/pip`
- llama.cpp: `llama.cpp/.venv/bin/python` (separate venv)

## Files

| File | Purpose |
|---|---|
| `preprocess_zeek.py` | Builds `zeek_dataset.jsonl` (300k samples) from 4 dataset sources |
| `train.py` | QLoRA fine-tuning via SFTTrainer, saves adapter to `v4-ids-lora-adapter/` |
| `benchmark.py` | Head-to-head: vanilla Qwen vs fine-tuned on CICIDS2017 samples |
| `Modelfile` | Ollama config — currently `FROM ./v3-ids.gguf`, temperature 0, num_predict 80 |
| `.gitignore` | Excludes datasets, models, checkpoints, generated JSONL, venvs |
| `datasets/SOURCES.txt` | Download URLs for all 4 dataset sources |

**Gitignored (regenerable):**
- `datasets/` — IoT-23 (8.7 GB .tar.gz), CTU-13 (1.9 GB .tar.bz2), UNSW-NB15 (175 MB .parquet)
- `*.pcap_ISCX.csv` — 8 CICIDS2017 CSVs in project root
- `zeek_dataset.jsonl` — 300k training samples (208 MB)
- `v4-ids-model/` — training checkpoints
- `v4-ids-lora-adapter/` — final LoRA adapter
- `*.gguf`, `llama.cpp/`, `benchmark_samples.json`

## Model Architecture

- **Base:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Quantization:** 4-bit NF4 (BitsAndBytes), bf16 compute dtype
- **LoRA:** r=8, lora_alpha=16, dropout=0.05, bias=none
- **Target modules (v4):** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (7 modules — all attention + MLP)

## Training (v4)

Run: `.venv/bin/python train.py`

Dataset is split 90/10 train/test (`test_size=0.1, seed=42`).

```python
per_device_train_batch_size = 2
per_device_eval_batch_size  = 2
gradient_accumulation_steps = 8   # effective batch = 16
optim = "paged_adamw_8bit"        # 8-bit optimizer states — key VRAM saver
num_train_epochs = 3
learning_rate = 5e-5              # lower than v3's 1e-4
lr_scheduler_type = "cosine"
warmup_ratio = 0.03
weight_decay = 0.01
bf16 = True
gradient_checkpointing = True
eval_strategy = "no"              # IMPORTANT: prevents OOM at epoch boundary
load_best_model_at_end = False    # IMPORTANT: prevents OOM
save_strategy = "epoch"
max_length = 512
logging_steps = 200
dataloader_num_workers = 0        # must be 0 — Python 3.14 forkserver issue
dataloader_pin_memory = True
```

**OOM warning:** RTX 3070 uses ~6700/8192 MiB during training. Re-enabling
`eval_strategy="epoch"` or `load_best_model_at_end=True` will OOM.

**Time estimate:** ~11 hours/epoch at ~2.5 s/step for 300k samples.

After training, the adapter is saved via `trainer.model.save_pretrained()` which
produces `adapter_config.json` + `adapter_model.safetensors` in `v4-ids-lora-adapter/`.

## Preprocessing

Run: `.venv/bin/python preprocess_zeek.py`

Produces `zeek_dataset.jsonl` — 300k chat-format samples (180k ATTACK / 120k benign).

Each training sample includes a randomly selected reason from pools of 10 varied
ATTACK_REASONS and 10 BENIGN_REASONS to add diversity.

### Dataset sources

| Source | Format | Path | Label logic |
|---|---|---|---|
| IoT-23 | Zeek `conn.log.labeled` in tar.gz | `datasets/iot-23/iot_23_datasets_small.tar.gz` | `"Malicious"` in last field -> ATTACK, `"Benign"` -> FP |
| CTU-13 | Binetflow CSV in tar.bz2 | `datasets/ctu-13/CTU-13-Dataset.tar.bz2` | `"Botnet"` -> ATTACK, `"Normal"` -> FP, `"Background"` skipped |
| UNSW-NB15 | Parquet (HuggingFace) | `datasets/unsw-nb15/Network-Flows/UNSW_Flow.parquet` | `binary_label=1` -> ATTACK, `0` -> FP |
| CICIDS2017 | CICFlowMeter CSVs | `*.pcap_ISCX.csv` in project root | `Label != "BENIGN"` -> ATTACK |

### Caps

- `MAX_PER_SOURCE_CLASS = 80,000` — per (source, class) before merge
- `FINAL_ATTACK = 180,000` / `FINAL_BENIGN = 120,000` — final dataset targets
- `PER_FILE_CAP = 10,000` — CICIDS2017 only, per file per class (prevents DDoS CSV from dominating)

### Source-specific notes

**IoT-23:** Zeek conn.log has 21 tab-separated fields. The last field bundles
`tunnel_parents label detailed-label` as space-separated sub-tokens (IoT-23 specific).
Use `len(parts) < 21` (not 22) and search for `"Malicious"`/`"Benign"` in `parts[-1]`.
Skip lines starting with `#` (Zeek header comments). Dash placeholders (`"-"`) are
replaced with `"0"`.

**CTU-13:** Binetflow only has `TotPkts` (no per-direction split), so `orig_pkts` and
`resp_pkts` are both set to `TotPkts // 2`. `dst_bytes` is derived as
`TotBytes - SrcBytes`.

**UNSW-NB15:** Column names differ from Zeek — `protocol` (not `proto`),
`binary_label` (not `label`). Handled via fallback lists in `load_unsw()`.
The HuggingFace repo (`rdpahalavan/UNSW-NB15`) also contains `Packet-Bytes/` and
`Payload-Bytes/` (~95 GB each) — always exclude with `ignore_patterns` when downloading.

**CICIDS2017:** `conn_state` is always `"-"` (CICFlowMeter has no equivalent).
Protocol is numeric (e.g. `"6"` for TCP, `"17"` for UDP) unlike the other sources
which use text (`"tcp"`). `Flow Duration` is converted from microseconds to seconds.

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

Run: `.venv/bin/python benchmark.py`

- Generates or loads `benchmark_samples.json` (50 samples/class/CSV from 8 CICIDS2017 files)
- Runs batched inference (batch_size=8, max_new_tokens=80) on vanilla Qwen then fine-tuned
- Reports: classification report, confusion matrix, per-attack-type accuracy, format failure rate
- Parses model output for `VERDICT:` line — returns `UNKNOWN` if not found

**Adapter loading** supports two paths:
1. `adapter_config.json` exists -> `PeftModel.from_pretrained()` (clean, expected for v4)
2. No `adapter_config.json` -> legacy manual LoRA injection from `model.safetensors`
   (uses only `q_proj`, `v_proj` — 2 modules, leftover from v2)

**TODO:** benchmark.py still uses the old 15-feature CICFlowMeter prompt, not the
Zeek-native 10-field prompt. It must be rewritten before v4 benchmark results are meaningful.

## Ollama Deployment

```bash
# 1. Convert to GGUF (use llama.cpp's own venv)
llama.cpp/.venv/bin/python llama.cpp/convert_hf_to_gguf.py v4-ids-lora-adapter/ --outfile v4-ids.gguf

# 2. Update Modelfile: change FROM ./v3-ids.gguf to FROM ./v4-ids.gguf

# 3. Create Ollama model
ollama create ids-classifier -f Modelfile

# 4. Test
ollama run ids-classifier
```

## Version History

| Version | Status | Notes |
|---|---|---|
| v3 | Done | CICIDS2017 only, CICFlowMeter 15-feature prompt, ~89% accuracy |
| v4 | Dataset ready, training not started | 4-source Zeek-native 10-field prompt, 300k samples |
