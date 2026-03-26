# IDS Alert Classifier — Fine-Tuning Project

## Project Goal

Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` on consumer hardware to classify network flows as **ATTACK** or **FALSE POSITIVE**, targeting real-world deployment against Zeek/PCAP captures.

**Hardware:** Ryzen 7 3700X · 32 GB RAM · RTX 3070 (8 GB VRAM)

---

## Python Environment

**CRITICAL:** This is an Arch Linux managed environment. Never use system `python3` or `pip3`.

Always use:
- `.venv/bin/python`
- `.venv/bin/pip`

`llama.cpp/` has its own separate `.venv` — use `llama.cpp/.venv/bin/python` for any scripts run from that directory (e.g. `convert_hf_to_gguf.py`).

---

## File Map

| File | Purpose |
|---|---|
| `preprocess_zeek.py` | Builds `zeek_dataset.jsonl` from 4 dataset sources |
| `train.py` | QLoRA fine-tuning via SFTTrainer, saves adapter to `v4-ids-lora-adapter/` |
| `benchmark.py` | Evaluates vanilla Qwen vs fine-tuned model on CICIDS2017 samples |
| `Modelfile` | Ollama model definition (currently references v3 GGUF) |
| `zeek_dataset.jsonl` | 300k training samples (208 MB, gitignored) |

**Gitignored large files (all regenerable):**
- `datasets/` — raw archives: IoT-23 (8.8 GB .tar.gz), CTU-13 (1.9 GB .tar.bz2), UNSW-NB15 (175 MB .parquet in `datasets/unsw-nb15/Network-Flows/`)
- `*.pcap_ISCX.csv` — CICIDS2017 raw CSVs (8 files, live in project root)
- `*.jsonl` — generated training data
- `v4-ids-model/` — training checkpoints
- `v4-ids-lora-adapter/` — final LoRA adapter
- `*.gguf` — quantized model for Ollama
- `llama.cpp/` — external conversion tool

---

## Model Architecture

- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Quantization:** 4-bit NF4 (BitsAndBytes) — used during both training and inference
- **LoRA:** r=8, lora_alpha=16, dropout=0.05, bias=none, task_type=CAUSAL_LM
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

---

## Training (v4)

Run: `.venv/bin/python train.py`

Key config values in `train.py`:

```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 8   # effective batch = 16
optim = "paged_adamw_8bit"        # halves VRAM usage for optimizer states
num_train_epochs = 3
learning_rate = 5e-5
lr_scheduler_type = "cosine"
warmup_ratio = 0.03
bf16 = True
gradient_checkpointing = True
eval_strategy = "no"              # IMPORTANT: prevents OOM at epoch boundary
load_best_model_at_end = False    # IMPORTANT: prevents OOM (would reload checkpoint on top of model)
save_strategy = "epoch"
max_length = 512
```

**OOM note:** RTX 3070 uses ~6700/8192 MiB during training. `eval_strategy="epoch"` and
`load_best_model_at_end=True` both spike VRAM enough to OOM. Never re-enable them.

**Estimated time:** ~11 hours/epoch on RTX 3070 at 2.5 s/step for 300k samples.

---

## Preprocessing (v4)

Run: `.venv/bin/python preprocess_zeek.py`

Produces `zeek_dataset.jsonl` (300k samples, 180k attacks / 120k benign).

### Dataset sources

| Source | Format | Label logic |
|---|---|---|
| IoT-23 | Native Zeek `conn.log.labeled` (tar.gz) | `"Malicious"` in last field → ATTACK, `"Benign"` → FALSE POSITIVE |
| CTU-13 | Binetflow/Argus CSV (tar.bz2) | `"Botnet"` in Label → ATTACK, `"Normal"` → FALSE POSITIVE, `"Background"` skipped |
| UNSW-NB15 | Parquet (Bro/Zeek-generated, HuggingFace) | `binary_label=1` → ATTACK, `0` → FALSE POSITIVE |
| CICIDS2017 | CICFlowMeter CSVs (8 files in project root) | `Label != "BENIGN"` → ATTACK; capped 10k/class/file to prevent DDoS file dominating |

### Caps

- `MAX_PER_SOURCE_CLASS = 80_000` — per (source, class) before merge
- `FINAL_ATTACK = 180_000`, `FINAL_BENIGN = 120_000` — final dataset targets
- `PER_FILE_CAP = 10_000` — per CICIDS2017 file per class

### IoT-23 parsing gotchas

Zeek conn.log has 21 tab-separated fields. The last field bundles
`tunnel_parents label detailed-label` as space-separated sub-tokens (IoT-23 specific).
Check `if len(parts) < 21` (not 22) and search for `"Malicious"`/`"Benign"` directly in `parts[-1]`.
Skip lines starting with `#` (Zeek header comments).

### CTU-13 packet count approximation

Binetflow only has `TotPkts` (no per-direction split). `orig_pkts` and `resp_pkts` are both
set to `TotPkts // 2` as an approximation.

### UNSW-NB15 column names

Dataset uses `protocol` (not `proto`) and `binary_label` (not `label`).
Both are handled via fallback lists in `load_unsw()`.

### UNSW-NB15 download warning

The HuggingFace repo contains `Packet-Bytes/` (95 GB raw packet payloads) and `.cache/` (17 GB)
which must not be downloaded. Only `Network-Flows/UNSW_Flow.parquet` (175 MB) is needed.
Use `snapshot_download` with `ignore_patterns` to exclude them, or download the parquet manually.

---

## Prompt Format (Zeek-native features)

All four sources are normalized to the same 10-field prompt:

```
System:
  You are a network security analyst. Always respond with VERDICT: <ATTACK or FALSE POSITIVE>
  on the first line, followed by REASON: <brief explanation>.

User:
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

Assistant response example:
  VERDICT: ATTACK
  REASON: Traffic pattern matches known malicious behavior with anomalous packet ratios.

---

## Benchmark

Run: `.venv/bin/python benchmark.py`

- Loads or generates `benchmark_samples.json` (50 samples/class/CSV from CICIDS2017)
- NOTE: benchmark.py still uses the old 15-feature CICFlowMeter prompt, not the Zeek prompt
- Compares vanilla `Qwen2.5-1.5B-Instruct` vs fine-tuned adapter side-by-side
- Reports: accuracy, attack recall, benign recall, format failure rate, per-attack-type breakdown

**benchmark.py adapter loading:** Supports two fallback paths:
1. `adapter_config.json` present → `PeftModel.from_pretrained()` (clean adapter)
2. No `adapter_config.json` → legacy manual LoRA weight injection from `model.safetensors`

**`benchmark_samples.json`** is gitignored and auto-regenerates from CICIDS2017 CSVs on the next run if missing.

**TODO:** benchmark.py still uses the old 15-feature CICFlowMeter prompt. It needs to be rewritten to use the Zeek-native 10-field prompt before v4 results are meaningful.

---

## Ollama Deployment

After training, convert to GGUF and update Ollama:

```bash
# 1. Convert adapter to merged HF model, then to GGUF (use llama.cpp's own venv)
llama.cpp/.venv/bin/python llama.cpp/convert_hf_to_gguf.py v4-ids-lora-adapter/ --outfile v4-ids.gguf

# 2. Update Modelfile to point to new GGUF
# Edit: FROM ./v4-ids.gguf

# 3. Create/update Ollama model
ollama create ids-classifier -f Modelfile

# 4. Test
ollama run ids-classifier
```

---

## Version History

| Version | Notes |
|---|---|
| v3 | Trained on CICIDS2017 only (CICFlowMeter features); ~89% accuracy on benchmark |
| v4 | Multi-source Zeek-native features (IoT-23, CTU-13, UNSW-NB15, CICIDS2017); dataset preprocessed (`zeek_dataset.jsonl` ready), training not yet started |

**Git log:**
- `cbbacc4` — git: added gitignore for large/downloadable files
- `088f9f2` — feat: prepared v4 for training
- `9c6f037` — feat: updated to v3
