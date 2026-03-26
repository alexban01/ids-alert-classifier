# IDS Alert Classifier — Fine-Tuning Project

Fine-tune `Qwen/Qwen2.5-1.5B-Instruct` via QLoRA to classify network flows as
**ATTACK** or **FALSE POSITIVE**, targeting deployment against Zeek conn.log / PCAP captures.

**Local hardware:** Ryzen 7 3700X, 32 GB RAM, RTX 3070 (8 GB VRAM)
**Training (v5):** RunPod RTX 5090 (32 GB VRAM), ~$0.89/hr on-demand

## Python Environment

Arch Linux managed environment — system `python3`/`pip3` refuse to install packages.

- Project: `.venv/bin/python` and `.venv/bin/pip`
- llama.cpp: `llama.cpp/.venv/bin/python` (separate venv)

## Files

| File | Purpose |
|---|---|
| `preprocess_zeek.py` | Builds `zeek_dataset.jsonl` (300k samples) from 4 dataset sources |
| `train.py` | QLoRA fine-tuning via SFTTrainer (v5: targets RunPod 5090), saves adapter to `v5-ids-lora-adapter/` |
| `benchmark.py` | Fine-tuned model benchmark on CICIDS2017 samples (Zeek-native prompt) |
| `benchmark_vanilla.py` | Standalone vanilla Qwen benchmark (same Zeek-native prompt) |
| `setup_runpod.sh` | RunPod pod setup: pip installs + dataset check |
| `Modelfile` | Ollama config — currently `FROM ./v3-ids.gguf`, temperature 0, num_predict 80 |
| `.gitignore` | Excludes datasets, models, checkpoints, generated JSONL, venvs |
| `datasets/SOURCES.txt` | Download URLs for all 4 dataset sources |

**Gitignored (regenerable):**
- `datasets/` — IoT-23 (8.7 GB .tar.gz), CTU-13 (1.9 GB .tar.bz2), UNSW-NB15 (175 MB .parquet)
- `*.pcap_ISCX.csv` — 8 CICIDS2017 CSVs in project root
- `zeek_dataset.jsonl` — 300k training samples (208 MB)
- `v4-ids-model/`, `v5-ids-model/` — training checkpoints
- `v4-ids-lora-adapter/`, `v5-ids-lora-adapter/` — final LoRA adapters
- `*.gguf`, `llama.cpp/`, `benchmark_samples*.json`

## Model Architecture

- **Base:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Quantization:** 4-bit NF4 (BitsAndBytes), bf16 compute dtype
- **LoRA (v5):** r=16, lora_alpha=32, dropout=0.05, bias=none (v4 was r=8, lora_alpha=16)
- **Target modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (7 modules — all attention + MLP)

## Training (v5 — RunPod RTX 5090)

Run on RunPod pod: `python train.py` (after `bash setup_runpod.sh`)

Dataset is split 90/10 train/test (`test_size=0.1, seed=42`).

```python
per_device_train_batch_size = 16    # v4: 2 (VRAM-limited)
per_device_eval_batch_size  = 16    # v4: 2
gradient_accumulation_steps = 1     # v4: 8 — effective batch stays 16
optim = "paged_adamw_8bit"
num_train_epochs = 3                # v4: 1
learning_rate = 2e-4                # v4: 5e-5 — QLoRA paper sweet spot
lr_scheduler_type = "cosine_with_restarts"  # v4: cosine (decayed to ~0)
lr_scheduler_kwargs = {"num_cycles": 3}     # 1 restart per epoch
warmup_ratio = 0.03
weight_decay = 0.01
bf16 = True
gradient_checkpointing = False      # v4: True (32 GB VRAM — no need)
eval_strategy = "epoch"             # v4: "no" (OOM on 3070)
load_best_model_at_end = True       # v4: False (OOM on 3070)
metric_for_best_model = "eval_loss"
save_strategy = "epoch"
max_length = 512
logging_steps = 100
dataloader_num_workers = 4          # v4: 0 (Python 3.14 forkserver issue)
dataloader_pin_memory = True
```

**Time estimate:** ~2-3 hours total (3 epochs) on RTX 5090. Cost: ~$2-3.

After training, the best adapter (by eval_loss) is saved via
`trainer.model.save_pretrained()` → `adapter_config.json` + `adapter_model.safetensors`
in `v5-ids-lora-adapter/`. Download this to local machine for inference/GGUF conversion.

**Inference still runs locally on RTX 3070** — adapter is hardware-agnostic. The
4-bit base model + LoRA adapter loads the same way regardless of training GPU.

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

- Generates or loads `benchmark_samples_v4.json` (50 samples/class/CSV from 8 CICIDS2017 files)
- Uses Zeek-native 10-field prompt (same `build_prompt` as `preprocess_zeek.py`)
- Maps CICIDS2017 columns to Zeek schema (protocol number, µs→s duration, conn_state="-")
- Runs batched inference (batch_size=8, max_new_tokens=80) on fine-tuned model only
- Reports: classification report, confusion matrix, per-attack-type accuracy, format failure rate
- Parses model output for `VERDICT:` line — returns `UNKNOWN` if not found
- `benchmark_vanilla.py` — same setup but runs vanilla Qwen only (standalone)

**Adapter loading:** Requires `adapter_config.json` in the adapter dir → loads via
`PeftModel.from_pretrained()`.

## Ollama Deployment

```bash
# 1. Convert to GGUF (use llama.cpp's own venv)
llama.cpp/.venv/bin/python llama.cpp/convert_hf_to_gguf.py v5-ids-lora-adapter/ --outfile v5-ids.gguf

# 2. Update Modelfile: change FROM line to FROM ./v5-ids.gguf

# 3. Create Ollama model
ollama create ids-classifier -f Modelfile

# 4. Test
ollama run ids-classifier
```

## Version History

| Version | Status | Notes |
|---|---|---|
| v3 | Done | CICIDS2017 only, CICFlowMeter 15-feature prompt, ~89% accuracy |
| v4 | Done | 4-source Zeek-native 10-field prompt, 300k samples, r=8, 1 epoch, 82% accuracy |
| v5 | Training ready | RunPod 5090, r=16, 3 epochs, cosine_with_restarts, LR 2e-4, eval enabled |
