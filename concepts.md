# IDS Alert Classifier — Concepts to Learn

---

## 1. Networking & Security

- **Zeek (Bro)** — what it is, how it generates `conn.log`, what each field means (`proto`, `duration`, `orig_pkts`, `resp_pkts`, `orig_bytes`, `resp_bytes`, `conn_state`, `service`)
- **Zeek `conn_state` values** — what SF, S0, S1, RSTO, OTH, REJ, RSTOS0 each mean about the TCP/UDP handshake lifecycle
- **Network flows** — what a bidirectional flow is vs a packet; originator vs responder
- **IDS (Intrusion Detection System)** — signature-based vs anomaly-based; what a SOC analyst does with alerts
- **False positive vs false negative in security** — why a 95% FP rate is operationally unusable even if "accuracy" looks okay
- **MITRE ATT&CK framework** — tactic/technique taxonomy (Credential Access, lateral movement, etc.)
- **Argus / binetflow** — alternative flow capture tool; why its state notation differs from Zeek
- **CICFlowMeter** — what it produces vs Zeek; why the missing `conn_state` breaks this project's approach
- **Common attack types** — port scanning, botnet C2 communication, DDoS, credential access — and what their flow-level signatures look like

---

## 2. Classical ML Baseline (context for why LLM was chosen)

- **Random Forest / XGBoost** — how they classify tabular data; why they overfit CICIDS2017
- **BERT / encoder models** for classification — why they give no explanation
- **LIME / SHAP** — explainability add-ons for black-box models; the argument that LLM explanations are preferable

---

## 3. Neural Network Fundamentals

- **Supervised classification** — labels, loss functions, gradient descent
- **Train/eval split** — why stratified sampling matters here (source-stratified vs random)
- **Class imbalance** — why 1:1 ATTACK:benign made the model trigger-happy; what the 2:1 ratio fixes
- **Overfitting / underfitting** — epochs, eval_loss as the selection criterion
- **Precision, Recall, F1** — what each means; why recall on ATTACK is the critical metric for IDS
- **Matthews Correlation Coefficient (MCC)** — why it's more honest than accuracy on imbalanced datasets (range -1 to +1)
- **Confusion matrix** — TP/FP/TN/FN in the security context
- **Distribution shift** — why lab datasets don't generalise to real traffic

---

## 4. Transformer Architecture

- **Attention mechanism** — Q, K, V projections; what `q_proj`, `k_proj`, `v_proj`, `o_proj` are
- **Multi-head attention** — why multiple heads; what GQA (Grouped Query Attention) is (Qwen3 uses it)
- **MLP layers** — `gate_proj`, `up_proj`, `down_proj` (SwiGLU activation); what the MLP does in a transformer block
- **Decoder-only (causal LM)** vs encoder-decoder — why autoregressive generation works for this task
- **Tokenization** — subword tokens, vocabulary, special tokens (`<|im_start|>`, `<|im_end|>`), padding token
- **Chat templates** — system/user/assistant role formatting; why the template must match between training and inference
- **Autoregressive generation** — how `max_new_tokens` controls output length; why stopping at `<|im_end|>` matters
- **Numerical precision** — bf16 vs fp16 vs fp32; why bf16 is preferred for training

---

## 5. Fine-Tuning & LoRA

- **Supervised Fine-Tuning (SFT)** — training on instruction/response pairs; how it differs from pretraining
- **LoRA (Low-Rank Adaptation)** — the core technique: freeze W, train A×B; why weight updates are low-rank
- **Rank `r`** — what it means intuitively (number of "directions" of change); r=8 vs r=16 tradeoff
- **`lora_alpha`** — the scaling factor `alpha/r`; how it controls correction magnitude
- **Target modules** — why attention + MLP projections are targeted; what gets skipped (embeddings, layer norms)
- **LoRA dropout** — regularisation on the adapter matrices
- **Adapter files** — what `adapter_config.json` + `adapter_model.safetensors` contain; ~20 MB size

---

## 6. QLoRA Specifically

- **4-bit NF4 quantization** — Normal Float 4; how it differs from int4; why NF4 preserves more information for normally-distributed weights
- **BitsAndBytes** — the library that implements 4-bit loading and dequantization
- **Dequantization on the fly** — weights are stored at 4-bit but compute happens in bf16; memory vs compute tradeoff
- **VRAM budgeting** — model weights vs activations vs optimizer states; why activations dominate
- **Gradient checkpointing** — recompute activations during backward pass instead of storing them; ~20% compute cost for large VRAM saving
- **Gradient accumulation** — simulate larger batch size across multiple forward passes when VRAM limits physical batch size
- **Paged AdamW 8-bit** — optimizer states paged to CPU RAM under pressure; why this was chosen over fused AdamW

---

## 7. Training Stack

- **HuggingFace `transformers`** — `AutoModelForCausalLM`, `AutoTokenizer`, `BitsAndBytesConfig`
- **PEFT library** — `LoraConfig`, `PeftModel`; what PEFT means (Parameter-Efficient Fine-Tuning)
- **TRL `SFTTrainer`** — supervised fine-tuning wrapper; handles chat template formatting from JSONL
- **JSONL format** — one JSON object per line; `messages` array with role/content pairs
- **Cosine LR scheduler with restarts** — why restarts help with 3 epochs; warmup ratio
- **`load_best_model_at_end`** — saves checkpoint with lowest eval_loss; why this matters across 3 epochs

---

## 8. Deployment Pipeline

- **LoRA adapter merging** — why a GGUF converter needs the full model weights, not just the delta; `merge_adapter.py`
- **GGUF format** — llama.cpp's binary model format; metadata, tensor layout
- **llama.cpp** — CPU/GPU inference without Python/CUDA; `convert_hf_to_gguf.py`
- **Quantization levels** — Q4_K_M, Q5_K_M, Q6_K, Q8_0, F16; size vs quality tradeoff for inference
- **Ollama** — model serving tool; `Modelfile` (FROM, TEMPLATE, PARAMETER stop); why the TEMPLATE block is required when GGUF doesn't embed it
- **`/api/generate`** — raw Ollama HTTP endpoint; why `benchmark_ollama.py` formats the prompt manually instead of using the chat API

---

## 9. Datasets

- **IoT-23** — CTU university IoT traffic captures; Zeek `conn.log.labeled` format; 21-field TSV; labeled field structure
- **CTU-13** — botnet traffic from CTU; binetflow/Argus format; state mapping to Zeek equivalents
- **UNSW-NB15** — Bro/Zeek-generated dataset from UNSW; parquet format; `binary_label` column
- **UWF-ZeekData24** — real Zeek conn.log from University of West Florida cyber range; why attacks were dropped (100% Credential Access, unlearnable at flow level)
- **CTU-Normal** — benign-only real browsing traffic; why it was critical for fixing the v4 FP problem
- **CICIDS2017** — why it was dropped (CICFlowMeter: no `conn_state`, numeric proto codes, documented label errors)

---

## 10. Evaluation Concepts Specific to This Project

- **`conn_state` over-reliance** — the model learned to use `conn_state` as a near-sufficient feature; what breaks when it's missing or unfamiliar
- **Source-stratified eval split** — why a random 10% slice would be misleading; holding out per (source, class) bucket
- **Format failure rate** — outputs that don't contain `VERDICT:` at all; tracked separately from wrong verdicts
- **Per-source breakdown** — why aggregated accuracy hides source-specific failures (e.g. UWF 0% attack recall)
- **Train/inference mismatch** — the v4 bug where missing fields computed as 0.0 during training but N/A during inference
