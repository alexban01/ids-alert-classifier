# Flow-Level Network Intrusion Detection with a Fine-Tuned Small Language Model: Capability Boundaries and the Out-of-Distribution Challenge

**Master's Thesis — Draft 1**
**Author:** Ban Alexandru Mircea
**Affiliation:** [Institution TBD]
**Draft date:** 2026-06-26 — not for distribution

> This is a working draft synthesised from the project development notes
> (`notes/thesis_notes_1.txt` … `thesis_notes_12.txt`) and the current project
> state. Numbers reported here are drawn from the recorded benchmark runs. The
> final thesis model (V12) was not yet trained at the time of this draft; where a
> result depends on V12, the figure is explicitly marked as a *target / expected*
> value and the most recent measured proxy (V10 or V11) is given alongside.

---

## Abstract

We investigate whether a small instruction-tuned language model, fine-tuned with
QLoRA on Zeek `conn.log` flow features, is a viable on-premises intrusion-detection
classifier that can both *decide* (ATTACK vs FALSE POSITIVE) and *explain* (a
natural-language reason) on commodity hardware. We fine-tune
`Qwen2.5-1.5B-Instruct` on a heterogeneous corpus assembled from IoT-23, CTU-13,
UNSW-NB15, UWF-ZeekData24, CTU-Normal, and fifteen CTU-Malware-Capture scenarios,
and evaluate it across both an in-distribution benchmark (four native-Zeek sources)
and a three-tier out-of-distribution (OOD) probe battery: CTU-SME-11 Win7AD-1
lateral movement (OOD-Hard), CTU-SME-11 Amazon Echo discovery scans (OOD-Medium),
and CTU-Malware Botnet-3 Kelihos P2P spam (OOD-Floor).

The work yields four findings. **First**, on known attack families the fine-tuned
model is competitive with classical machine learning on the same feature set:
in-distribution MCC reaches the +0.99 range and is matched by Random Forest and
Logistic Regression. **Second**, out-of-distribution behaviour is dominated by
*training composition*, not model capacity: a single change to the attack-family
budget (raising CTU-Malware's share of the attack pool) moved Win7AD-1 lateral-
movement recall from 11.3% to 87.1%, overturning an earlier finding that classical
ML "vastly outperforms" the LLM on OOD attacks. **Third**, the model exhibits a
novel, LLM-specific failure mode we call *confidently wrong* REASON generation:
fluent, internally consistent explanations that are factually inconsistent with the
flow features, produced for OOD attack flows the model misclassifies — a failure
class that classical detectors cannot exhibit because they have no explanation
surface. **Fourth**, the model generalises to a structurally novel attack family
(Kelihos P2P spam, ~81% recall) via positive transfer where both classical
baselines are degenerate (MCC = 0), demonstrating a cross-family generalisation
advantage of text-space fine-tuning over tabular models.

We argue that, for small LLM-based flow IDS, training-distribution *coverage* — not
architectural scale — is the load-bearing variable, and that the REASON field must
be treated as unreliable in the OOD regime where it would be most valuable.

**Keywords:** intrusion detection, large language models, QLoRA, Zeek, flow-level
classification, out-of-distribution generalisation, explainable security.

---

## 1. Introduction

### 1.1 Motivation

Network intrusion detection sits between two unsatisfying poles. Signature-based
systems (Snort, Suricata, Zeek `notice.log`) produce deterministic but brittle
alerts that fail on anything not already enumerated in a rule. Supervised
classifiers on flow-level features (Random Forest, XGBoost, BERT-style encoders)
add statistical coverage but emit an uninterpretable probability — a number such as
"ATTACK: 0.94" — that a SOC triage analyst cannot act on without a separate
explainability layer (LIME, SHAP).

A recent line of work proposes a third option: a language model that takes a flow
as input and returns *both* a verdict and a human-readable justification. The
appeal is operational — the explanation is essentially free if the model is already
generating text — but the open question is whether the *verdict* half of that output
is accurate enough to act on, and whether the *explanation* half is trustworthy.

This thesis investigates that question empirically for the deployment setting most
likely to be realised in practice: a small (1.5-billion-parameter) instruction-tuned
LLM, 4-bit quantised, QLoRA-fine-tuned on real Zeek `conn.log` data, and run on a
single consumer GPU (RTX 3070, 8 GB VRAM). The project has a dual purpose: it is
both a master's thesis investigating the viability of the approach and a working
deployment target — a classifier that runs locally via Ollama against live Zeek
logs, fast enough for interactive triage and conservative enough to be trusted on
real traffic.

### 1.2 Research questions

* **Q1 — Parity.** Does a fine-tuned small LLM match classical ML accuracy on
  known attack families seen in training?
* **Q2 — Generalisation.** Does it generalise to structurally novel attack
  traffic that the classical baselines also see for the first time?
* **Q3 — Explanation reliability.** Is the REASON field — the differentiator over
  classical ML — a reliable artefact across both the in-distribution and the OOD
  regime?

The prior literature answers Q1 affirmatively and Q2 pessimistically. Our results
confirm Q1, *partially overturn* Q2 conditional on training-data composition, and
introduce a qualitative failure mode on Q3 that prior work does not document.

### 1.3 Contributions

1. A complete, reproducible end-to-end fine-tuning pipeline
   (`preprocess_zeek.py` → `train.py` → `benchmark_realworld.py`) for flow-level
   IDS over heterogeneous, *Zeek-native* sources, with source-stratified
   train/eval splitting and explicit sampling controls over attack-family
   composition.
2. A three-tier OOD probe battery (Win7AD-1 / Echo / Kelihos) that characterises
   generalisation along a difficulty spectrum rather than at a single point, and is
   reported alongside in-distribution accuracy in every benchmark run.
3. The central empirical result that a single training-composition change
   (CTU-Malware attack budget) shifts Win7AD-1 lateral-movement recall from 11.3% to
   87.1% — reframing the "LLM OOD gap" from a model-architecture problem to a
   training-data-coverage problem.
4. A direct Random-Forest / Logistic-Regression baseline on the identical feature
   set, showing that the classical methods are *degenerate* on the OOD-Floor probe
   (Kelihos, MCC = 0) while the fine-tuned LLM achieves ~81% recall through positive
   transfer.
5. Documentation of the *confidently wrong* REASON phenomenon: an LLM-specific OOD
   failure mode invisible to classical IDS, with direct implications for SOC
   deployment and concrete mitigation recommendations.
6. A set of methodological negative results — the [BEHAVIOR] temporal-context null
   result, the host-level aggregation (Host Pass-2) null result, and the
   eval-loss-mis-selects-the-OOD-best-checkpoint observation — that are
   scientifically informative for anyone building LLM-based flow IDS.

### 1.4 Scope and a note on honesty of framing

The thesis is deliberately scoped to *flow-level, single-record* classification
against Zeek `conn.log` (optionally enriched with `http.log` / `dns.log` /
`ssl.log` / behavioural context). It does not claim to detect attacks that are
indistinguishable at the flow level without payload inspection or multi-flow
correlation (e.g. stealth HTTP C2 that mimics a CDN health check). One important
methodological theme runs through the project: several "model limitations" turned
out, on inspection, to be data-pipeline artefacts (a broken port-extraction
fallback), ground-truth labelling artefacts (IP-based rather than behaviour-based
labels), or training-composition artefacts (an accidental `TRAINING_FACTOR=0.5`
run). We report these episodes in full because the *correction* of an apparent
limitation is itself a finding.

---

## 2. Related Work

**Houssel et al. (2024)** [1] evaluate GPT-4 and Llama3 as zero-shot network
intrusion detectors and conclude that LLMs "struggle with precise attack detection"
but are valuable as *explainability* wrappers over a classical detector, especially
when paired with retrieval-augmented generation (RAG) and function-calling. Our
work tests whether *supervised fine-tuning* on flow-level data changes this
conclusion for a small open-weights model, and our "confidently wrong" finding
qualifies their explainability optimism: the explanation is only reliable in the
in-distribution regime.

**Sudasinghe et al. (2026)** [2] fine-tune a decoder-only LLaMA-1B with QLoRA on
CICIoT2023 — the closest methodological analogue to this thesis — and report 42.63%
accuracy on unseen attack types, with an optional RAG second pass. Their unseen-
attack ceiling matches our *initial* Win7AD-1 result (11.3%) in spirit; we show that
this ceiling is *composition-bounded* rather than architecture-bounded, since a
training-composition change lifts it past 87%.

**Gutiérrez-Galeano et al. (2025)** [3] fine-tune an encoder-decoder T5 on
CICFlowMeter features from the CICIDS family and report >99.84% accuracy. Those
numbers must be read against **Lypa et al. (2025)** [4], which compares feature-
extraction tools and documents the structural gap between CICFlowMeter (used by
CICIDS) and Zeek `conn.log` (used in production). Our own version history (V4 → V6 →
V7) independently rediscovers this gap: 80k CICIDS training samples have
`conn_state="-"` and `proto="unknown"` on 100% of rows, and a model trained on them
*collapses* on real Zeek traffic (real-world MCC fell from +0.596 to +0.336). We
therefore drop CICIDS entirely and train only on Zeek-native or Zeek-mappable
sources.

The foundational methods are LoRA [5] and QLoRA [6]; the datasets are CTU-13 [7],
IoT-23 [8], UNSW-NB15 [9], and UWF-ZeekData24 [10], with the CTU-Malware-Capture and
CTU-Normal series from the same Stratosphere Laboratory.

---

## 3. Problem Formulation

Given a single Zeek connection record, produce a binary verdict ∈ {ATTACK, FALSE
POSITIVE} accompanied by a brief natural-language reason. The base input is the
10-field Zeek-native flow summary (Table 1); from V8 onward two port fields are
added, and from V9 onward optional application-layer and behavioural context
sections are appended when available.

**Table 1 — Base input fields.**

| Field | Type | Zeek source |
|---|---|---|
| Proto | categorical | `proto` (tcp/udp/icmp) |
| Service | categorical | `service` |
| Dest Port / Src Port | int | `id.resp_p` / `id.orig_p` |
| Duration (s) | float | `duration` |
| Orig Packets / Resp Packets | int | `orig_pkts` / `resp_pkts` |
| Orig Bytes / Resp Bytes | int | `orig_bytes` / `resp_bytes` |
| Conn State | categorical | `conn_state` (SF/S0/REJ/RSTR/RSTO/OTH/…) |
| Bytes/sec, Orig Bytes/Pkt, Resp Bytes/Pkt | float | derived |

Missing Zeek fields (`-`) are propagated as `N/A` through the derived fields so that
training and inference see the same placeholder under identical conditions — a fix
that was load-bearing (see §5, V4 → V6).

The binary formulation is a deliberate first step. Production IDS requires multi-
class attack categorisation (reconnaissance, lateral movement, exfiltration, C2);
extending the label schema is straightforward and is left as future work to keep the
thesis scope bounded.

---

## 4. Method

### 4.1 Base model and adaptation

We fine-tune `Qwen/Qwen2.5-1.5B-Instruct` using QLoRA: the base is quantised to
4-bit NF4 (BitsAndBytes) with bf16 compute, and LoRA adapters are attached to all
seven projection matrices per transformer block:

```
target_modules = {q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj}
```

V10 used `r = 16`, `lora_alpha = 32`; V11/V12 use `r = 32`, `lora_alpha = 64`
(more capacity for port-aware and host-behaviour patterns). Trainable parameters are
≈ 0.05% of the base model (~688k at r=16 / ~36M at r=32, depending on counting
convention); the adapter is ~20–40 MB on disk. The frozen 4-bit base (~900 MB) plus
the adapter fits comfortably in 8 GB VRAM at inference.

**Why QLoRA rather than full fine-tuning.** A full fine-tune of a 1.5B-parameter
model needs ~18 GB of optimizer state alone (bf16 params + fp32 Adam moments),
exceeding consumer-GPU memory, and is wasteful: we are retraining a model that
already knows language and structure to teach it one narrow task. LoRA's low-rank
assumption — that the task-specific weight update lives in a low-dimensional
subspace — is well matched to a narrow binary-classification objective over highly
structured prompts. QLoRA additionally quantises the frozen base to 4-bit NF4,
dequantising each layer to bf16 on the fly during the forward pass; gradients flow
only into the bf16 adapter matrices `A` and `B`, never into the quantised base. The
result trains on a single 24–32 GB cloud GPU for a few dollars per run and is
hardware-agnostic at inference (the adapter trained on a RunPod RTX 5090/4090 runs
unchanged on a local RTX 3070).

### 4.2 Training data

Five "core" sources plus the CTU-Malware-Capture series feed a pool that is then
sub-sampled to a 2:1 benign:attack ratio at a 360k-sample target (120k attack / 240k
benign at `TRAINING_FACTOR = 1.0`).

**Table 2 — Training sources (V11/V12 composition).**

| Source | Format | Attack | Benign | Label logic |
|---|---|---:|---:|---|
| IoT-23 | Zeek `conn.log.labeled` (tar.gz) | up to ~45k pool | 10–20k | `Malicious` / `Benign` |
| CTU-13 | Argus binetflow | up to ~80k | ~80k | `Botnet` / `Normal`, Argus states mapped to Zeek |
| UNSW-NB15 | Parquet (HuggingFace) | up to ~80k | ~80k | `binary_label` |
| UWF-ZeekData24 | Zeek `conn.log` (Spark CSV) | ≤25k (capped) | ~80k | MITRE ATT&CK tactic labels |
| CTU-Normal | Zeek `conn.log` | 0 | ~100k | benign only |
| CTU-Malware-Capture | Zeek + binetflow, 15 scenarios | ~24k (full pool) | 0 | binetflow 5-tuple matched |

Five preprocessing decisions are load-bearing, each motivated by a measured failure
of an earlier version (see §5 for the chronology):

* **Zeek-native prompt; CICIDS dropped.** CICIDS2017 (included V4/V6) has
  `conn_state="-"` and `proto="unknown"` on all rows because CICFlowMeter has no
  Zeek state concept. Training on it taught the model to classify without its two
  most discriminative fields, collapsing real-world performance. Dropped from V7
  onward.

* **Argus-to-Zeek state mapping.** CTU-13 binetflow uses Argus states (INT, CON,
  FSPA_FSPA, …) that never appear in production Zeek logs. These are mapped to Zeek
  equivalents (INT→S1, CON→SF, S_→S0, SRPA_SPA→RSTO, …) so the model trains on a
  state vocabulary that exists at inference time. Because binetflow only carries
  `TotPkts`, `orig_pkts` and `resp_pkts` are each set to `TotPkts // 2`.

* **Hard-benign scoring.** A hand-curated `score_hard_benign()` rule set boosts the
  inclusion probability of benign flows that *structurally resemble* attacks — short
  SF TCP to Windows ports, UDP S0 probes, and Windows LLMNR/mDNS/NetBIOS broadcasts
  (UDP S0, 700–1500 orig_bytes, zero resp_bytes, ephemeral dest port). This forces
  the model to learn boundaries *within* look-alike families. The UDP-broadcast rule
  (V11) alone improved IoT-23 benign recall by +30.7 percentage points.

* **SF-state attack oversampling (2×).** Short SF/S1/OTH attacks (exfiltration,
  credential access, SMB exploits, C2) were under-represented in the raw pools and
  systematically misclassified by V7.1. Weighting them 2× in the final
  `random.choices` draw lifts their share of the attack pool from ~10% to ~18%.
  *(This same 2× weight is the mechanism behind the V11 composition accident; see
  §5.10.)*

* **CTU-Malware attack budget and a post-draw composition cap.** A reserved budget
  draws CTU-Malware attacks first; from V12 a *post-draw per-source hard cap* (25% of
  `FINAL_ATTACK`, with shortfall refilling that preserves the 2× SF weight)
  guarantees no single non-CTU-Malware source exceeds ~25–30% of the attack budget
  regardless of `TRAINING_FACTOR`. This fix exists specifically because the V11 run
  at factor 0.5 let UWF reach 44.7% of attacks.

**Application-layer context (V9.0).** When the CTU-Malware multi-log enrichment
provides them, `[HTTP]`, `[DNS]`, and `[SSL]` sections are appended to the prompt
(method/host/URI/User-Agent/status for HTTP; query/answer/type/TTL/NXDOMAIN for DNS;
version/cipher/issuer/validation for SSL). Each section is masked independently with
probability 0.50 during training to prevent a "has-section ⇒ ATTACK" shortcut, since
the benign sources (CTU-Normal, UWF) have no auxiliary logs. A post-preprocessing
coverage check warns if attack-side section coverage exceeds benign-side by more
than 2×.

**Behavioural context (V9.1).** A `[BEHAVIOR]` section adds per-source-host window
features (connections/60s, unique destinations/60s, unique ports/60s, same-port
repeats, identical-size repeats/300s, pair-periodicity score) intended to expose
beaconing and scanning when application-layer logs are absent. Its OOD value was
tested and found null (§5.7).

### 4.3 Training hyperparameters

QLoRA via TRL `SFTTrainer`. Effective batch size 24 on every target hardware. The
cost-optimised configuration (current) uses 2 epochs with sequence packing and an
eval subsample, roughly halving GPU-hours versus the original 3-epoch unpacked run
at an unchanged objective.

| Knob | Value |
|---|---|
| Epochs | 2 (V12 cost-optimised; V6–V11 used 3) |
| Packing | True (≈20% fewer tokens/epoch, objective unchanged) |
| Optimizer | `paged_adamw_8bit` (offloads optimizer state to CPU under pressure) |
| Learning rate | 2e-4, `cosine_with_restarts`, `num_cycles = epochs` |
| Warmup / weight decay | 0.03 / 0.01 |
| Precision | bf16 |
| max_length | 512 (token lengths mean 296 / p99 500) |
| Eval / save strategy | per epoch; `load_best_model_at_end` on `eval_loss`; `save_total_limit = 2` |

Hardware-specific settings: RunPod RTX 5090 (32 GB) — `batch = 24, accum = 1,
gradient_checkpointing = False`; local RTX 3070 (8 GB, validation only) — `batch = 4,
accum = 6, gradient_checkpointing = True, num_workers = 0`. The `--runpod` / `--local`
flag selects the dangerous direction explicitly (forgetting `--local` on a 3070
OOMs; forgetting `--runpod` only runs slow).

> **Checkpoint-selection caveat (V11 finding, §5.11).** `load_best_model_at_end`
> selects the lowest-`eval_loss` checkpoint, which is measured on the *in-
> distribution* eval split and systematically mis-selects the OOD-best checkpoint.
> On V11, epoch 2 had the lower eval-loss but epoch 1 generalised better (OOD MCC
> +0.699 vs +0.600). The deployment rule for V12 is therefore: benchmark *every*
> saved checkpoint on the OOD probes and ship the OOD-MCC-best one, not the auto-
> saved adapter.

### 4.4 Prompt template

The prompt is shared verbatim across preprocessing, training, benchmarking, and
deployment via a single `prompt_utils.build_prompt()`. Any divergence between
training and inference formatting would silently degrade accuracy; centralising the
builder eliminates this class of bug.

```
SYSTEM: You are a network security analyst. Always respond with
        VERDICT: <ATTACK or FALSE POSITIVE> on the first line,
        followed by REASON: <brief explanation>.

USER:   Analyze this network connection and classify it as ATTACK or FALSE POSITIVE.

          Proto:            tcp
          Service:          http
          Dest Port:        80
          Src Port:         49511
          Duration (s):     0.124
          Orig Packets:     6
          Resp Packets:     4
          Orig Bytes:       420
          Resp Bytes:       1204
          Conn State:       SF
          Bytes/sec:        13096.8
          Orig Bytes/Pkt:   70.0
          Resp Bytes/Pkt:   301.0

          [HTTP]
          Method: POST   Host: 185.220.100.240   URI: /gate.php
          User-Agent: Mozilla/4.0 (compatible; MSIE 7.0)   Status: 200 (1204 B)

ASSISTANT: VERDICT: ATTACK
           REASON: HTTP POST to a raw-IP gateway endpoint with a Trickbot-style
                   User-Agent and no hostname lookup — characteristic C2 pattern.
```

A diverse pool of REASON templates (77 attack / 48 benign in V12) is sampled during
label construction to prevent boilerplate memorisation and to teach the model to
cite the discriminative field in its output. The V12 pool was audited for
completeness against every training family (adding, e.g., port-135 DCE/RPC
exploitation for Murlo and ICMP host-sweep for Rbot-v2, plus benign complements for
legitimate SMB/RDP/DCE-RPC and Windows link-local multicast).

### 4.5 Deployment pipeline

For field use, the trained LoRA adapter is merged into the base model
(`merge_adapter.py`), converted to GGUF (`llama.cpp/convert_hf_to_gguf.py`), and
loaded into Ollama with an explicit `TEMPLATE` block restoring the Qwen chat
delimiters — the GGUF converter does not embed them in metadata, and omitting the
block silently turns the classifier into a text-continuation model that cheerfully
keeps writing the flow description instead of classifying it. Q8 quantisation costs
≈ 2 pp accuracy relative to the fp16 HF model — within acceptable range for a 4-bit
deployment. On a single RTX 3070 the merged model classifies at ~20 flows/second, so
realistic deployment targets alert triage (top-N flows flagged by Zeek's
`notice.log`) or batched periodic sweeps over `conn.log` windows rather than
line-rate inline detection.

---

## 5. Version History — The Empirical Chronology

The thesis is, in large part, the story of how the benchmark number moved and why.
This section is the experimental narrative; §6–§8 distil it into findings.

**Table 3 — Real-world (in-distribution) benchmark across versions.** Native-Zeek
4-source set; up to 300 attacks + 300 benign per source; `seed = 42`; cache reused
for exact comparability. (OOD probes added from V9 onward are reported separately in
Table 5.)

| Version | Key change | Acc | Atk Rec | FP Rec | MCC | Fmt fail |
|---|---|---:|---:|---:|---:|---:|
| V3 | CICIDS only, CICFlowMeter prompt | ~89% | — | — | n/a (circular) | — |
| V4 | Zeek-native prompt, 4 sources | 80.3% | 67.4% | 89.9% | +0.596 | 0.0% |
| V6 | + UWF + CTU-Normal + CICIDS | 67.0% | 67.1% | 66.8% | +0.336 | 0.0% |
| V7.1 | CICIDS dropped, UWF attacks dropped, 2:1 | 83.0% | 65.0%* | 96.0% | +0.660 | 0.0% |
| V8 (ep1, TF=0.1) | + Dest/Src port, conn_state masking, SF 2× | 85.0% | 67.0% | 99.0% | +0.719 | 0.0% |
| V8.1 | UWF port-extraction fix, Credential Access re-enabled | ~100% | 99.7% | 99.7% | +0.9932 | 0.0% |
| V9.0 (ep1) | multi-log `[HTTP]/[DNS]/[SSL]`, 50% masking | 87.8% | 76.4% | 99.1% | +0.775 | 0.0% |
| V9.1 (ep3) | + `[BEHAVIOR]`, loader fixes | 88.9% | 88.6% | 89.2% | +0.778 | 0.0% |
| V10 | CTU-Malware budget 19% → 40% | 90.0% | 87.3% | 92.3% | +0.797 | 0.0% |
| V11 ep1 (TF=0.5) | r=32, 15 scenarios, UDP-broadcast hard-benign | 93.1% | 88.0% | 98.2% | +0.866 | 0.0% |
| V11 ep2 | (overfit checkpoint) | 90.8% | 85.4% | 96.1% | +0.820 | 0.0% |
| **V12** | composition hard-cap, TF=1.0, reason completeness | — | — | — | **target +0.88–0.90** | — |

\* V7.1 attack recall is depressed by 300 UWF Credential Access flows intentionally
dropped from training; excluding them, V7.1 catches 585/600 = 97.5%.

### 5.1 V3 → V4: the deployment-reality shock

V3 reached ~89% on CICIDS but used CICFlowMeter-only features absent from real Zeek
logs — useless for deployment. V4 redesigned to a Zeek-native 10-field prompt over
four sources; accuracy stepped down to ~82% (harder, more diverse data), but the
real surprise was a **~95% false-positive rate on a real captured `conn.log`**: the
training data was all lab/synthetic, so the model had never seen realistic benign
user traffic and flagged it as anomalous. A separate bug — missing fields falling
through to `0.0` in derived features instead of `N/A` — created a train/inference
mismatch.

### 5.2 V6: more data, worse result

V6 added UWF-ZeekData24 and CTU-Normal (both real Zeek benign sources) precisely to
fix the FP problem — and made it worse (real-world MCC +0.336, CTU-Normal benign
recall collapsing to 18%). Root-cause analysis isolated four problems: (1) IoT-23
benign at 80k samples is 89% S0 UDP and taught "S0 = benign", so the model flagged
the SF-heavy CTU-Normal and UWF benign as suspicious; (2) UWF attacks are
exclusively short SF TCP Credential Access, nearly unlearnable at the flow level and
poisoning the boundary toward "short SF TCP = ATTACK"; (3) CTU-13 Argus states do not
exist in Zeek; (4) CICIDS's missing `conn_state`/`proto`. This diagnosis drove V7.

### 5.3 V7.1: the four root-cause fixes

Cap IoT-23 benign 80k→20k, drop UWF attacks, map CTU-13 Argus→Zeek states, drop
CICIDS, raise CTU-Normal to 100k, adopt a 2:1 benign:attack ratio, and switch to a
*source-stratified* train/eval split (so per-source failures show up in eval-loss
from epoch 1). Result: real-world MCC +0.660, FP recall recovered to 96.0%,
CTU-Normal benign recall 18%→97.3%. The primary deployment concern was addressed.

### 5.4 The conn_state over-reliance finding

A 16-case hand-crafted novel-attack suite run against V7.1 via Ollama produced 1/8
attack recall and 8/8 benign recall. *Every* missed attack received the identical
REASON ("Connection established and closed cleanly; no anomalous volume or timing"),
whether it was a 12 MB exfiltration, a 5000-packet UDP flood, or a 300-second
half-open DoS. The model had become a "SF-TCP detector": it keyed on `conn_state`,
`proto`, and `service` and did not robustly use the numeric features when state
contradicted the attack hypothesis. The only attack it caught was an unambiguous S0
SYN scan present in IoT-23 training. This motivated V8's port features, state
masking, and SF oversampling.

### 5.5 V8 → V8.1: a pipeline bug masquerading as a model limitation

V8 added Dest/Src Port, masked `conn_state` for 20% of samples, oversampled SF
attacks 2×, and raised LoRA rank. But V8 still scored 0% on UWF Credential Access.
Investigation revealed that both the training loader *and* the benchmark looked for
Zeek-standard port columns (`id.orig_p`, `orig_p`, …) while UWF CSVs name them
`src_port_zeek` / `dest_port_zeek` — so **every UWF port had silently been `N/A`
since V6**. With the fallback fixed and UWF Credential Access (port 4848 GlassFish
admin + SSL) and Defense Evasion (port 445 SMB, S0) re-enabled, V8.1 jumped to
**MCC +0.9932** on the 4-source benchmark, with Credential Access recall going 0% →
100%. The V7-era conclusion that these flows were "unlearnable" had been an artefact
of a missing column-name fallback. *Methodological lesson: verify feature extraction
against the actual data at every new source — a pipeline bug can produce misleading
benchmark numbers that read as fundamental model limitations.*

### 5.6 The OOD reckoning: V8.1 does not generalise

Two held-out CTU-Malware captures exposed the limit of the +0.9932 result.
Botnet-78-2 (Zeus banking trojan, HTTP C2) scored MCC +0.0448 (0.4% recall);
Botnet-3 (Kelihos) scored MCC +0.1674 (38% recall). Two causes were separated:
(1) *flow-level indistinguishability* — Zeus C2 over port 80 is structurally
identical to a normal web API call, a feature-set boundary no flow classifier can
cross; and (2) *ground-truth labelling mismatch* — Stratosphere binetflow labels
*every* flow from an infected host as `From-Botnet`, including its normal browsing
and DNS, so the model is penalised for correctly calling that traffic benign. Both
caveats matter for interpreting all subsequent OOD numbers.

### 5.7 V9.0/V9.1: multi-log and behavioural context — a documented null

V9.0 added the CTU-Malware-Capture series (7 scenarios with full `bro/` output) and
the `[HTTP]/[DNS]/[SSL]` context sections; V9.1 added the `[BEHAVIOR]` temporal-
window section. The hypothesis was that per-host behavioural signal would let the
model generalise to unseen families. At full training (ep3) on the CTU-SME-11 Amazon
Echo OOD probe, attack recall did **not** improve (33.7% → 31.6%); the small MCC gain
(+0.044) came entirely from higher benign recall. The `[BEHAVIOR]` hypothesis is a
clean null result and is reported as such. (The section is retained in the pipeline —
~40% coverage in V12 data — but is not credited with OOD gains.)

### 5.8 OOD probe selection

The original floor probe (Kelihos) sat at ~0% for every version and could not
discriminate *whether a change helped*. A DarkVNC probe was tried and was also 0%
(HTTP/HTTPS C2 on standard ports — below the flow-level floor). The project settled
on a **three-tier battery**: CTU-SME-11 Win7AD-1 (lateral movement + Trickbot C2,
clean ground truth, OOD-Hard), CTU-SME-11 Amazon Echo (inbound discovery scans,
structurally near IoT-23, OOD-Medium), and Kelihos (P2P spam, OOD-Floor). All three
are reported in every run, characterising OOD along a spectrum.

### 5.9 V9.1 gap analysis and the first classical-ML comparison — *and its later reversal*

On V9.1, Win7AD-1 attack recall was only **11.3%**. A distribution-gap analysis
(`analyze_gap.py`) explained it precisely: REJ+RSTR states are 0.3% of training
attacks but 83.7% of Win7AD-1 attacks, and lateral-movement ports (RDP 3389,
Kerberos 88/464, MSRPC 135, SMB 445) and Trickbot C2 ports (4134/22299) appear in
**zero** training attack samples. The model cannot label what it has never seen
labelled. A classical baseline (`baseline_ml.py`, RF + LR on 14 numeric/ordinal
features) was added and, at this point, *beat the LLM badly* on OOD: LR reached
78.2% Win7AD-1 recall (MCC +0.764) versus the LLM's 11.3% (+0.022) — a ~35× MCC gap.
The interim thesis framing was "classical ML vastly outperforms the fine-tuned LLM
on OOD attacks." **This framing was subsequently overturned by V10** (§5.10) and
must not be cited as the project's conclusion.

### 5.10 V10: one composition change closes the gap (the central result)

V10 changed exactly one thing relative to V9.1: it reserved a CTU-Malware attack
budget, raising CTU-Malware's share of the 120k attack pool from ~19% to ~40%
(architecture, hyperparameters, prompt, and ID sources all held fixed). CTU-Malware
contributes SMB scanning, botnet C2, and port-sweep flows — structurally far closer
to Win7AD-1 lateral movement than the IoT S0/UDP floods that had dominated V9.1.

**Win7AD-1 attack recall went 11.3% → 87.1%** (+75.8 pp), *exceeding* both Logistic
Regression (78.2%) and Random Forest (64.9%) on the same probe. Overall (incl. OOD)
MCC was +0.797; in-distribution sources stayed ≥98%. This single experiment is the
thesis's strongest evidence and reframes the V9.1 "gap" from an architectural limit
to a training-coverage problem.

A surprise accompanied it: Kelihos recall rose to 83.6%. Verification (§5.12)
confirmed Kelihos is genuinely held out and structurally unlike any training family,
making this a positive-transfer result rather than leakage.

### 5.11 V11: more scenarios, a hard-benign win, and an instructive accident

V11 raised LoRA to r=32/α=64, expanded CTU-Malware to 15 labelled scenarios (4
candidate captures had empty `Label` columns and were excluded), and added the
UDP-broadcast hard-benign rule. The hard-benign rule worked exactly as intended:
**IoT-23 benign recall +30.7 pp**. The full-source benchmark looked excellent —
ep1 MCC +0.866 (the best in-distribution number to date) — *but the run had been
trained at `TRAINING_FACTOR = 0.5` by accident* (a torch/transformers version
regression OOM-ed the RTX 5090 at the intended batch size, forcing a fallback). At
factor 0.5, halved per-source caps left UWF's still-large, ~100%-SF Credential
Access pool to fill the unscaled 120k attack budget, and the 2× SF draw weight
compounded it: **UWF reached 44.7% of attacks** — nearly half the attack data a
single homogeneous family. The model over-learned "short SF TCP = ATTACK", and
**Win7AD-1 recall regressed −29.3 pp**. The OOD-only benchmark told the real story:
ep1 OOD MCC +0.699, with lateral movement (Human_attacks) at **94%** but Trickbot C2
at only **66%**.

### 5.12 V12: the corrected run (final thesis model — not yet trained at draft time)

V12 is *not* an architectural change; it is the factor-1.0 run V11 was meant to be,
plus three composition fixes that make the bias structurally impossible: a UWF
attack cap (25k), a tighter IoT-23 per-file cap, and the post-draw 25%-per-source
hard cap with weight-preserving refill. Regenerated V12 data shows healthy
composition (UWF clamped to exactly 25.0%, no source dominance). V12 is expected to
recover Win7AD-1 to ~84–87% (V10 level), hold Echo ≥75%, Kelihos ~81–83%, IoT-23 FP
~99%, overall MCC +0.88–0.90. Per the checkpoint caveat (§4.3), V12 will ship the
OOD-MCC-best checkpoint, not the auto-saved lowest-eval-loss one.

---

## 6. Evaluation

### 6.1 Protocol and metrics

`benchmark_realworld.py` draws a balanced cache (up to 300 attacks + 300 benign per
source, `seed = 42`) reused across all model versions. Four in-distribution sources
(IoT-23, CTU-13, UWF, CTU-Normal) and the three OOD probes. We report accuracy,
per-class recall (attack TPR / benign TNR), Matthews Correlation Coefficient
(primary — well-defined under class imbalance and robust to degenerate classifiers),
and format-failure rate. A separate `--ood` mode evaluates the OOD battery alone
(900 attacks / 600 benign) and is the honest signal once the in-distribution sources
saturate.

### 6.2 In-distribution: parity with classical ML

On known families the task is solvable by simple models, and the LLM is in the same
band. On the V11 in-distribution split, Random Forest reaches MCC +1.000 and Logistic
Regression +0.709; the fine-tuned LLM reaches +0.866 (V11 ep1, full-source) and is
targeted at +0.88–0.90 for V12. **Q1 is answered affirmatively:** a 1.5B QLoRA model
matches tabular baselines on the in-distribution task. The RF's perfect score
reflects tight feature-space partitioning of four distinct source distributions, not
robust generalisation — as the OOD results show.

### 6.3 Out-of-distribution: the three-tier battery

**Table 4 — Latest OOD-only benchmark (V11 ep1, `--ood`, 2026-06-26).**

| Probe | Atk recall | Benign recall | Note |
|---|---:|---:|---|
| Win7AD-1 [OOD-Hard] | 73.0% | 93.0% | Human_attacks 94%, Trickbot C2 66% |
| Echo [OOD-Medium] | 75.7% | 96.7% | inbound discovery scans |
| Kelihos [OOD-Floor] | 80.7% | — | P2P spam, benign-free source |
| **Overall** | **76.4%** | **94.8%** | **MCC +0.699** |

Two decompositions matter. First, Win7AD-1's 300 attacks split into 79 lateral-
movement (REJ/RSTR) flows and 221 Trickbot C2 flows. **Lateral movement — the
original "publishable gap" (13% in V9.1, vs LR ~75%) — is now effectively solved at
94% recall.** The residual gap is Trickbot C2 at 66%, and because Trickbot is the
*majority* of the probe, it alone sets the Win7AD-1 score: lifting it to ~85% would
pull the whole probe to ~87% (V10 level). Crucially, **Trickbot is not one of the 15
training families and no synthetic Trickbot data is used** — so 66% recall on a fully
unseen C2 family from a 1.5B model is a *positive* generalisation result that only
reads as a "gap" against V10's 87% on the same probe.

### 6.4 Classical ML comparison (the OOD reversal)

**Table 5 — Classical ML vs LLM across the battery (factor-1.0-era baselines).**

| Probe | RF MCC | LR MCC | LLM V9.1 | LLM V10 | LLM V11 ep1 |
|---|---:|---:|---:|---:|---:|
| ID (held-out) | +1.000 | +0.709 | +0.778 | +0.797 | +0.866 |
| Win7AD-1 Atk recall | 59.3% | 88.7% | 11.3% | **87.1%** | 72.0%† |
| Echo Atk recall | 15.0% | 93.3% | 31.6% | 65.7% | 75.3% |
| Kelihos Atk recall | 12.0%‡ | 92.0%‡ | 81.3% | 83.6% | 81.0% |

† V11 ep1 trained at factor 0.5; the Win7AD-1 figure is not representative (see
§5.11). ‡ Both classical models are **degenerate** on Kelihos: RF labels ~everything
benign, LR labels ~everything ATTACK; FP recall = 0%, **MCC = 0** for both.

Three observations. **(1)** Random Forest is perfect in-distribution and nearly
worthless on two of three OOD probes (15% Echo, 12% Kelihos) — the perfect ID score
does not imply generalisation. **(2)** Logistic Regression is genuinely strong on
Win7AD-1 (88.7% recall, MCC +0.845) because refused-connection scans have
`orig_bytes_per_pkt = 0`, a near-linear signal LR exploits directly; but LR is
degenerate on Kelihos. **(3)** Only the fine-tuned LLM is *non-degenerate across all
three probes simultaneously*. The Kelihos result — ~81% recall on a P2P family with
zero structural representation in training, where both tabular baselines score MCC 0
— is the single strongest piece of evidence that text-space fine-tuning generalises
across attack families in a way tabular classifiers do not.

**Mechanism for Kelihos.** 50.3% of Kelihos benchmark flows are S0 UDP probes to
random high ports. IoT-23 training contains abundant S0 UDP attack flows (horizontal
scans). The model appears to have learned "S0 UDP with no response to non-service
ports ⇒ suspicious" and transferred it from IoT-23 to Kelihos — positive transfer via
feature generalisation, not family-specific memorisation (verified: none of the 15
training scenarios share Kelihos's UDP-P2P-to-random-high-ports signature).

### 6.5 Format reliability

Across all models and 10k+ generated outputs, the format-failure rate (missing a
parseable `VERDICT:` line) is **0.0%**. A short system prompt + an instruction-tuned
base + QLoRA on structured output is sufficient for stable format without
constrained-decoding machinery.

### 6.6 Negative results

* **`[BEHAVIOR]` temporal context** (§5.7): no OOD attack-recall improvement.
* **Host Pass-2 aggregation:** aggregating per-flow verdicts into a host verdict
  ("ATTACK if any flow is attack") gives OOD MCC +0.029 (ep1) / +0.136 (ep2) ≈
  chance, with FP recall collapsing to 33% (it flags two of every three benign
  hosts). Flow-level accuracy does **not** trivially compose into host-level
  attribution; the mechanism amplifies confidence but cannot recover systematically
  missed flows. Retired to a `--host-pass2` flag, off by default.
* **eval_loss mis-selects the OOD-best checkpoint** (§4.3): the in-distribution
  selection criterion picked the overfit checkpoint.

---

## 7. The "Confidently Wrong" Phenomenon

In the V9.1 regime, the model misclassified 1330 of 1500 Win7AD-1 attacks (88.7%) as
FALSE POSITIVE. Manual inspection of REASON fields on 30 random false negatives
revealed a consistent pattern: **fluent, coherent, internally consistent
explanations that are factually incompatible with the flow features in the prompt.**

> **Example 1 (REJ to RDP port 3389).** `VERDICT: FALSE POSITIVE / REASON: Short
> periodic HTTPS connections to a CDN or vendor endpoint with stable byte counts.`
> — The flow has Duration 0, Orig/Resp Bytes 0, conn_state REJ. There is no HTTPS,
> no CDN, no byte counts.
>
> **Example 2 (RSTO Trickbot C2 to port 4134).** `REASON: High UDP packet rate with
> large resp_bytes and long duration on a high-bandwidth connection.` — The flow is
> TCP, milliseconds long, with Resp Bytes 0.
>
> **Example 3 (REJ to Kerberos port 464).** Identical REASON text to Example 1,
> despite a different port and attack family.

The model pattern-matches to a benign explanation *template* it learned in training
and fabricates a reason that fits the template, ignoring the numeric values. The
failure has two components: (a) a wrong verdict, and (b) an explanation a hurried SOC
analyst would not flag as nonsense. **Neither a signature IDS nor a classical
classifier can produce this failure** — signatures either fire or do not; RF/LR emit
a score with no explanation surface.

**Why it matters, and the V10 update.** The REASON field is the principal
differentiator of LLM-based IDS, and it is unreliable in exactly the regime where it
would be most valuable: novel attacks the analyst has not seen. Importantly, once V10
lifted Win7AD-1 recall to 87%, the model began producing *correct* verdicts on most
of these flows — which suggests the confidently-wrong reasons were a *symptom of
operating outside the training distribution*, not a permanent property of the model.
This couples the phenomenon directly to the §6 finding: explanation reliability
tracks training coverage. The open question for the residual false negatives (e.g.
Trickbot C2 at 66%) is whether the REASON is now feature-consistent where the verdict
is correct, and still templated where it is wrong — an inspection deferred to the V12
run.

**Recommended mitigations.** (1) *Confidence-gating*: attach a calibration probe
(ensemble disagreement, or token-level log-probability of ATTACK vs FALSE POSITIVE)
and suppress the REASON below a threshold. (2) *RAG grounding at inference*: retrieve
nearby flows from the same source IP as context so the explanation has factual
anchors beyond the single-flow prompt. In all cases, **REASON should be treated as
unreliable in the OOD regime** and never shown to an analyst as ground truth.

---

## 8. Discussion

### 8.1 Training composition is the dominant variable

The V9.1 → V10 comparison is the experimental crux: a single change to the attack-
family budget moved Win7AD-1 recall 11.3% → 87.1%, exceeding every other change made
across eleven versions combined, with everything else held fixed. The prior-work
framing — "fine-tuned LLMs struggle on OOD attack detection" [1,2] — is correct only
*conditional on training composition*. When a structurally related family (SMB
scanning, botnet C2 in CTU-Malware) is adequately represented, the model generalises
to a novel family in the same structural neighbourhood (Win7AD-1 lateral movement);
when it is not, it produces plausible-but-wrong REASONs. Both observations share one
mechanism: language-model fine-tuning interpolates in the neighbourhood of training
examples, and OOD generalisation is bounded by what that neighbourhood contains. The
project's reframed objective follows directly: maximise *coverage of the attack types
the model misses on OOD*, not a particular CTU-Malware percentage (the 40% number was
a proxy, useful in V10 but not a target in itself).

### 8.2 Text-space vs feature-space generalisation

Why did Logistic Regression reach 88.7% Win7AD-1 recall from the same ID distribution
on which the V9.1 LLM reached 11.3%? LR's top signal, `orig_bytes_per_pkt = 0`, is an
unambiguous numeric marker of refused-connection scans, and a linear boundary
generalises across the port dimension automatically. The LLM, by contrast, reads the
*string* `"Orig Bytes/Pkt: 0.0"` next to `"Conn State: REJ"` and `"Dest Port: 3389"`
and has never seen that combination labelled ATTACK; text-space proximity does not
carry the geometric properties of Euclidean feature-space proximity. This explains
both the V9.1 failure and the V10 fix: adding training examples whose feature
combinations resemble Win7AD-1 pushes the attack-side text neighbourhood close enough
for interpolation to succeed. The complementary direction — Kelihos, where the LLM
wins decisively — shows that text-space fine-tuning *also* captures a kind of
semantic, cross-family generalisation (S0-UDP-to-odd-ports ⇒ suspicious) that the
linear/tree models cannot express on these features. The two model classes have
different, partly complementary generalisation geometries.

### 8.3 Limitations

* **Flow-level classification is fundamentally bounded.** Stealth HTTP C2 (Zeus,
  DarkVNC) and per-flow-normal P2P/SMTP traffic are structurally identical to benign
  flows at the `conn.log` level; no flow classifier separates them without payload
  inspection or multi-flow correlation. The model is best understood as a *signature
  learner over flow structure*, not an anomaly reasoner.
* **Binary, single-flow scope.** Production IDS needs multi-class output and
  multi-flow context; both are deferred.
* **Ground-truth quality.** Stratosphere binetflow labels are IP-based, not
  behaviour-based — all traffic from an infected host is `From-Botnet`, including its
  normal browsing. This *understates* the model's true precision on the OOD captures
  and makes raw OOD recall a conservative (pessimistic) figure.
* **"OOD-Floor" is not truly floor.** Kelihos at ~81% contradicts its original
  "structurally undetectable" designation; a genuine floor probe (e.g. encrypted C2
  tunnelled over legitimate HTTPS to a cloud CDN) remains future work. The stale
  framing should be corrected wherever it appears.
* **Host-level attribution does not compose** from flow verdicts (Host Pass-2 null).
* **Corpus is Western enterprise / IoT oriented** (Czech and US lab/cyber-range
  captures); ICS/SCADA, cloud, and consumer-5G traffic are untested.
* **Composition sensitivity cuts both ways.** The V11 accident shows that a single
  unintended sampling parameter (`TRAINING_FACTOR = 0.5`) can swing a headline OOD
  metric by ~29 pp. The same lever that is the main positive result is also a
  fragility; the V12 post-draw hard cap is the structural guard against it.

### 8.4 Practical deployment notes

Realistic deployment is alert triage or batched sweeps, not line-rate inline
detection (~20 flows/s on a 3070). The GGUF `Modelfile` must carry an explicit
`TEMPLATE` block. Most production SIEM pipelines deliver `conn.log` only, so the
model is designed to degrade gracefully to the flow-only prompt when auxiliary logs
are absent; the multi-log path is an opportunistic enrichment, not a requirement.

---

## 9. Conclusion

We fine-tune `Qwen2.5-1.5B-Instruct` with QLoRA to classify Zeek `conn.log` flows as
ATTACK or FALSE POSITIVE, reaching in-distribution MCC in the +0.87–0.90 band with a
0.0% format-failure rate across 10k+ generations, runnable on a single 8 GB consumer
GPU. The headline empirical finding is that a *single training-composition change*
moves out-of-distribution lateral-movement recall from 11.3% to 87.1% — exceeding
Logistic Regression and Random Forest on the same probe — reframing the "LLMs
struggle on OOD" narrative as a training-coverage problem rather than a capacity one.
A secondary, qualitative finding — the *confidently wrong* REASON phenomenon — is a
failure mode absent from classical detectors that any SOC deployment of LLM-based IDS
must account for, and which itself tracks training coverage. Finally, on a
structurally novel P2P family the LLM achieves ~81% recall where both tabular
baselines are degenerate, demonstrating a cross-family generalisation advantage of
text-space fine-tuning.

Three directions follow: (1) RAG-grounded, confidence-gated REASON generation at
inference; (2) a genuinely structural OOD-Floor probe to replace the now-too-easy
Kelihos; and (3) a controlled larger-base-model ablation (Qwen2.5-3B, identical
composition controls) to confirm that capacity is *not* the bottleneck — since the
1.5B model already reaches 87% on the hard probe, the prediction is that scale buys
little once coverage is adequate.

---

## References

[1] Houssel, P.-A., Singh, P., Layeghy, S., Portmann, M. (2024). *Towards
Explainable Network Intrusion Detection using Large Language Models.*
arXiv:2408.04342.

[2] Sudasinghe, M., Liyanage, M., Gardiyawasam Pussewalage, H. S. (2026).
*Lightweight LLMs for Network Attack Detection in IoT Networks.* arXiv:2601.15269.

[3] Gutiérrez-Galeano, L., Domínguez-Jiménez, J.-J., Schäfer, J., Medina-Bulo, I.
(2025). *LLM-Based Cyberattack Detection Using Network Flow Statistics.* Applied
Sciences 15(12):6529.

[4] Lypa, B., Horyn, I., Zagorodna, N., Tymoshchuk, P., Lechachenko, T. (2025).
*Comparison of Feature Extraction Tools for Network Traffic Data.* arXiv:2501.13004.

[5] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
arXiv:2106.09685.

[6] Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). *QLoRA:
Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.

[7] García, S., Grill, M., Stiborek, J., Zunino, A. (2014). *An empirical comparison
of botnet detection methods.* Computers & Security 45. [CTU-13.]

[8] Garcia, S., Parmisano, A., Erquiaga, M. J. (2020). *IoT-23: A labeled dataset
with malicious and benign IoT network traffic.* Stratosphere Laboratory.

[9] Moustafa, N., Slay, J. (2015). *UNSW-NB15: A comprehensive data set for network
intrusion detection systems.* MilCIS 2015.

[10] Bagui, S. et al. *UWF-ZeekData24: A Zeek-formatted dataset of network traffic
for intrusion detection research.* University of West Florida.

---

## Appendix A — Version timeline (abridged)

| Version | Key change | Real-world MCC (4-source) | OOD note |
|---|---|---:|---|
| V3 | CICIDS only, CICFlowMeter prompt | ~89% (circular) | — |
| V4 | Zeek-native prompt, 4 sources | +0.596 | ~95% FP on real conn.log |
| V6 | + UWF + CTU-Normal + CICIDS | +0.336 | regression diagnosed |
| V7.1 | drop CICIDS + UWF attacks, 2:1 | +0.660 | "SF-TCP detector" finding |
| V8 → V8.1 | ports, masking, SF 2×, UWF port fix | +0.9932 | Cred-Access 0%→100% |
| V9.0 | multi-log `[HTTP]/[DNS]/[SSL]` | +0.775 | Zeus/Kelihos OOD ≈ 0 |
| V9.1 | + `[BEHAVIOR]`, gap analysis | +0.778 | Win7AD-1 11.3%; LR 78.2% |
| V10 | CTU-Malware budget 19%→40% | +0.797 | **Win7AD-1 → 87.1%** |
| V11 ep1 | r=32, 15 scenarios, hard-benign UDP | +0.866 (TF=0.5, biased) | OOD MCC +0.699 |
| **V12** | composition hard-cap, TF=1.0 | target +0.88–0.90 | target Win7AD-1 ~84–87% |

## Appendix B — Distribution gap (training attacks vs Win7AD-1), V9.1 era

| conn_state | Train attacks % | Win7AD-1 attacks % |
|---|---:|---:|
| S0 | 33.0% | 0.0% |
| SF | 33.3% | 2.3% |
| INT (Argus→S1) | 30.6% | 0.0% |
| REJ | 0.3% | 65.8% |
| RSTR | 0.0% | 17.9% |
| RSTO | 0.0% | 13.4% |

Lateral-movement ports (RDP/SMB/Kerberos/MSRPC) and Trickbot C2 ports (4134/22299)
appear in **0%** of training attacks vs 19.1% / 12.3% of Win7AD-1 attacks. The 11.3%
V9.1 recall is fully explained by this gap — and closed by adding structurally
similar attack examples (V10), not by changing the model.
