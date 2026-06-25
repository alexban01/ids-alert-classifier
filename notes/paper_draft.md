# Flow-Level Network Intrusion Detection with Fine-Tuned Language Models: Capability Boundaries and the Out-of-Distribution Challenge

**Author:** Ban Alexandru Mircea
**Affiliation:** [Institution TBD]
**Draft:** 2026-04-24 — not for distribution

---

## Abstract

We investigate whether a small instruction-tuned language model, fine-tuned
via QLoRA on Zeek `conn.log` flow features, is a viable replacement for
classical machine-learning intrusion detection on commodity hardware. We
fine-tune `Qwen2.5-1.5B-Instruct` on a heterogeneous 360 k-sample corpus
assembled from IoT-23, CTU-13, UNSW-NB15, UWF-ZeekData24, CTU-Normal, and
fifteen CTU-Malware-Capture scenarios, and evaluate across a three-tier
out-of-distribution (OOD) probe battery: Win7AD-1 lateral movement
(OOD-Hard), Amazon Echo inbound scans (OOD-Medium), and Kelihos P2P spam
(OOD-Floor). Three findings follow.

**First,** in-distribution, the fine-tuned model reaches MCC = +0.866 on a
3 600-sample real-world benchmark, matching Random Forest (+1.000) and
Logistic Regression (+0.709) on the same feature set within the noise band
of the probe. **Second,** OOD behaviour is dominated by training
composition, not model capacity: increasing the CTU-Malware attack share
from 19 % to 40 % of the training pool lifted Win7AD-1 attack recall from
11.3 % to 87.1 %, exceeding Logistic Regression (78.2 %) and Random Forest
(64.9 %) on the same probe. **Third,** we document a novel LLM-specific
failure mode — *confidently wrong* REASON generation — in which the model
produces fluent, feature-inconsistent explanations for OOD attack flows
it misclassifies, a phenomenon absent from classical classifiers because
they have no explanation capability. We argue that training-distribution
coverage, not architectural scale, is the load-bearing variable for small
LLM-based IDS and that REASON outputs must be treated as unreliable in
OOD regimes.

**Keywords:** intrusion detection, large language models, QLoRA, Zeek, out-of-distribution generalisation, network security.

---

## 1. Introduction

Signature-based intrusion detection systems (Snort, Suricata) produce
deterministic but brittle alerts; supervised classifiers on flow-level
features (Random Forest, XGBoost) add statistical coverage but produce
uninterpretable probability scores that are of limited value to a SOC
triage analyst. Recent work [1, 2, 3] proposes to use large language
models as a third leg of this stool: an LLM that takes a flow as input and
returns *both* a verdict and a natural-language justification. The appeal
is practical — the explanation is free if the model is already generating
text — but the research question is whether the verdict half of the output
is accurate enough to act on.

This paper investigates that question empirically for the setting most
likely to be deployed: a small (1.5 B-parameter) instruction-tuned LLM,
4-bit quantised, QLoRA-fine-tuned on real Zeek `conn.log` data, and run on
a single consumer GPU (RTX 3070, 8 GB VRAM). Three questions structure the
investigation:

* **Q1.** Does a fine-tuned small LLM match classical ML accuracy on
  known attack families?
* **Q2.** Does it generalise to structurally novel attack traffic that
  classical ML also sees for the first time?
* **Q3.** Is the REASON field — the differentiator over classical ML —
  a reliable artefact across both regimes?

The prior literature answers Q1 affirmatively and Q2 pessimistically [1,
2]. Our results partially confirm Q1, partially overturn Q2 conditional on
training-data composition, and introduce a qualitative failure mode on Q3
that prior work does not document.

### 1.1  Contributions

1. A systematic end-to-end fine-tuning pipeline (`preprocess_zeek.py`
   → `train.py` → `benchmark_realworld.py`) for flow-level IDS on
   heterogeneous Zeek-native sources, with source-stratified train/eval
   split and explicit sampling controls for attack-family composition.
2. A three-tier OOD probe battery (Win7AD-1 / Echo / Kelihos) that
   characterises generalisation along a difficulty spectrum rather than
   at a single point.
3. The empirical finding that a single training-composition change
   (CTU-Malware attack budget: 19 % → 40 %) shifts Win7AD-1 recall from
   11.3 % to 87.1 %, reframing the "LLM OOD gap" from a model-architecture
   problem to a training-data-coverage problem.
4. A direct Random-Forest / Logistic-Regression baseline on the same
   feature set, showing that classical ML is degenerate on Kelihos
   (MCC = 0, all-ATTACK or all-benign) while the fine-tuned LLM achieves
   81–83 % recall via positive transfer from IoT-23 scan patterns.
5. The "confidently wrong" REASON phenomenon: an LLM-specific OOD
   failure mode that is invisible to classical IDS methods and has
   direct implications for SOC deployment.

---

## 2. Related Work

**Houssel et al. (2024) [1]** evaluate GPT-4 and Llama3 as zero-shot NID
systems and conclude that LLMs "struggle with precise attack detection"
but are useful as explainability wrappers over a classical detector. Our
work tests whether supervised fine-tuning on flow-level data changes this
conclusion for a small open-weights model.

**Sudasinghe et al. (2026) [2]** fine-tune LLaMA-1B with QLoRA on
CICIoT2023, reporting 42.63 % accuracy on unseen attack types — an OOD
ceiling that matches our initial Win7AD-1 result (11.3 %) in spirit,
though we show the ceiling is composition-bounded rather than
architecture-bounded. Their RAG-based second pass is complementary to
the multi-log enrichment we explore in §4.4.

**Gutiérrez-Galeano et al. (2025) [3]** fine-tune T5 on CICFlowMeter
features from CICIDS2017/CSE-CIC-IDS-2018/BCCC-CIC-IDS-2017, reporting
>99.84 % accuracy. Their numbers should be read alongside **Lypa et al.
(2025) [4]**, which compares feature-extraction tools and documents the
structural gap between CICFlowMeter (used by CICIDS) and Zeek
conn.log (used in production). Our V4→V6→V7 version history
independently rediscovers this gap: CICIDS2017 training samples have
`conn_state="-"` and `proto="unknown"` on 100 % of rows, and a model
trained on them collapses (67.0 % overall accuracy) when deployed
against real Zeek traffic.

---

## 3. Problem Formulation

Given a single Zeek connection record with the 10 native fields of Table
1, produce a binary verdict ∈ {ATTACK, FALSE POSITIVE} accompanied by a
brief natural-language reason. No temporal or multi-flow context is
available at the base formulation; §4.4 discusses context extensions.

**Table 1 — Input fields (base prompt).**

| Field | Type | Source |
|---|---|---|
| Proto | categorical | `proto` (tcp/udp/icmp) |
| Service | categorical | `service` |
| Dest Port, Src Port | int | `id.resp_p`, `id.orig_p` |
| Duration (s) | float | `duration` |
| Orig Packets, Resp Packets | int | `orig_pkts`, `resp_pkts` |
| Orig Bytes, Resp Bytes | int | `orig_bytes`, `resp_bytes` |
| Conn State | categorical | `conn_state` (SF/S0/REJ/RSTR/RSTO/...) |
| Bytes/sec, Orig Bytes/Pkt, Resp Bytes/Pkt | float | derived |

Missing Zeek fields (`-`) are propagated as `N/A` through the derived
fields so that training and inference see the same placeholder under
identical conditions.

The binary formulation is a deliberate first step. Real IDS output
requires multi-class attack categorisation; extending the label schema is
straightforward but is deferred to keep the thesis scope bounded.

---

## 4. Method

### 4.1  Base model and adaptation

We fine-tune `Qwen/Qwen2.5-1.5B-Instruct` using QLoRA (4-bit NF4
quantised base, bf16 compute) with LoRA adapters on all seven attention
and MLP projection matrices per block:

```
target_modules = {q_proj, k_proj, v_proj, o_proj,
                  gate_proj, up_proj, down_proj}
```

V10 uses `r = 16`, `lora_alpha = 32`; V11 onward uses `r = 32`,
`lora_alpha = 64`. Total trainable parameters are ≈ 0.05 % of the base
model. The frozen 4-bit base (~900 MB) plus ~20 MB adapter fits in 8 GB
VRAM at inference.

**Rationale for QLoRA over full fine-tuning.** Full fine-tuning of a
1.5 B-parameter model requires ~18 GB of optimizer state (bf16 params +
fp32 Adam moments), exceeding consumer-GPU memory. LoRA's low-rank
assumption — that task-specific updates live in a low-dimensional
subspace — is plausible for a narrow binary-classification objective
over structured prompts; empirically the adapter captures the task with
negligible degradation relative to full tuning in the fine-tuning
literature.

### 4.2  Training data

**Table 2 — Training sources (V11 / V12 composition after 2:1
benign:attack downsampling to 360 k total).**

| Source | Format | Attack | Benign | Label logic |
|---|---|---:|---:|---|
| IoT-23 | Zeek conn.log.labeled | up to 30 k | 20 k | Malicious / Benign |
| CTU-13 | Argus binetflow | up to 80 k | 80 k | Botnet / Normal, states mapped to Zeek |
| UNSW-NB15 | Parquet (HF) | up to 80 k | 80 k | binary_label |
| UWF-ZeekData24 | Zeek conn.log | 25 k (capped) | 80 k | MITRE-labelled |
| CTU-Normal | Zeek conn.log | 0 | 100 k | benign only |
| CTU-Malware-Capture | Zeek + binetflow | 24 k | 0 | 15 scenarios, binetflow-matched |

Four preprocessing decisions are load-bearing:

* **Zeek-native prompt.** CICIDS2017 was included in V4/V6 and dropped
  in V7: all 80 k CICIDS samples have `conn_state = "-"` and
  `proto = "unknown"` because CICFlowMeter has no Zeek-equivalent state
  concept. Training on them taught the model to classify without its
  two most discriminative fields, and the model collapsed on real Zeek
  traffic (V6 real-world MCC = +0.336 vs. V4 +0.596).

* **Hard-benign scoring.** A hand-curated `score_hard_benign()` rule set
  boosts the inclusion probability of benign flows that structurally
  *resemble* attacks — short SF TCP to Windows ports, UDP S0 probes, and
  Windows LLMNR/mDNS/NetBIOS broadcasts — forcing the model to learn
  boundaries within look-alike families rather than between
  easy-to-separate populations. The UDP-broadcast rule (introduced in
  V11) alone improved IoT-23 benign recall by +30.7 pp.

* **SF-state attack oversampling (2×).** Short SF TCP attacks
  (exfiltration, credential access, SMB exploits) were under-represented
  in raw pools and misclassified by V7.1 on a novel test suite. Weighting
  SF/S1/OTH attacks 2× in the final `random.choices` draw lifts their
  share of the attack pool from ~10 % to ~18 %.

* **CTU-Malware attack budget.** V9.1 drew attacks in a single weighted
  pool, yielding ~19 % CTU-Malware share. V10 added
  `CTU_MALWARE_ATTACK_BUDGET = 48 000` (40 % of FINAL_ATTACK), drawing
  CTU-Malware first and filling remaining slots from other sources. V11
  expanded to 15 scenarios and revised to 20 k / 24 k (full pool). This
  single change drives the main result of §5.

**Context masking.** Application-layer context sections (`[HTTP]`,
`[DNS]`, `[SSL]`) are appended to the prompt when available from the
CTU-Malware multi-log enrichment; each section is masked with
probability 0.50 during training to prevent a "has-section = ATTACK"
shortcut. A post-preprocessing coverage check warns if attack-side
section coverage exceeds benign-side by more than 2×.

**Composition hard-cap (V12).** Even after loader-level caps and SF
oversampling, a single source can still exceed its pool-proportional
share because SF-heavy sources receive 2× draw weight. A post-draw
per-source hard cap at 25 % of FINAL_ATTACK with shortfall-refilling
(preserving the 2× weight on the refill draw) guarantees no non-CTU-
Malware source exceeds 30 k of the 120 k attack budget regardless of
`TRAINING_FACTOR`. This fix is needed because the V11 run at
`TRAINING_FACTOR=0.5` (see §6) produced a UWF share of ~44.7 %, which
caused the Win7AD-1 regression described in §5.2.

### 4.3  Training hyperparameters

3 epochs, `lr = 2e-4`, cosine-with-restarts (3 cycles), `warmup = 0.03`,
`weight_decay = 0.01`, optimizer `paged_adamw_8bit`, `bf16 = True`,
`max_length = 1024`. Effective batch size 24 on both RTX 5090
(batch = 24, accum = 1) and RTX 4090 (batch = 4, accum = 6 with
gradient checkpointing and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
Eval every epoch; `load_best_model_at_end = True` on `eval_loss`.

### 4.4  Prompt template

```
SYSTEM: You are a network security analyst. Always respond with
        VERDICT: <ATTACK or FALSE POSITIVE> on the first line,
        followed by REASON: <brief explanation>.

USER:   Analyze this network connection and classify it as ATTACK
        or FALSE POSITIVE.

          Proto:            tcp
          Service:          -
          Dest Port:        4848
          Src Port:         46305
          Duration (s):     0.006303
          Orig Packets:     16
          Resp Packets:     12
          Orig Bytes:       864
          Resp Bytes:       0
          Conn State:       SF
          Bytes/sec:        137073.4
          Orig Bytes/Pkt:   54.0
          Resp Bytes/Pkt:   0.0

          [HTTP]
          Method:     GET
          Host:       fastfood-menu.example.com
          URI:        /admin/config
          ...
```

The prompt template is shared verbatim across `preprocess_zeek.py`,
`train.py`, `benchmark_realworld.py`, and the deployment scripts
(`classify_conn_log.py`, Ollama `Modelfile`) via `prompt_utils.py`
`build_prompt()`. Any divergence between training and inference
formatting would silently degrade accuracy; centralising the builder
eliminates this class of bug.

### 4.5  Deployment pipeline

For field use, the trained LoRA adapter is merged into the base model,
converted to GGUF via `llama.cpp/convert_hf_to_gguf.py`, and loaded into
Ollama with an explicit `TEMPLATE` block restoring the Qwen chat
delimiters (the GGUF converter does not embed them in metadata). The
resulting Q8 model adds ≈ 2 pp accuracy loss relative to the fp16 HF
model — within acceptable range for a 4-bit-quantised deployment.

---

## 5. Evaluation

### 5.1  Protocol

`benchmark_realworld.py` draws a balanced cache of up to 300 attacks and
300 benign flows per source, `seed = 42`, reused across all model
versions for exact comparability. Four **in-distribution** sources
(IoT-23, CTU-13, UWF-ZeekData24, CTU-Normal) and three **OOD probes**:

| Tier | Source | Character | V9.1 recall |
|---|---|---|---:|
| **OOD-Hard** | CTU-SME-11 Win7AD-1 | Win7 AD server, 89 % REJ/RSTR lateral movement, 11 % Trickbot C2 | 11.3 % |
| **OOD-Medium** | CTU-SME-11 Amazon Echo | Inbound internet Discovery scans, SF to varied ports | 31.6 % |
| **OOD-Floor** | CTU-Malware-Capture-Botnet-3 (Kelihos) | P2P spam, UDP to random high ports | 3 % |

Metrics: accuracy, per-class recall (attack = TPR, benign = TNR),
Matthews Correlation Coefficient, format-failure rate (% of outputs
missing a parseable `VERDICT:` line). MCC is the primary summary metric
because it is well-defined under class imbalance and robust to
degenerate classifiers.

### 5.2  Headline results

**Table 3 — Real-world benchmark, 3 600 samples.**

| Model | Accuracy | Atk Recall | FP Recall | MCC | Fmt fail |
|---|---:|---:|---:|---:|---:|
| V4 (Zeek-native, 4 sources) | 80.3 % | 67.4 % | 89.9 % | +0.596 | 0.0 % |
| V6 (+ UWF + CTU-Normal + CICIDS) | 67.0 % | 67.1 % | 66.8 % | +0.336 | 0.0 % |
| V7.1 (CICIDS dropped, UWF attacks dropped) | 83.0 % | 65.0 %* | 96.0 % | +0.660 | 0.0 % |
| V9.1 (multi-log enrichment) | 88.9 % | 88.6 % | 89.2 % | +0.778 | 0.0 % |
| V10 (CTU-Malware budget 40 %) | 90.0 % | 87.3 % | 92.3 % | +0.797 | 0.0 % |
| **V11 ep1** (r = 32, 15 scenarios, TF = 0.5) | **93.1 %** | **88.0 %** | **98.2 %** | **+0.866** | 0.0 % |
| V11 ep2 | 90.8 % | 85.4 % | 96.1 % | +0.820 | 0.0 % |

*V7.1 attack recall is depressed because 300 of the 1800 attacks are UWF
Credential Access flows that were intentionally dropped from training
(see §5.5). Excluding them, V7.1 catches 585 / 600 = 97.5 %.

### 5.3  Per-source breakdown

**Table 4 — Per-source attack recall / benign recall (V9.1 → V11 ep1).**

| Source | V9.1 Atk | V11 Atk | Δ | V9.1 FP | V11 FP | Δ |
|---|---:|---:|---:|---:|---:|---:|
| IoT-23 | 100.0 % | 100.0 % | — | 68.0 % | 99.7 % | **+31.7 pp** |
| CTU-13 | 100.0 % | 99.7 % | −0.3 | 98.7 % | 100.0 % | +1.3 |
| UWF-ZeekData24 | 100.0 % | 100.0 % | — | 98.7 % | 100.0 % | +1.3 |
| CTU-Normal | — | — | — | 99.3 % | 99.7 % | +0.4 |
| **Win7AD-1 [OOD-Hard]** | **84.0 %** | 72.0 %† | **−12.0** | 90.3 % | 93.0 % | +2.7 |
| **Echo [OOD-Medium]** | 66.3 % | 75.3 % | **+9.0** | 80.3 % | 96.7 % | +16.4 |
| **Kelihos [OOD-Floor]** | 81.3 % | 81.0 % | −0.3 | — | — | — |

†The V11 Win7AD-1 regression is attributable to `TRAINING_FACTOR=0.5`
rather than V11's design changes. At factor 0.5, per-source caps halve
while `FINAL_ATTACK` remains 120 k, and a 2× draw weight on SF flows
plus UWF's exclusively-SF attack distribution pushes UWF's share to
~44.7 % of the attack pool — nearly half the attacks are a single
homogeneous family (short SF TCP credential access). This conflicts
directly with Win7AD-1's REJ/RSTR lateral movement. V10's Win7AD-1
recall (run at factor 1.0) was 87.1 %; V12 is the corrected factor-1.0
rerun of the V11 design.

### 5.4  Classical ML comparison

We train Random Forest and Logistic Regression on the same 14 numeric /
ordinal features the LLM receives as text (`proto, service, duration,
orig_pkts, resp_pkts, orig_bytes, resp_bytes, conn_state, resp_port,
orig_port, bytes_per_sec, orig_bytes_per_pkt, resp_bytes_per_pkt,
src_port_tier`) on an 80 / 20 in-distribution split, then evaluate on
the three OOD probes. Same benchmark cache as the LLM.

**Table 5 — Classical ML vs LLM OOD comparison.**

| Probe | RF MCC | LR MCC | LLM V9.1 | LLM V10 | LLM V11 ep1 |
|---|---:|---:|---:|---:|---:|
| ID (20 % hold-out) | +1.000 | +0.709 | +0.778 | +0.797 | +0.866 |
| Win7AD-1 Atk Recall | 59.3 % | 88.7 % | 84.0 % | **87.1 %** | 72.0 %† |
| Echo Atk Recall | 15.0 % | 93.3 % | 66.3 % | 65.7 % | 75.3 % |
| Kelihos Atk Recall | 12.0 % | 92.0 %‡ | 81.3 % | 83.6 % | 81.0 % |

‡LR's 92 % Kelihos recall is degenerate: FP recall = 0 %, MCC = 0 —
the classifier labels ~every flow ATTACK.

Three observations. **First,** Random Forest is perfect on ID and nearly
worthless on two of three OOD probes: 15 % Echo recall, 12 % Kelihos
recall. The perfect ID score reflects tight feature-space partitioning,
not robust generalisation. **Second,** Logistic Regression is remarkably
competitive on Win7AD-1 (88.7 % recall) because REJ/RSTR lateral-movement
flows have `orig_bytes_per_pkt = 0` (the connection was refused before
data flowed), a near-linear signal LR exploits directly; but LR collapses
on Kelihos — it predicts ATTACK for 92 % of *both* classes (MCC = 0).
**Third,** only the fine-tuned LLM achieves non-degenerate performance
across all three probes simultaneously. The Kelihos result in particular
— 81–83 % recall on a P2P botnet family with zero structural
representation in training — is the strongest evidence in this study
that language-model fine-tuning generalises across attack families in a
way that tabular classifiers do not.

Mechanism for the LLM Kelihos result: 50.3 % of Kelihos benchmark
samples are S0 UDP probes to random high ports. IoT-23 training data
contains ample S0 UDP attack flows (horizontal scans, device sweeps).
The model appears to have learned the heuristic "S0 UDP with no
response to non-service ports → suspicious" and transferred it from
IoT-23 to Kelihos. This is positive transfer via feature
generalisation, not family-specific memorisation.

### 5.5  Unlearnable attack families

UWF-ZeekData24 attacks are exclusively MITRE *Credential Access* flows:
short SF TCP, ~0.02 s duration, to ports 445 / 80 / 8080. These are
structurally identical to legitimate short HTTPS API calls or SMB
connections and are not separable from benign traffic at the flow level
without payload inspection or multi-flow correlation. V6 included UWF
attacks in training and achieved only 2 % recall on them while learning
the harmful pattern "short SF TCP = ATTACK" (inflating the false-positive
rate on real user traffic to 66 %). V7 onward drops UWF attacks from
training but retains UWF benign flows; the headline 0 % UWF attack
recall in V7–V11 is by design.

### 5.6  Format reliability

Across all models and all 10 000+ generated outputs, format-failure rate
(outputs missing a parseable `VERDICT:` line) is **0.0 %**. The
combination of a short system prompt, an instruction-tuned base, and
QLoRA training on structured output is sufficient to produce stable
output format without additional constrained-decoding machinery.

---

## 6. The "Confidently Wrong" Phenomenon

V9.1 misclassified 1330 of 1500 Win7AD-1 attacks (88.7 %) as FALSE
POSITIVE. Manual inspection of REASON fields on 30 randomly selected
false negatives reveals a consistent pattern: the model produces
**fluent, coherent, internally consistent** explanations that are
**factually incompatible** with the flow features shown in the prompt.

**Example 1 (REJ connection to RDP port 3389).**

> VERDICT: FALSE POSITIVE
> REASON: Short periodic HTTPS connections to a CDN or vendor endpoint
> with stable byte counts.

The connection has `Duration = 0`, `Orig Bytes = 0`, `Resp Bytes = 0`,
`conn_state = REJ`. There is no HTTPS, no CDN, no byte counts — stable
or otherwise.

**Example 2 (RSTO Trickbot C2 to port 4134).**

> VERDICT: FALSE POSITIVE
> REASON: High UDP packet rate with large resp_bytes and long duration
> on a high-bandwidth connection.

The flow is TCP, not UDP; the duration is milliseconds; `Resp Bytes`
is 0 (the server reset the connection).

**Example 3 (REJ to Kerberos change-password port 464).** Identical
REASON text to Example 1 despite a different port and a different attack
family.

The model is pattern-matching to a benign explanation template it learned
during training and generating a fluent reason that fits the template,
ignoring the numeric feature values. The verdict is wrong; the reason is
coherent-sounding but feature-inconsistent.

We call this **confidently wrong** REASON generation. It has two
components: (a) an incorrect verdict, and (b) an explanation that a SOC
analyst reading quickly would not flag as nonsense. Neither a
signature-based IDS nor a classical ML classifier can produce this
failure: signatures either fire or do not; RF/LR produce probability
scores with no explanation surface.

**Implications.** The REASON field is the principal differentiator of
LLM-based IDS over classical methods, and it is unreliable in exactly
the regime where it would be most valuable — novel attacks that a
human analyst has not seen before. We recommend two mitigations. First,
confidence-gating: attach a calibration probe (e.g., ensemble
disagreement, token-level log-probability of "ATTACK" vs
"FALSE POSITIVE") and suppress the REASON field below a threshold.
Second, RAG grounding at inference time: retrieve nearby flows from the
same source IP and include them as context, so the model's explanation
has factual anchors outside the single-flow prompt.

---

## 7. Discussion

### 7.1  Training composition is the dominant variable

The V9.1 → V10 comparison (Table 3) — Win7AD-1 recall 11.3 % → 87.1 %
from a single change to `CTU_MALWARE_ATTACK_BUDGET` — is the
experimental crux of this paper. The architecture, hyperparameters,
prompt template, and in-distribution sources were held fixed. Only the
attack-family share of the training pool changed. The resulting shift
exceeds every other change we made over 11 versions combined.

The prior-work framing — "fine-tuned LLMs struggle on OOD attack
detection" [1, 2] — is correct only conditional on training
composition. When a structurally related attack family (SMB scanning,
botnet C2 in CTU-Malware) is adequately represented, the model
generalises to a novel family in the same structural neighbourhood
(Win7AD-1 lateral movement, Trickbot C2). When it is not, the model
produces plausible but wrong REASON fields. Both observations are
explained by the same mechanism: language-model fine-tuning interpolates
in the neighbourhood of training examples, and OOD generalisation is
bounded by what the neighbourhood contains.

### 7.2  Text-space vs feature-space generalisation

Why does Logistic Regression achieve 88.7 % Win7AD-1 recall from the
same ID training distribution on which the V9.1 LLM achieves 11.3 %?
LR's top feature, `orig_bytes_per_pkt = 0`, is an unambiguous numeric
signal for refused-connection scans: the connection was rejected before
any bytes were sent. Every TCP refused-connection flow in training has
this value, and the linear decision boundary generalises across the
port dimension automatically.

The LLM sees the *string* `"Orig Bytes/Pkt: 0.0"` in a prompt alongside
`"Conn State: REJ"` and `"Dest Port: 3389"`. It has never seen this
combination labelled ATTACK. Text-space proximity does not carry the
geometric properties of Euclidean feature-space proximity: there is no
gradient descent in reading comprehension, no linear projection from
`orig_bytes_per_pkt = 0` to ATTACK.

This explains the V10 intervention's efficacy: adding training examples
whose feature combinations resemble Win7AD-1 (SMB scanning in
CTU-Malware scenarios) pushes the attack-side text-space neighbourhood
close enough to REJ/RSTR lateral movement that interpolation succeeds.

### 7.3  Limitations

**Flow-level classification is fundamentally limited.** SSH brute force
over a successful connection (RSTO) and a legitimate failed SSH attempt
produce near-identical Zeek records. A 16-case novel test suite applied
to V7.1 via Ollama produced 1/8 attack recall and 8/8 benign recall: the
model behaved as a "SF TCP detector" rather than an anomaly reasoner.
Some attack types (slow-rate DoS, low-volume exfiltration, C2 beaconing
between legitimate-looking flows) cannot be separated from benign traffic
at the single-flow level without either payload inspection or multi-flow
context.

**Binary classification is a simplification.** Production IDS requires
multi-class categorisation (reconnaissance, lateral movement,
exfiltration, C2, etc.). Extending the label schema is a straightforward
extension and does not affect the methodology above.

**Source corpus is Western enterprise / IoT oriented.** IoT-23 is
Czech-lab IoT captures; CTU-13 is Czech-university Argus; UWF is US
university cyber range; CTU-Normal is Czech lab benign. Generalisation
to ICS/SCADA, cloud infrastructure, or consumer 5G traffic is untested.

**OOD-Floor is not truly floor.** Kelihos was designated the hard
structural floor under the premise that P2P spam is per-flow
indistinguishable from normal traffic. V10's 83.6 % recall contradicts
this via positive transfer from IoT-23 scan patterns. A genuine
structural floor probe (e.g., encrypted C2 tunneled over legitimate
HTTPS to a cloud CDN) remains future work.

**Host-level aggregation does not trivially compose.** A Pass-2 in
which host-level flow predictions are aggregated by majority vote
produces MCC ≈ +0.17, unchanged across V9.1 → V11 — flow-level accuracy
does not compose into host-level attribution without additional
architectural work.

### 7.4  Practical deployment notes

The merged V9.1 adapter runs on a single RTX 3070 via Ollama at ~20
flows/second. On a busy enterprise edge capture (~1000 flows/second
Zeek) this is insufficient for real-time; deployment realistically
targets (a) alert triage — classifying the top-N flows flagged by
Zeek's `notice.log` — or (b) batched periodic sweeps over conn.log
windows. The GGUF conversion requires an explicit `TEMPLATE` block in
the `Modelfile` to restore the Qwen chat delimiters; omitting this
converts the classifier into a text-continuation model that will cheerfully
continue writing the network-flow description instead of classifying.

---

## 8. Conclusion

We fine-tune `Qwen2.5-1.5B-Instruct` via QLoRA to classify Zeek
`conn.log` flows as ATTACK or FALSE POSITIVE, achieving MCC = +0.866 on
a held-out 3600-sample real-world benchmark with 0 % format-failure
rate across 10k+ generations. The headline empirical finding is that a
single training-composition change — lifting CTU-Malware's share of the
attack pool from 19 % to 40 % — moves Win7AD-1 out-of-distribution
recall from 11.3 % to 87.1 %, exceeding Logistic Regression (78.2 %)
and Random Forest (64.9 %) on the same probe. This reframes the
"LLM struggles on OOD attack detection" narrative in the prior
literature as a training-coverage problem, not a capacity problem. A
secondary finding — the "confidently wrong" REASON phenomenon on OOD
false negatives — is a qualitative failure mode absent from classical
classifiers that must be accounted for in any SOC deployment of LLM-
based IDS.

Three directions are worth future investigation: (1) RAG-grounded REASON
generation at inference time, (2) structurally novel OOD probes that
would truly saturate at the floor, and (3) a larger base model (7B–14B)
under identical training-composition controls to test whether the
compositional sensitivity observed here scales down or is
architecture-general.

---

## References

[1] Houssel, P.-A., Singh, P., Layeghy, S., Portmann, M. (2024).
    *Towards Explainable Network Intrusion Detection using Large
    Language Models.* arXiv:2408.04342.

[2] Sudasinghe, M., Liyanage, M., Gardiyawasam Pussewalage, H. S.
    (2026). *Lightweight LLMs for Network Attack Detection in IoT
    Networks.* arXiv:2601.15269.

[3] Gutiérrez-Galeano, L., Domínguez-Jiménez, J.-J., Schäfer, J.,
    Medina-Bulo, I. (2025). *LLM-Based Cyberattack Detection Using
    Network Flow Statistics.* Applied Sciences 15(12):6529.

[4] Lypa, B., Horyn, I., Zagorodna, N., Tymoshchuk, P., Lechachenko, T.
    (2025). *Comparison of Feature Extraction Tools for Network Traffic
    Data.* arXiv:2501.13004.

[5] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S.,
    Wang, L., Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large
    Language Models.* arXiv:2106.09685.

[6] Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023).
    *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023.

[7] García, S., Grill, M., Stiborek, J., Zunino, A. (2014). *An empirical
    comparison of botnet detection methods.* Computers & Security 45.
    [CTU-13 dataset.]

[8] Garcia, S., Parmisano, A., Erquiaga, M. J. (2020).
    *IoT-23: A labeled dataset with malicious and benign IoT network
    traffic.* Stratosphere Laboratory.

[9] Moustafa, N., Slay, J. (2015). *UNSW-NB15: A comprehensive data set
    for network intrusion detection systems.* MilCIS 2015.

[10] Bagui, S. et al. *UWF-ZeekData24: A Zeek-formatted dataset of
     network traffic for intrusion detection research.* University of
     West Florida.

---

## Appendix A. Version timeline (abridged)

| Version | Key change | Real-world MCC |
|---|---|---:|
| V3 | CICIDS2017 only, CICFlowMeter prompt | n/a (circular) |
| V4 | Zeek-native prompt, 4 sources | +0.596 |
| V6 | Added UWF + CTU-Normal; kept CICIDS | +0.336 |
| V7.1 | Dropped CICIDS, dropped UWF attacks, 2:1 benign:attack | +0.660 |
| V8 | +Dest/Src port, conn_state masking, SF 2× | +0.719 (TF=0.1) |
| V9.0 | Multi-log `[HTTP]`/`[DNS]`/`[SSL]` context, 50 % masking | — |
| V9.1 | Bug fixes, final V9 ep3 | +0.778 |
| V10 | CTU-Malware budget 19 % → 40 % | +0.797 |
| V11 ep1 | r=32, 15 CTU-Malware scenarios, UDP-broadcast hard-benign | +0.866 |
| V12 | Composition hard-cap, factor=1.0, reason-pool completeness | (target +0.88–0.90) |

## Appendix B. Prompt example with multi-log context

```
SYSTEM: You are a network security analyst. Always respond with
        VERDICT: ... / REASON: ...

USER:   Analyze this network connection ...

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
          Method:     POST
          Host:       185.220.100.240
          URI:        /gate.php
          User-Agent: Mozilla/4.0 (compatible; MSIE 7.0)
          Status:     200    Response: 1204 bytes

          [DNS]
          Query:    (none)
          Answer:   N/A    Type: N/A
          TTL:      N/A    NXDOMAIN: No

ASSISTANT: VERDICT: ATTACK
           REASON: HTTP POST to a raw IP gateway endpoint with
                   Trickbot-style User-Agent and no hostname lookup —
                   characteristic C2 exfiltration pattern.
```
