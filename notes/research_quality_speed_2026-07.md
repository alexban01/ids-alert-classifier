# Research: quality & speed levers (literature review, 2026-07-15)

Context at time of writing: best model V11 ep1 (MCC +0.777), current family v13.x
(r=16, 50% data) best = v13.1 no-pack (+0.7465). Planned: v13.3 (FA2+packing),
v13.2 (2ep), v14a (completion-only loss), v14b (logit calibration), v16
(grounded reasons + GRPO pilot). Priority: quality > speed.

---

## A. Quality levers, ranked by (evidence × fit × cost)

### A1. Refine v14a: don't fully mask the prompt — use a small prompt-loss weight (~0.1)

Two papers directly cover our exact regime (long prompt ~300 tok, short
completion ~10–50 tok):

- **"Instruction Fine-Tuning: Does Prompt Loss Matter?"** (arXiv:2401.13586):
  prompt-loss weight (PLW) has a significant negative *quadratic* relationship
  with downstream performance on **short-completion** data — i.e. both extremes
  are suboptimal: PLW=1.0 (our status quo, full-sequence loss) *and* PLW=0
  (pure completion-only, the current v14a plan). Optimum ≈ **0.1**. For long
  completions PLW doesn't matter.
- **"Instruction Tuning With Loss Over Instructions"** (NeurIPS 2024,
  arXiv:2405.14394): same conclusion from the other direction — keeping *some*
  instruction loss acts as a regularizer and beats full masking precisely when
  completions are short.

**Action:** make v14a a 3-arm ablation instead of 2: PLW=1.0 (status quo,
already have), PLW=0 (TRL `completion_only_loss=True`, planned), PLW≈0.1
(custom `compute_loss` with token weights — ~20 lines). The literature predicts
0.1 wins. This is the highest-confidence quality lever we haven't tried.

### A2. Checkpoint averaging / LoRA soup of ep1+ep2 (free, retroactive)

The recurring "ep1 beats ep2 on benchmark but ep2 wins eval_loss" pattern is a
classic case for weight averaging: average the two LoRA adapters' tensors
element-wise (same shapes, same init → averaging is valid) and benchmark the
soup. Model-souping literature consistently shows averaged checkpoints
generalize better OOD than either endpoint. Cost: ~30 min, no training. Try on
V11 ep1+ep2 first (largest known gap). If the soup beats ep1's +0.777, it's a
free MCC gain on every future 2-epoch run.

### A3. Base model: Qwen3-1.7B swap (biggest single-model lever)

The Qwen3 technical report (arXiv:2505.09388) shows **Qwen3-1.7B-Base
outperforming Qwen2.5-3B-Base** on more than half the benchmarks. STATE.md's
"base swap deferred" decision (2026-06-26) was about *capacity* (3B) — Qwen3 is
a *quality-per-parameter* upgrade at the same deploy footprint (edge/3070
constraint intact). Same tokenizer family/chat-template style → near drop-in.
distil-labs' 12-small-model fine-tuning benchmark also finds small Qwen3 models
among the most tunable. **Action:** after v13.2/v13.3 settle the recipe, run
one arm with `Qwen/Qwen3-1.7B` (use non-thinking mode / `enable_thinking=False`
templating) on identical data. Keep as a controlled ablation, not a mid-stream
switch.

### A4. Grounded reasons (v16a) — literature agrees, one ordering caveat

The non-IID network traffic instruction-tuning paper (arXiv:2505.20866) and
the PEFT flow-IDS chapter (Springer, 91% acc / 90% macro-F1 with LoRA on 10k
flows) both credit *richer instruction context* — not model size — for OOD
gains. This supports v16a (feature-derived template reasons) over random
reasons. **Caveat:** if reasons move to reason-first (CoT-style) format, the
v14b first-token logit calibration breaks (verdict token no longer first).
Sequence them: calibrate verdict-first models now (v14b), evaluate reason-first
separately in the GRPO pilot.

### A5. NEFTune — one-flag regularizer, cheap A/B

`SFTConfig(neftune_noise_alpha=5)` in TRL. Adds uniform noise to embeddings
during training; consistently +5–10 points on instruction-following evals
(arXiv:2310.05914; 2026 follow-ups combine it with LoRA + FA2 routinely,
e.g. arXiv:2606.10392). Evidence is on open-ended generation, not binary
classification — expectation modest, but the cost is literally one flag on the
next scheduled run. Worth folding into an existing arm, not a dedicated run.

### A6. DoRA — one-flag PEFT variant, modest expectation

`LoraConfig(use_dora=True)`. Decomposes weight updates into magnitude +
direction; closes ~half the LoRA→full-FT gap in some settings at 5–10% VRAM
overhead (2026 PEFT guides). BUT the empirical multilingual-LoRA study
(arXiv:2606.10428) finds gains are model-dependent and **smallest on Qwen**;
"Which LoRA?" work shows swapping to DoRA/PiSSA does not reliably help.
~20–30% slower training. Low priority — try only if A1–A3 plateau.

### A7. Verdict-token class weighting (focal/weighted CE)

For steering the attack-recall vs benign-precision trade *in training* rather
than post-hoc: weight the CE loss on the verdict token by class (or focal
loss). Imbalanced-LLM-classification literature (arXiv:2510.09783 + 2025
guides) supports class-weighted loss + per-class metric tracking over
oversampling for LLM fine-tuning. We already do 2:1 benign:attack sampling;
loss weighting is finer-grained and composable with A1's custom compute_loss.
Medium effort. Note v14b threshold calibration achieves a similar trade
post-hoc for free — do that first.

### Validation of current config against "LoRA Without Regret" (Thinking Machines)

Our setup already matches their recommendations — no changes needed:
- all-linear targets (7 modules incl. MLP) ✓ (their strongest finding:
  attention-only LoRA underperforms; MLP coverage is what matters)
- LR 2e-4 in their optimal 1e-4–5e-4 band (~10× full-FT) ✓
- effective batch 24 — well under their <512 warning (LoRA tolerates large
  batches worse than full FT) ✓
- r=16 has ample capacity: their rule of thumb is ~1 bit/token of dataset
  information vs rank×2 bits/param; our near-duplicate-heavy 46M-token
  (50%) set has far less information content than r=16's ~36M-bit capacity ✓

### Benchmark-methodology note from the literature

"Beware of the Batch Size" (arXiv:2602.09492) documents how LoRA A/Bs get
confounded by effective-batch differences. Relevant here: **packing changes
samples/step** (~1.46 samples/slot × 24 ≈ 35 effective vs 24 unpacked), so
part of v13.1-no-pack's win over v12.2 could be batch dynamics, not only the
attention leak. v13.3 (FA2 + packing, same slot count) disambiguates — one more
reason it's the right next run.

---

## B. Speed levers (quality-neutral or better)

1. **v13.3 FA2 + packing (already planned)** — literature-confirmed correct
   fix; TRL bfd packing + FA2 varlen isolates samples. Recovers the measured
   3.4× throughput if quality holds. Do first.
2. **Liger kernels** — `SFTConfig(use_liger_kernel=True)`. Layer-level Triton
   kernels (fused RMSNorm/RoPE/SwiGLU/CE); ~20% faster, up to 60% less
   activation memory; explicitly compatible with QLoRA/PEFT. On the 3070 the
   memory saving may also allow batch 6–8 (fewer grad-accum steps). One flag,
   stacks with FA2. **Do not combine with Unsloth** (double-patching).
3. **Unsloth** — ~2× faster QLoRA, ~70% less VRAM; the strongest local-3070
   lever but replaces the training stack (train.py, provenance sidecars all
   need rework). Only worth it if many more local runs are planned after the
   recipe freezes. Skip for now.
4. **Data volume** — 50% already validated (v12.2 gate). If more speed is
   needed, a 25% arm is cheap to test; semantic near-dup dedup of flows would
   be the principled version (flow data is massively redundant), but plain
   random downsampling has already captured most of the win.

Recommended stack for the next runs: **FA2 + packing + Liger + 50% data** —
multiplicative, all quality-neutral per the literature, ≈4× over v13.1's
throughput.

---

## Sources

- LoRA Without Regret — https://thinkingmachines.ai/blog/lora/ (HF TRL guide: https://huggingface.co/docs/trl/lora_without_regret)
- Does Prompt Loss Matter? — https://arxiv.org/html/2401.13586v2
- Instruction Tuning With Loss Over Instructions — https://arxiv.org/pdf/2405.14394
- Beware of the Batch Size — https://arxiv.org/html/2602.09492v1
- Qwen3 Technical Report — https://arxiv.org/pdf/2505.09388
- distil-labs small-model tunability benchmark — https://www.distillabs.ai/blog/we-benchmarked-12-small-language-models-across-8-tasks-to-find-the-best-base-model-for-fine-tuning/
- Instruction-tuning for Non-I.I.D. Network Traffic — https://arxiv.org/pdf/2505.20866
- Parameter-Efficient LLMs for Flow-Based IDS — https://link.springer.com/chapter/10.1007/978-3-032-27993-4_4
- LLMs for flow-based IDS vs ML/DL baselines — https://link.springer.com/article/10.1007/s10462-025-11432-2
- Lightweight LLMs for IoT attack detection (QLoRA+RAG) — https://arxiv.org/pdf/2601.15269
- Which LoRA? empirical study — https://arxiv.org/pdf/2606.10428
- PEFT methods 2026 guide (DoRA/PiSSA/VeRA) — https://www.spheron.network/blog/peft-methods-2026-dora-galore-pissa-vera-guide/
- NEFTune — https://github.com/neelsjain/NEFTune ; LoRA+NEFTune+FA — https://arxiv.org/abs/2606.10392
- Liger kernel — https://www.spheron.network/blog/liger-kernel-llm-training-gpu-cloud/
- Unsloth benchmarks — https://developers.redhat.com/articles/2026/04/01/unsloth-and-training-hub-lightning-fast-lora-and-qlora-fine-tuning
- LLMs for Imbalanced Classification — https://arxiv.org/pdf/2510.09783
