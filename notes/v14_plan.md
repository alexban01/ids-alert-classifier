# v14 / v14.1 plan — dense-supervision fixes, then grounded reasons + GRPO pilot

_Drafted 2026-07-13. Research session: SFT (dense) vs RL (sparse) — conclusions below.
Prerequisite: **finish v13.2 first** (2-epoch no-pack, queued in STATE.md Next steps #4);
v14 builds on whichever v13.x checkpoint wins on MCC (currently v13.1 ep1, +0.7465;
V11 ep1 +0.777 remains overall best)._

## Research conclusions this plan is based on

1. **Current training is full-sequence CE** — loss on every token including the ~300-token
   prompt (`train.py`, no completion-only masking). For a classifier whose target is
   effectively one token, most of the loss budget is spent modeling the prompt. Fixing this
   is the *dense* improvement to make before any sparse/RL experiment.
2. **Straight RL on the current verdict-first format is not worth running.** For a
   single-token output with ground-truth labels, cross-entropy is already the right
   estimator; GRPO on the same token is a noisier, k×-costlier gradient toward the same
   objective. RL's documented edge (Chu et al. 2025, "SFT Memorizes, RL Generalizes") shows
   up when a *reasoning chain* must be discovered — hence reason-first or nothing.
3. **GRPO needs no reason labels.** Reward checks only the verdict; the model's own
   reasoning is reinforced when it precedes correct verdicts (R1-Zero recipe). The random
   `pick_reason()` pools are irrelevant to it.
4. **Attack-type labels exist in the raw sources but training loaders discard them**:
   IoT-23 `detailed-label` (`loader_iot23.py` keeps only Malicious/Benign), UWF
   `label_tactic` (used as filter, then binarized), CTU-13/CTU-Malware `flow=` labels,
   UNSW `attack_cat`. The benchmark's per-label table proves they're recoverable
   (`bench_loaders.py` keeps them). Grounded reasons are a loader change away, not new data.
5. **Main GRPO risk: decorative reasoning.** 10-feature tabular prompts may not need a
   chain; reward can't tell grounded reasoning from boilerplate (CoT unfaithfulness —
   Turpin et al. 2023). Treat as a pilot with an explicit faithfulness check, not a bet.

---

## v14 — two changes, both cheap

### v14a: completion-only loss (train.py change, one retrain)

Mask loss to the assistant turn so all gradient lands on `VERDICT: <X>`.

- **Implementation:** the dataset is conversational messages-format JSONL. TRL options:
  - `SFTConfig(assistant_only_loss=True)` — requires the chat template to carry
    `{% generation %}` markers; **Qwen2.5's template doesn't**, so this errors unless the
    template is patched.
  - Preferred: map messages → prompt/completion form **at load time in train.py**
    (`{"prompt": [system,user msgs], "completion": [assistant msg]}`) and set
    `completion_only_loss=True`. Keeps `zeek_dataset_50pct.jsonl` byte-identical on disk
    → dataset hash/provenance in `run.json` stays valid.
- One variable vs the v13.x winner: same data (50%), r=16/α=32, no-pack, same epochs as
  the winning arm.
- **eval_loss will NOT be comparable to any prior run** (loss over ~5 target tokens instead
  of ~300) — footnote it in EXPERIMENTS.md like the reason-on/off rows; compare by MCC only.
- Command sketch: `train.py --tag v14 --no-pack --dataset zeek_dataset_50pct.jsonl`
  (+ whatever flag the completion-only change lands as, e.g. `--completion-only`).

### v14b: logit-based verdict + threshold calibration (inference-only, no retrain)

Replace greedy decode with a score: single forward pass, read the logits at the first
verdict-content token position, `score = logp("ATTACK") − logp("FALSE")`, classify by
`score > τ`.

- **Tune τ on the eval split (`zeek_dataset_eval.jsonl`), never on the benchmark** — tuning
  on the benchmark is leakage and invalidates the number.
- Applies retroactively to ANY checkpoint (v11, v13.1, v14) — run it on all recent ones;
  it separates "model ranks flows correctly but the argmax cutpoint is off" from real
  misranking, and yields ROC/PR curves for the thesis.
- Implementation: new mode in `benchmark_realworld.py` / `ids/infer_utils.py` (needs a
  logits path, not just `generate()`). Current benchmark cache stores generated text only,
  so a `--logits` run bypasses/regenerates the cache.
- **Deploy caveat:** Ollama/GGUF path doesn't expose logits through the current scripts;
  llama.cpp can return logprobs but that's extra plumbing. Calibrated threshold is
  HF-inference-only until that's built. Greedy decode stays the deploy default.

**Expected value:** v14a concentrates training signal (real but unquantified upside);
v14b is free MCC — with ATTACK precision/recall at 0.91/0.82 vs FP 0.84/0.92 (v13.1),
the operating point is visibly off-center and a tuned τ typically buys points.

---

## v14.1 — grounded reasons + GRPO reason-first pilot

### v14.1a: grounded template reasons (preprocess change)

Replace random `pick_reason()` with reasons **derived from the sample's own features**,
optionally enriched with the harvested attack-type label:

- Loader changes: keep IoT-23 `detailed-label`, UWF `label_tactic`, CTU-13/CTU-Malware
  `flow=` label, UNSW `attack_cat` → pass through to `make_sample()` as e.g. `attack_type`.
- Template from features first, type second: "S0 state, 200 orig pkts, 0 resp bytes →
  scan pattern (PartOfAHorizontalPortScan)". Feature-derived means the reason is
  *checkable against the prompt* — that's the interpretability thesis contribution.
- **Open decision (decide at run time, both defensible):** keep verdict-first (reason =
  post-hoc explanation, no CoT effect, safest) vs flip to reason-first (potential CoT
  benefit + sets up GRPO format, but changes the output contract for all inference paths
  incl. `extract_verdict()` / Modelfile).
- Costs ~11–14% more training tokens than verdict-only (known REASON overhead).
- New dataset ⇒ full preprocess rerun ⇒ breaks the "hold composition constant" chain —
  which is why this is v14.1, after v14 isolates the loss-masking variable.

### v14.1b: GRPO reason-first pilot (new RL stage)

- **Recipe:** TRL `GRPOTrainer`, LoRA, starting from the best SFT checkpoint (post-v14).
  Reason-first output format (system prompt change). Reward: +1 correct verdict, small
  format-compliance term (parseable REASON→VERDICT), nothing on reason content.
  k = 4–8 rollouts/prompt; completions are short so this is tractable.
- **Hardware:** RunPod 5090 recommended (k rollouts + ref model on 8 GB local would crawl).
- **Success criterion:** OOD MCC (Win7AD-1, Kelihos slices) vs the same checkpoint without
  RL, same benchmark, threshold-calibrated both. Overall MCC alone can hide OOD movement.
- **Faithfulness check (required before claiming interpretability):** perturbation test —
  for a sample of outputs, flip the feature the REASON cites (e.g. change S0→SF) and check
  the verdict/reason respond; if verdicts are invariant to the cited evidence, the
  reasoning is decorative and the thesis claim is only "RL for accuracy", not
  interpretability.
- **Known risks:** reasoning collapses to boilerplate (most likely failure), reward
  hacking on format, high run-to-run variance at 1.5B. Timebox it; a null result is
  still a reportable thesis finding ("outcome-reward RL does not improve tabular-flow
  classification at 1.5B").

---

## Ordering

_(2026-07-14: v13.3 inserted before v13.2 — retry packing with FlashAttention-2,
same config as v12.2, to isolate the attention-leak hypothesis and recover the
~3.4× packed throughput if confirmed. See STATE.md Next steps #4 / TASK-018.
v13.2 then runs with whichever arm wins. If v13.3 wins, v14a inherits
`--flash-attn` + packing instead of `--no-pack`.)_

1. v13.3 → v13.2 (queued) → settles the v13.x winner.
2. **v14b threshold calibration** — free, retroactive, do it first; re-score v11 ep1,
   v12.2, v13.1, v13.2 while v14a trains.
3. **v14a completion-only retrain** — one variable vs v13.x winner.
4. **v14.1a grounded reasons** → retrain (verdict-first or reason-first, decide then).
5. **v14.1b GRPO pilot** on top of the best of the above.
