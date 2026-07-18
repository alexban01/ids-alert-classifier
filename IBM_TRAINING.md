# IBM Cloud training log — v13.3 → v14

Autonomous-session log (started 2026-07-17). Every change/decision made while
driving the IBM Cloud runs is recorded here. Instance: `ids-training`,
gx3-24x120x1l40s (1× L40S 48 GB), Frankfurt, floating IP 158.176.5.120,
user `ubuntu`, $3.749/hr.

## Timeline & decisions

### 2026-07-17 — v13.3 launch, OOM, fix, relaunch
- **v13.3** (r=16, packed + FlashAttention-2, 2 epochs, `zeek_dataset_50pct.jsonl`,
  no-reason) launched on the L40S via `scripts/setup_ibm.sh`.
- **Crash: CUDA OOM at step 0** in `cross_entropy` at `--ibm`'s original batch 24×1.
  Cause: packed seqs are all 512 tokens + no gradient checkpointing on the cloud
  path + fp32 logit upcast over the 152k vocab (~7.5 GB single tensor).
- **Decision: batch 12 × grad_accum 2** (train.py `--ibm` path) — keeps effective
  batch 24 for cross-run comparability instead of re-enabling gradient
  checkpointing (~30% slower). Relaunched; stable at ~1.03 s/it, ~41.7 GB VRAM,
  ETA ~2h50m for 9,986 steps. Crashed log kept on VM as `train.log.oom`.
- Monitoring: local watcher polls the VM every 5 min; epoch-1 snapshot →
  notify owner; completion → pull models.

### 2026-07-17 — plan for after v13.3 finishes (decided up front)
1. **Soup on the VM**: `scripts/soup_adapters.py epoch-1 epoch-2 -o
   models/v13.3-soup-adapter` (w=0.5, the untuned/honest default).
2. **Benchmark on the VM** (owner's call: "all processed on the ibm cloud"):
   upload `benchmarks/` + `results/benchmark_realworld_cache.json` (cache means
   no raw datasets needed) and run `benchmark_realworld.py` over
   v13.3 epoch-1 / epoch-2 / soup. L40S ≈ several× faster than the local 3070.
3. **v14a next** (per `notes/v14_plan.md`): completion-only loss, ONE variable
   vs the v13.x winner. Arm inherits from the v13.3 result. 2 epochs either way
   (epoch snapshots make epoch-1 free and give soup ingredients).
   **Decision rule (OOD-weighted, owner's instruction 2026-07-17):** compare the
   best v13.3 variant against v13.1 on BOTH overall MCC and the OOD slices
   (Win7AD-1 attack recall primary, Echo/Kelihos secondary). Concordant → that
   arm wins. Discordant (overall MCC says one thing, OOD says the other) →
   **OOD wins** — the thesis objective is OOD generalization, and this project
   has already been burned twice by in-distribution metrics (eval_loss
   mis-selecting ep2; overall MCC hiding Win7AD-1 regressions). Same rule for
   picking which v14a variant (ep1/ep2/soup) is "best" in later comparisons.
4. Pull all adapters/snapshots/run.jsons back to the local machine, update
   STATE.md / EXPERIMENTS.md, leave instance running for the owner to delete.

### 2026-07-17 — v14a implementation (while v13.3 trains)
- **train.py `--completion-only` added**: maps messages → `{prompt, completion}`
  at load time (dataset file on disk untouched ⇒ run.json hash provenance
  intact); TRL 0.29.1 auto-enables `completion_only_loss` for prompt/completion
  datasets, and its packing collator honors the completion mask, so the flag
  composes with packed+FA2. Recorded in run.json as `completion_only`.
- eval_loss of completion-only runs is NOT comparable to any full-sequence run
  (loss over ~5 target tokens vs ~300) — MCC only.
- Local 3070 smoke test of `--completion-only --flash-attn` before use.

### 2026-07-17 — workflow rule from owner: ALL compute on the IBM cloud
- Local 3070 is off-limits until further notice (a local `--completion-only`
  smoke was killed mid-run on request). Local machine is used only to push
  training inputs and pull finished models/soups for the owner's own testing.
- Benchmarks run on the VM first; models are downloaded after, as copies.
- Consequence: the `--completion-only` validation moves to the VM — 100 steps
  on the L40S after v13.3 frees the GPU, immediately before the full v14a run.

### 2026-07-17 — overnight autonomous queue (owner away until tomorrow)
Sequential on the L40S: (1) v13.3 → soup → benchmark → pull; (2) v14a — arm by
the logged OOD rule, 100-step smoke, full 2-ep run, soup, benchmark, pull;
(3) v14a OTHER arm (unconditional — owner wants the compute used; completes the
pack×completion-only cross), same smoke-first protocol; (4) soup w-sweep
benchmarks (w=0.40/0.60) for v13.3 + both v14a arms. Owner clarified: planned
stuff only — the Qwen-3B capacity control I floated is OUT (not part of v14).
Every training config smoke-tested ~100 steps on the L40S before its full run
(OOM guard — the batch-24 crash was at step 0, so smokes catch peak memory).
NOT doing unattended: v16a (needs local preprocess + open thesis decision),
GRPO. All watchers poll at 5-min cadence, silent to owner; models pulled to
the local machine as each benchmark completes.
**Teardown (owner authorized 2026-07-17):** after ALL data is pulled AND
sha256-verified local-vs-VM (every adapter_model.safetensors + parseable
run.jsons + benchmark report/results/cache), `sudo poweroff` the VM (stops GPU
billing). Full deletion (VSI + floating IP release) needs the console — owner
does that tomorrow. No checksum match → NO poweroff.

### 2026-07-17 ~02:00 — v13.3 COMPLETE
- Trained clean: 10,544 s (2.93 h ≈ $11), eval_loss 0.1881 (NOT comparable to
  v13.1/v13.2 — those were... actually comparable, all no-reason 50pct; noted:
  0.1881 vs v13.2's 0.1822 — but eval_loss has mis-selected before; MCC decides).
- epoch-1 + epoch-2 snapshots present with auto-copied run.json (new train.py
  code worked). Soup (w=0.5) cooked on VM. Benchmark of ep1/ep2/soup running on
  the VM (`bench_v133.log`), watcher at 5-min polls.
- v13.3 adapters + snapshots + soup pulled to local `models/` (82M/154M/36M);
  run.jsons will be re-pulled after the benchmark writes MCC back into them.

### 2026-07-17 ~02:40 — v13.3 BENCHMARKED (on VM) + v14a ARM DECISION
| variant | MCC | Win7AD-1 atk | Kelihos | notes |
|---|---|---|---|---|
| ep1 | +0.675 | 52.3% | 46% | |
| ep2 (final) | +0.714 | 79.0% | 45% | ep2 > ep1 — FIRST run where epoch 2 helps |
| soup w=0.5 | **+0.722** | **80.0%** | 46% | soup beats both parents again |

- vs v13.1 (no-pack 1ep): MCC +0.7465 / Win7AD-1 65.3% — **discordant metrics**.
  Per the OOD-weighted rule: v13.3-soup wins Win7AD-1 by +14.7pp → **v14a arm =
  packed + FA2** (`--flash-attn`, no `--no-pack`).
- Two pattern flips worth thesis attention: (1) with leak-free FA2 packing,
  epoch-2 HELPS (+0.675→+0.714, Win7AD 52→79) — opposite of every no-pack/leaky
  run; ep1 looks undertrained rather than ep2 overfit. (2) v13.3 ep1 (+0.675)
  is well below leaky-packed v12.2 1-ep (+0.731) on MCC — the "attention leak
  hurt quality" story is NOT cleanly confirmed at 1 epoch; but v13.3's OOD at
  2 ep (80%) matches/beats everything in the v12/v13 family. Kelihos stable ~46%.
- Benchmark artifacts pulled local: `results/bench_v133.log`, updated report/
  results JSONs, MCC-carrying run.jsons for ep1/final/soup.
- Next: v14a smoke (100 steps, `--completion-only --flash-attn --ibm`) → full
  2-ep v14a run.

### 2026-07-17 ~03:00 — v14a smoke PASSED, full run LAUNCHED
- Smoke (100 steps, `--ibm --completion-only --flash-attn`): checkpoint-100
  saved, "completion-only loss: dataset mapped" confirmed, loss at verdict-token
  scale (0.073→0.024 by step 200, token acc 99%), zero attention/mask warnings.
  Confirms TRL's bfd packing collator carries the completion mask.
- **v14a full run launched** (PID 9760): `--ibm --completion-only --flash-attn
  --epochs 2 --dataset zeek_dataset_50pct.jsonl --tag v14`. ETA ~2.9h (same
  shape as v13.3). v13.3's train log preserved as `train_v13.3.log` on the VM.
- After completion: soup ep1+ep2 → benchmark ep1/final/soup on VM → pull →
  then the OTHER arm (`--completion-only --no-pack`, tag v14-nopack).

### 2026-07-17 ~06:00 — v14a COMPLETE, benchmark running
- v14a trained clean: 10,569 s (2.94 h), eval_loss 0.0018 (completion-only
  scale — footnote in EXPERIMENTS.md, MCC only). epoch-1/epoch-2 snapshots +
  run.jsons in place; soup cooked on VM; benchmark of ep1/final/soup running
  (`bench_v14.log`). v14a adapters+snapshots+soup pulled local (82M/71M/36M).
- Next after benchmark: v14a-nopack arm (`--completion-only --no-pack`,
  tag v14-nopack), same smoke-first protocol (no-pack + completion-only was
  never smoke-tested — the killed local smoke was packed).

### 2026-07-17 ~06:50 — v14a BENCHMARKED: soup rescues a big OOD win
| variant | MCC | Win7AD-1 atk | Kelihos |
|---|---|---|---|
| ep1 | +0.694 | 47.7% | 43% |
| ep2 (final) | +0.652 | 25.7% | 41% |
| soup w=0.5 | **+0.722** | **86.0%** | 43% |

- **Win7AD-1 86.0% = best OOD result since V10's 87.1%**, from souping two
  individually OOD-poor checkpoints (47.7/25.7%) — strongest soup effect seen
  in this project. Overall MCC ties v13.3 soup (+0.722) but +6pp on the primary
  OOD probe → per the OOD rule, **v14a soup is the current front-runner**.
- Completion-only + 2 epochs overfits OOD hard as single checkpoints (ep2
  Win7AD 25.7%) — souping is now clearly load-bearing, not an optimization.
  Thesis-worthy pattern.
- Artifacts + MCC-carrying run.jsons pulled local. Benchmark log:
  `results/bench_v14.log`.
- No-pack arm smoke launched (`--completion-only --no-pack`, tag smoke14n).

### 2026-07-17 ~07:20 — v14-nopack LAUNCHED (last training run of the queue)
- No-pack smoke passed (checkpoint-100, no crash). Full run live (PID 17210):
  `--ibm --completion-only --no-pack --epochs 2`, tag v14-nopack. No-pack means
  ~15k steps; expect ~3-4 h. v14a's log preserved as `train_v14.log` on VM.
- Remaining after it: soup + benchmark v14-nopack → w-sweep soup benchmarks
  (v13.3, v14a, v14-nopack; w=0.40/0.60) → final pull + sha256 verify →
  ledgers → poweroff.

### 2026-07-17 ~12:30 — v14-nopack COMPLETE; final 9-model benchmark running
- v14-nopack trained clean: 15,145 s (4.2 h), completion-only + no-pack
  confirmed in run.json. All adapters/snapshots pulled local.
- 7 new soups cooked on VM: v14-nopack w=0.5 + w-sweeps (0.40/0.60) for
  v13.3 / v14a / v14-nopack. All pulled local.
- Final benchmark (9 models: v14-nopack ep1/final/w50/w40/w60 + four w-sweep
  soups of v13.3+v14a) running on VM → `bench_final.log`. After it: pull
  results + run.jsons, sha256 verify EVERYTHING, ledgers, poweroff.

### 2026-07-17 ~14:40 — FINAL BENCHMARK + CLOSEOUT
| variant | MCC | Win7AD-1 atk |
|---|---|---|
| v14-nopack ep1 (=final; load_best picked ep1) | +0.713 | 61.7% |
| v14-nopack soup w=0.5 | **+0.734** (night's best MCC) | 55.3% |
| v14-nopack soup w=0.40 / w=0.60 | +0.724 / +0.726 | 48.0% / 54.0% |
| v13.3 soup w=0.40 / w=0.60 | +0.720 / +0.716 | 80.0% / 78.0% |
| v14a soup w=0.40 | +0.718 | 78.3% |
| **v14a soup w=0.60** | +0.720 | **87.0% — ties V10's all-time record (87.1%)** |

- **Final verdict (OOD rule): v14a soup is the overnight winner.** Honest
  untuned number: w=0.5, MCC +0.722 / Win7AD-1 86.0%. w=0.60's 87.0% is
  test-set-tuned w (same caveat as v11's w=0.40) — report as analysis.
- Pack × completion-only cross complete: no-pack+completion-only gives the
  best overall MCC (+0.734) but poor OOD; packed(FA2)+completion-only soups
  dominate OOD. Composition-vs-OOD trade visible across all 15 variants.
- ALL data local & verified: 18/18 adapter safetensors sha256-identical
  local-vs-VM; benchmark logs (bench_v133/v14/final.log) + report/results
  JSONs + every run.json/soup.json pulled. EXPERIMENTS.md regenerated (16 runs).
- Total spend ≈ 13.5 GPU-hours ≈ $51. VM powered off after verification.
- **Owner TODO:** delete the VSI + release floating IP 158.176.5.120 in the
  IBM console (poweroff stops GPU billing; disk+IP still trickle).

## Open items
- v14b (logit-threshold calibration, inference-only) deliberately NOT started —
  separate work item, needs a logits path in the benchmark; queued after v14a.
- Instance teardown (floating IP + VSI) stays manual — owner decides when.
