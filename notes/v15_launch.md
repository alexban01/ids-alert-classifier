# v15 launch checklist (prepared 2026-07-18, ready to run)

v15 = v11 recipe + v14.x fixes, quality-first: **full reason-on dataset (360k samples),
r=32/α=64, completion-only loss, no packing, 2 epochs, IBM Cloud L40S.**

Everything is prepared — dataset built & verified (`zeek_dataset.jsonl` 360k train /
`zeek_dataset_eval.jsonl` 46.2k eval, `reason: true` in meta, built 2026-07-18 03:56),
`train.py --lora-r` added, `scripts/setup_ibm.sh` updated to launch v15.

## Steps

1. Start the IBM Cloud gx3 L40S VSI (needs a floating IP).

2. Upload (~330 MB — only what training needs, not the repo):
   ```bash
   cd ~/fine_tunning && rsync -avR -e "ssh -i ~/.ssh/ibm_cloud" \
       train.py ids scripts/setup_ibm.sh \
       zeek_dataset.jsonl zeek_dataset_eval.jsonl \
       ubuntu@<FLOATING_IP>:~/fine_tunning/
   ```

3. On the VM (bootstraps driver/env/pre-flight, then launches v15 under nohup):
   ```bash
   ./scripts/setup_ibm.sh
   ```
   Effective train args: `--ibm --flash-attn --epochs 2 --no-pack --completion-only
   --lora-r 32 --tag v15` (full `zeek_dataset.jsonl` is the default dataset).
   Monitor: `tail -f train.log`. If the driver install forces a reboot, rerun the
   script — it resumes.

4. Download when done: `models/v15-ids-lora-adapter/` + `models/v15-ids-model/epoch-*/`
   (sha256-verify like the overnight campaign), then power off the VM.

## After training (local 3070)

1. Soup ep1+ep2 (w=0.5 is the honest number; w-sweep optional with the usual
   test-set-tuning caveat).
2. `benchmarks/benchmark_realworld.py` (greedy — canonical numbers).
3. `scripts/calibrate_threshold.py models/v15-soup-adapter` (τ on eval split),
   then `benchmark_realworld.py --logits` for the calibrated number.
4. Update `STATE.md` + `EXPERIMENTS.md` (`scripts/experiments.py`).

## Notes / caveats

- Eval split is now reason-on ⇒ v15 `eval_loss` NOT comparable to v12–v14 (no-reason
  eval). Compare via MCC only.
- 360k samples vs the ~241k the old cost estimates assumed ⇒ expect proportionally
  more GPU-hours (quality-first, accepted trade-off).
- Baselines to beat: v11 soup w=0.40 τ=−0.5 → MCC +0.8171 / Win7AD-1 89.3%
  (all-time record, see `results/benchmark_realworld_results_logits.json`).
- Uncommitted work on master: v14b logits path (`ids/infer_utils.py`,
  `benchmarks/benchmark_realworld.py`, `scripts/calibrate_threshold.py`),
  `train.py --lora-r`, `setup_ibm.sh` v15 args — commit before/after launch as desired.
