# Experiments

_Generated from `models/*/run.json` — do not edit by hand; rebuild with `python scripts/experiments.py`._

| Run | Date | Git | Reason | Epochs | Pack | TF | LoRA r | Eff.batch | Eval loss¹ | MCC | Atk R | FP R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v13.1-ids-lora-adapter | 2026-07-11 | 9aaf78e | no | 1 | no | – | 16 | 24 | 0.1878 | +0.7465 | 82.2% | 92.1% |
| v12.2-ids-lora-adapter | 2026-07-05 | 9f55b5c | no | 1 | yes | – | 16 | 24 | 0.1942 | +0.7308 | 84.2% | 88.8% |

¹ Eval loss is computed over the full sequence and is comparable only *within* a run (best-checkpoint selection) — **not** across Reason on/off rows, which have different target token counts. Use **MCC** for cross-run comparison.
