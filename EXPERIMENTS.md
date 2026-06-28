# Experiments

_Generated from `models/*/run.json` — do not edit by hand; rebuild with `python scripts/experiments.py`._

| Run | Date | Git | Reason | Epochs | Pack | TF | LoRA r | Eff.batch | Eval loss¹ | MCC | Atk R | FP R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| _(no runs yet)_ | | | | | | | | | | | | |

¹ Eval loss is computed over the full sequence and is comparable only *within* a run (best-checkpoint selection) — **not** across Reason on/off rows, which have different target token counts. Use **MCC** for cross-run comparison.
