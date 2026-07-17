# Experiments

_Generated from `models/*/run.json` — do not edit by hand; rebuild with `python scripts/experiments.py`._

| Run | Date | Git | Reason | Epochs | Pack | TF | LoRA r | Eff.batch | Eval loss¹ | MCC | Atk R | FP R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v14-nopack-soup-w60-adapter | 2026-07-17 | ef5e235 | no | 2 | no | – | 16 | 24 | 0.0020 | +0.7260 | 75.3% | 95.8% |
| v14-nopack-soup-w40-adapter | 2026-07-17 | ef5e235 | no | 2 | no | – | 16 | 24 | 0.0020 | +0.7238 | 75.8% | 95.2% |
| v14-nopack-soup-adapter | 2026-07-17 | ef5e235 | no | 2 | no | – | 16 | 24 | 0.0020 | +0.7337 | 76.6% | 95.4% |
| v14-nopack-ids-lora-adapter | 2026-07-17 | ef5e235 | no | 2 | no | – | 16 | 24 | 0.0020 | +0.7128 | 74.9% | 94.9% |
| v14-soup-w60-adapter | 2026-07-17 | ef5e235 | no | 2 | yes | – | 16 | 24 | 0.0018 | +0.7203 | 80.8% | 90.8% |
| v14-soup-w40-adapter | 2026-07-17 | ef5e235 | no | 2 | yes | – | 16 | 24 | 0.0018 | +0.7176 | 79.2% | 92.0% |
| v14-soup-adapter | 2026-07-17 | ef5e235 | no | 2 | yes | – | 16 | 24 | 0.0018 | +0.7216 | 80.8% | 91.0% |
| v14-ids-lora-adapter | 2026-07-17 | ef5e235 | no | 2 | yes | – | 16 | 24 | 0.0018 | +0.6523 | 67.4% | 95.2% |
| v13.3-soup-w60-adapter | 2026-07-17 | ef5e235 | no | 2 | yes | – | 16 | 24 | 0.1881 | +0.7162 | 81.7% | 89.7% |
| v13.3-soup-w40-adapter | 2026-07-17 | ef5e235 | no | 2 | yes | – | 16 | 24 | 0.1881 | +0.7197 | 81.7% | 90.0% |
| v13.3-soup-adapter | 2026-07-17 | ef5e235 | no | 2 | yes | – | 16 | 24 | 0.1881 | +0.7216 | 82.1% | 89.8% |
| v13.3-ids-lora-adapter | 2026-07-17 | ef5e235 | no | 2 | yes | – | 16 | 24 | 0.1881 | +0.7144 | 80.8% | 90.3% |
| v13.2-soup-adapter | 2026-07-16 | cc6cbd4 | no | 2 | no | – | 16 | 24 | 0.1822 | +0.7343 | 85.0% | 88.4% |
| v13.2-ids-lora-adapter | 2026-07-16 | cc6cbd4 | no | 2 | no | – | 16 | 24 | 0.1822 | +0.7180 | 82.8% | 88.8% |
| v13.1-ids-lora-adapter | 2026-07-11 | 9aaf78e | no | 1 | no | – | 16 | 24 | 0.1878 | +0.7465 | 82.2% | 92.1% |
| v12.2-ids-lora-adapter | 2026-07-05 | 9f55b5c | no | 1 | yes | – | 16 | 24 | 0.1942 | +0.7308 | 84.2% | 88.8% |

¹ Eval loss is computed over the full sequence and is comparable only *within* a run (best-checkpoint selection) — **not** across Reason on/off rows, which have different target token counts. Use **MCC** for cross-run comparison.
