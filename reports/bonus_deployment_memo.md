# Bonus 5D - Deployment memo (export + edge benchmark)

Model: `results/task5a_finetune_top30/best.keras` -> TFLite. Protocol: CPU, 1 thread(s), batch 1, 20 warm-up + 100 timed runs, test split for accuracy.
Quantization calibration data: 100 images from the **train** split.

## Benchmark
| variant | size MB | lat mean ms | lat median | lat p95 | test acc | macro F1 | acc drop (pp) | drop <= 1 pp |
|---|---|---|---|---|---|---|---|---|
| keras (.keras) | 10.27 | 102.16 | 101.18 | 113.67 | 0.9944 | 0.9944 | +0.00 | yes |
| tflite float32 | 9.54 | 6.64 | 6.54 | 7.27 | 0.9944 | 0.9944 | +0.00 | yes |
| tflite float16 | 4.83 | 7.26 | 7.16 | 7.76 | 0.9944 | 0.9944 | +0.00 | yes |
| tflite int8_dynamic | 2.69 | 58.40 | 57.86 | 65.53 | 0.9944 | 0.9944 | +0.00 | yes |
| tflite int8_full | 2.90 | 7.86 | 7.91 | 8.21 | 0.9722 | 0.9722 | +2.22 | NO |

## Comparison with capstone targets
- Quality stage budget (end-to-end): 52 ms. Best acceptable variant (`tflite float32`): 6.64 ms = 12.8 % of the budget (model only; add pre/post-processing).
- YOLOv8 reference: 2.1 ms / 2.8 MB vs this model: 6.64 ms / 9.54 MB.
- Accuracy drop vs Keras: see column above; acceptable limit <= 1 pp.

## Explanation checkpoints
- Single number that decides edge feasibility: per-image latency on the *target* hardware vs the stage budget.
- Why INT8 can lose accuracy: weights and activations are rounded to 256 levels, so logits shift slightly and samples near the decision boundary can flip.
- Why warm-up: first runs include memory allocation, lazy init and cold caches; timing them overstates latency.
