# Model comparison (built from results.json)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Latency ms/img | Size MB | Params | Train time s | Epochs run |
|---|---|---|---|---|---|---|---|---|---|
| efficientnet_b0_frozen | 0.9944 | 0.9945 | 0.9944 | 0.9944 | 96.2 | 18.1 | 4,213,668 | 311 | 25 |
| mobilenetv2_frozen | 0.9889 | 0.9889 | 0.9889 | 0.9889 | 80.5 | 11.1 | 2,422,081 | 347 | 32 |
| week7_yolov8n_cls | 0.9167 | 0.9192 | 0.9167 | 0.9165 | 33.3 | 2.8 | 1,440,850 | 886 | 7 |
| week7_yolov8s_cls | 0.9889 | 0.9891 | 0.9889 | 0.9889 | 77.0 | 9.8 | 5,083,298 | 6155 | 20 |

All numbers are on the TEST split (same split, seed and fixed parameters for all models). Latency: mean of 100 timed single-image predictions after 20 warm-up predictions; devices: efficientnet_b0_frozen=cpu, mobilenetv2_frozen=cpu, week7_yolov8n_cls=cpu, week7_yolov8s_cls=cpu.
