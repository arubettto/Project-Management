| variant | size MB | lat mean ms | lat median | lat p95 | test acc | macro F1 | acc drop (pp) | drop <= 1 pp |
|---|---|---|---|---|---|---|---|---|
| keras (.keras) | 10.27 | 102.16 | 101.18 | 113.67 | 0.9944 | 0.9944 | +0.00 | yes |
| tflite float32 | 9.54 | 6.64 | 6.54 | 7.27 | 0.9944 | 0.9944 | +0.00 | yes |
| tflite float16 | 4.83 | 7.26 | 7.16 | 7.76 | 0.9944 | 0.9944 | +0.00 | yes |
| tflite int8_dynamic | 2.69 | 58.40 | 57.86 | 65.53 | 0.9944 | 0.9944 | +0.00 | yes |
| tflite int8_full | 2.90 | 7.86 | 7.91 | 8.21 | 0.9722 | 0.9722 | +2.22 | NO |