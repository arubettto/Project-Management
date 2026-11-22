# Bonus 5B - Threshold and cost memo

**Model:** `results/task5a_finetune_top30/best.keras` - **Split used for the sweep:** validation only (n = 180, bad = 90); the test split was not loaded.

## Sweep (0.10-0.90, step 0.05; "positive" = bad)
| thr | precision(bad) | recall(bad) | F1(bad) | false accept | false reject | cost |
|---|---|---|---|---|---|---|
| 0.1 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 |
| 0.15 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 |
| 0.2 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 |
| 0.25 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 |
| 0.3 | 1.000 | 0.989 | 0.994 | 1 | 0 | 5 |
| 0.35 | 1.000 | 0.989 | 0.994 | 1 | 0 | 5 |
| 0.4 | 1.000 | 0.989 | 0.994 | 1 | 0 | 5 |
| 0.45 | 1.000 | 0.989 | 0.994 | 1 | 0 | 5 |
| 0.5 | 1.000 | 0.989 | 0.994 | 1 | 0 | 5 |
| 0.55 | 1.000 | 0.989 | 0.994 | 1 | 0 | 5 |
| 0.6 | 1.000 | 0.978 | 0.989 | 2 | 0 | 10 |
| 0.65 | 1.000 | 0.978 | 0.989 | 2 | 0 | 10 |
| 0.7 | 1.000 | 0.978 | 0.989 | 2 | 0 | 10 |
| 0.75 | 1.000 | 0.978 | 0.989 | 2 | 0 | 10 |
| 0.8 | 1.000 | 0.978 | 0.989 | 2 | 0 | 10 |
| 0.85 | 1.000 | 0.978 | 0.989 | 2 | 0 | 10 |
| 0.9 | 1.000 | 0.967 | 0.983 | 3 | 0 | 15 |

PR curve with 0.50 and the chosen point: `reports/threshold/pr_curve.png`

## Cost story
- **False accept** (bad photo passes the gate): OCR reads garbage and an operator has to rework the record. The damage is absorbed
  *downstream* by the pipeline, and it is expensive in time and money.
- **False reject** (good photo rejected): the user is asked to retake the photo. Annoying, but safe - nothing wrong enters the pipeline.

## Choice and effect (validation)
| | threshold 0.50 | chosen 0.10 | change |
|---|---|---|---|
| false accepts | 1 | 0 | -1 |
| false rejects | 0 | 0 | +0 |
| precision (bad) | 1.000 | 1.000 | +0.000 |
| recall (bad) | 0.989 | 1.000 | +0.011 |
| false accepts per 1000 photos | 5.6 | 0.0 | -5.6 |
| false rejects per 1000 photos | 0.0 | 0.0 | +0.0 |

**Operator impact:** at this threshold the operator receives 5.6 fewer bad photos per 1000 uploads,
at the price of 0.0 more retakes per 1000. 

## Why test must not be used to pick the threshold (data leakage)
The threshold is a hyper-parameter fitted to the data it is chosen on. If it is tuned on test, the test set stops being unseen:
the reported test metrics become optimistically biased and no longer estimate real-world performance. Per course rules this is
data leakage (-> Unsatisfactory). Selection is done on validation; test stays untouched.

`chosen_threshold = 0.10` was added to the results row.
