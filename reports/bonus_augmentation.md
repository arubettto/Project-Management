# Bonus 5C - Error-driven augmentation (glare focus)

**Augmentation (train split only):** `{"glare": {"enabled": true, "target_regex": "glare", "p": 0.7, "n_ellipses": [1, 3], "alpha": [150, 230], "size_frac": [0.15, 0.5]}, "motion_blur": {"enabled": false, "target_regex": "blur", "p": 0.5, "kernel": [7, 21], "angle": [0, 180]}}`
Applied only to training images matching `glare`; validation and test are unchanged. Same protocol and seed (123), regime = frozen.
Examples: `results/task5c_augment_glare_frozen/aug_examples.png`

## Per-class delta (test split)
| class | n(test) | F1 before | F1 after | dF1 | dPrecision | dRecall |
|---|---|---|---|---|---|---|
| bad | 90 | 0.9944 | 0.9944 | +0.0000 | +0.0000 | +0.0000 |
| good | 90 | 0.9945 | 0.9945 | +0.0000 | +0.0000 | +0.0000 |

| metric | before | after | delta |
|---|---|---|---|
| accuracy | 0.9944 | 0.9944 | +0.0000 |
| precision_macro | 0.9945 | 0.9945 | +0.0000 |
| recall_macro | 0.9944 | 0.9944 | +0.0000 |
| f1_macro | 0.9944 | 0.9944 | +0.0000 |

Test files matching 'glare' (n=16): accuracy 0.9375 -> 0.9375


## Why val/test must stay clean
Validation drives early stopping and model selection, test is the final estimate of real-world performance. If they were augmented
the model would be judged on artificial data that we generated ourselves, the numbers would no longer measure real glare, and
before/after would not be comparable. Augmentation is a *training-time* tool only.

## Why this is the Monitoring & Control phase
measure (Task 4 metrics) -> analyze errors (failure gallery) -> correct (targeted augmentation) -> re-measure (this table).
