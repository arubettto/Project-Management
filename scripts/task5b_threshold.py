import csv
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve

from common import *

MODEL_PATH = os.environ.get("MODEL_PATH", BEST_CKPT)   
MODEL_REGIME = os.environ.get("MODEL_REGIME", BEST_REGIME)
COST_FA = float(os.environ.get("COST_FA", 5))  
COST_FR = float(os.environ.get("COST_FR", 1))   
OUT = os.path.join(REPORTS_DIR, "threshold")


def counts(y_bad, p_bad, t):
    pred_bad = p_bad >= t
    tp = int(np.sum(pred_bad & y_bad)); fp = int(np.sum(pred_bad & ~y_bad))
    fn = int(np.sum(~pred_bad & y_bad)); tn = int(np.sum(~pred_bad & ~y_bad))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"thr": round(float(t), 2), "precision": prec, "recall": rec, "f1": f1,
            "false_accept": fn, "false_reject": fp, "tp": tp, "tn": tn,
            "cost": COST_FA * fn + COST_FR * fp}


def main():
    set_seed(); ensure(OUT); ensure(REPORTS_DIR)
    names = class_names_from_dir("train")
    assert len(names) == 2, "5B is a binary good/bad task; adapt p_bad for more classes"
    bad = bad_index(names)

    model = tf.keras.models.load_model(MODEL_PATH)
    val_ds, _, _ = load_split("val")                    
    y_true, probs = predict(model, val_ds)
    p1 = probs[:, 0]                                       
    p_bad = p1 if bad == 1 else 1 - p1
    y_bad = y_true == bad

    thrs = np.round(np.arange(0.10, 0.9001, 0.05), 2)
    sweep = [counts(y_bad, p_bad, t) for t in thrs]
    best = min(sweep, key=lambda r: (r["cost"], -r["f1"]))  
    base = counts(y_bad, p_bad, 0.5)

    with open(os.path.join(OUT, "sweep_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys())); w.writeheader(); w.writerows(sweep)
    table = md_table(["thr", "precision(bad)", "recall(bad)", "F1(bad)", "false accept", "false reject", "cost"],
                     [[r["thr"], fmt(r["precision"], 3), fmt(r["recall"], 3), fmt(r["f1"], 3),
                       r["false_accept"], r["false_reject"], f"{r['cost']:.0f}"] for r in sweep])
    open(os.path.join(OUT, "sweep_table.md"), "w").write(table)

    pr, rc, _ = precision_recall_curve(y_bad, p_bad)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(rc, pr, label="PR curve (class 'bad', validation)")
    ax.scatter([base["recall"]], [base["precision"]], c="tab:orange", s=70, zorder=3, label=f"threshold 0.50")
    ax.scatter([best["recall"]], [best["precision"]], c="tab:red", marker="*", s=160, zorder=3,
               label=f"chosen {best['thr']:.2f}")
    ax.set_xlabel("recall (bad)"); ax.set_ylabel("precision (bad)"); ax.set_title("Precision-recall, validation split")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(OUT, "pr_curve.png"), dpi=140); plt.close(fig)

    n = len(y_bad)
    fa0, fa1 = base["false_accept"], best["false_accept"]
    fr0, fr1 = base["false_reject"], best["false_reject"]
    per1000 = lambda c: c / n * 1000

    update_row(MODEL_REGIME, {"chosen_threshold": best["thr"], "threshold_selected_on": "validation",
                              "threshold_cost_ratio_FA_to_FR": f"{COST_FA:g}:{COST_FR:g}"})

    memo = f"""# Bonus 5B - Threshold and cost memo

**Model:** `{MODEL_PATH}` - **Split used for the sweep:** validation only (n = {n}, bad = {int(y_bad.sum())}); the test split was not loaded.

## Sweep (0.10-0.90, step 0.05; "positive" = bad)
{table}

PR curve with 0.50 and the chosen point: `reports/threshold/pr_curve.png`

## Cost story
- **False accept** (bad photo passes the gate): OCR reads garbage and an operator has to rework the record. The damage is absorbed
  *downstream* by the pipeline, and it is expensive in time and money.
- **False reject** (good photo rejected): the user is asked to retake the photo. Annoying, but safe - nothing wrong enters the pipeline.
- **Which error I penalize:** false accepts, with a relative cost of {COST_FA:g} : {COST_FR:g}. <<justify the ratio: operator minutes per rework vs seconds per retake, in your own numbers>>

## Choice and effect (validation)
| | threshold 0.50 | chosen {best['thr']:.2f} | change |
|---|---|---|---|
| false accepts | {fa0} | {fa1} | {fa1 - fa0:+d} |
| false rejects | {fr0} | {fr1} | {fr1 - fr0:+d} |
| precision (bad) | {fmt(base['precision'],3)} | {fmt(best['precision'],3)} | {best['precision'] - base['precision']:+.3f} |
| recall (bad) | {fmt(base['recall'],3)} | {fmt(best['recall'],3)} | {best['recall'] - base['recall']:+.3f} |
| false accepts per 1000 photos | {per1000(fa0):.1f} | {per1000(fa1):.1f} | {per1000(fa1) - per1000(fa0):+.1f} |
| false rejects per 1000 photos | {per1000(fr0):.1f} | {per1000(fr1):.1f} | {per1000(fr1) - per1000(fr0):+.1f} |

**Operator impact:** at this threshold the operator receives {abs(per1000(fa0) - per1000(fa1)):.1f} {'fewer' if fa1 <= fa0 else 'more'} bad photos per 1000 uploads,
at the price of {abs(per1000(fr1) - per1000(fr0)):.1f} {'more' if fr1 >= fr0 else 'fewer'} retakes per 1000. <<convert to operator-minutes per day using your pipeline volume>>

## Why test must not be used to pick the threshold (data leakage)
The threshold is a hyper-parameter fitted to the data it is chosen on. If it is tuned on test, the test set stops being unseen:
the reported test metrics become optimistically biased and no longer estimate real-world performance. Per course rules this is
data leakage (-> Unsatisfactory). Selection is done on validation; test stays untouched.

`chosen_threshold = {best['thr']:.2f}` was added to the results row.
"""
    open(os.path.join(REPORTS_DIR, "bonus_threshold_memo.md"), "w").write(memo)
    print(table)
    print(f"\nchosen threshold = {best['thr']:.2f}  (FA {fa0}->{fa1}, FR {fr0}->{fr1})")


if __name__ == "__main__":
    main()
