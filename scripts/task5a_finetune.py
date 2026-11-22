import json
import os
import time

import numpy as np
import tensorflow as tf

from common import *

RUN = os.path.join(RESULTS_ROOT, "task5a_finetune_top30")


def main():
    set_seed()
    ensure(RUN); ensure(REPORTS_DIR)

    train_ds, _, names = load_split("train", shuffle=True)
    val_ds, _, _ = load_split("val")
    test_ds, test_files, _ = load_split("test")

    frozen = tf.keras.models.load_model(BEST_CKPT)
    m_fr, *_ = full_eval(frozen, test_ds, test_files, names, os.path.join(RUN, "frozen_reference"), "frozen (test)")
    sample = get_sample("val")
    lat_fr = keras_latency(frozen, sample)
    size_fr = clean_size_mb(BEST_CKPT, os.path.join(RUN, "_tmp_frozen"))
    gap_fr = fit_gap(read_history_csv(BEST_HISTORY_CSV))
    params = frozen.count_params()

    model = tf.keras.models.load_model(BEST_CKPT)
    unfreeze_top(model)
    trainable = count_trainable(model)
    print(f"trainable params: {trainable:,} / {model.count_params():,}")
    hist, secs = train(model, train_ds, val_ds, RUN, lr=LR_FINETUNE)
    gap_ft = fit_gap(hist)

    best_path = os.path.join(RUN, "best.keras")
    best = tf.keras.models.load_model(best_path)
    m_ft, *_ = full_eval(best, test_ds, test_files, names, RUN, "fine-tuned top30 (test)")
    lat_ft = keras_latency(best, sample)
    size_ft = clean_size_mb(best_path, os.path.join(RUN, "_tmp_ft"))

    row = base_row("finetune_top30", lr=LR_FINETUNE, unfrozen_layers=UNFREEZE_LAST, freeze_bn=FREEZE_BN,
                   trainable_params=trainable, total_params=int(best.count_params()),
                   metrics={k: m_ft[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
                   per_class=m_ft["per_class"], confusion_matrix=m_ft["confusion_matrix"],
                   latency_cpu_ms=lat_ft, model_size_mb=size_ft, train_time_s=secs,
                   train_val_gap=gap_ft, init_from=BEST_CKPT)
    save_row(row, RUN)

    def d(a, b): return f"{b - a:+.4f}"
    rows = [[k, fmt(m_fr[k]), fmt(m_ft[k]), d(m_fr[k], m_ft[k])]
            for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")]
    rows += [["latency CPU mean (ms)", fmt(lat_fr["mean_ms"], 2), fmt(lat_ft["mean_ms"], 2), d(lat_fr["mean_ms"], lat_ft["mean_ms"])],
             ["model size (MB, no optimizer)", fmt(size_fr, 2), fmt(size_ft, 2), d(size_fr, size_ft)],
             ["total params", f"{params:,}", f"{best.count_params():,}", "0"],
             ["trainable params", "head only (see Task 2)", f"{trainable:,}", ""]]
    if gap_fr:
        rows.append(["train-val acc gap", fmt(gap_fr["acc_gap"]), fmt(gap_ft["acc_gap"]), d(gap_fr["acc_gap"], gap_ft["acc_gap"])])
    else:
        rows.append(["train-val acc gap", "n/a (set BEST_HISTORY_CSV)", fmt(gap_ft["acc_gap"]), ""])
    rows.append(["val loss - train loss", fmt(gap_fr["loss_gap"]) if gap_fr else "n/a", fmt(gap_ft["loss_gap"]),
                 d(gap_fr["loss_gap"], gap_ft["loss_gap"]) if gap_fr else ""])
    table = md_table(["metric (test split)", "frozen", "fine-tuned top30", "delta"], rows)

    df1 = m_ft["f1_macro"] - m_fr["f1_macro"]
    gap_up = (gap_ft["acc_gap"] - gap_fr["acc_gap"]) if gap_fr else None
    verdict_hint = ("DEPLOY fine-tuned" if df1 > 0 and (gap_up is None or gap_up <= 0.03) else "KEEP frozen")

    memo = f"""# Bonus 5A - Fine-tuning delta memo (frozen vs finetune_top30)

Protocol: top {UNFREEZE_LAST} backbone layers unfrozen (BatchNorm frozen: {FREEZE_BN}), Adam lr = {LR_FINETUNE:g}
(10x lower than head training {LR_HEAD:g}), epochs <= {EPOCHS}, patience {PATIENCE}, batch {BATCH}, seed {SEED},
same train/val/test split as the frozen run. Test split used once, after training.

## Delta table
{table}

Curves: `results/task5a_finetune_top30/curves.png` - Errors: `results/task5a_finetune_top30/errors/`

## Memo (5 lines)
1. **Improved:** macro-F1 {fmt(m_fr['f1_macro'])} -> {fmt(m_ft['f1_macro'])} ({d(m_fr['f1_macro'], m_ft['f1_macro'])}); accuracy {fmt(m_fr['accuracy'])} -> {fmt(m_ft['accuracy'])}. <<which class/metric improved most, from the per-class table / confusion matrix>>
2. **Got worse:** <<precision/recall that dropped, errors that appeared in errors/gallery.png; training time {secs/60:.1f} min vs frozen>>
3. **Latency/size unchanged:** latency {fmt(lat_fr['mean_ms'],2)} -> {fmt(lat_ft['mean_ms'],2)} ms, size {fmt(size_fr,2)} -> {fmt(size_ft,2)} MB, because the architecture and number of weights are identical; unfreezing only changes the *values* of weights, not the graph, FLOPs or file format. <<mention measurement noise>>
4. **Train-val gap (overfitting sign):** at the best epoch (epoch {gap_ft['best_epoch']} of {gap_ft['epochs_run']}) train acc {fmt(gap_ft['train_acc'])} vs val acc {fmt(gap_ft['val_acc'])}, gap {fmt(gap_ft['acc_gap'])}; frozen gap {fmt(gap_fr['acc_gap']) if gap_fr else 'n/a'}. <<does the gap grow? what do the curves show?>>
5. **Deploy verdict:** <<{verdict_hint} - this is only a draft rule; justify with your Task 1 acceptance criteria and the numbers above>>
"""
    open(os.path.join(REPORTS_DIR, "bonus_finetune_delta.md"), "w").write(memo)
    print(table)
    print("\nDraft memo written to reports/bonus_finetune_delta.md - finish the <<...>> parts by hand.")


if __name__ == "__main__":
    main()
