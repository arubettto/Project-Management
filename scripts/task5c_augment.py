import argparse
import glob
import json
import os
import re

import numpy as np
import tensorflow as tf
from PIL import Image

from augment_glare import AUG_CONFIG, augment
from common import *

REGIME = os.environ.get("REGIME", BEST_REGIME)     
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def make_train_ds(names, aug_on):
    binary = len(names) == 2
    root = os.path.join(DATA_DIR, "train")
    files, labels = [], []
    for i, n in enumerate(names):
        for p in sorted(glob.glob(os.path.join(root, n, "*"))):
            if p.lower().endswith(EXT):
                files.append(p); labels.append(i)
    rel = [os.path.relpath(f, root) for f in files]
    fg = [bool(re.search(AUG_CONFIG["glare"]["target_regex"], r, re.I)) for r in rel]
    fb = [bool(re.search(AUG_CONFIG["motion_blur"]["target_regex"], r, re.I)) for r in rel]
    print(f"train images: {len(files)} | glare-targeted: {sum(fg)} | blur-targeted: {sum(fb)} | augmentation on: {aug_on}")
    if aug_on and AUG_CONFIG["glare"]["enabled"] and sum(fg) == 0:
        raise SystemExit("No training file matches AUG_CONFIG['glare']['target_regex'] - fix the regex in augment_glare.py")

    pre = preprocess_fn()

    def load(path, y, g, b):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)                      
        if aug_on:
            img = tf.numpy_function(augment, [img, g, b], tf.float32)
            img.set_shape(IMG_SIZE + (3,))
        y = tf.cast(y, tf.float32)[None] if binary else tf.cast(y, tf.int32)
        return img, y

    ds = tf.data.Dataset.from_tensor_slices((files, np.array(labels), np.array(fg), np.array(fb)))
    ds = ds.shuffle(len(files), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=1 if aug_on else AUTOTUNE)   
    ds = ds.map(lambda x, y: (pre(x), y), num_parallel_calls=AUTOTUNE).batch(BATCH).prefetch(AUTOTUNE)
    return ds, files, fg


def save_aug_examples(files, flags, path, n=6):
    idx = [i for i, f in enumerate(flags) if f][:n]
    if not idx:
        return
    fig, axes = plt.subplots(2, len(idx), figsize=(2.6 * len(idx), 5.4))
    axes = np.array(axes).reshape(2, -1)
    for c, i in enumerate(idx):
        img = np.array(Image.open(files[i]).convert("RGB").resize(IMG_SIZE)).astype(np.float32)
        axes[0, c].imshow(img.astype(np.uint8)); axes[0, c].set_title("original", fontsize=8)
        axes[1, c].imshow(augment(img, True, True, force=True).astype(np.uint8)); axes[1, c].set_title("augmented", fontsize=8)
        axes[0, c].axis("off"); axes[1, c].axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def train_pipeline(train_ds, val_ds, names, run_dir):
    model = build_frozen_model(len(names))
    stage1 = run_dir if REGIME == "frozen" else os.path.join(run_dir, "stage1_head")
    hist, secs = train(model, train_ds, val_ds, stage1, LR_HEAD)
    if REGIME == "frozen":
        return hist, secs
    model = tf.keras.models.load_model(os.path.join(stage1, "best.keras"))
    unfreeze_top(model)
    hist2, secs2 = train(model, train_ds, val_ds, run_dir, LR_FINETUNE)
    return hist2, secs + secs2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no-aug", action="store_true"); args = ap.parse_args()
    aug_on = not args.no_aug
    set_seed(); ensure(REPORTS_DIR)
    tag = "control" if not aug_on else "glare"
    run = os.path.join(RESULTS_ROOT, f"task5c_augment_{tag}_{REGIME}")
    ensure(run)

    val_ds, _, names = load_split("val")                  
    test_ds, test_files, _ = load_split("test")             
    train_ds, train_files, train_flags = make_train_ds(names, aug_on)   
    if aug_on:
        save_aug_examples(train_files, train_flags, os.path.join(run, "aug_examples.png"))

    base_model = tf.keras.models.load_model(BEST_CKPT)
    m0, yt, yp0, _ = full_eval(base_model, test_ds, test_files, names, os.path.join(run, "baseline_reference"), "baseline (test)")

    hist, secs = train_pipeline(train_ds, val_ds, names, run)
    model = tf.keras.models.load_model(os.path.join(run, "best.keras"))
    m1, _, yp1, _ = full_eval(model, test_ds, test_files, names, run, f"{tag} (test)")

    regex = AUG_CONFIG["glare"]["target_regex"]
    sub0, nsub = subset_accuracy(test_files, yt, yp0, regex)
    sub1, _ = subset_accuracy(test_files, yt, yp1, regex)

    row = base_row(f"augment_{tag}_{REGIME}", lr_head=LR_HEAD, lr_finetune=LR_FINETUNE if REGIME != "frozen" else None,
                   augmentation=AUG_CONFIG if aug_on else None, augmentation_split="train only",
                   metrics={k: m1[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")},
                   per_class=m1["per_class"], confusion_matrix=m1["confusion_matrix"],
                   glare_subset_accuracy=sub1, train_time_s=secs, train_val_gap=fit_gap(hist))
    save_row(row, run)

    ctrl_path = os.path.join(RESULTS_ROOT, f"task5c_augment_control_{REGIME}", "test_metrics.json")
    ctrl = json.load(open(ctrl_path))["per_class"] if (aug_on and os.path.exists(ctrl_path)) else None
    rows = []
    for c in names:
        a, b = m0["per_class"][c], m1["per_class"][c]
        r = [c, a["support"], fmt(a["f1"]), fmt(b["f1"]), f"{b['f1'] - a['f1']:+.4f}",
             f"{b['precision'] - a['precision']:+.4f}", f"{b['recall'] - a['recall']:+.4f}"]
        if ctrl:
            r.append(f"{ctrl[c]['f1'] - a['f1']:+.4f}")
        rows.append(r)
    head = ["class", "n(test)", "F1 before", "F1 after", "dF1", "dPrecision", "dRecall"] + (["dF1 of control (noise)"] if ctrl else [])
    table = md_table(head, rows)
    overall = md_table(["metric", "before", "after", "delta"],
                       [[k, fmt(m0[k]), fmt(m1[k]), f"{m1[k] - m0[k]:+.4f}"] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")])
    sub_line = (f"Test files matching '{regex}' (n={nsub}): accuracy {fmt(sub0)} -> {fmt(sub1)}" if sub0 is not None else "")

    memo = f"""# Bonus 5C - Error-driven augmentation (glare focus)

**Evidence for the choice (Task 4):** <<the dominant failure in your failure gallery is glare; your per-class F1 for glare = ...; reference F1 in the capstone report = 0.92>>

**Augmentation (train split only):** `{json.dumps(AUG_CONFIG)}`
Applied only to training images matching `{regex}`; validation and test are unchanged. Same protocol and seed ({SEED}), regime = {REGIME}.
Examples: `results/task5c_augment_{tag}_{REGIME}/aug_examples.png`

## Per-class delta (test split)
{table}

{overall}

{sub_line}

## Conclusion (5 lines)
1. **Did it help?** glare F1 {fmt(m0['per_class'].get('glare', {}).get('f1'))} -> {fmt(m1['per_class'].get('glare', {}).get('f1'))}. <<yes/no/marginal; compare the change with the control-run noise>>
2. **Did it hurt another class?** <<look at dF1/dPrecision/dRecall of 'suitable' and the other defect classes in the table>>
3. **Why (cross-class side effect):** <<e.g. if glare recall rises but 'suitable' precision falls, the model became conservative and rejects bright-but-clean cards>>
4. **Trade-off:** <<what you gain on glare vs what you lose elsewhere, in numbers>>
5. **Verdict:** <<keep / drop the augmentation, justified by the table>>

## Why val/test must stay clean
Validation drives early stopping and model selection, test is the final estimate of real-world performance. If they were augmented
the model would be judged on artificial data that we generated ourselves, the numbers would no longer measure real glare, and
before/after would not be comparable. Augmentation is a *training-time* tool only.

## Why this is the Monitoring & Control phase
measure (Task 4 metrics) -> analyze errors (failure gallery) -> correct (targeted augmentation) -> re-measure (this table).
"""
    open(os.path.join(REPORTS_DIR, "bonus_augmentation.md"), "w").write(memo)
    print(table); print(overall); print(sub_line)
    print("\nDraft memo written to reports/bonus_augmentation.md - finish the <<...>> parts by hand.")


if __name__ == "__main__":
    main()
