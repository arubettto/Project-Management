import csv
import json
import math
import os
import pathlib
import random
import time
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

SEED = int(os.environ.get("SEED", 123))        
DATA_DIR = os.environ.get("DATA_DIR", "data/quality_split_ready")
SPLIT_DIRS = {"train": "train", "val": "validation", "test": "test"}  
IMG_SIZE = (224, 224)
BATCH = 32
EPOCHS = int(os.environ.get("EPOCHS", 50))
PATIENCE = 5
MONITOR = "val_loss"
PREPROCESS = os.environ.get("PREPROCESS", "mobilenet_v2")
WEIGHTS = None if os.environ.get("WEIGHTS", "imagenet") == "none" else os.environ.get("WEIGHTS", "imagenet") 

HEAD_UNITS = 128
DROPOUT = 0.3
LR_HEAD = 1e-3

LR_FINETUNE = 1e-5
UNFREEZE_LAST = 30
FREEZE_BN = True      

BEST_REGIME = os.environ.get("BEST_REGIME", "frozen") 
BEST_CKPT = os.environ.get("BEST_CKPT", "results/mobilenetv2_frozen/best.keras")
BEST_HISTORY_CSV = "results/mobilenetv2_frozen/history.csv"
BAD_CLASS = "bad"                                   

RESULTS_ROOT = "results"
REPORTS_DIR = "reports"
RESULTS_JSON = os.path.join(RESULTS_ROOT, "results.json")   

AUTOTUNE = tf.data.AUTOTUNE


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def now():
    return datetime.now().isoformat(timespec="seconds")


def ensure(path):
    os.makedirs(path, exist_ok=True)
    return path


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def fmt(x, nd=4):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def preprocess_fn():
    if PREPROCESS == "mobilenet_v2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    if PREPROCESS == "rescale":
        return lambda x: x / 255.0
    return lambda x: x


def class_names_from_dir(split="train"):
    return sorted(d.name for d in pathlib.Path(DATA_DIR, SPLIT_DIRS.get(split, split)).iterdir() if d.is_dir())


def bad_index(names):
    return names.index(BAD_CLASS)


def load_split(split, shuffle=False):
    """Returns (dataset, file_paths, class_names). Images are RGB, 224x224, preprocessed like Task 2."""
    names = class_names_from_dir("train")
    mode = "binary" if len(names) == 2 else "int"
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DATA_DIR, SPLIT_DIRS.get(split, split)), labels="inferred", label_mode=mode, class_names=names,
        color_mode="rgb", image_size=IMG_SIZE, batch_size=BATCH, shuffle=shuffle, seed=SEED)
    files = list(ds.file_paths)
    pre = preprocess_fn()
    ds = ds.map(lambda x, y: (pre(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return ds, files, names


def get_sample(split="val"):
    """One preprocessed image, shape (1,H,W,3), for latency benchmarks."""
    ds, _, _ = load_split(split)
    for x, _ in ds.take(1):
        return x[:1].numpy().astype(np.float32)


def build_frozen_model(num_classes):
    """Task-2 style model: frozen MobileNetV2 -> GAP -> Dense -> Dropout -> sigmoid/softmax."""
    base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights=WEIGHTS)
    base.trainable = False
    inp = tf.keras.Input(IMG_SIZE + (3,))
    x = base(inp, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(HEAD_UNITS, activation="relu")(x)
    x = tf.keras.layers.Dropout(DROPOUT)(x)
    out = (tf.keras.layers.Dense(1, activation="sigmoid")(x) if num_classes == 2
           else tf.keras.layers.Dense(num_classes, activation="softmax")(x))
    return tf.keras.Model(inp, out)


def find_base(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested backbone model found - adapt find_base() to your architecture.")


def unfreeze_top(model, n=UNFREEZE_LAST, freeze_bn=FREEZE_BN):
    """Unfreeze only the last n layers of the backbone."""
    base = find_base(model)
    base.trainable = True
    for layer in base.layers[:-n]:
        layer.trainable = False
    if freeze_bn:
        for layer in base.layers[-n:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
    return base


def count_trainable(model):
    return int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))


def compile_model(model, lr):
    loss = "binary_crossentropy" if model.output_shape[-1] == 1 else "sparse_categorical_crossentropy"
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=["accuracy"])


def train(model, train_ds, val_ds, run_dir, lr):
    """Identical protocol everywhere: epochs=EPOCHS, early stopping patience=PATIENCE, batch=BATCH, val only."""
    ensure(run_dir)
    compile_model(model, lr)
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor=MONITOR, patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(os.path.join(run_dir, "best.keras"), monitor=MONITOR, save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(run_dir, "history.csv")),
    ]
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs, verbose=2).history
    secs = time.time() - t0
    plot_curves(hist, os.path.join(run_dir, "curves.png"))
    return hist, secs


def plot_curves(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ep = range(1, len(h["loss"]) + 1)
    ax[0].plot(ep, h["loss"], label="train"); ax[0].plot(ep, h["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(ep, h["accuracy"], label="train"); ax[1].plot(ep, h["val_accuracy"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def read_history_csv(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0] if k != "epoch"}


def fit_gap(h):
    """Train-val gap at the best epoch (the epoch whose weights are restored)."""
    if h is None:
        return None
    i = int(np.argmin(h["val_loss"]))
    return {"best_epoch": i + 1, "epochs_run": len(h["loss"]),
            "train_acc": h["accuracy"][i], "val_acc": h["val_accuracy"][i],
            "acc_gap": h["accuracy"][i] - h["val_accuracy"][i],
            "train_loss": h["loss"][i], "val_loss": h["val_loss"][i],
            "loss_gap": h["val_loss"][i] - h["loss"][i]}


def predict(model, ds):
    ys, ps = [], []
    for x, y in ds:
        ps.append(model(x, training=False).numpy())
        ys.append(y.numpy())
    return np.concatenate(ys).ravel().astype(int), np.concatenate(ps)


def to_pred(probs, thr=0.5):
    return (probs[:, 0] >= thr).astype(int) if probs.shape[1] == 1 else probs.argmax(1)


def compute_metrics(y_true, y_pred, names):
    labels = list(range(len(names)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pc, rc, fc, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
        "per_class": {n: {"precision": float(pc[i]), "recall": float(rc[i]), "f1": float(fc[i]), "support": int(sup[i])}
                      for i, n in enumerate(names)},
        "confusion_matrix": cm.tolist(),
    }


def save_confusion(cm, names, path, title):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 3, 1.2 * len(names) + 2.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def error_gallery(files, y_true, y_pred, names, out_dir, max_n=16):
    ensure(out_dir)
    idx = np.where(y_true != y_pred)[0]
    with open(os.path.join(out_dir, "errors.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["file", "true", "pred"])
        for i in idx:
            w.writerow([files[i], names[y_true[i]], names[y_pred[i]]])
    sel = idx[:max_n]
    if len(sel):
        cols = 4; rows = math.ceil(len(sel) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, i in zip(axes, sel):
            ax.imshow(Image.open(files[i]).convert("RGB"))
            ax.set_title(f"true: {names[y_true[i]]}\npred: {names[y_pred[i]]}", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "gallery.png"), dpi=120); plt.close(fig)
    return int(len(idx))


def full_eval(model, ds, files, names, out_dir, title="", thr=0.5):
    """Accuracy + macro P/R/F1 + confusion matrix + error gallery. Call ONLY on the test split, once, after training."""
    ensure(out_dir)
    y_true, probs = predict(model, ds)
    y_pred = to_pred(probs, thr)
    m = compute_metrics(y_true, y_pred, names)
    m["n_errors"] = error_gallery(files, y_true, y_pred, names, os.path.join(out_dir, "errors"))
    save_confusion(m["confusion_matrix"], names, os.path.join(out_dir, "confusion_matrix.png"), title or "test")
    json.dump(m, open(os.path.join(out_dir, "test_metrics.json"), "w"), indent=2)
    return m, y_true, y_pred, probs


def subset_accuracy(files, y_true, y_pred, regex):
    """Accuracy on test files whose path matches regex (e.g. 'glare') - useful when labels are only good/bad."""
    import re
    idx = [i for i, f in enumerate(files) if re.search(regex, f, re.I)]
    if not idx:
        return None, 0
    return float(np.mean(y_true[idx] == y_pred[idx])), len(idx)


def clean_size_mb(ckpt_path, tmp_dir=None):
    """Size of the model WITHOUT optimizer state (fair before/after comparison). tmp_dir is ignored (temp dir is used)."""
    import tempfile
    m = tf.keras.models.load_model(ckpt_path, compile=False)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "clean_model.keras")
        m.save(p)
        return os.path.getsize(p) / 1e6


def benchmark(fn, sample, warmup=20, runs=100):
    """Warm-up protocol: 20 untimed runs, then 100 timed runs, batch size 1."""
    for _ in range(warmup):
        fn(sample)
    ts = []
    for _ in range(runs):
        t = time.perf_counter(); fn(sample); ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {"mean_ms": float(ts.mean()), "median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
            "std_ms": float(ts.std()), "warmup": warmup, "runs": runs, "batch": 1}


def keras_latency(model, sample):
    with tf.device("/CPU:0"):
        x = tf.constant(sample)
        return benchmark(lambda s: model(s, training=False).numpy(), x)


def load_rows(path=RESULTS_JSON):
    if not os.path.exists(path):
        return []
    rows = json.load(open(path))
    return [rows] if isinstance(rows, dict) else rows


def save_row(row, run_dir, path=RESULTS_JSON):
    """Writes run_dir/results.json and appends the row to the global results table."""
    ensure(run_dir)
    json.dump(row, open(os.path.join(run_dir, "results.json"), "w"), indent=2)
    rows = load_rows(path)
    rows.append(row)
    ensure(os.path.dirname(path) or ".")
    json.dump(rows, open(path, "w"), indent=2)


def update_row(regime, fields, path=RESULTS_JSON):
    """Add fields (e.g. chosen_threshold, deploy_benchmark) to the latest row with this regime."""
    rows = load_rows(path)
    for r in reversed(rows):
        if r.get("regime") == regime:
            r.update(fields)
            json.dump(rows, open(path, "w"), indent=2)
            return True
    print(f"[warn] no row with regime={regime!r} in {path}; add these fields by hand: {fields}")
    return False


def base_row(regime, **extra):
    row = {"regime": regime, "model": "MobileNetV2", "seed": SEED, "batch": BATCH, "epochs_max": EPOCHS,
           "patience": PATIENCE, "split": "test", "preprocess": PREPROCESS, "timestamp": now()}
    row.update(extra)
    return row
