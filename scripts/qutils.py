import json, os, random, shutil, time, traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

CLASSES = ["bad", "good"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WARMUP, TIMED = 20, 100


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))


def split_dir(root, split):
    """Return the folder of a split; accepts 'val' as an alias of 'validation'."""
    root = Path(root)
    cand = [root / split] + ([root / "val"] if split == "validation" else [])
    for c in cand:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"Split folder not found: {cand[0]} (run make_split_folders.py)")


def list_split(root, split):
    """[(path, label_idx)] sorted, for one split."""
    items = []
    d = split_dir(root, split)
    for idx, cls in enumerate(CLASSES):
        for p in sorted((d / cls).glob("*")):
            if p.suffix.lower() in IMG_EXT:
                items.append((str(p), idx))
    return items


def print_counts(root):
    print(f"Image counts in {root}:")
    for split in ("train", "validation", "test"):
        items = list_split(root, split)
        per = {c: sum(1 for _, l in items if l == i) for i, c in enumerate(CLASSES)}
        print(f"  {split:<11} total={len(items):>6}  " + "  ".join(f"{c}={n}" for c, n in per.items()))


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(fm),
        "precision_macro": float(pm),
        "recall_macro": float(rm),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(CLASSES)},
    }
    return metrics, cm.tolist()


def copy_errors(paths, y_true, y_pred, out_dir):
    """Copy misclassified test images to out_dir/errors, name = true-X__pred-Y__<orig>."""
    err_dir = Path(out_dir) / "errors"
    if err_dir.exists():
        shutil.rmtree(err_dir)
    err_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for pth, t, p in zip(paths, y_true, y_pred):
        if int(t) != int(p):
            shutil.copy2(pth, err_dir / f"true-{CLASSES[int(t)]}__pred-{CLASSES[int(p)]}__{Path(pth).name}")
            n += 1
    print(f"Copied {n} misclassified test images to {err_dir}")
    return n


def measure_latency(predict_fn, pil_images, warmup=WARMUP, timed=TIMED):
    """Mean ms/image. predict_fn(PIL.Image) must do preprocessing + forward pass."""
    n = len(pil_images)
    for i in range(warmup):
        predict_fn(pil_images[i % n])
    t = []
    for i in range(timed):
        img = pil_images[i % n]
        t0 = time.perf_counter()
        predict_fn(img)
        t.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(t))


def load_latency_images(test_items, seed, k=WARMUP + TIMED):
    rng = random.Random(seed)
    chosen = [test_items[i][0] for i in rng.sample(range(len(test_items)), min(k, len(test_items)))]
    return [Image.open(p).convert("RGB") for p in chosen]


def save_results(out_dir, res):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    log_run(res["experiment"], "ok", res.get("note", ""))
    print(f"Saved {out_dir / 'results.json'}")


def log_run(experiment, status, note=""):
    """Append one line per run (including failures) to results/experiment_log.jsonl."""
    Path("results").mkdir(exist_ok=True)
    with open("results/experiment_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                            "experiment": experiment, "status": status, "note": note},
                           ensure_ascii=False) + "\n")


def run_logged(experiment, note, fn):
    """Run fn(); on exception write results/<experiment>/FAILED.json + log line, then re-raise."""
    try:
        return fn()
    except Exception as e:
        out = Path("results") / experiment
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "FAILED.json", "w", encoding="utf-8") as f:
            json.dump({"experiment": experiment, "status": "failed", "error": repr(e),
                       "note": note or "ADD A ONE-LINE NOTE (--note) ABOUT WHY IT FAILED",
                       "traceback": traceback.format_exc()}, f, indent=2)
        log_run(experiment, "failed", f"{e!r} | {note}")
        raise


def plot_curves(history, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    keys = [("loss", "val_loss", "Loss"), ("accuracy", "val_accuracy", "Accuracy")]
    for a, (tr, va, name) in zip(ax, keys):
        if tr in history:
            a.plot(range(1, len(history[tr]) + 1), history[tr], label="train")
        if va in history:
            a.plot(range(1, len(history[va]) + 1), history[va], label="validation")
        a.set_xlabel("epoch"); a.set_title(name); a.grid(alpha=.3); a.legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_confusion(cm, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    fig, a = plt.subplots(figsize=(4, 3.6))
    a.imshow(cm, cmap="Blues")
    a.set_xticks([0, 1]); a.set_xticklabels(CLASSES); a.set_yticks([0, 1]); a.set_yticklabels(CLASSES)
    a.set_xlabel("predicted"); a.set_ylabel("true"); a.set_title(title, fontsize=9)
    for i in range(2):
        for j in range(2):
            a.text(j, i, int(cm[i, j]), ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def base_result(experiment, week, model, regime, input_size, seed, device, epochs_planned):
    return {"experiment": experiment, "week": week, "model": model, "regime": regime,
            "input_size": input_size, "seed": seed, "device": device,
            "epochs_planned": epochs_planned}
