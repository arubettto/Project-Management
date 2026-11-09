"""Task 3: YOLOv8 classification (yolov8n-cls and yolov8s-cls) on the SAME data/quality_split.

Exact command (put it in your report):
  python scripts/train_yolo_quality.py --seed <STUDENT_ID> --variants n s --week 7
If CUDA runs out of memory:  add  --batch 16  and write that in the report.

Fixed by the course: epochs 50, patience 5, imgsz 640, batch 32, seed = student ID.
Metrics are computed by us (sklearn) on the TEST split from per-image predictions, so they are
fractions (0.945, not 94.5).
"""
import argparse, csv, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image
import qutils as Q


def read_history(csv_path):
    """Ultralytics results.csv -> history dict with the same keys as the Keras runs."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    col = lambda name: [float(r[name]) for r in rows if r.get(name) not in (None, "")]
    hist = {"loss": col("train/loss"), "val_loss": col("val/loss"),
            "val_accuracy": col("metrics/accuracy_top1")}   # fractions already (0-1)
    return {k: v for k, v in hist.items() if v}, len(rows)


def run_variant(args, variant):
    from ultralytics import YOLO
    import torch
    exp = f"week{args.week}_yolov8{variant}_cls"
    Q.run_logged(exp, args.note, lambda: _run(args, variant, exp, YOLO, torch))


def _run(args, variant, exp, YOLO, torch):
    Q.set_seed(args.seed)
    Q.print_counts(args.data)                     # R1: same folders as Task 2, no re-splitting
    out = Path("results").resolve() / exp
    data = str(Path(args.data).resolve())
    kw = dict(data=data, epochs=args.epochs, patience=args.patience, imgsz=args.imgsz,
              batch=args.batch, seed=args.seed, deterministic=True,
              project=str(out.parent), name=exp, exist_ok=True, plots=True)
    if args.device:
        kw["device"] = args.device
    print("Ultralytics train args:", kw)

    t0 = time.perf_counter()
    model = YOLO(f"yolov8{variant}-cls.pt")
    model.train(**kw)
    train_time = time.perf_counter() - t0

    best = out / "weights" / "best.pt"
    assert best.exists(), f"best.pt not found at {best}"
    model = YOLO(str(best))
    names = model.names                           # {0: 'bad', 1: 'good'} - from folder names
    assert sorted(names.values()) == sorted(Q.CLASSES), f"class names must be good/bad, got {names}"

    # TEST evaluation only
    test_items = Q.list_split(args.data, "test")
    paths = [p for p, _ in test_items]
    y_true = np.array([l for _, l in test_items])
    y_pred = []
    for i in range(0, len(paths), 32):
        for r in model.predict(source=paths[i:i + 32], imgsz=args.imgsz, verbose=False):
            y_pred.append(Q.CLASSES.index(names[int(r.probs.top1)]))
    y_pred = np.array(y_pred)
    assert len(y_pred) == len(y_true)
    metrics, cm = Q.compute_metrics(y_true, y_pred)
    print(metrics, cm)
    Q.plot_confusion(cm, out / "confusion_matrix_test.png", f"{exp} - test")
    Q.copy_errors(paths, y_true, y_pred, out)

    # latency: 20 warm-up + 100 timed, real test images, PIL in memory -> probabilities
    imgs = Q.load_latency_images(test_items, args.seed)
    latency = Q.measure_latency(lambda im: model.predict(im, imgsz=args.imgsz, verbose=False), imgs)
    device = f"cuda ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    hist, epochs_run = read_history(out / "results.csv")
    params = int(sum(p.numel() for p in model.model.parameters()))

    res = Q.base_result(exp, args.week, f"yolov8{variant}-cls", "pretrained_full_finetune",
                        args.imgsz, args.seed, device, args.epochs)
    res.update({
        "epochs_run": epochs_run, "train_time_s": round(train_time, 1),
        "latency_ms_per_image": round(latency, 2), "latency_device": device,
        "model_size_mb": round(os.path.getsize(best) / (1024 * 1024), 3),
        "params": params, "test_samples": int(len(y_true)), "batch_size": args.batch,
        "metrics": metrics, "confusion_matrix": cm, "history": hist, "note": args.note,
    })
    Q.save_results(out, res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, help="YOUR STUDENT ID")
    ap.add_argument("--variants", nargs="+", default=["n", "s"], choices=["n", "s", "m"])
    ap.add_argument("--week", type=int, default=7)
    ap.add_argument("--data", default="data/quality_split")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None, help="e.g. 0 or cpu (default: Ultralytics auto)")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    for v in a.variants:
        run_variant(a, v)
