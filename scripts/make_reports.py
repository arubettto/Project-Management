"""Fill report skeletons with FACTS from results.json. The interpretation text is yours to write.

  python scripts/make_reports.py baseline      # -> reports/baseline.md        (Task 2)
  python scripts/make_reports.py week7         # -> reports/week7_comparison.md (Task 3)
"""
import json, sys
from pathlib import Path


def load(exp):
    return json.load(open(Path("results") / exp / "results.json"))


def facts_curves(r):
    h = r["history"]; n = len(h["loss"])
    va = h.get("val_accuracy", []); vl = h.get("val_loss", [])
    best = (min(range(len(vl)), key=lambda i: vl[i]) + 1) if vl else None
    s = [f"- epochs run: {r['epochs_run']} of {r['epochs_planned']} planned (early stopping {'triggered' if r['epochs_run'] < r['epochs_planned'] else 'not triggered'}); best val_loss at epoch {best}"]
    if "accuracy" in h and va:
        s.append(f"- final train accuracy {h['accuracy'][-1]:.3f} vs final val accuracy {va[-1]:.3f} (gap {h['accuracy'][-1] - va[-1]:+.3f})")
    if vl:
        s.append(f"- final train loss {h['loss'][-1]:.3f} vs val loss {vl[-1]:.3f}; min val loss {min(vl):.3f}")
    return "\n".join(s)


def baseline():
    r = load("mobilenetv2_frozen"); m = r["metrics"]; pc = m["per_class"]; cm = r["confusion_matrix"]
    harder = min(pc, key=lambda c: pc[c]["f1"])
    txt = f"""# Task 2 - MobileNetV2 frozen baseline

Command: `python scripts/train_keras_quality.py --model mobilenetv2 --regime frozen --seed {r['seed']} --week {r['week']}`
Seed = {r['seed']}; epochs <= 50, patience 5, batch 32, Adam lr 1e-3, input 224x224, preprocess_input mode "tf", threshold 0.5, split 70/15/15.

## Test-split results ({r['test_samples']} images)
| accuracy | precision macro | recall macro | F1 macro | latency ms/img | size MB | params | train time s |
|---|---|---|---|---|---|---|---|
| {m['accuracy']:.4f} | {m['precision_macro']:.4f} | {m['recall_macro']:.4f} | {m['f1_macro']:.4f} | {r['latency_ms_per_image']} | {r['model_size_mb']} | {r['params']:,} | {r['train_time_s']} |

Per class: bad P/R/F1 = {pc['bad']['precision']:.3f}/{pc['bad']['recall']:.3f}/{pc['bad']['f1']:.3f}; good P/R/F1 = {pc['good']['precision']:.3f}/{pc['good']['recall']:.3f}/{pc['good']['f1']:.3f}.
Confusion matrix (rows true bad/good, cols predicted bad/good): {cm}. Images: curves.png, confusion_matrix.png; errors in results/mobilenetv2_frozen/errors/.

## Facts from the curves (use them, do not just copy them)
{facts_curves(r)}
- harder class by F1: **{harder}**

## Interpretation (5-10 lines, WRITE IN YOUR OWN WORDS)
TODO 1. What do the curves show: overfitting (train acc >> val acc, val loss rising) or underfitting (both low, still improving)? Cite the numbers above.
TODO 2. Which class is harder ({harder}?) and why: look at the images in errors/ and name the defect that fools the model (blur / glare / skew / occlusion / exposure).
TODO 3. One sentence on what you would change (e.g. fine-tuning with a low learning rate, augmentation).
"""
    Path("reports").mkdir(exist_ok=True)
    Path("reports/baseline.md").write_text(txt, encoding="utf-8"); print("reports/baseline.md written")


def week7(baseline_exp="mobilenetv2_frozen"):
    b = load(baseline_exp); ys = [load(f"week7_yolov8{v}_cls") for v in "ns" if (Path("results") / f"week7_yolov8{v}_cls" / "results.json").exists()]
    rows = "| model | accuracy | F1 macro | latency ms/img | size MB | params | input | epochs run | train time s |\n|---|---|---|---|---|---|---|---|---|\n"
    for r in [b] + ys:
        rows += f"| {r['model']} ({r['regime']}) | {r['metrics']['accuracy']:.4f} | {r['metrics']['f1_macro']:.4f} | {r['latency_ms_per_image']} | {r['model_size_mb']} | {r['params']:,} | {r['input_size']} | {r['epochs_run']} | {r['train_time_s']} |\n"
    deltas = ""
    for y in ys:
        deltas += (f"- {y['model']} vs MobileNetV2: accuracy {y['metrics']['accuracy'] - b['metrics']['accuracy']:+.4f}, F1 {y['metrics']['f1_macro'] - b['metrics']['f1_macro']:+.4f}, "
                   f"latency {y['latency_ms_per_image'] - b['latency_ms_per_image']:+.1f} ms ({y['latency_ms_per_image'] / b['latency_ms_per_image']:.1f}x), "
                   f"size {y['model_size_mb'] - b['model_size_mb']:+.1f} MB, params {y['params'] - b['params']:+,}\n")
    seed = b["seed"]
    txt = f"""# Week 7 - YOLOv8 classification vs MobileNetV2 (same split, same test set)

Commands (seed = student ID = {seed}):
```
python scripts/train_keras_quality.py --model mobilenetv2 --regime frozen --seed {seed} --week 6
python scripts/train_yolo_quality.py --seed {seed} --variants n s --week 7
```
YOLO: epochs 50, patience 5, imgsz 640, batch 32 (note here if you had to use 16 because of CUDA OOM). Latency: 20 warm-up + 100 timed predictions on test images.

## Mini table
{rows}
## Numeric deltas (facts)
{deltas}
## Comparison paragraph (5+ sentences, WRITE YOURSELF)
TODO. Cover: (1) accuracy delta, (2) latency delta, (3) size delta, with the numbers above;
(4) WHY they differ - architecture (frozen ImageNet features + small head in MobileNetV2 vs full fine-tuning of the whole YOLOv8-cls network, depth/width of n vs s);
(5) input size 640 vs 224 (more pixels keep fine defects such as glare and blur visible but cost compute, which explains latency);
(6) head design (custom Dense(128)+Dropout head with sigmoid vs YOLO classification head with softmax over 2 classes);
(7) one sentence on which one you would prefer for the 100 ms stage-1 gate.
"""
    Path("reports").mkdir(exist_ok=True)
    Path("reports/week7_comparison.md").write_text(txt, encoding="utf-8"); print("reports/week7_comparison.md written")


if __name__ == "__main__":
    {"baseline": baseline, "week7": week7}[sys.argv[1]]()
