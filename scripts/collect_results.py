import argparse, json, glob, shutil, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import qutils as Q

REPORTS = Path("reports")
# acceptance criteria from the Task 1 memo (+ the course budget of the first gate)
CRIT = {"accuracy": 0.91, "f1_macro": 0.90, "latency_ms_per_image": 100.0, "model_size_mb": 50.0}


def load_results():
    rows = []
    for f in sorted(glob.glob("results/*/results.json")):
        r = json.load(open(f))
        m = r["metrics"]
        rows.append({"experiment": r["experiment"], "model": r["model"], "regime": r["regime"],
                     "input": r["input_size"], "accuracy": m["accuracy"], "precision_macro": m["precision_macro"],
                     "recall_macro": m["recall_macro"], "f1_macro": m["f1_macro"],
                     "latency_ms_per_image": r["latency_ms_per_image"], "model_size_mb": r["model_size_mb"],
                     "params": r["params"], "train_time_s": r["train_time_s"], "epochs_run": r["epochs_run"],
                     "device": r.get("device"), "latency_device": r.get("latency_device", r.get("device")),
                     "cm": r["confusion_matrix"], "dir": str(Path(f).parent), "test_samples": r["test_samples"],
                     "bad_recall": m["per_class"]["bad"]["recall"], "good_recall": m["per_class"]["good"]["recall"]})
    return pd.DataFrame(rows)


def passes(r):
    return {"accuracy": r.accuracy >= CRIT["accuracy"], "f1_macro": r.f1_macro >= CRIT["f1_macro"],
            "latency": r.latency_ms_per_image < CRIT["latency_ms_per_image"],
            "size": r.model_size_mb <= CRIT["model_size_mb"], "bad_recall": r.bad_recall >= 0.85}


def table(df):
    cols = ["experiment", "accuracy", "precision_macro", "recall_macro", "f1_macro", "latency_ms_per_image",
            "model_size_mb", "params", "train_time_s", "epochs_run"]
    d = df[cols].copy()
    d.to_csv(REPORTS / "comparison.csv", index=False)
    hdr = "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Latency ms/img | Size MB | Params | Train time s | Epochs run |\n|" + "---|" * 10 + "\n"
    body = "".join(f"| {r.experiment} | {r.accuracy:.4f} | {r.precision_macro:.4f} | {r.recall_macro:.4f} | {r.f1_macro:.4f} | "
                   f"{r.latency_ms_per_image:.1f} | {r.model_size_mb:.1f} | {int(r.params):,} | {r.train_time_s:.0f} | {int(r.epochs_run)} |\n"
                   for r in d.itertuples())
    note = ("\nAll numbers are on the TEST split (same split, seed and fixed parameters for all models). "
            "Latency: mean of 100 timed single-image predictions after 20 warm-up predictions; devices: "
            + ", ".join(sorted(set(f"{r.experiment}={r.latency_device}" for r in df.itertuples()))) + ".\n")
    (REPORTS / "comparison.md").write_text("# Model comparison (built from results.json)\n\n" + hdr + body + note, encoding="utf-8")
    return hdr + body


def charts(df):
    for xcol, fname, xl in [("latency_ms_per_image", "acc_vs_latency.png", "latency, ms/image"),
                            ("model_size_mb", "acc_vs_size.png", "model size, MB")]:
        fig, a = plt.subplots(figsize=(6.2, 4.2))
        a.scatter(df[xcol], df["accuracy"], s=70)
        for r in df.itertuples():
            a.annotate(r.experiment, (getattr(r, xcol), r.accuracy), fontsize=7, xytext=(4, 4), textcoords="offset points")
        if xcol == "latency_ms_per_image":
            a.axvline(100, ls="--", c="r", lw=1); a.text(100, a.get_ylim()[0], " 100 ms budget", color="r", fontsize=8, va="bottom")
        a.set_xlabel(xl); a.set_ylabel("test accuracy"); a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(REPORTS / fname, dpi=140); plt.close(fig)
    n = len(df)
    fig, ax = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    for a, r in zip(ax[0], df.itertuples()):
        cm = np.array(r.cm); a.imshow(cm, cmap="Blues")
        a.set_xticks([0, 1]); a.set_xticklabels(Q.CLASSES); a.set_yticks([0, 1]); a.set_yticklabels(Q.CLASSES)
        a.set_title(r.experiment, fontsize=8); a.set_xlabel("predicted"); a.set_ylabel("true")
        for i in range(2):
            for j in range(2):
                a.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(REPORTS / "confusion_matrices.png", dpi=140); plt.close(fig)


# ---------- metadata / defects ----------
FILE_KEYS = ["filename", "file", "file_name", "path", "image", "image_path", "img", "name"]
DEFECT_KEYS = ["defect", "defect_type", "defects", "quality_issue", "issue", "defect_name", "label_name"]


def load_defects(meta_path):
    p = Path(meta_path) if meta_path else None
    if not p or not p.exists():
        found = glob.glob("**/metadata.jsonl", recursive=True)
        p = Path(found[0]) if found else None
    if not p:
        print("metadata.jsonl not found -> defects must be labelled by hand"); return {}, None
    mapping, fk, dk = {}, None, None
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        fk = fk or next((k for k in FILE_KEYS if k in d), None)
        dk = dk or next((k for k in DEFECT_KEYS if k in d), None)
        if fk and dk:
            v = d[dk]; v = "+".join(map(str, v)) if isinstance(v, list) else str(v)
            mapping[Path(str(d[fk])).name] = v
    print(f"metadata: {p} file key={fk} defect key={dk} entries={len(mapping)}")
    return mapping, (fk, dk)


def gallery(df, defects, data, n_show=6):
    gal = REPORTS / "failure_gallery"; gal.mkdir(parents=True, exist_ok=True)
    test = Q.list_split(data, "test") if Path(data).exists() else []
    test_def = Counter(defects.get(Path(p).name, "unknown") for p, _ in test)
    lines, per_model, overall = [], {}, Counter()
    for r in df.itertuples():
        errs = sorted((Path(r.dir) / "errors").glob("*"))
        errs = [e for e in errs if e.suffix.lower() in Q.IMG_EXT]
        orig = lambda e: e.name.split("__", 2)[2]
        dcount = Counter(defects.get(orig(e), "unknown") for e in errs)
        per_model[r.experiment] = dcount
        overall.update(dcount)
        d = gal / r.experiment; d.mkdir(exist_ok=True)
        shown = errs[:max(n_show, 5)]
        for e in shown:
            shutil.copy2(e, d / e.name)
        if shown:
            cols = 3; rows_ = (len(shown) + cols - 1) // cols
            fig, ax = plt.subplots(rows_, cols, figsize=(3.4 * cols, 3.0 * rows_), squeeze=False)
            for a in ax.ravel(): a.axis("off")
            for a, e in zip(ax.ravel(), shown):
                t, p_ = e.name.split("__")[0][5:], e.name.split("__")[1][5:]
                a.imshow(Image.open(e).convert("RGB")); a.set_title(f"true: {t} -> pred: {p_}\ndefect: {defects.get(orig(e), 'n/a')}", fontsize=7)
            fig.suptitle(r.experiment, fontsize=10); fig.tight_layout(); fig.savefig(gal / f"{r.experiment}.png", dpi=120); plt.close(fig)
        dom = dcount.most_common(1)[0] if dcount else ("n/a", 0)
        lines.append(f"### {r.experiment}\n- misclassified test images: {len(errs)} (gallery shows {len(shown)}; need >= 5)\n"
                     f"- failures by defect: {dict(dcount) if dcount else 'n/a'}\n- **dominant failure defect: {dom[0]}** ({dom[1]} images)\n"
                     + ("- error rate per defect (errors / test images with that defect): "
                        + ", ".join(f"{k}: {v}/{test_def[k]}" for k, v in dcount.items() if test_def.get(k)) + "\n" if test_def and dcount else ""))
    hardest = overall.most_common(1)[0][0] if overall else "n/a"
    txt = ("# Error analysis (failure gallery)\n\n" + "\n".join(lines) +
           f"\n## Hardest defect overall\n\n**{hardest}** (most failures summed over all models). Expected from the thesis: glare (F1 = 0.92). "
           "Check that your data agrees; if not, say so.\n\n## How to reduce it (WRITE 3-5 SENTENCES YOURSELF)\n\n"
           "TODO: which augmentation or data change would help for the hardest defect, and why (e.g. more glare variants "
           "with different position / intensity / size, glare on laminated pages, hard-negative mining from the errors above, "
           "real glare photos in train, threshold tuning).\n")
    (REPORTS / "error_analysis.md").write_text(txt, encoding="utf-8")
    return hardest


def memo(df, hardest):
    df = df.copy(); df["ok"] = [all(passes(r).values()) for r in df.itertuples()]
    okdf = df[df.ok].sort_values(["f1_macro", "latency_ms_per_image"], ascending=[False, True])
    chosen = (okdf if len(okdf) else df.sort_values("f1_macro", ascending=False)).iloc[0]
    why_none = "" if len(okdf) else "**No model meets all acceptance criteria; the model below is the closest and the decision is conditional (see Risks).** "
    mark = lambda b: "pass" if b else "FAIL"
    ev = "| Model | Accuracy | F1 macro | Recall bad | Latency ms | Size MB | All criteria |\n|---|---|---|---|---|---|---|\n"
    for r in df.itertuples():
        p = passes(r)
        ev += (f"| {r.experiment} | {r.accuracy:.3f} ({mark(p['accuracy'])}) | {r.f1_macro:.3f} ({mark(p['f1_macro'])}) | {r.bad_recall:.3f} ({mark(p['bad_recall'])}) | "
               f"{r.latency_ms_per_image:.1f} ({mark(p['latency'])}) | {r.model_size_mb:.1f} ({mark(p['size'])}) | {'YES' if r.ok else 'no'} |\n")
    rej = ""
    for r in df.itertuples():
        if r.experiment == chosen.experiment:
            continue
        fails = [k for k, v in passes(r).items() if not v]
        rej += (f"- **{r.experiment}**: F1 {r.f1_macro:.3f} vs {chosen.f1_macro:.3f}, latency {r.latency_ms_per_image:.1f} vs {chosen.latency_ms_per_image:.1f} ms, "
                f"size {r.model_size_mb:.1f} vs {chosen.model_size_mb:.1f} MB"
                + (f"; fails: {', '.join(fails)}." if fails else "; meets all criteria but is not better on the deciding metrics (F1, then latency)."))
        rej += " TODO: add the real reason in your own words.\n"
    skipped = [("ResNet50", "resnet50", "≈98 MB FP32 weights > 50 MB limit"), ("VGG19", "vgg19", "≈548 MB"), ("ViT-B/16", "vit", "≈330 MB, data-hungry")]
    skipped = [f"{n} ({why})" for n, key, why in skipped if not df.experiment.str.contains(key).any()]
    if skipped:
        rej += "- Not trained (excluded in the Task 1 memo before any experiment): " + "; ".join(skipped) + " - they violate the size/latency limits of a stage-1 CPU gate.\n"
    text = f"""# Decision memo: quality gate model (stage 1 of the pipeline)

## 1. Decision
{why_none}**Use {chosen.experiment} ({chosen.model}, input {int(chosen.input)}x{int(chosen.input)}) as the quality gate.** TODO: confirm this is your decision; if you pick another model, justify it below.

## 2. Evidence
All values are on the same TEST split ({int(chosen.test_samples)} images), same seed and fixed training parameters; source: results/*/results.json.

{ev}
Criteria (from the Task 1 memo): accuracy >= {CRIT['accuracy']}, macro F1 >= {CRIT['f1_macro']}, recall of the "bad" class >= 0.85, latency < {CRIT['latency_ms_per_image']:.0f} ms/image (budget of the first pipeline gate), size <= {CRIT['model_size_mb']:.0f} MB.
The chosen model: accuracy {chosen.accuracy:.3f}, F1 {chosen.f1_macro:.3f}, latency {chosen.latency_ms_per_image:.1f} ms ({abs(100 - chosen.latency_ms_per_image):.0f} ms {'below' if chosen.latency_ms_per_image < 100 else 'ABOVE'} the 100 ms budget), size {chosen.model_size_mb:.1f} MB.
Latency devices: {', '.join(sorted(set(df.latency_device.astype(str))))} (run scripts/remeasure_latency.py if the budget must be proven on the CPU server).

## 3. Rejected alternatives and why
{rej}
## 4. Risks and mitigation
- **Synthetic-to-real gap (thesis Table 7: 2.8-5.8 %).** The test split is synthetic, so real-photo accuracy will probably be lower by 3-6 points. Mitigation: keep a small real-photo check set and re-measure before release; stricter synthetic threshold (>= 0.91 for a real target of 0.85).
- **Glare weakness** (thesis F1 = 0.92; hardest defect in our failure analysis: {hardest}). Mitigation: stronger glare augmentation (varied position, size, intensity, laminate-style reflections), add real glare photos, hard-negative mining from the failure gallery, threshold tuning for the "bad" class so that uncertain glare photos are rejected rather than accepted.
- **Operator verification loop:** photos that are borderline (probability close to 0.5) go to a human operator instead of silently passing, and operator decisions are logged to grow the training set.

## 5. Deployment note
CPU server now: run the model with the framework used for training ({chosen.model}); latency above includes preprocessing. Mobile/edge later: export to TFLite (Keras models, with optional quantisation) or ONNX (YOLOv8 via `model.export(format="onnx")`, Keras via tf2onnx), then re-measure latency with the same protocol (20 warm-up + 100 timed) and re-check accuracy on the test split after export/quantisation.
"""
    (REPORTS / "week8_decision_memo.md").write_text(text, encoding="utf-8")
    shutil.copy(REPORTS / "week8_decision_memo.md", REPORTS / "decision_memo.md")
    json.dump({"chosen": chosen.experiment, "hardest_defect": hardest, "any_pass": bool(len(okdf)),
               "chosen_stats": {k: (float(chosen[k]) if k not in ("experiment",) else chosen[k]) for k in
                                ["accuracy", "f1_macro", "latency_ms_per_image", "model_size_mb"]}},
              open(REPORTS / "decision_summary.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/quality_split")
    ap.add_argument("--metadata", default=None)
    a = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)
    df = load_results()
    assert len(df) >= 3, f"Need >= 3 results.json (MobileNetV2, one YOLOv8, third model); found {len(df)}"
    print(table(df)); charts(df)
    defects, _ = load_defects(a.metadata)
    hardest = gallery(df, defects, a.data)
    memo(df, hardest)
    print("Reports written to", REPORTS)
