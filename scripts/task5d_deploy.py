import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")    
import csv
import json
import shutil

import numpy as np
import tensorflow as tf

from common import *

OUT = os.path.join(RESULTS_ROOT, "task5d_deploy")
THREADS = int(os.environ.get("THREADS", 1))
DROP_LIMIT_PP = 1.0                     
TARGET_STAGE_MS = 52.0                   
TARGET_YOLO_MS, TARGET_YOLO_MB = 2.1, 2.8  


def representative_data(n=100):
    ds, _, _ = load_split("train", shuffle=True)
    got = 0
    for x, _ in ds:
        for i in range(x.shape[0]):
            yield [x[i:i + 1].numpy().astype(np.float32)]
            got += 1
            if got >= n:
                return


def convert(saved_dir, variant):
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    if variant == "float32":
        pass
    elif variant == "float16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif variant == "int8_dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif variant == "int8_full":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_data      
    return conv.convert()


class TFLiteModel:
    def __init__(self, path):
        try:                                                   
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            Interpreter = tf.lite.Interpreter
        self.it = Interpreter(model_path=path, num_threads=THREADS)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]

    def __call__(self, x):
        self.it.set_tensor(self.inp, x.astype(np.float32))
        self.it.invoke()
        return self.it.get_tensor(self.out)


def evaluate_tflite(tfl, ds, names):
    ys, ps = [], []
    for x, y in ds:
        x = x.numpy()
        ps.append(np.concatenate([tfl(x[i:i + 1]) for i in range(len(x))]))
        ys.append(y.numpy())
    y_true = np.concatenate(ys).ravel().astype(int)
    return compute_metrics(y_true, to_pred(np.concatenate(ps)), names)


def main():
    set_seed(); ensure(OUT); ensure(REPORTS_DIR)
    test_ds, test_files, names = load_split("test")
    sample = get_sample("val")

    model = tf.keras.models.load_model(BEST_CKPT, compile=False)
    m_k, *_ = full_eval(model, test_ds, test_files, names, os.path.join(OUT, "keras_reference"), "keras (test)")
    size_k = clean_size_mb(BEST_CKPT, os.path.join(OUT, "_tmp"))
    lat_k = keras_latency(model, sample)

    saved = os.path.join(OUT, "saved_model")
    shutil.rmtree(saved, ignore_errors=True)
    model.export(saved) if hasattr(model, "export") else tf.saved_model.save(model, saved)

    results = [{"variant": "keras (.keras)", "size_mb": size_k, "latency": lat_k, "metrics": m_k}]
    for variant in ["float32", "float16", "int8_dynamic", "int8_full"]:
        try:
            blob = convert(saved, variant)
        except Exception as e:                                   
            print(f"[skip] {variant}: {e}")
            continue
        path = os.path.join(OUT, f"model_{variant}.tflite")
        open(path, "wb").write(blob)
        tfl = TFLiteModel(path)
        m = evaluate_tflite(tfl, test_ds, names)
        lat = benchmark(tfl, sample)
        results.append({"variant": f"tflite {variant}", "size_mb": len(blob) / 1e6, "latency": lat, "metrics": m})
        print(variant, "acc", m["accuracy"], "lat", lat["mean_ms"])

    shutil.rmtree(saved, ignore_errors=True)                   
    shutil.rmtree(os.path.join(OUT, "_tmp"), ignore_errors=True)

    # ---- tables ----
    acc0, f10 = m_k["accuracy"], m_k["f1_macro"]
    trows = []
    for r in results:
        drop = (acc0 - r["metrics"]["accuracy"]) * 100
        r["acc_drop_pp"] = drop
        r["ok_drop"] = drop <= DROP_LIMIT_PP
        trows.append([r["variant"], fmt(r["size_mb"], 2), fmt(r["latency"]["mean_ms"], 2), fmt(r["latency"]["median_ms"], 2),
                      fmt(r["latency"]["p95_ms"], 2), fmt(r["metrics"]["accuracy"]), fmt(r["metrics"]["f1_macro"]),
                      f"{drop:+.2f}", "yes" if r["ok_drop"] else "NO"])
    table = md_table(["variant", "size MB", "lat mean ms", "lat median", "lat p95", "test acc", "macro F1", "acc drop (pp)", f"drop <= {DROP_LIMIT_PP:g} pp"], trows)
    open(os.path.join(OUT, "benchmark.md"), "w").write(table)
    json.dump(results, open(os.path.join(OUT, "benchmark.json"), "w"), indent=2)
    with open(os.path.join(OUT, "benchmark.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["variant", "size_mb", "lat_mean_ms", "lat_median_ms", "lat_p95_ms", "acc", "f1_macro", "acc_drop_pp"])
        for r in results:
            w.writerow([r["variant"], r["size_mb"], r["latency"]["mean_ms"], r["latency"]["median_ms"], r["latency"]["p95_ms"],
                        r["metrics"]["accuracy"], r["metrics"]["f1_macro"], r["acc_drop_pp"]])

    tfl_rows = [r for r in results if r["variant"].startswith("tflite")]
    best = min([r for r in tfl_rows if r["ok_drop"]] or tfl_rows, key=lambda r: r["latency"]["mean_ms"]) if tfl_rows else None
    update_row(BEST_REGIME, {"deploy_benchmark": {
        "cpu_threads": THREADS, "warmup": 20, "runs": 100, "batch": 1,
        "variants": [{"variant": r["variant"], "size_mb": r["size_mb"], "latency_mean_ms": r["latency"]["mean_ms"],
                      "accuracy": r["metrics"]["accuracy"], "acc_drop_pp": r["acc_drop_pp"]} for r in results]}})

    share = (best["latency"]["mean_ms"] / TARGET_STAGE_MS * 100) if best else None
    memo = f"""# Bonus 5D - Deployment memo (export + edge benchmark)

Model: `{BEST_CKPT}` -> TFLite. Protocol: CPU, {THREADS} thread(s), batch 1, 20 warm-up + 100 timed runs, test split for accuracy.
Quantization calibration data: 100 images from the **train** split.

## Benchmark
{table}

## Comparison with capstone targets
- Quality stage budget (end-to-end): {TARGET_STAGE_MS:g} ms. Best acceptable variant (`{best['variant'] if best else 'n/a'}`): {fmt(best['latency']['mean_ms'],2) if best else 'n/a'} ms = {fmt(share,1) if share is not None else 'n/a'} % of the budget (model only; add pre/post-processing).
- YOLOv8 reference: {TARGET_YOLO_MS:g} ms / {TARGET_YOLO_MB:g} MB vs this model: {fmt(best['latency']['mean_ms'],2) if best else 'n/a'} ms / {fmt(best['size_mb'],2) if best else 'n/a'} MB.
- Accuracy drop vs Keras: see column above; acceptable limit <= {DROP_LIMIT_PP:g} pp. <<state explicitly which variants pass, "0.00" counts as "zero drop">>

## Recommendation
**Server vs edge:** <<decide with the numbers: does per-image latency on the target hardware fit the stage budget? Remember this laptop/Colab CPU is NOT the target device - say how you would re-measure there>>
**Quantization risks:** <<int8 rounds weights/activations, which shifts decision boundaries near the threshold (especially for hard classes such as glare); calibration data must resemble production; verify per-class recall, not only accuracy; re-check the chosen_threshold from 5B after quantization>>

## Explanation checkpoints
- Single number that decides edge feasibility: per-image latency on the *target* hardware vs the stage budget.
- Why INT8 can lose accuracy: weights and activations are rounded to 256 levels, so logits shift slightly and samples near the decision boundary can flip.
- Why warm-up: first runs include memory allocation, lazy init and cold caches; timing them overstates latency.
"""
    open(os.path.join(REPORTS_DIR, "bonus_deployment_memo.md"), "w").write(memo)
    print(table)
    print("\nDraft memo written to reports/bonus_deployment_memo.md - finish the <<...>> parts by hand.")


if __name__ == "__main__":
    main()
