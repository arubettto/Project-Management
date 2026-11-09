"""Re-measure latency on CPU (the deployment target) and update results.json in place.
Use it when you trained on a GPU but the budget (<100 ms) is for the CPU server.

  python scripts/remeasure_latency.py results/mobilenetv2_frozen results/week7_yolov8n_cls ...
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"           # hide GPU BEFORE importing TF / torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from PIL import Image
import qutils as Q


def main(dirs, data="data/quality_split"):
    for d in map(Path, dirs):
        res = json.load(open(d / "results.json"))
        items = Q.list_split(data, "test")
        imgs = Q.load_latency_images(items, res["seed"])
        size = res["input_size"]
        if res["model"].startswith("yolov8"):
            from ultralytics import YOLO
            m = YOLO(str(d / "weights" / "best.pt"))
            fn = lambda im: m.predict(im, imgsz=size, verbose=False, device="cpu")
        else:
            import tensorflow as tf
            from train_keras_quality import make_preprocess, MODELS, build_base
            mode = res.get("preprocessing_mode")
            mode = None if mode in (None, "builtin") else mode
            m = tf.keras.models.load_model(str(d / "best.keras"))
            pre = make_preprocess(mode)
            fn = lambda im: m.predict(pre(np.asarray(im.resize((size, size), Image.NEAREST), dtype=np.float32)[None]), verbose=0)
        res["latency_ms_per_image"] = round(Q.measure_latency(fn, imgs), 2)
        res["latency_device"] = "cpu"
        json.dump(res, open(d / "results.json", "w"), indent=2)
        print(d, "->", res["latency_ms_per_image"], "ms/image on CPU")


if __name__ == "__main__":
    main(sys.argv[1:])
