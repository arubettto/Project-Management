import argparse, inspect, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
except Exception:  # Keras 3 without tf.keras legacy alias
    from keras.src.legacy.preprocessing.image import ImageDataGenerator

import qutils as Q

# name -> (constructor, input size, preprocessing mode required by the model family)
MODELS = {
    "mobilenetv2":      ("MobileNetV2",      224, "tf"),
    "efficientnet_b0":  ("EfficientNetB0",   224, "torch"),
    "resnet50":         ("ResNet50",         224, "caffe"),
    "mobilenetv3large": ("MobileNetV3Large", 224, "tf"),
    "inceptionv3":      ("InceptionV3",      299, "tf"),
}
# families whose Keras implementation normalises the input INSIDE the model by default
BUILTIN_PREPROC = {"efficientnet_b0", "mobilenetv3large"}


def build_base(name, weights):
    ctor_name, size, mode = MODELS[name]
    ctor = getattr(tf.keras.applications, ctor_name)
    kwargs = dict(weights=weights, include_top=False, input_shape=(size, size, 3))
    if name in BUILTIN_PREPROC:
        if "include_preprocessing" in inspect.signature(ctor).parameters:
            kwargs["include_preprocessing"] = False      # we apply the family mode ourselves
        else:
            mode = None                                  # model normalises internally: feed raw 0-255
            print(f"NOTE: this Keras version's {ctor_name} normalises inputs internally; "
                  f"feeding raw 0-255 pixels (equivalent to the '{MODELS[name][2]}' mode).")
    base = ctor(**kwargs)
    return base, size, mode


def make_preprocess(mode):
    if mode is None:
        return lambda x: x
    return lambda x: imagenet_utils.preprocess_input(x, mode=mode)


def build_model(base, regime):
    if regime == "frozen":
        base.trainable = False
        lr = 1e-3
    else:  # finetune: unfreeze everything except BatchNorm, low lr
        base.trainable = True
        for l in base.layers:
            if isinstance(l, layers.BatchNormalization):
                l.trainable = False
        lr = 1e-5
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--regime", default="frozen", choices=["frozen", "finetune"])
    ap.add_argument("--seed", type=int, required=True, help="YOUR STUDENT ID")
    ap.add_argument("--week", type=int, default=6)
    ap.add_argument("--data", default="data/quality_split")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' (default) or 'none' (smoke tests only)")
    ap.add_argument("--note", default="", help="one-line note stored in the log (use it for failed attempts)")
    args = ap.parse_args()
    exp = f"{args.model}_{args.regime}"
    Q.run_logged(exp, args.note, lambda: run(args, exp))


def run(args, exp):
    Q.set_seed(args.seed); tf.keras.utils.set_random_seed(args.seed % (2 ** 32))
    out = Path("results") / exp
    out.mkdir(parents=True, exist_ok=True)

    # 1. data check
    Q.print_counts(args.data)
    base, size, mode = build_base(args.model, None if args.weights == "none" else args.weights)
    pre = make_preprocess(mode)
    print(f"Model={args.model} input={size}x{size} preprocessing mode={mode} regime={args.regime} seed={args.seed}")

    # 2. generators (shuffle True only for train)
    def gen(split, shuffle):
        return ImageDataGenerator(preprocessing_function=pre).flow_from_directory(
            str(Q.split_dir(args.data, split)), target_size=(size, size), batch_size=args.batch,
            class_mode="binary", classes=Q.CLASSES, shuffle=shuffle, seed=args.seed)
    train_g, val_g, test_g = gen("train", True), gen("validation", False), gen("test", False)
    assert train_g.class_indices == {"bad": 0, "good": 1}, train_g.class_indices

    # 3-4. model + compile
    model, lr = build_model(base, args.regime)
    model.summary()
    ckpt = out / "best.keras"
    cbs = [EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
           ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True)]

    # 5. train
    t0 = time.perf_counter()
    h = model.fit(train_g, validation_data=val_g, epochs=args.epochs, callbacks=cbs, verbose=2)
    train_time = time.perf_counter() - t0
    history = {k: [float(x) for x in v] for k, v in h.history.items()}
    epochs_run = len(history["loss"])

    # 6. TEST evaluation only now
    test_g.reset()
    probs = model.predict(test_g, verbose=0).ravel()
    y_true = np.asarray(test_g.classes)
    assert len(probs) == len(y_true), "generator/prediction length mismatch"
    y_pred = (probs > 0.5).astype(int)                  # threshold fixed at 0.5
    metrics, cm = Q.compute_metrics(y_true, y_pred)
    print(metrics); print("confusion matrix [rows=true bad,good]:", cm)

    # 7. plots, 8. errors
    Q.plot_curves(history, out / "curves.png", f"{exp} (seed {args.seed})")
    Q.plot_confusion(cm, out / "confusion_matrix.png", f"{exp} - test")
    Q.copy_errors(test_g.filepaths, y_true, y_pred, out)

    # 9. latency (20 warm-up + 100 timed), size, params
    test_items = Q.list_split(args.data, "test")
    imgs = Q.load_latency_images(test_items, args.seed)

    def predict_one(img):
        x = np.asarray(img.resize((size, size), Image.NEAREST), dtype=np.float32)[None]
        return model.predict(pre(x), verbose=0)
    latency = Q.measure_latency(predict_one, imgs)
    gpus = tf.config.list_physical_devices("GPU")
    device = "gpu" if gpus else "cpu"

    # 10. results.json
    res = Q.base_result(exp, args.week, args.model, args.regime, size, args.seed, device, args.epochs)
    res.update({
        "epochs_run": epochs_run, "train_time_s": round(train_time, 1),
        "latency_ms_per_image": round(latency, 2), "latency_device": device,
        "model_size_mb": round(os.path.getsize(ckpt) / (1024 * 1024), 3),
        "params": int(model.count_params()), "test_samples": int(len(y_true)),
        "preprocessing_mode": mode or "builtin", "learning_rate": lr, "batch_size": args.batch,
        "metrics": metrics, "confusion_matrix": cm, "history": history, "note": args.note,
    })
    Q.save_results(out, res)


if __name__ == "__main__":
    main()
