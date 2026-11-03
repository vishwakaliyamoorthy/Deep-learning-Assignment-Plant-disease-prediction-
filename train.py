"""
Leaf Image-Based Plant Disease Detection Using Deep Learning
Assignment 3 - Implementation (Deep Learning, U24ADB565)
Vishwa Kaliyamoorthy | TU6241103111168

Pipeline (as planned in Assignments 1 and 2)
  PlantVillage images -> resize 224x224 -> augmentation -> MobileNetV2 (ImageNet) + custom head
  Phase 1: train head only (base frozen)       lr 1e-3
  Phase 2: fine-tune last 30 base layers       lr 1e-5  (BatchNorm kept frozen)
  Evaluate on the untouched test split and save every result into the output folder.

Run:
  python train.py --quick          # small demo run, fine on a laptop CPU (~15-30 min)
  python train.py --max-per-class 300   # medium run
  python train.py                  # full run as per Assignment 2 (best on a GPU / Colab)
"""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

IMG_EXT = {".jpg", ".jpeg", ".png"}


# ----------------------------------------------------------------------------------------
# 1. Arguments
# ----------------------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Plant disease detection - MobileNetV2 transfer learning")
    p.add_argument("--data-dir", default="dataset/PlantVillage-Dataset/raw/color",
                   help="folder that contains one sub-folder per class")
    p.add_argument("--out-dir", default="outputs", help="where results are saved")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs-head", type=int, default=10, help="Phase 1 epochs")
    p.add_argument("--epochs-ft", type=int, default=15, help="Phase 2 epochs")
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-ft", type=float, default=1e-5)
    p.add_argument("--unfreeze-layers", type=int, default=30)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--max-per-class", type=int, default=0, help="0 = use all images of every class")
    p.add_argument("--quick", action="store_true",
                   help="demo mode: 100 images/class, 4 + 3 epochs (laptop friendly)")
    p.add_argument("--baseline", action="store_true",
                   help="also train the small custom CNN baseline for comparison")
    p.add_argument("--baseline-epochs", type=int, default=8)
    p.add_argument("--no-dedupe", action="store_true", help="skip exact-duplicate image removal")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--no-tflite", action="store_true")
    p.add_argument("--weights", default="imagenet", choices=["imagenet", "none"],
                   help="'none' = random init (only for offline code testing)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.quick:
        a.max_per_class = a.max_per_class or 100
        a.epochs_head, a.epochs_ft, a.baseline_epochs = 4, 3, 5
    return a


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ----------------------------------------------------------------------------------------
# 2. Dataset manifest (path, label, split)
# ----------------------------------------------------------------------------------------
def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_manifest(a):
    root = Path(a.data_dir)
    if not root.exists():
        raise SystemExit(
            f"\nDataset folder not found: {root}\n"
            "Run  python download_dataset.py  first (or see README.md for the manual download),\n"
            "or pass the folder with  --data-dir <path-to-folder-with-class-subfolders>\n")
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if len(class_dirs) < 2:
        raise SystemExit(f"Expected one sub-folder per class inside {root}")
    rng = random.Random(a.seed)
    rows = []
    for ci, d in enumerate(class_dirs):
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        if a.max_per_class and len(files) > a.max_per_class:
            files = rng.sample(files, a.max_per_class)
        rows += [(str(p), d.name, ci) for p in files]
    df = pd.DataFrame(rows, columns=["path", "class_name", "label"])

    if not a.no_dedupe:                                  # data-leakage guard (Assignment 2, Sec. 8)
        df["hash"] = [md5(p) for p in df.path]
        before = len(df)
        df = df.drop_duplicates("hash").drop(columns="hash").reset_index(drop=True)
        print(f"Removed {before - len(df)} exact duplicate images")

    # stratified 70 / 15 / 15 split
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df.label, random_state=a.seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df.label, random_state=a.seed)
    df.loc[train_df.index, "split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"
    class_names = [d.name for d in class_dirs]
    return df, class_names


def pretty(name):
    return name.replace("___", " - ").replace("_", " ").replace(",", "").strip()


# ----------------------------------------------------------------------------------------
# 3. tf.data input pipeline
# ----------------------------------------------------------------------------------------
def make_dataset(df, n_classes, a, training, class_weights=None):
    paths = df.path.values
    labels = df.label.values.astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(df), 4096), seed=a.seed, reshuffle_each_iteration=True)
    size = (a.img_size, a.img_size)
    cw = tf.constant(class_weights, tf.float32) if class_weights is not None else None

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, size)                 # float32, range 0..255
        onehot = tf.one_hot(label, n_classes)
        if cw is not None and training:
            return img, onehot, tf.gather(cw, label)
        return img, onehot

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(a.batch_size).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------------------------
# 4. Models
# ----------------------------------------------------------------------------------------
def build_mobilenet(n_classes, a):
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1, value_range=(0, 255)),
    ], name="augmentation")                              # active only while training

    shape = (a.img_size, a.img_size, 3)
    try:
        base = keras.applications.MobileNetV2(input_shape=shape, include_top=False, weights=None if a.weights == "none" else "imagenet")
    except Exception as e:                               # no internet for the weight download
        raise SystemExit(f"Could not download ImageNet weights ({e}).\n"
                         "Connect to the internet once so Keras can fetch MobileNetV2 weights.")
    base.trainable = False                               # Phase 1

    inputs = keras.Input(shape)
    x = augment(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)        # same as mobilenet_v2.preprocess_input -> [-1, 1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="plant_disease_mobilenetv2"), base


def build_baseline(n_classes, a):
    inputs = keras.Input((a.img_size, a.img_size, 3))
    x = layers.Rescaling(1 / 255.0)(inputs)
    for f in (32, 64, 128, 256):
        x = layers.Conv2D(f, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def compile_model(model, lr, a):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.CategoricalCrossentropy(label_smoothing=a.label_smoothing),
                  metrics=["accuracy"])


def callbacks(out, tag):
    cbs = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, verbose=1),
        keras.callbacks.ModelCheckpoint(str(out / f"best_{tag}.keras"), monitor="val_loss", save_best_only=True),
    ]
    try:                                                 # TensorBoard is optional
        import tensorboard  # noqa: F401
        cbs.append(keras.callbacks.TensorBoard(log_dir=str(out / "logs" / tag)))
    except ImportError:
        pass
    return cbs


def merge_history(histories):
    merged = {}
    for h in histories:
        for k, v in h.history.items():
            merged.setdefault(k, []).extend(v)
    return merged


# ----------------------------------------------------------------------------------------
# 5. Plots and evaluation
# ----------------------------------------------------------------------------------------
def plot_curves(hist, boundary, path, title="MobileNetV2"):
    ep = range(1, len(hist["loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, key, name in ((ax[0], "accuracy", "Accuracy"), (ax[1], "loss", "Loss")):
        axis.plot(ep, hist[key], label=f"Train {name.lower()}")
        axis.plot(ep, hist["val_" + key], label=f"Validation {name.lower()}")
        if boundary:
            axis.axvline(boundary + 0.5, color="gray", ls="--", label="Start of fine-tuning")
        axis.set_title(f"{title} - {name}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(name)
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(model, test_df, test_ds, class_names, out, tag=""):
    probs = model.predict(test_ds, verbose=0)
    y_true = test_df.label.values
    y_pred = probs.argmax(1)
    conf = probs.max(1)
    present = sorted(set(y_true) | set(y_pred))
    names = [pretty(class_names[i]) for i in range(len(class_names))]

    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    test_loss = float(model.evaluate(test_ds, verbose=0)[0])

    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                   target_names=names, digits=4, zero_division=0)
    (out / f"classification_report{tag}.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=names, columns=names).to_csv(out / f"confusion_matrix{tag}.csv")
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(14, 12) if n > 15 else (8, 7))
    sns.heatmap(cm_norm, cmap="Blues", ax=ax, vmin=0, vmax=1,
                annot=(n <= 15), fmt=".2f" if n <= 15 else "",
                xticklabels=range(n) if n > 15 else names,
                yticklabels=range(n) if n > 15 else names,
                cbar_kws={"label": "Fraction of true class"})
    ax.set_xlabel("Predicted class (index)")
    ax.set_ylabel("Actual class (index)")
    ax.set_title(f"Confusion matrix (row-normalised){' - ' + tag.strip('_') if tag else ''}")
    fig.tight_layout()
    fig.savefig(out / f"confusion_matrix{tag}.png", dpi=150)
    plt.close(fig)

    pva = test_df[["path"]].copy()
    pva["actual"] = [pretty(class_names[i]) for i in y_true]
    pva["predicted"] = [pretty(class_names[i]) for i in y_pred]
    pva["confidence_%"] = (conf * 100).round(2)
    pva["correct"] = y_true == y_pred
    pva.to_csv(out / f"predicted_vs_actual{tag}.csv", index=False)

    # most confused class pairs
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], i, j) for i in range(n) for j in range(n) if off[i, j] > 0), reverse=True)[:10]
    lines = [f"{c} x  actual '{names[i]}'  ->  predicted '{names[j]}'" for c, i, j in pairs]
    (out / f"top_confusions{tag}.txt").write_text("\n".join(lines) or "No misclassifications.", encoding="utf-8")

    return {"test_accuracy": acc, "test_loss": test_loss, "macro_f1": macro_f1,
            "weighted_f1": weighted_f1, "test_images": int(len(test_df)),
            "wrong_predictions": int((y_true != y_pred).sum())}, pva


def sample_predictions_grid(pva, path, k=12, seed=0):
    sample = pva.sample(min(k, len(pva)), random_state=seed)
    cols = 4
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, sample.iterrows()):
        ax.imshow(Image.open(r.path).convert("RGB"))
        ok = r.correct
        ax.set_title(f"Predicted: {r.predicted} ({r['confidence_%']:.0f}%)\nActual: {r.actual}",
                     fontsize=8, color="green" if ok else "red")
    fig.suptitle("Sample test predictions (green = correct, red = wrong)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def class_distribution_plot(df, class_names, path):
    counts = df.groupby("label").size().reindex(range(len(class_names)), fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(class_names)), counts.values, color="#2a7f62")
    ax.set_xlabel("Class index")
    ax.set_ylabel("Images")
    ax.set_title("Images per class")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------------------
def main():
    a = get_args()
    set_seed(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("TensorFlow", tf.__version__, "| GPUs:", [g.name for g in tf.config.list_physical_devices("GPU")] or "none (CPU)")
    df, class_names = build_manifest(a)
    n_classes = len(class_names)
    df.to_csv(out / "dataset_manifest.csv", index=False)
    (out / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (out / "class_names.txt").write_text(
        "\n".join(f"{i}: {pretty(c)}" for i, c in enumerate(class_names)), encoding="utf-8")
    class_distribution_plot(df, class_names, out / "class_distribution.png")

    train_df, val_df, test_df = (df[df.split == s] for s in ("train", "val", "test"))
    print(f"Classes: {n_classes} | images: {len(df)} | train {len(train_df)} / val {len(val_df)} / test {len(test_df)}")

    cw = None
    if not a.no_class_weights:
        w = compute_class_weight("balanced", classes=np.arange(n_classes), y=train_df.label.values)
        cw = w.astype("float32")
    train_ds = make_dataset(train_df, n_classes, a, True, cw)
    val_ds = make_dataset(val_df, n_classes, a, False)
    test_ds = make_dataset(test_df, n_classes, a, False)

    # ---------------- MobileNetV2: Phase 1 (feature extraction) ----------------
    model, base = build_mobilenet(n_classes, a)
    total_params = model.count_params()
    compile_model(model, a.lr_head, a)
    trainable_p1 = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print(f"\n=== Phase 1: training the new head (base frozen) | params {total_params:,} | trainable {trainable_p1:,}")
    t0 = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs_head,
                   callbacks=callbacks(out, "phase1"), verbose=2)

    # ---------------- Phase 2: fine-tuning ----------------
    base.trainable = True
    for layer in base.layers[:-a.unfreeze_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False                      # keep BN statistics frozen
    compile_model(model, a.lr_ft, a)                     # must re-compile after changing trainable
    trainable_p2 = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print(f"\n=== Phase 2: fine-tuning last {a.unfreeze_layers} base layers | trainable {trainable_p2:,}")
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs_ft,
                   callbacks=callbacks(out, "phase2"), verbose=2)
    train_minutes = (time.time() - t0) / 60

    hist = merge_history([h1, h2])
    boundary = len(h1.history["loss"])
    pd.DataFrame(hist).assign(epoch=lambda d: np.arange(1, len(d) + 1)).to_csv(out / "training_history.csv", index=False)
    plot_curves(hist, boundary, out / "accuracy_loss_curves.png")

    # ---------------- Evaluation on the test split ----------------
    print("\n=== Evaluating on the held-out test set")
    metrics, pva = evaluate(model, test_df, test_ds, class_names, out)
    sample_predictions_grid(pva, out / "sample_predictions.png")
    print(f"Test accuracy {metrics['test_accuracy']*100:.2f}% | macro-F1 {metrics['macro_f1']:.4f} | "
          f"loss {metrics['test_loss']:.4f}")

    model.save(out / "plant_disease_mobilenetv2.keras")

    # ---------------- Optional baseline CNN ----------------
    baseline_metrics = None
    if a.baseline:
        print("\n=== Baseline custom CNN (for comparison)")
        bl = build_baseline(n_classes, a)
        compile_model(bl, 1e-3, a)
        bh = bl.fit(train_ds, validation_data=val_ds, epochs=a.baseline_epochs,
                    callbacks=callbacks(out, "baseline"), verbose=2)
        plot_curves(bh.history, 0, out / "baseline_accuracy_loss_curves.png", "Baseline CNN")
        baseline_metrics, _ = evaluate(bl, test_df, test_ds, class_names, out, tag="_baseline")

    # ---------------- Optional TensorFlow Lite export ----------------
    tflite_kb = None
    if not a.no_tflite:
        try:
            conv = tf.lite.TFLiteConverter.from_keras_model(model)
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
            conv.target_spec.supported_types = [tf.float16]
            blob = conv.convert()
            (out / "plant_disease_mobilenetv2_fp16.tflite").write_bytes(blob)
            tflite_kb = round(len(blob) / 1024, 1)
        except Exception as e:
            print("TFLite export skipped:", e)

    # ---------------- Training summary ----------------
    summary = {
        "framework": f"TensorFlow {tf.__version__} / Keras {keras.__version__}",
        "model": "MobileNetV2 (ImageNet) + GAP + Dense(256, ReLU) + Dropout(0.4) + Dense(softmax)",
        "classes": n_classes,
        "images_total": int(len(df)),
        "split_train_val_test": [int(len(train_df)), int(len(val_df)), int(len(test_df))],
        "input_size": f"{a.img_size}x{a.img_size}",
        "batch_size": a.batch_size,
        "optimizer": "Adam",
        "loss": f"categorical cross-entropy, label smoothing {a.label_smoothing}",
        "class_weights_used": cw is not None,
        "phase1": {"lr": a.lr_head, "epochs_planned": a.epochs_head, "epochs_run": boundary,
                   "trainable_params": trainable_p1},
        "phase2": {"lr": a.lr_ft, "epochs_planned": a.epochs_ft, "epochs_run": len(hist["loss"]) - boundary,
                   "unfrozen_layers": a.unfreeze_layers, "trainable_params": trainable_p2},
        "total_params": int(total_params),
        "training_time_minutes": round(train_minutes, 1),
        "best_val_accuracy": round(float(max(hist["val_accuracy"])), 4),
        "final_train_accuracy": round(float(hist["accuracy"][-1]), 4),
        "test_results": metrics,
        "baseline_cnn_test_results": baseline_metrics,
        "tflite_fp16_size_kb": tflite_kb,
        "seed": a.seed,
    }
    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    txt = [
        "MODEL TRAINING SUMMARY", "=" * 40,
        f"Framework        : {summary['framework']}",
        f"Model            : {summary['model']}",
        f"Classes / images : {n_classes} / {len(df)}  (train {len(train_df)}, val {len(val_df)}, test {len(test_df)})",
        f"Input size       : {summary['input_size']}",
        f"Batch size       : {a.batch_size}",
        f"Optimizer / loss : Adam / {summary['loss']}",
        f"Phase 1          : {boundary} epochs run (max {a.epochs_head}), lr {a.lr_head}, head only",
        f"Phase 2          : {len(hist['loss']) - boundary} epochs run (max {a.epochs_ft}), lr {a.lr_ft}, last {a.unfreeze_layers} layers",
        f"Total parameters : {total_params:,}",
        f"Training time    : {train_minutes:.1f} minutes",
        f"Best val acc     : {summary['best_val_accuracy']*100:.2f}%",
        f"TEST accuracy    : {metrics['test_accuracy']*100:.2f}%",
        f"TEST macro-F1    : {metrics['macro_f1']:.4f}",
        f"TEST loss        : {metrics['test_loss']:.4f}",
    ]
    if baseline_metrics:
        txt.append(f"Baseline CNN test accuracy : {baseline_metrics['test_accuracy']*100:.2f}%")
    (out / "training_summary.txt").write_text("\n".join(txt), encoding="utf-8")
    print("\n" + "\n".join(txt))
    print(f"\nAll results saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
