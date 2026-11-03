"""
Predict the disease of ONE leaf image with the trained model (Assignment 1 output format).

Usage:  python predict.py path/to/leaf.jpg
        python predict.py path/to/leaf.jpg --model outputs/plant_disease_mobilenetv2.keras
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image
from tensorflow import keras


def pretty(name):
    return name.replace("___", " - ").replace("_", " ").replace(",", "").strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("image")
    p.add_argument("--model", default="outputs/plant_disease_mobilenetv2.keras")
    p.add_argument("--classes", default="outputs/class_names.json")
    p.add_argument("--threshold", type=float, default=0.50, help="below this confidence -> 'uncertain'")
    a = p.parse_args()

    model = keras.models.load_model(a.model)
    names = json.loads(Path(a.classes).read_text(encoding="utf-8"))
    size = model.input_shape[1]

    img = Image.open(a.image).convert("RGB").resize((size, size))
    x = np.expand_dims(np.asarray(img, dtype="float32"), 0)
    probs = model.predict(x, verbose=0)[0]
    top = probs.argsort()[::-1][:3]

    label, conf = names[top[0]], float(probs[top[0]])
    print(f"\nInput : {a.image}")
    if conf < a.threshold:
        print(f"Result: UNCERTAIN ({conf*100:.1f}% confidence) - please retake the photo.")
    else:
        status = "Healthy" if label.lower().endswith("healthy") else "Diseased"
        print(f"Result: {pretty(label)} ({conf*100:.1f}% confidence)  ->  {status}")
    print("Top 3 :", "; ".join(f"{pretty(names[i])} {probs[i]*100:.1f}%" for i in top))


if __name__ == "__main__":
    main()
