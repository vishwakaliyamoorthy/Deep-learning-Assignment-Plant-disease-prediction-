"""
Reusable inference service for plant disease classification.
Wraps the trained MobileNetV2 model and class mapping.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
from PIL import Image
from tensorflow import keras


def format_label(name: str) -> str:
    """Clean raw dataset class names like Apple___Apple_scab -> Apple - Apple scab."""
    return name.replace("___", " - ").replace("_", " ").replace(",", "").strip()


class DiseasePredictor:
    def __init__(
        self,
        model_path: Union[str, Path] = "outputs/plant_disease_mobilenetv2.keras",
        classes_path: Union[str, Path] = "outputs/class_names.json",
        threshold: float = 0.50,
    ):
        self.model_path = Path(model_path)
        self.classes_path = Path(classes_path)
        self.threshold = threshold

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at '{self.model_path}'.")
        if not self.classes_path.exists():
            raise FileNotFoundError(f"Class names file not found at '{self.classes_path}'.")

        self.model = keras.models.load_model(str(self.model_path))
        self.class_names: List[str] = json.loads(self.classes_path.read_text(encoding="utf-8"))
        self.input_size: int = self.model.input_shape[1]

    def predict(self, image_input: Union[str, Path, Image.Image]) -> Dict[str, Any]:
        """
        Runs inference on an image file path or PIL Image.
        Returns prediction details including primary diagnosis, status, and top-3 classes.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        resized = img.resize((self.input_size, self.input_size))
        x = np.expand_dims(np.asarray(resized, dtype="float32"), axis=0)
        probs = self.model.predict(x, verbose=0)[0]

        top_indices = probs.argsort()[::-1][:3]
        top_k: List[Tuple[str, float]] = [
            (format_label(self.class_names[i]), float(probs[i]))
            for i in top_indices
        ]

        primary_raw = self.class_names[top_indices[0]]
        primary_label = format_label(primary_raw)
        confidence = float(probs[top_indices[0]])

        if confidence < self.threshold:
            status = "Uncertain"
        elif primary_raw.lower().endswith("healthy"):
            status = "Healthy"
        else:
            status = "Diseased"

        return {
            "primary_label": primary_label,
            "raw_label": primary_raw,
            "confidence": confidence,
            "status": status,
            "top_k": top_k,
        }
