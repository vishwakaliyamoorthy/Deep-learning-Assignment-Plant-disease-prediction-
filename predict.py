"""
Predict the disease of ONE leaf image with the trained model (Assignment output format).
Uses the shared DiseasePredictor service.

Usage:  python predict.py path/to/leaf.jpg
        python predict.py path/to/leaf.jpg --model outputs/plant_disease_mobilenetv2.keras
"""
import argparse
from predictor import DiseasePredictor


def main():
    p = argparse.ArgumentParser(description="Predict plant disease from a leaf image.")
    p.add_argument("image", help="Path to leaf image")
    p.add_argument("--model", default="outputs/plant_disease_mobilenetv2.keras")
    p.add_argument("--classes", default="outputs/class_names.json")
    p.add_argument("--threshold", type=float, default=0.50, help="below this confidence -> 'uncertain'")
    a = p.parse_args()

    predictor = DiseasePredictor(model_path=a.model, classes_path=a.classes, threshold=a.threshold)
    res = predictor.predict(a.image)

    print(f"\nInput : {a.image}")
    if res["status"] == "Uncertain":
        print(f"Result: UNCERTAIN ({res['confidence']*100:.1f}% confidence) - please retake the photo.")
    else:
        print(f"Result: {res['primary_label']} ({res['confidence']*100:.1f}% confidence)  ->  {res['status']}")

    top_str = "; ".join(f"{label} {prob*100:.1f}%" for label, prob in res["top_k"])
    print("Top 3 :", top_str)


if __name__ == "__main__":
    main()
