import unittest
from pathlib import Path
from PIL import Image
import numpy as np

class TestPredictor(unittest.TestCase):
    def setUp(self):
        from predictor import DiseasePredictor, format_label
        self.format_label = format_label
        self.predictor = DiseasePredictor(
            model_path="outputs/plant_disease_mobilenetv2.keras",
            classes_path="outputs/class_names.json",
            threshold=0.50
        )

    def test_format_label(self):
        self.assertEqual(self.format_label("Apple___Apple_scab"), "Apple - Apple scab")
        self.assertEqual(self.format_label("Tomato___Late_blight"), "Tomato - Late blight")
        self.assertEqual(self.format_label("Cherry_(including_sour)___healthy"), "Cherry (including sour) - healthy")

    def test_predict_real_image(self):
        # Test using an actual image from the dataset
        sample_path = "dataset/PlantVillage-Dataset/raw/color/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG"
        res = self.predictor.predict(sample_path)
        self.assertIn("primary_label", res)
        self.assertIn("raw_label", res)
        self.assertIn("confidence", res)
        self.assertIn("status", res)
        self.assertIn("top_k", res)
        self.assertIsInstance(res["top_k"], list)
        self.assertEqual(len(res["top_k"]), 3)
        self.assertIn(res["status"], ["Healthy", "Diseased", "Uncertain"])

    def test_predict_pil_image(self):
        # Test passing PIL image object directly
        dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        res = self.predictor.predict(dummy)
        self.assertIn("primary_label", res)
        self.assertIn("status", res)

if __name__ == "__main__":
    unittest.main()
