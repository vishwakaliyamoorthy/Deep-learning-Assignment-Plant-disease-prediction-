import os
import unittest
import tkinter as tk
from PIL import Image
import numpy as np

class TestDemoApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure display is available for Tkinter
        if not os.environ.get("DISPLAY"):
            raise unittest.SkipTest("No DISPLAY available for Tkinter GUI test")

    def test_app_initialization_and_prediction_display(self):
        from demo_app import PlantDiseaseApp
        root = tk.Tk()
        root.withdraw()  # keep hidden during headless unit test
        try:
            app = PlantDiseaseApp(root)
            self.assertIsNotNone(app.predictor)

            # Test processing a dummy image
            dummy_img = Image.fromarray(np.full((224, 224, 3), 120, dtype=np.uint8))
            app.process_image(dummy_img, source_title="Test Dummy")

            # Check that diagnosis title and status were populated
            diag = app.diagnosis_var.get()
            status = app.status_var.get()
            self.assertNotEqual(diag, "No image loaded")
            self.assertTrue(any(s in status for s in ["HEALTHY", "DISEASE", "UNCERTAIN"]))
        finally:
            root.destroy()

if __name__ == "__main__":
    unittest.main()
