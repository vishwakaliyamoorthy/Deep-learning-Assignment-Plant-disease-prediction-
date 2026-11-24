import csv
import unittest
from pathlib import Path

class TestEvalData(unittest.TestCase):
    def test_eval_dataset_integrity(self):
        csv_path = Path("eval_test_data/test_data.csv")
        self.assertTrue(csv_path.exists(), f"Missing {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected_fields = {"sample_id", "filename", "image_path", "crop", "condition", "true_class", "status"}
            self.assertTrue(expected_fields.issubset(set(reader.fieldnames or [])))
            rows = list(reader)

        self.assertGreaterEqual(len(rows), 25, "Expected at least 25 evaluation samples")

        # Verify each image file exists and is valid
        for row in rows:
            img_path = Path(row["image_path"])
            self.assertTrue(img_path.exists(), f"Image path '{img_path}' does not exist on disk.")
            self.assertIn(row["status"], ["Healthy", "Diseased"])

if __name__ == "__main__":
    unittest.main()
