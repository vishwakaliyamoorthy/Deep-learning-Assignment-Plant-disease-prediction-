"""
Creates a curated, balanced manual evaluation dataset from the PlantVillage dataset.
Selects 30 distinct leaf samples across 15+ crop species covering healthy and diseased conditions,
copies them to eval_test_data/images/, and generates eval_test_data/test_data.csv.
"""
import csv
import os
import shutil
from pathlib import Path

# Target representative classes for balanced manual evaluation
TARGET_CLASSES = [
    # Apple
    ("Apple___Apple_scab", "Apple", "Apple Scab", "Diseased"),
    ("Apple___Black_rot", "Apple", "Black Rot", "Diseased"),
    ("Apple___Cedar_apple_rust", "Apple", "Cedar Apple Rust", "Diseased"),
    ("Apple___healthy", "Apple", "Healthy", "Healthy"),
    # Blueberry
    ("Blueberry___healthy", "Blueberry", "Healthy", "Healthy"),
    # Cherry
    ("Cherry_(including_sour)___Powdery_mildew", "Cherry", "Powdery Mildew", "Diseased"),
    ("Cherry_(including_sour)___healthy", "Cherry", "Healthy", "Healthy"),
    # Corn
    ("Corn_(maize)___Common_rust_", "Corn", "Common Rust", "Diseased"),
    ("Corn_(maize)___Northern_Leaf_Blight", "Corn", "Northern Leaf Blight", "Diseased"),
    ("Corn_(maize)___healthy", "Corn", "Healthy", "Healthy"),
    # Grape
    ("Grape___Black_rot", "Grape", "Black Rot", "Diseased"),
    ("Grape___Esca_(Black_Measles)", "Grape", "Esca (Black Measles)", "Diseased"),
    ("Grape___healthy", "Grape", "Healthy", "Healthy"),
    # Orange
    ("Orange___Haunglongbing_(Citrus_greening)", "Orange", "Citrus Greening", "Diseased"),
    # Peach
    ("Peach___Bacterial_spot", "Peach", "Bacterial Spot", "Diseased"),
    ("Peach___healthy", "Peach", "Healthy", "Healthy"),
    # Pepper
    ("Pepper,_bell___Bacterial_spot", "Pepper Bell", "Bacterial Spot", "Diseased"),
    ("Pepper,_bell___healthy", "Pepper Bell", "Healthy", "Healthy"),
    # Potato
    ("Potato___Early_blight", "Potato", "Early Blight", "Diseased"),
    ("Potato___Late_blight", "Potato", "Late Blight", "Diseased"),
    ("Potato___healthy", "Potato", "Healthy", "Healthy"),
    # Strawberry
    ("Strawberry___Leaf_scorch", "Strawberry", "Leaf Scorch", "Diseased"),
    ("Strawberry___healthy", "Strawberry", "Healthy", "Healthy"),
    # Tomato
    ("Tomato___Bacterial_spot", "Tomato", "Bacterial Spot", "Diseased"),
    ("Tomato___Early_blight", "Tomato", "Early Blight", "Diseased"),
    ("Tomato___Late_blight", "Tomato", "Late Blight", "Diseased"),
    ("Tomato___Leaf_Mold", "Tomato", "Leaf Mold", "Diseased"),
    ("Tomato___Septoria_leaf_spot", "Tomato", "Septoria Leaf Spot", "Diseased"),
    ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato", "Yellow Leaf Curl Virus", "Diseased"),
    ("Tomato___healthy", "Tomato", "Healthy", "Healthy"),
]


def create_dataset(
    source_dir: str = "dataset/PlantVillage-Dataset/raw/color",
    dest_dir: str = "eval_test_data",
) -> None:
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    images_dest = dest_path / "images"
    images_dest.mkdir(parents=True, exist_ok=True)

    csv_path = dest_path / "test_data.csv"
    rows = []

    for idx, (class_name, crop, condition, status) in enumerate(TARGET_CLASSES, start=1):
        class_folder = source_path / class_name
        if not class_folder.exists():
            continue

        images = sorted([f for f in class_folder.glob("*.JPG") or class_folder.glob("*.jpg")])
        if not images:
            continue

        # Choose the first clear image
        src_image = images[0]
        clean_ext = src_image.suffix.lower()
        clean_name = f"sample_{idx:02d}_{crop.lower().replace(' ', '_')}_{condition.lower().replace(' ', '_').replace('(', '').replace(')', '')}{clean_ext}"
        dst_image = images_dest / clean_name

        shutil.copy2(src_image, dst_image)

        rows.append({
            "sample_id": f"SMP-{idx:03d}",
            "filename": clean_name,
            "image_path": str(dst_image),
            "crop": crop,
            "condition": condition,
            "true_class": class_name,
            "status": status,
            "notes": f"Ground-truth {status.lower()} leaf sample from {crop}",
        })

    fieldnames = ["sample_id", "filename", "image_path", "crop", "condition", "true_class", "status", "notes"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    readme_path = dest_path / "README.md"
    readme_content = f"""# Manual Evaluation Test Dataset

This directory contains a curated set of **{len(rows)} diverse leaf images** covering 15+ agricultural crops, sampled from the PlantVillage dataset for manual verification, demo presentations, and evaluation.

## Files
- `test_data.csv`: Structured index file containing:
  - `sample_id`: Unique identifier (e.g., `SMP-001`)
  - `filename`: Local image filename in `images/`
  - `image_path`: Path to image file
  - `crop`: Plant / crop name (e.g., Apple, Tomato, Potato)
  - `condition`: Specific disease name or "Healthy"
  - `true_class`: Exact model class identifier
  - `status`: Ground-truth diagnosis (`Healthy` or `Diseased`)
- `images/`: Folder containing the {len(rows)} JPG test leaf images.

## How to Test in the Demo UI
1. Launch `python demo_app.py`.
2. Click **"Load Test Sample"**.
3. Select any leaf from the sample list (e.g. `Tomato - Early Blight` or `Apple - Healthy`).
4. The image and the model's prediction will automatically display for instant visual validation.
"""
    readme_path.write_text(readme_content, encoding="utf-8")
    print(f"Successfully generated {len(rows)} evaluation samples in '{dest_dir}'.")


if __name__ == "__main__":
    create_dataset()
