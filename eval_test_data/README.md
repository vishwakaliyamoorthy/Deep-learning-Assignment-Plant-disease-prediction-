# Manual Evaluation Test Dataset

This directory contains a curated set of **30 diverse leaf images** covering 15+ agricultural crops, sampled from the PlantVillage dataset for manual verification, demo presentations, and evaluation.

## Files
- `test_data.csv`: Structured index file containing:
  - `sample_id`: Unique identifier (e.g., `SMP-001`)
  - `filename`: Local image filename in `images/`
  - `image_path`: Path to image file
  - `crop`: Plant / crop name (e.g., Apple, Tomato, Potato)
  - `condition`: Specific disease name or "Healthy"
  - `true_class`: Exact model class identifier
  - `status`: Ground-truth diagnosis (`Healthy` or `Diseased`)
- `images/`: Folder containing the 30 JPG test leaf images.

## How to Test in the Demo UI
1. Launch `python demo_app.py`.
2. Click **"Load Test Sample"**.
3. Select any leaf from the sample list (e.g. `Tomato - Early Blight` or `Apple - Healthy`).
4. The image and the model's prediction will automatically display for instant visual validation.
