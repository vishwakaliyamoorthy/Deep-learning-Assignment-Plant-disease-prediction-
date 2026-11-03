# Leaf Image-Based Plant Disease Detection (Assignment 3)

TensorFlow/Keras, MobileNetV2 transfer learning, PlantVillage dataset (38 classes, ~54,300 colour images).

## 1. Set up (once)
Python 3.10 - 3.12 is recommended (TensorFlow does not support very new Python versions).

```
python -m venv venv
venv\Scripts\activate          # Windows      (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## 2. Get the dataset (once, ~1 GB)

**Option A - automatic (needs git):**
```
python download_dataset.py
```
This puts the images in `dataset/PlantVillage-Dataset/raw/color/<class_name>/`.

**Option B - manual (no git):** download "PlantVillage Dataset" from Kaggle
(https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset), unzip it, and run training with
`--data-dir "<path to the folder named color>"`. The folder must contain one sub-folder per class.

## 3. Train and evaluate
```
python train.py --quick                 # demo run: 100 images/class, 4+3 epochs - good for a laptop
python train.py --quick --baseline      # same, plus the small custom CNN for comparison
python train.py --max-per-class 300     # medium run (better accuracy, longer)
python train.py                         # FULL run (10 + 15 epochs, all images) - use a GPU / Colab
```
The first run downloads the MobileNetV2 ImageNet weights automatically, so you need internet once.

Everything is written to the `outputs/` folder:

| File | Use in the assignment |
|---|---|
| `training_summary.txt / .json` | Model training summary (epochs, batch size, learning rates, parameters, time) |
| `accuracy_loss_curves.png` | Accuracy / loss curves |
| `confusion_matrix.png / .csv` | Confusion matrix |
| `classification_report.txt` | Precision, recall, F1 for every class |
| `sample_predictions.png` | Sample predictions (green = correct, red = wrong) |
| `predicted_vs_actual.csv` | Predicted vs actual output for every test image |
| `top_confusions.txt` | Most confused class pairs (for the reflection) |
| `plant_disease_mobilenetv2.keras` | Trained model |
| `plant_disease_mobilenetv2_fp16.tflite` | Mobile (TensorFlow Lite) version |

## 4. Predict one image
```
python predict.py path\to\leaf.jpg
```
Example output: `Result: Tomato - Early blight (94.1% confidence)  ->  Diseased`

## Troubleshooting
* **"Dataset folder not found"** - run step 2, or pass `--data-dir`.
* **Out of memory** - add `--batch-size 16`.
* **Too slow** - use `--quick`, or run the same files on Google Colab / Kaggle with a free GPU.
# Deep-learning-Assignment-Plant-disease-prediction-
