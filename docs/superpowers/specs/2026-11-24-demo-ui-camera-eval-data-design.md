# Design Specification: Plant Disease Demo UI, Camera Module & Manual Evaluation Dataset

**Date:** 2026-11-24  
**Status:** Approved  
**Author:** Antigravity & User  

---

## 1. Overview and Problem Statement
The Plant Disease Detection project currently provides a CLI script (`predict.py`) and trained deep learning models (`outputs/plant_disease_mobilenetv2.keras`). To demonstrate the model interactively and evaluate its real-world performance, two key capabilities are required:
1. **Interactive Demo UI with Camera Module**: A standalone desktop application where users can upload leaf images, capture live photos via a connected webcam, and view real-time disease classification results with confidence scores.
2. **Manual Evaluation Dataset**: A dedicated, curated dataset of diverse leaf images accompanied by an index file (`test_data.csv`) covering healthy and diseased conditions across multiple crop types for manual evaluation and demonstration.

---

## 2. Goals & Constraints
- **Zero-Friction Desktop Experience**: Clean, responsive Python Tkinter desktop GUI running on Linux/macOS/Windows with standard Python libraries.
- **Hardware-Integrated Camera**: Fast webcam capture with live video preview powered by `opencv-python`, with graceful fallback if no camera hardware is present.
- **Single Source of Truth for Inference**: Refactor inference code into a reusable `DiseasePredictor` module so `predict.py` CLI and `demo_app.py` share identical preprocessing and prediction logic.
- **Curated Test Suite**: Create an isolated `eval_test_data/` folder containing 30 representative test images across various crops (healthy and diseased) with a structured `test_data.csv` index.

---

## 3. Architecture & Module Design

```
+-----------------------------------------------------------+
|                        demo_app.py                        |
|                                                           |
|  +-----------------------------+  +--------------------+  |
|  |     Image Canvas Frame      |  |   Results Panel    |  |
|  |  (Displays active leaf)     |  | - Class & Status   |  |
|  +-----------------------------+  | - Confidence Bar   |  |
|                                   | - Top 3 Probabilities|
|  +-----------------------------+  +--------------------+  |
|  |       Action Toolbar        |                          |
|  | [Upload] [Camera] [Samples] |                          |
|  +-----------------------------+                          |
+-----------------------------------------------------------+
        |                  |                    |
        v                  v                    v
+---------------+  +---------------+  +---------------------+
|   camera.py   |  |  predictor.py |  |   eval_test_data/   |
| (OpenCV feed  |  | (Keras Mobile |  | (Curated images     |
|  & snapshot)  |  |   NetV2)      |  |  & test_data.csv)   |
+---------------+  +---------------+  +---------------------+
```

### 3.1 `camera.py` (Camera Module)
- **Class**: `CameraModule`
- **Responsibilities**:
  - Detect available video capture devices (`/dev/video0`, `/dev/video1`, etc.) via `cv2.VideoCapture`.
  - Capture single snapshot frames as PIL `Image` objects.
  - Provide continuous video frame generator or callback for live Tkinter canvas streaming.
  - Handle safe release of device handles on window close or error.
- **Methods**:
  - `is_available() -> bool`: Probes video devices and returns status.
  - `get_frame() -> Optional[PIL.Image]`: Grabs current frame from camera as RGB PIL Image.
  - `release()`: Closes hardware camera streams.

### 3.2 `predictor.py` (Reusable Inference Module)
- **Class**: `DiseasePredictor`
- **Responsibilities**:
  - Loads model (`outputs/plant_disease_mobilenetv2.keras`) and class names (`outputs/class_names.json`).
  - Implements image normalization and MobileNetV2 input formatting.
  - Predicts top-k classes and formats human-readable labels (e.g., `Tomato___Late_blight` -> `Tomato - Late blight`).
  - Classifies health status: `"Healthy"`, `"Diseased"`, or `"Uncertain"` if confidence is below threshold (default: 0.50).
- **Methods**:
  - `predict_image(image_input: Union[str, Path, PIL.Image]) -> PredictionResult`:
    Returns dictionary with:
    - `primary_label`: clean disease name string
    - `raw_label`: raw dataset class string
    - `confidence`: float between 0.0 and 1.0
    - `status`: `"Healthy"`, `"Diseased"`, or `"Uncertain"`
    - `top_k`: list of `(label, confidence)` tuples (default 3)

### 3.3 `demo_app.py` (Tkinter Desktop Application)
- **Class**: `PlantDiseaseApp(tk.Tk)`
- **Key UI Elements**:
  - **Left Section (Image & Actions)**:
    - Canvas (400x400) rendering the currently loaded image (auto-resizing maintaining aspect ratio).
    - Toolbar containing 4 distinct buttons:
      - `Upload Leaf Image`: Opens standard file picker (`.jpg`, `.jpeg`, `.png`).
      - `Open Camera`: Launches the live camera viewfinder modal.
      - `Load Test Sample`: Opens the sample selector modal to choose from curated manual evaluation data.
      - `Clear`: Resets the display.
  - **Right Section (Prediction & Diagnostics)**:
    - Disease Diagnosis Header: Formatted text displaying predicted condition.
    - Status Badge: High-contrast visual indicator:
      - Green tag for **Healthy**
      - Red tag for **Diseased**
      - Amber tag for **Uncertain** (<50% confidence)
    - Confidence Percentage & visual progress bar.
    - Top-3 Ranked Predictions with percentage probabilities.
  - **Camera Viewfinder Modal (`CameraDialog`)**:
    - Toplevel window with 640x480 live webcam stream.
    - `Capture Snapshot` button + `Spacebar` keybind.
    - `Cancel` button + `Escape` keybind.
    - Once snapped, the frame is immediately passed to the main canvas and evaluated.
  - **Sample Selector Modal (`SampleSelectorDialog`)**:
    - Dialog displaying table or dropdown of test images from `eval_test_data/test_data.csv`.
    - Shows Crop, Condition, and Health Status.
    - Selecting an entry automatically loads the image and runs inference.

### 3.4 `create_eval_data.py` & `eval_test_data/` (Manual Evaluation Suite)
- **`create_eval_data.py`**:
  - Samples 30 representative test leaf images from `dataset/PlantVillage-Dataset/raw/color/` across 15 crops:
    - Apple (Scab, Black rot, Rust, Healthy)
    - Corn (Cercospora, Common rust, Northern blight, Healthy)
    - Grape (Black rot, Esca, Leaf blight, Healthy)
    - Potato (Early blight, Late blight, Healthy)
    - Tomato (Bacterial spot, Early blight, Late blight, Septoria, Healthy)
    - Peach (Bacterial spot, Healthy)
    - Pepper (Bacterial spot, Healthy)
    - Strawberry (Leaf scorch, Healthy)
  - Copies images to `eval_test_data/images/<clean_filename>.jpg`.
  - Generates `eval_test_data/test_data.csv` with columns:
    `sample_id,filename,image_path,crop,condition,true_class,status,notes`.
  - Generates `eval_test_data/README.md` documenting the dataset and how to use it for manual grading/demo.

---

## 4. Dependencies
- Existing: `tensorflow>=2.16`, `numpy`, `pillow`.
- Added: `opencv-python` (for live video feed capture and camera device access).
- Python Standard Library: `tkinter`, `csv`, `json`, `pathlib`, `shutil`, `threading`.

---

## 5. Error Handling & Edge Cases
1. **No Camera Detected**: If no `/dev/video*` or OpenCV camera device opens, the `Open Camera` button shows an informative messagebox: *"No active camera detected. Please plug in a webcam or use image upload."* The app remains fully functional.
2. **Camera Resource In Use**: If the camera device is locked by another process, catch the exception, notify the user, and release handles cleanly.
3. **Invalid/Corrupt Image File**: File loader catches `UnidentifiedImageError` and alerts user without crashing the app.
4. **Missing Model File**: If `outputs/plant_disease_mobilenetv2.keras` is missing, display a prompt guiding the user to run `python train.py --quick` first.
5. **Non-blocking Execution**: Model prediction is executed efficiently; image rendering uses cached PhotoImages to avoid memory leaks.

---

## 6. Verification Plan
1. **Unit/Headless Tests**:
   - `test_predictor.py`: Verify `DiseasePredictor` initializes, preprocesses images, and outputs correct dictionary schema on sample images.
   - `test_camera.py`: Test device detection and error handling when cameras are present or simulated.
   - `test_eval_data.py`: Verify `create_eval_data.py` generates exactly 30 valid images and a well-formed CSV file with matching headers and file existence.
2. **Integration & Manual UI Verification**:
   - Run `python create_eval_data.py` and inspect `eval_test_data/test_data.csv`.
   - Launch `python demo_app.py`.
   - Test image upload via file dialog.
   - Test sample selection via test data picker.
   - Test camera capture modal and verify prediction displays accurately.
