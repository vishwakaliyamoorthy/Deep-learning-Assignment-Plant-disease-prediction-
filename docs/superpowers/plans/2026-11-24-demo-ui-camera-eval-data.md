# Plant Disease Demo UI, Camera Module & Manual Evaluation Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Tkinter desktop demo UI with live webcam capture and image upload for plant disease prediction, alongside a curated manual evaluation dataset and CSV index.

**Architecture:** A reusable inference engine (`predictor.py`) loaded once by both the CLI (`predict.py`) and a desktop UI (`demo_app.py`). Live camera acquisition is handled by a modular camera wrapper (`camera.py`). A generator script (`create_eval_data.py`) curates 30 representative test images into `eval_test_data/` with a descriptive `test_data.csv` index.

**Tech Stack:** Python 3.12, TensorFlow / Keras 2.21, Pillow, OpenCV (`opencv-python`), Tkinter, CSV, JSON.

**Spec:** [docs/superpowers/specs/2026-11-24-demo-ui-camera-eval-data-design.md](file:///home/vishwa/Desktop/Vishwa%20Projects/plant_disease_project/docs/superpowers/specs/2026-11-24-demo-ui-camera-eval-data-design.md)

## Global Constraints
- Target python environment: `.venv-tf/bin/python`.
- Minimal third-party dependencies: standard library wherever possible, `opencv-python` for camera feed.
- Model path: `outputs/plant_disease_mobilenetv2.keras`, classes path: `outputs/class_names.json`.
- Zero remote push until the user reviews and confirms the working demo.

## Review Focus
1. **Camera unavailable or locked**: The camera dialog must display a clean user alert and not crash the desktop GUI.
2. **Corrupted or non-image file uploaded**: Must be trapped gracefully without crashing the UI.
3. **Low-confidence prediction (<50%)**: Result badge must clearly indicate `Uncertain` with an alert to retake the photo.
4. **Missing model weights file**: Must show an actionable error message rather than a raw Python stack trace.
5. **Image aspect ratio handling**: Images of non-square resolutions must resize into the UI preview canvas without stretching or distortion.

---

### Task 1: Environment & Dependency Setup

**Files:**
- Modify: `requirements.txt`
- Verification: `.venv-tf/bin/python -c "import cv2; print(cv2.__version__)"`

- [ ] **Step 1: Update requirements.txt to include opencv-python**
- [ ] **Step 2: Install opencv-python into .venv-tf via uv**
  Run: `uv pip install opencv-python --python .venv-tf/bin/python`
- [ ] **Step 3: Verify import and camera access**
  Run: `.venv-tf/bin/python -c "import cv2; print('OpenCV version:', cv2.__version__)"`
- [ ] **Step 4: Commit**
  Run: `git add requirements.txt && git commit -m "chore: add opencv-python dependency"`

---

### Task 2: Reusable Inference Service (`predictor.py`) & CLI Refactoring

**Files:**
- Create: `predictor.py`
- Modify: `predict.py`
- Test: `tests/test_predictor.py`

**Interfaces:**
- Produces: `DiseasePredictor(model_path, classes_path, threshold=0.50)`
  - `predict(image_input: Union[str, Path, Image.Image]) -> dict`
    Returning: `{"primary_label": str, "raw_label": str, "confidence": float, "status": str, "top_k": list[tuple[str, float]]}`

- [ ] **Step 1: Write unit test for DiseasePredictor**
  Create `tests/test_predictor.py` asserting inference output structure, healthy vs diseased classification, and thresholding.
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv-tf/bin/python -m unittest tests/test_predictor.py`
  Expected: FAIL (No module named 'predictor')
- [ ] **Step 3: Implement DiseasePredictor in `predictor.py`**
- [ ] **Step 4: Refactor `predict.py` to use `predictor.py`**
- [ ] **Step 5: Run tests to verify they pass**
  Run: `.venv-tf/bin/python -m unittest tests/test_predictor.py`
  Run: `.venv-tf/bin/python predict.py "dataset/PlantVillage-Dataset/raw/color/Apple___Apple_scab/0a5e9323-dbad-432d-ac58-d291718345d9___FREC_Scab 3417.JPG"`
- [ ] **Step 6: Commit**
  Run: `git add predictor.py predict.py tests/test_predictor.py && git commit -m "refactor: extract reusable DiseasePredictor service"`

---

### Task 3: Camera Module (`camera.py`)

**Files:**
- Create: `camera.py`
- Test: `tests/test_camera.py`

**Interfaces:**
- Produces: `CameraModule(device_index=0)`
  - `is_available() -> bool`
  - `get_frame() -> Optional[Image.Image]`
  - `release()`

- [ ] **Step 1: Write unit test for CameraModule in `tests/test_camera.py`**
  Tests camera initialization, frame capture or safe fallback handling when device is absent.
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv-tf/bin/python -m unittest tests/test_camera.py`
  Expected: FAIL (No module named 'camera')
- [ ] **Step 3: Implement `camera.py`**
- [ ] **Step 4: Run test to verify it passes**
  Run: `.venv-tf/bin/python -m unittest tests/test_camera.py`
- [ ] **Step 5: Commit**
  Run: `git add camera.py tests/test_camera.py && git commit -m "feat: add CameraModule for webcam capture"`

---

### Task 4: Manual Evaluation Dataset & Generator (`create_eval_data.py`)

**Files:**
- Create: `create_eval_data.py`
- Create: `eval_test_data/test_data.csv`
- Create: `eval_test_data/README.md`
- Test: `tests/test_eval_data.py`

- [ ] **Step 1: Write test for evaluation dataset generator in `tests/test_eval_data.py`**
  Validates that `eval_test_data/test_data.csv` has 30 rows, all image files exist on disk, and headers match schema.
- [ ] **Step 2: Implement `create_eval_data.py`**
  Selects 30 diverse images (healthy and diseased across 15 plant species) and copies them to `eval_test_data/images/`.
- [ ] **Step 3: Execute `create_eval_data.py`**
  Run: `.venv-tf/bin/python create_eval_data.py`
- [ ] **Step 4: Run test to verify dataset integrity**
  Run: `.venv-tf/bin/python -m unittest tests/test_eval_data.py`
- [ ] **Step 5: Commit**
  Run: `git add create_eval_data.py eval_test_data/ tests/test_eval_data.py && git commit -m "feat: curate manual evaluation dataset and CSV manifest"`

---

### Task 5: Desktop Demo Application (`demo_app.py`)

**Files:**
- Create: `demo_app.py`
- Test: `tests/test_demo_app.py` (headless smoke test of UI components and controllers)

**Features:**
- Main Window: Two-column layout (Left: Image Canvas + Controls; Right: Diagnosis Header, Status Badge, Confidence Bar, Top 3 List).
- Buttons: `Upload Leaf Image`, `Open Camera`, `Load Test Sample`, `Clear`.
- Live Camera Viewfinder Dialog with real-time stream and capture button.
- Test Sample Selector Dialog to pick any of the 30 curated test leaves.

- [ ] **Step 1: Write headless test for `demo_app.py`**
- [ ] **Step 2: Implement `demo_app.py`**
- [ ] **Step 3: Run headless test to verify UI initialization and controller flows**
  Run: `.venv-tf/bin/python -m unittest tests/test_demo_app.py`
- [ ] **Step 4: Commit**
  Run: `git add demo_app.py tests/test_demo_app.py && git commit -m "feat: add Tkinter desktop demo application"`

---

### Task 6: End-to-End Verification and Demonstration

- [ ] **Step 1: Run complete test suite**
  Run: `.venv-tf/bin/python -m unittest discover -s tests -v`
- [ ] **Step 2: Test live inference on manual evaluation samples**
- [ ] **Step 3: Launch demo app and verify live interactive performance**
- [ ] **Step 4: Update README.md with instructions to launch demo and run manual evaluation**
- [ ] **Step 5: Commit changes**
