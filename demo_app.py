"""
Plant Leaf Disease Detection - Desktop Demo UI
Built with Tkinter, OpenCV, and MobileNetV2.
Supports:
1. Uploading leaf images from disk
2. Live webcam feed capture and snapshot
3. Loading curated test samples from eval_test_data/
"""
import csv
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

from PIL import Image, ImageTk

from camera import CameraModule
from predictor import DiseasePredictor


class CameraDialog(tk.Toplevel):
    """Modal live webcam dialog to frame leaf and capture photo."""
    def __init__(self, parent, on_capture_callback):
        super().__init__(parent)
        self.title("📷 Live Camera Viewfinder")
        self.transient(parent)
        self.resizable(False, False)
        self.on_capture = on_capture_callback
        self.current_frame: Optional[Image.Image] = None
        self._photo_image = None
        self._is_running = True

        self.cam = CameraModule(device_index=0)
        if not self.cam.is_available():
            messagebox.showwarning(
                "Camera Unavailable",
                "Could not connect to a webcam device (/dev/video0).\n"
                "Please verify camera permissions or use 'Upload Image' instead.",
                parent=self
            )
            self.destroy()
            return

        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", lambda e: self.close())
        self.bind("<space>", lambda e: self.capture())

        # Viewfinder Canvas (640x480)
        self.canvas = tk.Canvas(self, width=640, height=480, bg="#111111", highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)

        # Toolbar
        btn_bar = ttk.Frame(self)
        btn_bar.pack(fill=tk.X, padx=10, pady=(0, 10))

        lbl_hint = ttk.Label(btn_bar, text="Center the leaf in good lighting. Press Space or click Capture.", font=("Helvetica", 9))
        lbl_hint.pack(side=tk.LEFT)

        btn_cancel = ttk.Button(btn_bar, text="Cancel (Esc)", command=self.close)
        btn_cancel.pack(side=tk.RIGHT, padx=5)

        btn_snap = ttk.Button(btn_bar, text="📸 Capture Snapshot", command=self.capture)
        btn_snap.pack(side=tk.RIGHT, padx=5)

        self._stream_loop()

    def _stream_loop(self):
        if not self._is_running:
            return
        frame = self.cam.get_frame()
        if frame is not None:
            self.current_frame = frame
            # Scale to fit 640x480 canvas
            display_img = frame.resize((640, 480), Image.Resampling.LANCZOS)
            self._photo_image = ImageTk.PhotoImage(display_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo_image)
        self.after(33, self._stream_loop)

    def capture(self):
        if self.current_frame is not None:
            captured = self.current_frame.copy()
            self.close()
            self.on_capture(captured, "Webcam Snapshot")
        else:
            messagebox.showwarning("No Frame", "No camera frame available to capture.", parent=self)

    def close(self):
        self._is_running = False
        self.cam.release()
        self.destroy()


class SampleSelectorDialog(tk.Toplevel):
    """Modal dialog allowing one-click selection of curated manual test leaf images."""
    def __init__(self, parent, csv_path: str, on_select_callback):
        super().__init__(parent)
        self.title("🧪 Select Manual Evaluation Leaf Sample")
        self.transient(parent)
        self.geometry("700x420")
        self.resizable(True, True)
        self.on_select = on_select_callback
        self.csv_path = Path(csv_path)

        lbl = ttk.Label(
            self,
            text="Choose a sample from the curated test set to evaluate model performance:",
            font=("Helvetica", 10, "bold")
        )
        lbl.pack(anchor=tk.W, padx=12, pady=(12, 6))

        # Table
        cols = ("ID", "Crop", "Condition", "Status", "Filename")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=12)
        self.tree.heading("ID", text="Sample ID")
        self.tree.heading("Crop", text="Crop")
        self.tree.heading("Condition", text="Condition / Disease")
        self.tree.heading("Status", text="True Status")
        self.tree.heading("Filename", text="Filename")

        self.tree.column("ID", width=70, anchor=tk.CENTER)
        self.tree.column("Crop", width=100)
        self.tree.column("Condition", width=200)
        self.tree.column("Status", width=90, anchor=tk.CENTER)
        self.tree.column("Filename", width=200)

        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=(12, 0), pady=6)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 12))

        self.tree.bind("<Double-1>", lambda e: self.load_selected())

        # Buttons
        btn_bar = ttk.Frame(self)
        btn_bar.pack(fill=tk.X, padx=12, pady=10)

        btn_cancel = ttk.Button(btn_bar, text="Cancel", command=self.destroy)
        btn_cancel.pack(side=tk.RIGHT, padx=5)

        btn_load = ttk.Button(btn_bar, text="Load Selected Leaf", command=self.load_selected)
        btn_load.pack(side=tk.RIGHT, padx=5)

        self._load_csv()

    def _load_csv(self):
        if not self.csv_path.exists():
            messagebox.showerror("Missing Data", f"File '{self.csv_path}' not found.", parent=self)
            return

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.tree.insert(
                    "",
                    tk.END,
                    values=(row["sample_id"], row["crop"], row["condition"], row["status"], row["filename"]),
                    tags=(row["image_path"],)
                )

    def load_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("Select Item", "Please select a row from the list.", parent=self)
            return
        item = self.tree.item(sel[0])
        image_path = item["tags"][0]
        label_text = f"{item['values'][1]} - {item['values'][2]} ({item['values'][3]})"
        self.destroy()
        self.on_select(image_path, label_text)


class PlantDiseaseApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Leaf Image-Based Plant Disease Detection Demo")
        self.root.geometry("980x640")
        self.root.minsize(860, 560)
        self.root.configure(bg="#f5f7fa")

        # Load Predictor
        try:
            self.predictor = DiseasePredictor(
                model_path="outputs/plant_disease_mobilenetv2.keras",
                classes_path="outputs/class_names.json",
                threshold=0.50
            )
        except Exception as e:
            messagebox.showerror(
                "Model Loading Error",
                f"Failed to load model weights:\n{e}\n\nPlease verify 'outputs/plant_disease_mobilenetv2.keras' exists."
            )
            self.predictor = None

        self._active_image: Optional[Image.Image] = None
        self._preview_photo: Optional[ImageTk.PhotoImage] = None

        # Variables
        self.diagnosis_var = tk.StringVar(value="No image loaded")
        self.status_var = tk.StringVar(value="--")
        self.confidence_var = tk.StringVar(value="Confidence: --")
        self.advice_var = tk.StringVar(value="Upload a leaf photo, take a camera snapshot, or select a test sample to begin.")
        self.source_var = tk.StringVar(value="Source: None")

        self.top1_var = tk.StringVar(value="1. --")
        self.top2_var = tk.StringVar(value="2. --")
        self.top3_var = tk.StringVar(value="3. --")

        self._setup_ui()

    def _setup_ui(self):
        # Header banner
        header = tk.Frame(self.root, bg="#1e3a29", height=56)
        header.pack(fill=tk.X, side=tk.TOP)
        lbl_app_title = tk.Label(
            header,
            text="🌿 Plant Leaf Disease Detection System",
            font=("Helvetica", 16, "bold"),
            fg="#ffffff",
            bg="#1e3a29"
        )
        lbl_app_title.pack(side=tk.LEFT, padx=16, pady=12)

        lbl_app_subtitle = tk.Label(
            header,
            text="MobileNetV2 Transfer Learning • 38 Classes",
            font=("Helvetica", 10),
            fg="#c8e6c9",
            bg="#1e3a29"
        )
        lbl_app_subtitle.pack(side=tk.RIGHT, padx=16, pady=14)

        # Main 2-column container
        main_pane = tk.Frame(self.root, bg="#f5f7fa")
        main_pane.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

        # ---------------- Left Column (Image & Controls) ----------------
        left_col = tk.Frame(main_pane, bg="#ffffff", bd=1, relief=tk.SOLID)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        lbl_preview_title = tk.Label(
            left_col,
            text="Leaf Image Preview",
            font=("Helvetica", 12, "bold"),
            bg="#ffffff",
            fg="#2c3e50"
        )
        lbl_preview_title.pack(anchor=tk.W, padx=14, pady=(12, 6))

        # Canvas for image preview
        self.canvas = tk.Canvas(left_col, bg="#222b35", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)
        self.canvas.bind("<Configure>", lambda e: self._redraw_preview())

        lbl_source = tk.Label(
            left_col,
            textvariable=self.source_var,
            font=("Helvetica", 9, "italic"),
            bg="#ffffff",
            fg="#7f8c8d"
        )
        lbl_source.pack(anchor=tk.W, padx=14, pady=(2, 8))

        # Action Buttons Toolbar
        btn_frame = tk.Frame(left_col, bg="#ffffff")
        btn_frame.pack(fill=tk.X, padx=14, pady=(0, 14))

        btn_upload = tk.Button(
            btn_frame,
            text="📁 Upload Image",
            font=("Helvetica", 10, "bold"),
            bg="#2980b9",
            fg="#ffffff",
            activebackground="#1f618d",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            command=self.upload_image
        )
        btn_upload.pack(side=tk.LEFT, padx=(0, 6))

        btn_cam = tk.Button(
            btn_frame,
            text="📷 Open Camera",
            font=("Helvetica", 10, "bold"),
            bg="#27ae60",
            fg="#ffffff",
            activebackground="#1e8449",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            command=self.open_camera
        )
        btn_cam.pack(side=tk.LEFT, padx=(0, 6))

        btn_samples = tk.Button(
            btn_frame,
            text="🧪 Load Test Sample",
            font=("Helvetica", 10, "bold"),
            bg="#8e44ad",
            fg="#ffffff",
            activebackground="#6c3483",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            command=self.open_samples
        )
        btn_samples.pack(side=tk.LEFT, padx=(0, 6))

        btn_clear = tk.Button(
            btn_frame,
            text="🔄 Reset",
            font=("Helvetica", 10),
            bg="#ecf0f1",
            fg="#2c3e50",
            activebackground="#bdc3c7",
            relief=tk.FLAT,
            padx=10,
            pady=6,
            command=self.reset
        )
        btn_clear.pack(side=tk.RIGHT)

        # ---------------- Right Column (Results & Insights) ----------------
        right_col = tk.Frame(main_pane, bg="#ffffff", bd=1, relief=tk.SOLID, width=420)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_col.pack_propagate(False)

        lbl_res_heading = tk.Label(
            right_col,
            text="Diagnosis & Classification",
            font=("Helvetica", 12, "bold"),
            bg="#ffffff",
            fg="#2c3e50"
        )
        lbl_res_heading.pack(anchor=tk.W, padx=16, pady=(12, 10))

        # Card: Primary Diagnosis
        diag_card = tk.Frame(right_col, bg="#f8f9fa", bd=1, relief=tk.GROOVE)
        diag_card.pack(fill=tk.X, padx=16, pady=(0, 12))

        lbl_card_title = tk.Label(diag_card, text="Primary Diagnosis", font=("Helvetica", 9, "bold"), fg="#7f8c8d", bg="#f8f9fa")
        lbl_card_title.pack(anchor=tk.W, padx=12, pady=(10, 2))

        self.lbl_diagnosis = tk.Label(
            diag_card,
            textvariable=self.diagnosis_var,
            font=("Helvetica", 14, "bold"),
            fg="#2c3e50",
            bg="#f8f9fa",
            wraplength=380,
            justify=tk.LEFT
        )
        self.lbl_diagnosis.pack(anchor=tk.W, padx=12, pady=(2, 6))

        # Status Tag Badge
        self.status_badge = tk.Label(
            diag_card,
            textvariable=self.status_var,
            font=("Helvetica", 10, "bold"),
            fg="#7f8c8d",
            bg="#e2e8f0",
            padx=12,
            pady=3
        )
        self.status_badge.pack(anchor=tk.W, padx=12, pady=(2, 10))

        # Confidence Bar
        self.lbl_confidence = tk.Label(
            diag_card,
            textvariable=self.confidence_var,
            font=("Helvetica", 10),
            fg="#34495e",
            bg="#f8f9fa"
        )
        self.lbl_confidence.pack(anchor=tk.W, padx=12, pady=(0, 4))

        self.progress_bar = ttk.Progressbar(diag_card, orient=tk.HORIZONTAL, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=12, pady=(0, 12))

        # Card: Top 3 Predictions
        top_card = tk.Frame(right_col, bg="#f8f9fa", bd=1, relief=tk.GROOVE)
        top_card.pack(fill=tk.X, padx=16, pady=(0, 12))

        lbl_top_title = tk.Label(top_card, text="Differential Diagnosis (Top 3 Candidates)", font=("Helvetica", 9, "bold"), fg="#7f8c8d", bg="#f8f9fa")
        lbl_top_title.pack(anchor=tk.W, padx=12, pady=(10, 6))

        lbl_t1 = tk.Label(top_card, textvariable=self.top1_var, font=("Helvetica", 10), fg="#2c3e50", bg="#f8f9fa", wraplength=370, justify=tk.LEFT)
        lbl_t1.pack(anchor=tk.W, padx=12, pady=2)
        lbl_t2 = tk.Label(top_card, textvariable=self.top2_var, font=("Helvetica", 10), fg="#555555", bg="#f8f9fa", wraplength=370, justify=tk.LEFT)
        lbl_t2.pack(anchor=tk.W, padx=12, pady=2)
        lbl_t3 = tk.Label(top_card, textvariable=self.top3_var, font=("Helvetica", 10), fg="#777777", bg="#f8f9fa", wraplength=370, justify=tk.LEFT)
        lbl_t3.pack(anchor=tk.W, padx=12, pady=(2, 10))

        # Card: Recommendation / Advice
        adv_card = tk.Frame(right_col, bg="#f8f9fa", bd=1, relief=tk.GROOVE)
        adv_card.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

        lbl_adv_title = tk.Label(adv_card, text="Evaluation Notes & Recommendations", font=("Helvetica", 9, "bold"), fg="#7f8c8d", bg="#f8f9fa")
        lbl_adv_title.pack(anchor=tk.W, padx=12, pady=(10, 4))

        self.lbl_advice = tk.Label(
            adv_card,
            textvariable=self.advice_var,
            font=("Helvetica", 9),
            fg="#2c3e50",
            bg="#f8f9fa",
            wraplength=370,
            justify=tk.LEFT
        )
        self.lbl_advice.pack(anchor=tk.W, padx=12, pady=4)

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Leaf Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.JPG *.PNG *.bmp")]
        )
        if path:
            self.process_image_path(path)

    def open_camera(self):
        CameraDialog(self.root, on_capture_callback=self.process_image)

    def open_samples(self):
        csv_file = "eval_test_data/test_data.csv"
        if not Path(csv_file).exists():
            messagebox.showinfo("Dataset Not Found", "Curated test dataset not found. Generating now...")
            from create_eval_data import create_dataset
            create_dataset()
        SampleSelectorDialog(self.root, csv_file, on_select_callback=self.process_image_path)

    def process_image_path(self, path: str, display_label: Optional[str] = None):
        try:
            img = Image.open(path)
            title = display_label or Path(path).name
            self.process_image(img, source_title=title)
        except Exception as e:
            messagebox.showerror("Image Error", f"Unable to read image:\n{e}")

    def process_image(self, img: Image.Image, source_title: str = "Image"):
        self._active_image = img.convert("RGB")
        self.source_var.set(f"Loaded: {source_title}")
        self._redraw_preview()

        if self.predictor is None:
            return

        try:
            res = self.predictor.predict(self._active_image)
            self._update_results(res)
        except Exception as e:
            messagebox.showerror("Prediction Error", f"Inference failed:\n{e}")

    def _redraw_preview(self):
        if self._active_image is None:
            self.canvas.delete("all")
            w = self.canvas.winfo_width() or 380
            h = self.canvas.winfo_height() or 380
            self.canvas.create_text(
                w // 2, h // 2,
                text="No Leaf Image Selected\nClick 'Upload Image', 'Open Camera', or 'Load Test Sample'",
                fill="#8892b0",
                font=("Helvetica", 11),
                justify=tk.CENTER
            )
            return

        cw = self.canvas.winfo_width() or 380
        ch = self.canvas.winfo_height() or 380

        # Preserve aspect ratio
        img = self._active_image.copy()
        img.thumbnail((cw, ch), Image.Resampling.LANCZOS)
        self._preview_photo = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        x = (cw - img.width) // 2
        y = (ch - img.height) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self._preview_photo)

    def _update_results(self, res: dict):
        status = res["status"]
        conf = res["confidence"]
        primary = res["primary_label"]

        self.diagnosis_var.set(primary)
        self.confidence_var.set(f"Confidence: {conf * 100:.1f}%")
        self.progress_bar["value"] = conf * 100

        if status == "Healthy":
            self.status_var.set("✓ HEALTHY")
            self.status_badge.configure(bg="#c8e6c9", fg="#1b5e20")
            self.lbl_diagnosis.configure(fg="#1b5e20")
            self.advice_var.set(
                "Plant tissue shows healthy characteristics with no visible fungal, bacterial, or viral disease symptoms."
            )
        elif status == "Diseased":
            self.status_var.set("⚠ DISEASE DETECTED")
            self.status_badge.configure(bg="#ffcdd2", fg="#b71c1c")
            self.lbl_diagnosis.configure(fg="#b71c1c")
            self.advice_var.set(
                f"Pathological infection identified: {primary}. "
                "Inspect surrounding foliage and isolate or treat with recommended agricultural remedies."
            )
        else:  # Uncertain
            self.status_var.set("? UNCERTAIN")
            self.status_badge.configure(bg="#ffe0b2", fg="#e65100")
            self.lbl_diagnosis.configure(fg="#e65100")
            self.advice_var.set(
                f"Model confidence is low ({conf*100:.1f}%). "
                "Please retake the photo with clean lighting, closer framing, and a plain background."
            )

        top_k = res.get("top_k", [])
        if len(top_k) >= 1:
            self.top1_var.set(f"1. {top_k[0][0]} — {top_k[0][1]*100:.1f}%")
        if len(top_k) >= 2:
            self.top2_var.set(f"2. {top_k[1][0]} — {top_k[1][1]*100:.1f}%")
        if len(top_k) >= 3:
            self.top3_var.set(f"3. {top_k[2][0]} — {top_k[2][1]*100:.1f}%")

    def reset(self):
        self._active_image = None
        self._preview_photo = None
        self.canvas.delete("all")
        self.source_var.set("Source: None")
        self.diagnosis_var.set("No image loaded")
        self.status_var.set("--")
        self.status_badge.configure(bg="#e2e8f0", fg="#7f8c8d")
        self.lbl_diagnosis.configure(fg="#2c3e50")
        self.confidence_var.set("Confidence: --")
        self.progress_bar["value"] = 0
        self.advice_var.set("Upload a leaf photo, take a camera snapshot, or select a test sample to begin.")
        self.top1_var.set("1. --")
        self.top2_var.set("2. --")
        self.top3_var.set("3. --")
        self._redraw_preview()


def main():
    root = tk.Tk()
    app = PlantDiseaseApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
