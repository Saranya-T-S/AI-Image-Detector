"""
ai_detector_gui.py

Tkinter-based GUI for detecting AI-generated vs real images.

"""

import os
import importlib
import importlib.util
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# --- Default configuration ---
DEFAULT_MODEL_PATH = "C:/Users/gokku/Desktop/aiml_proj/novelty.h5"
DEFAULT_FEATURE_MODULE_NAME = "ai_img_detector"
DEFAULT_FEATURE_MODE = "ela"


class AIDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI vs Real Image Detector")
        self.root.geometry("720x520")
        self.root.resizable(False, False)

        self.model = None
        self.model_path = None
        self.feature_module = None
        self.feature_mode = DEFAULT_FEATURE_MODE
        self.input_shape = None  # (H, W, C)

        self._build_ui()
        self._try_autoload_default_model()
        self._try_auto_import_feature_module()

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(pady=8, fill="x")

        btn_frame = tk.Frame(top)
        btn_frame.pack()

        tk.Button(
            btn_frame,
            text="Load Keras Model (.h5)",
            command=self.load_model
        ).grid(row=0, column=0, padx=6)

        tk.Button(
            btn_frame,
            text="Load Feature Module (.py)",
            command=self.load_feature_module
        ).grid(row=0, column=1, padx=6)

        self.btn_upload = tk.Button(
            btn_frame,
            text="Upload Image",
            command=self.upload_image,
            state="disabled"
        )
        self.btn_upload.grid(row=0, column=2, padx=6)

        info = tk.Frame(self.root)
        info.pack(fill="x", padx=10, pady=6)

        self.lbl_model = tk.Label(info, text="Model: None", anchor="w")
        self.lbl_model.pack(fill="x")

        self.lbl_module = tk.Label(info, text="Feature module: Loading default...", anchor="w")
        self.lbl_module.pack(fill="x")

        self.lbl_input = tk.Label(info, text="Model input shape: Unknown", anchor="w")
        self.lbl_input.pack(fill="x")

        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True, padx=10, pady=6)

        preview_frame = tk.LabelFrame(main, text="Image Preview", width=360, height=360)
        preview_frame.pack(side="left", padx=8, pady=4)
        preview_frame.pack_propagate(False)

        self.preview_label = tk.Label(preview_frame, text="No image")
        self.preview_label.pack(expand=True)

        result_frame = tk.LabelFrame(main, text="Prediction", width=320, height=360)
        result_frame.pack(side="right", padx=8, pady=4)
        result_frame.pack_propagate(False)

        self.pred_label = tk.Label(result_frame, text="Prediction: N/A", font=("Arial", 16))
        self.pred_label.pack(pady=(30, 10))

        self.conf_label = tk.Label(result_frame, text="Confidence: N/A", font=("Arial", 14))
        self.conf_label.pack()

    # -------------------------
    # Auto loading
    # -------------------------
    def _try_autoload_default_model(self):
        if os.path.exists(DEFAULT_MODEL_PATH):
            try:
                self._load_model_from_path(DEFAULT_MODEL_PATH)
            except Exception:
                pass

    def _try_auto_import_feature_module(self):
        try:
            mod = importlib.import_module(DEFAULT_FEATURE_MODULE_NAME)
            self._set_feature_module(mod, DEFAULT_FEATURE_MODULE_NAME + ".py")
        except Exception:
            pass

    # -------------------------
    # Model and module loading
    # -------------------------
    def load_model(self):
        path = filedialog.askopenfilename(
            title="Select Keras model",
            filetypes=[("Keras Models", "*.h5 *.keras")]
        )
        if not path:
            return
        try:
            self._load_model_from_path(path)
            messagebox.showinfo("Model Loaded", os.path.basename(path))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _load_model_from_path(self, path):
        self.model = tf.keras.models.load_model(path, compile=False)
        self.model_path = path
        self.lbl_model.config(text=f"Model: {os.path.basename(path)}")

        ish = self.model.input_shape
        if isinstance(ish, list):
            ish = ish[0]

        if len(ish) == 4:
            _, a, b, c = ish
            if a in (1, 3):
                H, W, C = b, c, a
            else:
                H, W, C = a, b, c
        else:
            H, W, C = 256, 256, 3

        self.input_shape = (int(H), int(W), int(C))
        self.lbl_input.config(text=f"Model input shape: {self.input_shape}")
        self._update_upload_state()

    def load_feature_module(self):
        path = filedialog.askopenfilename(
            title="Select feature module",
            filetypes=[("Python Files", "*.py")]
        )
        if not path:
            return

        spec = importlib.util.spec_from_file_location("feature_module", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        filename = os.path.basename(path)  
        self._set_feature_module(mod, filename)
        messagebox.showinfo("Feature Module Loaded", filename)

    def _set_feature_module(self, mod, display_name=None):
        self.feature_module = mod
        if display_name:
            self.lbl_module.config(text=f"Feature module: {display_name}")
        self._update_upload_state()

    def _update_upload_state(self):
        if self.model is not None:
            self.btn_upload.config(state="normal")

    # -------------------------
    # Image upload and prediction
    # -------------------------
    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not path:
            return

        img = Image.open(path).convert("RGB")
        preview = img.copy()
        preview.thumbnail((340, 340))
        imgtk = ImageTk.PhotoImage(preview)

        self.preview_label.config(image=imgtk, text="")
        self.preview_label.image = imgtk

        self._predict(path)

    def _predict(self, path):
        H, W, C = self.input_shape

        if self.feature_module is not None:
            pil_img = self.feature_module.pil_load_rgb(path, (W, H))
            rgb = self.feature_module.to_numpy(pil_img)
            feat = self.feature_module.make_feature(rgb, self.feature_mode)
        else:
            img = Image.open(path).convert("RGB").resize((W, H))
            feat = np.asarray(img).astype(np.float32) / 255.0
            if C == 1:
                feat = np.mean(feat, axis=2, keepdims=True)

        x = np.expand_dims(feat, axis=0).astype(np.float32)

        try:
            pred = self.model.predict(x, verbose=0).ravel()[0]
        except Exception:
            x = np.transpose(x, (0, 3, 1, 2))
            pred = self.model.predict(x, verbose=0).ravel()[0]

        label = "AI Generated" if pred >= 0.5 else "Real Image"
        confidence = pred * 100.0

        self.pred_label.config(text=f"Prediction: {label}")
        self.conf_label.config(text=f"Confidence (AI): {confidence:.2f}%")


def main():
    root = tk.Tk()
    app = AIDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
