# ai_image_detector_ela_prnu_fusion.py

import argparse
import os
from pathlib import Path
from typing import List, Tuple
import gc
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pywt
from scipy.signal import wiener
from scipy.fftpack import fft2, fftshift
import pandas as pd

# ---------------------------
# GPU Memory Configuration
# ---------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# ---------------------------
# Utils
# ---------------------------
def list_images(root: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[str]:
    p = Path(root)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {root}")
    files = [str(x) for x in p.rglob("*") if x.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images found under: {root}")
    return sorted(files)

def pil_load_rgb(path: str, size: Tuple[int, int]) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize(size, Image.Resampling.LANCZOS)
    return im

def to_numpy(im: Image.Image) -> np.ndarray:
    return np.asarray(im)

# ---------------------------
# Feature Extraction
# ---------------------------
def extract_prnu_enhanced(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    # Wavelet decomposition for denoising
    coeffs = pywt.wavedec2(gray, 'db4', level=2)
    cA = coeffs[0]
    details = []
    for level_coeffs in coeffs[1:]:
        cH, cV, cD = level_coeffs
        details.extend([wiener(cH, mysize=5), wiener(cV, mysize=5), wiener(cD, mysize=5)])
    
    reconstructed_coeffs = [cA] + [tuple(details[i:i+3]) for i in range(0, len(details), 3)]
    denoised = pywt.waverec2(reconstructed_coeffs, 'db4')
    
    if denoised.shape != gray.shape:
        denoised = cv2.resize(denoised, (gray.shape[1], gray.shape[0]))
    
    # Calculate noise residual
    residual = gray - denoised
    residual = residual - residual.mean()
    std = residual.std()
    if std > 0:
        residual = residual / (3*std)
    residual = np.clip(residual, -1, 1)*0.5 + 0.5
    return residual.astype(np.float32)

def extract_ela_enhanced(rgb: np.ndarray, quality: int = 95) -> np.ndarray:
    
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result, encimg = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not result:
        raise ValueError("JPEG recompression failed in ELA.")
    
    dec = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    diff = cv2.absdiff(rgb, dec).astype(np.float32)
    
    ela_channels = []
    for c in range(3):
        channel_diff = diff[:, :, c]
        p95 = np.percentile(channel_diff, 95)
        scale = 255.0/p95 if p95>0 else 1.0
        ela_channels.append(np.clip(channel_diff*scale, 0, 255)/255.0)
    
    ela = np.stack(ela_channels, axis=-1).astype(np.float32)
    return ela

def extract_fft_spectrum(rgb: np.ndarray) -> np.ndarray:
    
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    fft = fft2(gray)
    fft_shifted = fftshift(fft)
    magnitude = np.abs(fft_shifted)
    magnitude_db = np.log1p(magnitude)
    magnitude_db = (magnitude_db - magnitude_db.min()) / (magnitude_db.max() - magnitude_db.min() + 1e-8)
    return magnitude_db.astype(np.float32)

def extract_noise_residual(rgb: np.ndarray) -> np.ndarray:
    
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    denoised_median = cv2.medianBlur((gray*255).astype(np.uint8), 5)/255.0
    denoised_bilateral = cv2.bilateralFilter((gray*255).astype(np.uint8), 9, 75, 75)/255.0
    residual_median = np.abs(gray - denoised_median)
    residual_bilateral = np.abs(gray - denoised_bilateral)
    combined_residual = np.maximum(residual_median, residual_bilateral)
    combined_residual = (combined_residual - combined_residual.min()) / (combined_residual.max() - combined_residual.min() + 1e-8)
    return combined_residual.astype(np.float32)

def make_feature(rgb: np.ndarray, mode: str) -> np.ndarray:
    
    if mode == "prnu":
        return extract_prnu_enhanced(rgb)[..., None]
    
    elif mode == "ela":
        return extract_ela_enhanced(rgb)
    
    elif mode == "fft":
        return extract_fft_spectrum(rgb)[..., None]
    
    elif mode == "fusion_ela_prnu":
        # ELA (3 channels) + PRNU (1 channel) = 4 channels total
        prnu = extract_prnu_enhanced(rgb)  # Single channel
        ela = extract_ela_enhanced(rgb)     # 3 channels (RGB)
        feat = np.concatenate([ela, prnu[..., None]], axis=-1)
        return feat.astype(np.float32)
    
    elif mode == "fusion_advanced":
        # Full fusion with all features: 5 channels
        prnu = extract_prnu_enhanced(rgb)
        fft_spec = extract_fft_spectrum(rgb)
        noise_res = extract_noise_residual(rgb)
        ela_gray = cv2.cvtColor((extract_ela_enhanced(rgb)*255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
        feat = np.stack([prnu, fft_spec, noise_res, ela_gray, gray], axis=-1)
        return feat.astype(np.float32)
    
    else:
        raise ValueError(f"Unknown feature_mode: {mode}")

# ---------------------------
# CNN Architecture
# ---------------------------
def build_cnn_standard(input_shape):
    
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out)

# ---------------------------
# Dataset Creation
# ---------------------------
def make_dataset(paths, labels, img_size, feature_mode, batch_size, shuffle=True):
  
    H, W = img_size[1], img_size[0]
    if feature_mode == "fusion_ela_prnu":
        C = 4  # ELA (3) + PRNU (1)
    elif feature_mode == "fusion_advanced":
        C = 5
    elif feature_mode == "ela":
        C = 3
    else:
        C = 1
    
    labels = np.array(labels, dtype=np.int32)

    def load_and_process(path, label):
        path_str = path.numpy().decode('utf-8') if isinstance(path, tf.Tensor) else path
        try:
            im = pil_load_rgb(path_str, (W, H))
            rgb = to_numpy(im)
            feat = make_feature(rgb, feature_mode)
            return feat.astype(np.float32), np.int32(label)
        except Exception as e:
            print(f"Error loading {path_str}: {e}")
            return np.zeros((H, W, C), dtype=np.float32), np.int32(0)

    def tf_load_and_process(path, label):
        feat, lbl = tf.py_function(load_and_process, [path, label], [tf.float32, tf.int32])
        feat.set_shape([H, W, C])
        lbl.set_shape([])
        return feat, lbl

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(paths), 1000))
    dataset = dataset.map(tf_load_and_process, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# ---------------------------
# Training & Evaluation
# ---------------------------
def train_eval(args):
    
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    print("Loading image paths...")
    try:
        ai_paths = list_images(args.ai_dir)
        real_paths = list_images(args.real_dir)
        print(f"Found {len(ai_paths)} AI images and {len(real_paths)} real images")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
       
    if args.balance:
        n = min(len(ai_paths), len(real_paths))
        ai_paths, real_paths = ai_paths[:n], real_paths[:n]
        print(f"Balanced dataset to {n} images per class")

    all_paths = ai_paths + real_paths
    all_labels = [1]*len(ai_paths) + [0]*len(real_paths)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=args.seed, stratify=all_labels
    )
    print(f"Training samples: {len(train_paths)}, Validation samples: {len(val_paths)}")

    print(f"Creating datasets with feature mode: {args.feature_mode}")
    ds_train = make_dataset(train_paths, train_labels, args.img_size, args.feature_mode, args.batch_size)
    ds_val = make_dataset(val_paths, val_labels, args.img_size, args.feature_mode, args.batch_size, shuffle=False)

    if args.feature_mode == "fusion_ela_prnu":
        in_channels = 4
    elif args.feature_mode == "fusion_advanced":
        in_channels = 5
    elif args.feature_mode == "ela":
        in_channels = 3
    else:
        in_channels = 1
    
    input_shape = (args.img_size[1], args.img_size[0], in_channels)
    print(f"Input shape: {input_shape}")
    
    model = build_cnn_standard(input_shape)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(f"./outputs/{timestamp}_{args.feature_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(out_dir / f"best_model_{args.feature_mode}.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    print("\nStarting training...")
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    model.save(out_dir / f"final_model_{args.feature_mode}.h5")
    pd.DataFrame(history.history).to_csv(out_dir / f"history_{args.feature_mode}.csv", index=False)
    print(f"Model saved to: {out_dir}")

    print("\nEvaluating model...")
    y_true, y_pred, y_proba = [], [], []
    for xb, yb in ds_val:
        proba = model.predict(xb, verbose=0)
        pred = (proba >= 0.5).astype(int).ravel()
        y_true.append(yb.numpy())
        y_pred.append(pred)
        y_proba.append(proba.ravel())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_proba = np.concatenate(y_proba)

    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, digits=4, target_names=["Real", "AI"])
    
    pd.DataFrame(cm, index=["Real", "AI"], columns=["Real", "AI"]).to_csv(
        out_dir / f"confusion_matrix_{args.feature_mode}.csv"
    )
    with open(out_dir / f"classification_report_{args.feature_mode}.txt", "w") as f:
        f.write(cr)
    
    results_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'probability': y_proba
    })
    results_df.to_csv(out_dir / f"predictions_{args.feature_mode}.csv", index=False)
    
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(cr)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nAll results saved to: {out_dir.resolve()}")

# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Image Detector with ELA+PRNU Fusion")
    parser.add_argument("--ai_dir", type=str, required=True, help="Directory containing AI-generated images")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory containing real images")
    parser.add_argument("--feature_mode", type=str, default="fusion_ela_prnu",
                        choices=["ela", "prnu", "fft", "fusion_ela_prnu", "fusion_advanced"],
                        help="Feature extraction mode")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--img_size", type=int, nargs=2, default=[256, 256], help="Image size (width height)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--balance", action="store_true", help="Balance dataset by taking min(AI, Real) samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("="*50)
    print("AI IMAGE DETECTOR - ELA+PRNU FUSION")
    print("="*50)
    print(f"Feature Mode: {args.feature_mode}")
    print(f"Image Size: {args.img_size[0]}x{args.img_size[1]}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Balance Dataset: {args.balance}")
    print("="*50 + "\n")
    

    train_eval(args)
