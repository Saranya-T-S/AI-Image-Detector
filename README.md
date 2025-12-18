# AI-Image-Detector

This project implements a deep learning-based system to detect AI-generated images using forensic feature extraction techniques. The system combines CNNs with image forensics to achieve high accuracy in identifying synthetic content.

**Project Files**

- novelty.h5 – Trained model for AI image detection.
- ai_detector_gui.py – Tkinter GUI for real-time image authenticity verification.
- ai_img_detector.py – Feature extraction module.
- demo_video.mp4 – Demo of GUI in action.

**Python Environment**

- TensorFlow 2.17
- Keras 3.10
- NumPy 1.26
- Other dependencies: OpenCV, Pillow, scikit-learn, pandas.

**Key Achievements**

- Successfully developed a deep learning system to detect AI-generated images using features like ELA, PRNU, FFT, and noise residuals.
- Achieved 97.54% accuracy with the Fusion Advanced Model (CNN + ELA + PRNU + FFT + Noise Removal).
- Developed a user-friendly Tkinter GUI for real-time image authenticity verification.

**Future Work**

- Update models with newer AI-generated datasets (Midjourney, DALL-E 3, etc.).
- Explore advanced architectures: ResNet, Vision Transformers, EfficientNet.
- Apply data augmentation to increase dataset size and diversity for better training.
- Perform hyperparameter fine-tuning: learning rate, batch size, dropout, number of layers.
- Use Grad-CAM visualizations for model interpretability.
