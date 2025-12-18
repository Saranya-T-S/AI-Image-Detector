# AI-Image-Detector

This project implements a deep learning-based system to detect AI-generated images using forensic feature extraction techniques. The system combines CNNs with image forensics to achieve high accuracy in identifying synthetic content.

**Team Members**

- 126004237 - Saranya T S, ECE
- 126180060 - Baranika R, Electronics Engineering (VLSI)
- 126180019 - K Parvathavardhini Priya Sadhvi, Electronics Engineering (VLSI)
- 126180029 - Chethana Nagalli, Electronics Engineering (VLSI)

**Demo Video**
Demo.mp4
**Project Files**

- novelty.h5 – Trained model for AI image detection.
- ai_detector_gui.py – Tkinter GUI for real-time image authenticity verification.
- ai_img_detector.py – Feature extraction module.
- demo_video.mp4 – Demo of GUI in action.

**Dataset Details**
- For AI-Generated Images:(1219)
https://www.kaggle.com/datasets/dibyarupdutta/dmimagesubset
- For Real Camera Captured Images:(1219)
  - https://lesc.dinfo.unifi.it/VISION/ (500)
  - https://www.kaggle.com/datasets/mariammarioma/midjourney-imagenet-real-vs-synth?select=Midjourney_Exp2 (485)
  - https://www.kaggle.com/datasets/rafsunahmad/camera-photos-vs-ai-generated-photos-classifier (234)

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
