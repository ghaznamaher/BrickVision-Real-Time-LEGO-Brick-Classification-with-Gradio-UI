# BrickVision-Real-Time-LEGO-Brick-Classification-with-Gradio-UI

**Objective:**
Build an end-to-end, beginner-friendly computer vision app that identifies LEGO brick types from images to aid sorting and building.

**Data:**

Source: Kaggle — pacogarciam3/lego-brick-sorting-image-recognition (downloaded via kagglehub).

Preprocessing: Auto-unpack archives, normalize to RGB JPG, and create a smaller, balanced dataset (cap images/class with MAX_PER_CLASS) to fit Colab storage.

Split: train/val/test ≈ 70/15/15.

**Model & Training:**

Architecture: YOLOv8 classification (yolov8n-cls.pt for speed; upgradeable to yolov8s/m/l-cls).

Settings: ~224×224 images, batch tuned to device (GPU/CPU), ~12–20 epochs.

Outputs: Trained weights saved as best.pt.

**Evaluation:**

Metrics computed on validation (and test if present): accuracy and class-wise performance (Ultralytics’ built-ins).

Quick sanity check: single-image top-3 predictions.

**User Interface (UI):**

Gradio web app: upload an image → returns class probabilities (top-k).

Runs inside Colab with an optional public share link.

**Deployment Options:**

Keep using Gradio share from Colab for demos.

Standalone app.py (Ultralytics + Gradio) to run locally/server.

Optional: host on Hugging Face Spaces (Gradio app + requirements.txt + best.pt).

Why Classification (not Detection) first?
Most images contain a single brick; classification is simpler, faster, and lighter. If later needed to find multiple bricks per image, extend to YOLO detection with bounding boxes.

**Constraints & Mitigations:**

Colab storage: controlled with MAX_PER_CLASS to avoid disk exhaustion.

No GPU: reduce batch size/epochs; still works on CPU (slower).



