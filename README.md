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

**Use Case**

* Problem. Identifying and sorting LEGO® bricks by type is slow and error-prone when done by hand. It gets in the way of building, cataloging collections, classroom activities, and maker projects.

* Solution. BrickVision is a web app that classifies a single LEGO brick from a photo using a fine-tuned YOLOv8 classification model. Upload an image → get the top predicted part class with confidence scores. It runs in the browser (no installs), so anyone can try it.

**Who benefits.**

Hobbyists/AFOLs: faster binning and inventory management

Educators/Makerspaces: quick demos of computer vision + hands-on sorting

Parents/Students: simple, kid-friendly way to learn about AI

**How it’s used.**

* Take/drag a photo of one brick (good lighting, plain background helps).

* The app returns the predicted class (e.g., Brick_2x4) with confidence bars.

* Use results to sort bins, label drawers, or log inventory.

**Notes & limits.**

* Trained for single-brick images; multi-brick scenes may confuse the classifier.

* Accuracy improves with more training data, balanced classes, and longer training.

**Future work:**

Switch to object detection to localize multiple bricks per image; integrate with a physical sorter.

**Live Deployment (Hugging Face Spaces)**

Public URLs

App (full screen): https://ghazna-brickvision-ai.hf.space

Project page: https://huggingface.co/spaces/Ghazna/brickvision-ai

Stack

UI: Gradio

Model: Ultralytics YOLOv8-cls (PyTorch CPU)

Hosting: Hugging Face Spaces (Gradio SDK)

**What we deployed.**

app.py — loads the trained weights, runs inference, and serves a Gradio interface.

requirements.txt — pinned deps (PyTorch CPU wheels) for faster, reliable builds.

Weights — your fine-tuned model (best.pt). We pointed the app to it via env var:

Space Settings → Variables: WEIGHTS_PATH = best.pt

