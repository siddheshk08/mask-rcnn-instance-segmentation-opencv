# Mask R-CNN Instance Segmentation with OpenCV

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://www.conventionalcommits.org/en/v1.0.0/)
[![Made with Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Release](https://img.shields.io/badge/Release-v0.1.0-orange.svg)](../../releases)

---

## 📌 Overview

This project demonstrates **instance segmentation** using **Mask R-CNN** with the **OpenCV DNN module**.
The model is pre-trained on the **COCO dataset** and can detect and segment multiple object classes in images.

It provides a **CLI-based demo** to run inference on sample images, visualize masks, and save outputs for reproducibility.
This repository is designed to be **portfolio-ready**, lightweight, and easy to reproduce.

---

## 📂 Repository Structure

```
mask-rcnn-instance-segmentation-opencv/
├─ src/                 # Source code (main script + utils)
│  ├─ main.py
│  └─ utils/
│     └─ util.py
├─ data/                # Dataset folder (ignored in git)
│  ├─ samples/          # Example images for quick testing
│  └─ README.md
├─ models/              # Pretrained weights (release assets)
│  └─ README.md
├─ build/               # Outputs and temporary files (ignored in git)
├─ docs/                # Documentation and diagrams
│  └─ assets/
├─ notebooks/           # Optional experiments
├─ requirements.txt     # Minimal dependencies
├─ LICENSE              # MIT License
└─ README.md            # Project documentation (this file)
```

---

## ⚙️ Getting Started

### 1. Prerequisites

* Python **3.9+** (tested on 3.9 and 3.10)
* PowerShell (for the commands below)

### 2. Clone the Repository

```powershell
git clone https://github.com/AlbertoMarquillas/mask-rcnn-instance-segmentation-opencv.git
cd mask-rcnn-instance-segmentation-opencv
```

### 3. Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 5. Download Model Files

The required Mask R-CNN weights and config are uploaded as **release assets**.

1. Go to the [Releases](../../releases) page.
2. Download these files:

   * `mask_rcnn_inception_v2_coco_2018_01_28.pbtxt`
   * `frozen_inference_graph.pb`
   * `class.names`
3. Place them inside the local `models/` folder.

---

## 🚀 Usage

### Run with default settings

```powershell
python .\src\main.py
```

### Run with sample image and visualize results

```powershell
python .\src\main.py --image .\data\samples\cat_and_dog.png --show
```

### Run with confidence threshold and custom output directory

```powershell
python .\src\main.py --thr 0.6 --outdir .\build\test_outputs
```

Outputs will be saved in `build/outputs/` by default:

* `mask.png` → segmented mask
* `overlay.png` → overlay of mask on the input image

---

## 📊 Features

* **Mask R-CNN (COCO)** with OpenCV DNN backend.
* Detects and segments **90+ classes**.
* CLI with configurable paths, thresholds, and outputs.
* Works in **headless mode** (saves results) or with **OpenCV GUI**.
* Clean and professional repository structure.

---

## 📚 What I Learned

* How to use the **OpenCV DNN module** for deep learning inference.
* Loading and running a **pretrained TensorFlow model** in OpenCV.
* Implementing **mask overlay blending** for visualization.
* Structuring a **portfolio-ready repository** with clear docs and placeholders.

---

## 🛣️ Roadmap

* [ ] Add Jupyter notebooks for interactive demos.
* [ ] Extend CLI with video input support.
* [ ] Benchmark inference time with GPU acceleration.
* [ ] Add Dockerfile for containerized usage.

---

## 📄 License

This project is licensed under the terms of the [MIT License](LICENSE).
