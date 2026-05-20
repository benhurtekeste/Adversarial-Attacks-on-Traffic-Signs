# Adversarial Attacks on Traffic Signs

An interactive web application for exploring adversarial robustness of deep learning models trained on traffic sign recognition. Upload a traffic sign image, apply state-of-the-art adversarial attacks, and inspect how subtle pixel perturbations fool a neural network with real-time visualizations and quantitative metrics.

---

## Demo

| Original | Adversarial | Perturbation Heatmap |
|:---:|:---:|:---:|
| ![Stop Sign](assets/stop_sign.png) | *(generated in app)* | *(generated in app)* |

> The app ships with six sample traffic signs (stop, yield, no entry, road work, slippery road, speed limit 100) so you can explore attacks immediately without your own images.

---

## Features

- **Four attack algorithms** — FGSM, PGD, Carlini & Wagner (C&W), and DeepFool
- **Configurable attack strength** — epsilon slider from 0.001 to 0.3
- **Side-by-side comparison** — original vs. adversarial image with prediction labels and confidence scores
- **Top-5 confidence chart** — interactive Plotly bar chart showing pre- and post-attack class probabilities
- **Perturbation heatmap** — pixel-level visualization of where and how much the image was modified
- **Robustness metrics** — L2 norm, L∞ norm, SSIM, and PSNR computed on every attack
- **Attack success indicator** — clearly flags whether the attack changed the model's prediction

---

## Attack Methods

| Attack | Type | Iterations | Notes |
|--------|------|-----------|-------|
| **FGSM** | Gradient sign | 1 | Fastest; single-step perturbation |
| **PGD** | Iterative gradient | 40 | Stronger multi-step variant of FGSM |
| **C&W** | Optimization (L2) | 10 | Finds minimal-norm adversarial examples |
| **DeepFool** | Boundary search | 50 | Minimal perturbation to cross decision boundary |

All attacks are implemented via the [Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox).

---

## Model

A custom CNN trained on the **German Traffic Sign Recognition Benchmark (GTSRB)**, a standard dataset of 43 traffic sign classes.

```
Input (3 × 48 × 48)
  └─ Conv(32, 3×3) → ReLU
  └─ Conv(64, 3×3) → ReLU → MaxPool(2×2)
  └─ Conv(128, 3×3) → ReLU → MaxPool(2×2)
  └─ Flatten → FC(512) → ReLU → Dropout(0.5)
  └─ FC(43) → Softmax
```

The pre-trained weights (`models/gtsrb_cnn.pth`) are included in the repository.

---

## Project Structure

```
Adversarial-Attacks-on-Traffic-Signs/
├── app.py                   # Streamlit web application (entry point)
├── requirements.txt         # Python dependencies
├── models/
│   └── gtsrb_cnn.pth        # Pre-trained GTSRB CNN weights
├── assets/                  # Sample traffic sign images
│   ├── stop_sign.png
│   ├── yield_sign.jpg
│   ├── no_entry.png
│   ├── road_work.png
│   ├── slippery_road.png
│   └── speed_100.png
└── utils/
    ├── attacks.py           # ART attack wrappers
    ├── model.py             # CNN architecture definition
    ├── preprocessing.py     # Image resize and tensor conversion
    ├── visualization.py     # Heatmaps and robustness metrics
    └── labels.py            # GTSRB class index → label mapping
```

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/benhurtekeste/Adversarial-Attacks-on-Traffic-Signs.git
cd Adversarial-Attacks-on-Traffic-Signs

# Create and activate a virtual environment (recommended)
python -m venv traffic
traffic\Scripts\activate        # Windows
# source traffic/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage

1. **Upload an image** — use the file uploader or select one of the bundled sample images.
2. **Choose an attack** — pick FGSM, PGD, C&W, or DeepFool from the sidebar.
3. **Set epsilon** — drag the slider to control perturbation magnitude (lower = harder to see, higher = stronger attack).
4. **Run** — click **Generate Adversarial Example** and inspect the results.

### Interpreting the Metrics

| Metric | Meaning | Attack is stronger when… |
|--------|---------|--------------------------|
| **L2 norm** | Euclidean distance between images | Higher |
| **L∞ norm** | Maximum pixel change | Higher |
| **SSIM** | Perceptual similarity (0–1) | Lower |
| **PSNR** | Image quality in dB | Lower |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `torch` / `torchvision` | CNN inference |
| `adversarial-robustness-toolbox` | Attack implementations |
| `numpy` / `pillow` / `opencv-python` | Image handling |
| `matplotlib` / `plotly` | Visualization |
| `scikit-image` | SSIM and PSNR metrics |
| `scikit-learn` | Utilities |

---

## Background

Adversarial examples are inputs to machine learning models that are intentionally designed to cause misclassification. For traffic sign recognition systems used in autonomous vehicles, this poses a real safety risk — a stop sign with a carefully crafted sticker could be classified as a speed limit sign.

This project demonstrates how accessible these attacks are and aims to build intuition for:

- Why neural networks are brittle to imperceptible perturbations
- How attack strength (epsilon) trades off between visibility and effectiveness
- How different attack strategies compare in terms of perturbation magnitude and success rate

---

## License

This project is released for educational and research purposes.

---

## Acknowledgements

- [German Traffic Sign Recognition Benchmark (GTSRB)](http://benchmark.ini.rub.de/index.php?section=gtsrb)
- [IBM Adversarial Robustness Toolbox (ART)](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [Goodfellow et al., 2014 — Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (FGSM)
- [Madry et al., 2017 — Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) (PGD)
- [Carlini & Wagner, 2017 — Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644) (C&W)
- [Moosavi-Dezfooli et al., 2016 — DeepFool](https://arxiv.org/abs/1511.04599)
