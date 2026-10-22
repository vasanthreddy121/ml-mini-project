# Automatic Image Colorization

A machine-learning colorization pipeline that predicts plausible chrominance channels ($U, V$) from a grayscale luminance image ($Y$) in OpenCV's YUV representation. The pipeline combines SLIC superpixel segmentation, local 2D-FFT and luminance statistical features, supervised regression models (Support Vector Regression and ExtraTrees ensembles), graph-based Markov Random Field (MRF/ICM) spatial refinement, and optional bilateral edge-preserving chrominance smoothing.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Model Directory Choices & Benchmarks](#model-directory-choices--benchmarks)
3. [Environment Setup](#environment-setup)
4. [Inference Commands](#inference-commands)
5. [Training and Evaluation Commands](#training-and-evaluation-commands)
6. [Running the Test Suite](#running-the-test-suite)
7. [Domain Limitations (Nature Specialist vs. General Domain)](#domain-limitations)
8. [Final Reports and Visual Deliverables](#final-reports-and-visual-deliverables)

---

## Architecture Overview

```text
Grayscale Image (Y)
       │
       ▼
SLIC Superpixel Segmentation (compactness: 0.1 - 10.0, ~200 segments)
       │
       ▼
Feature Extraction (10×10 patch around centroid: 2D FFT magnitudes + mean/std/contrast = 103 features)
       │
       ▼
Feature Standardization (StandardScaler)
       │
       ▼
Regression Models (U & V Chrominance Prediction)
  ├── Baseline SVR (models/)
  ├── Nature-Aware SVR (models_sprint_svr_nature/)
  └── Recommended: Nature-Aware ExtraTrees (models_sprint_et_nature/)
       │
       ▼
Spatial MRF/ICM Refinement (iterative graph-neighbor smoothing, gamma: 0.1 - 2.0)
       │
       ▼
Bilateral Edge-Preserving Chrominance Smoothing (--smooth-chroma, d=7, sigma_color=15, sigma_space=7)
       │
       ▼
Color Reconstruction (Y + U + V → RGB)
```

---

## Model Directory Choices & Benchmarks

The repository includes three verified model configurations evaluated on the untouched 6-image nature test set (`dataset/test/`):

| Model Directory | Architecture | Training Dataset | Test RGB MAE | Test PSNR | Characteristics |
|---|---|---|:---:|:---:|---|
| **`models/`** | RBF-SVR ($C=0.125, \epsilon=0.0625$) | 12 initial images | 16.912 | 20.833 dB | Baseline model. Preserved intact for grading and regression checks. |
| **`models_sprint_svr_nature/`** | RBF-SVR ($C=1.0, \epsilon=0.05$) | 40 balanced DIV2K nature images | 16.666 | 21.050 dB | Nature-specialist SVR alternative. |
| **`models_sprint_et_nature/`** *(Recommended)* | ExtraTrees Regressor (100 trees, max_depth=15) | 40 balanced DIV2K nature images | **14.170** | **22.593 dB** | **Best candidate**: +1.76 dB PSNR gain over baseline, vivid skies, rivers, and foliage. |

---

## Environment Setup

Use Python 3.12 in an isolated virtual environment:

```bash
# Clone the repository
git clone https://github.com/vasanthreddy121/ml-mini-project.git
cd ml-mini-project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

*Note: On Windows, activate with `.venv\Scripts\activate`.*

---

## Inference Commands

### Recommended: High-Quality Colorization on Nature Images
Run inference with the recommended **Nature-Aware ExtraTrees** model, compactness `0.1`, MRF gamma `0.1`, and bilateral smoothing enabled:

```bash
python main.py colorize \
  --input tree.jpg \
  --model-dir models_sprint_et_nature \
  --compactness 0.1 \
  --gamma 0.1 \
  --smooth-chroma
```

### Classical Baseline Colorization
To run the original untouched baseline model:

```bash
python main.py colorize \
  --input tree.jpg \
  --model-dir models \
  --compactness 10.0 \
  --gamma 2.0
```

### Output Artifacts
Inference writes results under `outputs/live/<input_stem>/`:
- `input_preview.png`: Grayscale luminance input
- `slic_superpixels.png`: Superpixel boundary visualization
- `svr_only_colorization.png`: Direct regressor colorization
- `mrf_refined_colorization.png`: Final colorized image (with MRF and bilateral smoothing if enabled)
- `predicted_uv.npz`: Raw predicted $U, V$ code arrays
- `refined_uv.npz`: Refined $U, V$ code arrays and metadata

---

## Training and Evaluation Commands

### Reproducing the Quality Loop Evaluation
Generate per-image benchmark metrics and side-by-side comparison panels across baseline, intermediate candidate, and smoothed candidate:

```bash
python scripts/run_quality_loop_evaluation.py
```

### Reproducing the Nature-Aware Sprint Training
To retrain the Nature-Aware SVR and ExtraTrees models from DIV2K nature subsets:

```bash
python scripts/run_sprint_experiments.py
```

### Running the Baseline Pipeline
To execute the complete legacy baseline pipeline from scratch on `dataset/`:

```bash
python main.py all
```

Or individual stages:
```bash
python main.py preprocess
python main.py segment
python main.py features
python main.py train
python main.py colorize --split test
python main.py mrf --split test
python main.py evaluate --split test
```

---

## Running the Test Suite

Execute the unit and regression test suite (13 passing tests):

```bash
python -m unittest discover tests/
```

Test coverage includes:
- [tests/test_sprint_models.py](tests/test_sprint_models.py): Ensures `models_sprint_et_nature` and `models_sprint_svr_nature` load, predict valid shapes, and perform live colorization.
- [tests/test_live_colorize.py](tests/test_live_colorize.py): Verifies end-to-end live colorization, argument parsing, and `--smooth-chroma` bilateral filtering.
- [tests/test_div2k_experiment.py](tests/test_div2k_experiment.py): Regression checks for DIV2K data preparation and validation logic.

---

## Domain Limitations

> [!WARNING]
> **Nature Specialist Domain Boundary:**
> The `models_sprint_et_nature/` model was trained exclusively on natural landscape, mountain, river, and forest photography. It excels at outdoor nature scenes (skies, foliage, water, rock surfaces).
>
> When presented with human portraits or indoor photos (e.g. human skin, synthetic textiles), the model encounters an out-of-distribution domain shift. Because bright outdoor pixels correlate with blue/cyan sky in nature training, human skin highlights will exhibit a cool/cyan hue. For general cross-domain colorization (portraits, urban scenes), deep learning colorization models (e.g., DDColor, ECCV16) are recommended.

---

## Final Reports and Visual Deliverables

- **Quality Optimization Loop Report**: [outputs/quality_loop/REPORT.md](outputs/quality_loop/REPORT.md)
  - Detailed diagnostic of superpixel plateau boundaries and gradient analysis.
  - Quantitative benchmark table comparing baseline vs. smoothed ExtraTrees.
- **Master Final Solution Report**: [outputs/final_solution/FINAL_REPORT.md](outputs/final_solution/FINAL_REPORT.md)
  - Complete architecture, design decisions, and ablation studies.
- **Run Guide**: [outputs/final_solution/RUN_GUIDE.md](outputs/final_solution/RUN_GUIDE.md)
  - Detailed command manual for training, evaluation, and live demonstrations.
- **Machine-Readable Metrics**:
  - [outputs/quality_loop/best_config.json](outputs/quality_loop/best_config.json)
  - [outputs/quality_loop/results.json](outputs/quality_loop/results.json)
  - [outputs/final_solution/results.json](outputs/final_solution/results.json)
- **Visual Comparison Panels**:
  - Waterfall: [outputs/quality_loop/comparisons/tree_quality_loop_comparison.png](outputs/quality_loop/comparisons/tree_quality_loop_comparison.png)
  - River Rapids: [outputs/quality_loop/comparisons/r1_quality_loop_comparison.png](outputs/quality_loop/comparisons/r1_quality_loop_comparison.png)
  - Clouds & Sky: [outputs/quality_loop/comparisons/l2_quality_loop_comparison.png](outputs/quality_loop/comparisons/l2_quality_loop_comparison.png)
  - Mountain Lake: [outputs/quality_loop/comparisons/m1_quality_loop_comparison.png](outputs/quality_loop/comparisons/m1_quality_loop_comparison.png)
