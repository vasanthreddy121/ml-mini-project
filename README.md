# Automatic Image Colorization

This college mini-project colorizes landscape images with a classical machine-learning pipeline. It uses the Y channel in OpenCV's YUV representation as the grayscale input, SLIC superpixels, local FFT/luminance features, two SVR regressors for U and V, and a lightweight MRF/ICM chrominance refinement.

## Setup

Activate the supplied Python 3.12 environment and install dependencies if needed:

```bash
pip install -r requirements.txt
```

Place colour source images in `dataset/train`, `dataset/validation`, and `dataset/test`. The original colour is intentional: its Y channel is the input while U and V are training targets.

## Project structure

```text
dataset/                  # Colour source images split into train/validation/test
models/                   # Training-only scaler and U/V SVR models
outputs/                  # Intermediate data, predictions, refinements, and reports
src/                      # Pipeline stages and configuration
main.py                   # Command-line entry point
```

Important output folders are `outputs/features`, `outputs/svr_results`,
`outputs/mrf_results`, `outputs/final`, `outputs/comparisons`, and
`outputs/evaluation`.

## Run the complete pipeline

Run from the project root. The default `all` command executes the complete
pipeline in this order:

```text
preprocessing → SLIC segmentation → FFT features → SVR training
→ SVR-only colorization → MRF/ICM refinement → evaluation
```

```bash
python main.py all
```

The test split is used for inference and final evaluation only; model fitting
uses the training split, while validation is used only for post-fit checking.

Individual stages can also be run when their required earlier outputs exist:

```bash
python main.py preprocess
python main.py segment
python main.py features
python main.py train
python main.py colorize --split test
python main.py mrf --split test
python main.py evaluate --split test
```

## Optional gamma sensitivity experiment

The gamma experiment runs MRF/ICM with gamma values 0.1, 0.5, 1.0, and 2.0
for 10 iterations, without overwriting the main SVR or MRF outputs:

```bash
python -m src.gamma_sensitivity
```

It writes gamma-specific artifacts under `outputs/gamma_sensitivity` and its
summary to `outputs/evaluation/test_gamma_sensitivity.csv`.

## Method

For each SLIC superpixel, the system takes a 10×10 patch around its centroid and uses log FFT magnitude plus mean, standard deviation, and contrast as its feature vector. Training targets are the mean U and V values within the corresponding superpixel. The MRF step minimizes a simple energy with a data term that preserves the SVR result and a pairwise term that encourages neighbouring superpixels to take similar chrominance values. Its ICM update is:

`xᵢ = (pᵢ + γ Σⱼ xⱼ) / (1 + γ |N(i)|)`

where `pᵢ` is the SVR prediction, `N(i)` is the set of adjacent superpixels, and γ is configured in `src/config.py`.

The MRF term promotes spatially compatible chrominance. Its effect on
quantitative reconstruction accuracy depends on the chosen gamma and should be
reported from the saved evaluation results rather than assumed to be an
improvement.
