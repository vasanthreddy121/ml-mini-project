# Automatic Image Colorization

This college mini-project colorizes landscape images with a classical machine-learning pipeline. It uses the Y channel in OpenCV's YUV representation as the grayscale input, SLIC superpixels, local FFT/luminance features, two SVR regressors for U and V, and a lightweight MRF/ICM chrominance refinement.

## Setup

Use Python 3.12 and an isolated virtual environment from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The tracked baseline models were serialized with scikit-learn 1.9.0, which is
pinned in `requirements.txt` for compatible loading. On Windows, activate with
`.venv\Scripts\activate` instead.

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

`all` refits and rewrites the baseline model/output artifacts. Use it only when
you intentionally want to regenerate the baseline; it is not the live-demo
command.

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

## Colorize a new grayscale image

Use the trained models to estimate colour for a new grayscale JPG, JPEG, PNG,
BMP, or WEBP image. RGB files that visually contain grayscale content are also
accepted. This command performs inference only; it does not retrain or alter
the dataset models.

```bash
python main.py colorize --input "/path/to/my_image.jpg"
```

For a faculty demonstration after setup, use the saved baseline models:

```bash
source .venv/bin/activate
python main.py colorize --input "/absolute/path/to/grayscale_image.png"
```

The image is resized to the configured width while preserving aspect ratio,
then follows the same YUV, SLIC, FFT-feature, SVR, and MRF/ICM stages as test
inference. It writes the following files under
`outputs/live/<input_stem>/`:

- `input_preview.png` — resized luminance input
- `slic_superpixels.png` — SLIC boundary visualization
- `svr_only_colorization.png` — SVR estimate before refinement
- `mrf_refined_colorization.png` — final SVR + MRF/ICM estimate
- `predicted_uv.npz` — raw per-superpixel SVR U/V predictions and IDs
- `refined_uv.npz` — raw per-superpixel MRF-refined U/V values and IDs

The live command also prints min, max, mean, and standard deviation for the
raw superpixel U and V values before and after MRF/ICM refinement.

These are estimated colours, not a reconstruction of the image's unknown true
colours.

## Separate DIV2K candidate experiment

DIV2K is kept entirely separate from the baseline: it writes preprocessing,
SLIC labels, features, candidate test outputs, and reports below
`outputs/div2k/`, and saves its scaler/SVRs to `models_div2k/`. The original
`dataset/`, `models/`, and baseline test outputs are never overwritten.

Start with the 50-source-image pilot (a deterministic 40/10 source-image
train/validation split with seed 42):

```bash
python main.py div2k-prepare --div2k-limit 50
python main.py div2k-segment
python main.py div2k-features
python main.py div2k-train
python main.py div2k-evaluate
```

Before scaling the dataset, compare a small safe RBF-SVR grid on the current
DIV2K source-level validation split. It evaluates the untouched original test
set only once, after choosing the lowest mean U/V validation MAE:

```bash
python main.py div2k-tune --div2k-max-train-samples 10000
```

Validation-grid artifacts are kept under `outputs/div2k/hyperparameter_search/`
and per-configuration models under `models_div2k/hyperparameter_search/`.

Or run those pilot steps in sequence:

```bash
python main.py div2k-all --div2k-limit 50
```

`div2k-train` retains the same RBF-SVR architecture but deterministically caps
training superpixels at 10,000 by default (`--div2k-max-train-samples`) because
exact RBF-SVR training grows quadratically with rows. The command prints the
kernel-memory lower-bound and writes counts, validation metrics, and training
time to `outputs/div2k/training_summary.json`.

For a larger prepared subset, choose the source count explicitly; do not start
the 800-image run until the pilot's time and memory report has been reviewed:

```bash
python main.py div2k-prepare --div2k-limit 800 --div2k-seed 42
```

Candidate evaluation uses the untouched six-image original test split and
writes per-image and average baseline-versus-candidate MAE, MSE, RMSE, and PSNR
to `outputs/div2k/evaluation/test_comparison_metrics.csv`.

The DIV2K candidate is experimental and underperformed the baseline on the
original test set. It remains opt-in only:

```bash
python main.py colorize --input "/path/to/my_image.jpg" --model-dir models_div2k
```

## Mixed original + DIV2K candidate

The mixed pilot combines only `dataset/train` with the manifest-listed DIV2K
training sources. Original validation images are used solely for selecting the
candidate hyperparameters; original test images are untouched until final
evaluation. Models and outputs are isolated in `models_mixed_div2k/` and
`outputs/mixed_div2k/`.

```bash
python main.py mixed-tune --div2k-max-train-samples 10000 --div2k-seed 42
python main.py mixed-evaluate
```

The pilot compares the original SVR settings and the DIV2K-tuned settings on
original validation, then reports baseline, DIV2K-only tuned, and mixed-model
metrics on the same untouched original test images. It never promotes a mixed
model automatically.

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
