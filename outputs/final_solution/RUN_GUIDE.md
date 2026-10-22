# Quick Run Guide: Automatic Image Colorization

This guide provides concise, copy-pasteable instructions for reproducing, training, evaluating, and running the optimized image colorization pipeline.

---

## 1. Environment & Setup

Ensure the virtual environment is activated or invoke Python directly via `.venv/bin/python`:

```bash
# Verify environment and dependencies
.venv/bin/python -c "import sklearn, cv2, numpy, joblib; print('Dependencies OK: sklearn', sklearn.__version__, 'cv2', cv2.__version__)"
```

---

## 2. Colorize a New Grayscale Image (Inference)

### Recommended: Selected Nature Specialist (ExtraTrees)
```bash
# Colorize unseen waterfall image with best ExtraTrees model (compactness=0.2, gamma=0.2)
.venv/bin/python main.py colorize \
  --input tree.jpg \
  --model-dir models_sprint_et_nature \
  --compactness 0.2 --gamma 0.2 \
  --output-dir outputs/live_nature_et

# Colorize any custom image
.venv/bin/python main.py colorize \
  --input /path/to/your/image.png \
  --model-dir models_sprint_et_nature \
  --compactness 0.2 --gamma 0.2 \
  --output-dir outputs/live
```

### Alternative: Classical RBF-SVR Specialist
```bash
# Colorize using the tuned Nature-Aware SVR model
.venv/bin/python main.py colorize \
  --input tree.jpg \
  --model-dir models_sprint_svr_nature \
  --compactness 0.2 --gamma 0.2 \
  --output-dir outputs/live_nature_svr
```

### Comparison: Original Baseline Model (models/)
```bash
# Baseline SVR (compactness=10, gamma=2.0)
.venv/bin/python main.py colorize \
  --input tree.jpg \
  --model-dir models \
  --output-dir outputs/live_baseline
```

### Deep Learning Benchmark (External Contrast)
```bash
# Run DDColor-tiny ONNX benchmark on CPU in ~2.5s
.venv/bin/python scripts/run_pretrained_benchmark.py --input tree.jpg human_test.png
```

---

## 3. Retrain and Reproduce All Models

To re-run the complete data-filtering, training, and multi-model evaluation pipeline from scratch:

```bash
.venv/bin/python scripts/run_sprint_experiments.py
```

This script:
1. Filters the 640 DIV2K training images for nature scenes ($\ge 30\%$ green/blue chroma).
2. Assembles the 10,000-sample balanced training set (2,343 original landscapes + 7,657 nature DIV2K).
3. Fits and saves `models_sprint_svr_nature/` and `models_sprint_et_nature/`.
4. Evaluates all 5 models on the 6 untouched test images and saves `outputs/improvement_sprint/test_comparison_all_models.csv`.
5. Renders 5-way visual comparison panels for nature scenes.

To regenerate the final solution panels in `outputs/final_solution/visual_comparisons/`:
```bash
.venv/bin/python scripts/generate_final_solution_artifacts.py
```

---

## 4. Run Regression Test Suite

All 12 unit tests verify data isolation, live colorization, model loading, and finite predictions:

```bash
.venv/bin/python -m unittest discover tests/
```

---

## 5. Locating and Interpreting Output Artifacts

| Output Path | Description |
| :--- | :--- |
| `outputs/final_solution/visual_comparisons/nature/tree_comparison.png` | 4-way panel comparing Input, Baseline, Nature SVR, Nature ExtraTrees on waterfall |
| `outputs/final_solution/visual_comparisons/nature/r1_comparison.png` | 4-way panel comparing models on river rapids |
| `outputs/final_solution/visual_comparisons/nature/l2_comparison.png` | 4-way panel comparing models on sunny landscape |
| `outputs/final_solution/visual_comparisons/nature/m1_comparison.png` | 4-way panel comparing models on mountain lake |
| `outputs/final_solution/visual_comparisons/portraits_general/` | Cross-domain evaluation on human portrait against DDColor benchmark |
| `outputs/final_solution/results.json` | Machine-readable metrics, split sizes, and configurations |
| `outputs/final_solution/FINAL_REPORT.md` | Comprehensive research and engineering report |

### Understanding Live Output Files:
When running `main.py colorize`, the following files are saved in `--output-dir/<stem>/`:
- `input_preview.png`: 2D luminance channel converted back to RGB preview.
- `slic_superpixels.png`: Luminance overlaid with yellow superpixel boundaries.
- `svr_only_colorization.png`: Colorized image directly from regressor chrominance (no smoothing).
- `mrf_refined_colorization.png`: Final output after MRF/ICM edge-preserving refinement.
- `predicted_uv.npz` & `refined_uv.npz`: Raw per-superpixel $U, V$ predictions and metadata.

