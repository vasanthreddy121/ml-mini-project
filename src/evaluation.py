"""Image-level evaluation of saved grayscale, SVR, and SVR+MRF test results."""

import csv
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing import SUPPORTED_EXTENSIONS, resize_image


METHODS = ("grayscale", "svr", "svr_mrf")


def load_rgb(path):
    """Load an image through OpenCV and return an RGB uint8 array."""
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def metrics(predicted, target):
    """Return RGB reconstruction MAE, MSE, RMSE, and PSNR."""
    error = predicted.astype(np.float32) - target.astype(np.float32)
    mse = float(np.mean(error ** 2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rgb_rmse": float(np.sqrt(mse)),
        "psnr": float("inf") if mse == 0 else float(10 * np.log10((255.0 ** 2) / mse)),
    }


def save_rgb(path, image_rgb):
    """Save RGB data using OpenCV's BGR file convention."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def make_comparison(name, original, grayscale, svr, final, split):
    """Save a labeled Original | Grayscale | SVR | SVR + MRF 2x2 panel."""
    images = (("Original", original), ("Grayscale", grayscale),
              ("SVR", svr), ("SVR + MRF", final))
    height, width = original.shape[:2]
    title_height = 30
    panels = []
    for title, image in images:
        panel = np.full((height + title_height, width, 3), 255, dtype=np.uint8)
        panel[title_height:] = image
        cv2.putText(panel, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(panel)
    save_rgb(Path("outputs/comparisons") / split / f"{name}_comparison.png",
             np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))


def method_images(name, split):
    """Load saved inference results and construct the Y-channel grayscale baseline."""
    processed = np.load(Path("outputs/processed") / split / f"{name}.npz")
    y = processed["y"]
    grayscale = cv2.cvtColor(y, cv2.COLOR_GRAY2RGB)
    svr = load_rgb(Path("outputs/svr_results") / split / f"{name}_svr.png")
    svr_mrf = load_rgb(Path("outputs/final") / split / f"{name}_final.png")
    return {"grayscale": grayscale, "svr": svr, "svr_mrf": svr_mrf}


def original_test_rgb(name, split):
    """Load the original colour test image and resize exactly as preprocessing did."""
    source_dir = Path("dataset") / split
    matches = sorted(
        path for path in source_dir.iterdir()
        if path.is_file()
        and path.stem == name
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one supported source image named '{name}' in {source_dir}; found {matches}"
        )
    source_path = matches[0]
    image_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read ground-truth source image: {source_path}")
    return resize_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def write_results(rows, averages, split):
    """Save all per-image rows and average rows as a CSV file."""
    output = Path("outputs/evaluation") / f"{split}_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("image", "method", "mae", "mse", "rgb_rmse", "psnr")
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        for method, result in averages.items():
            writer.writerow({"image": "AVERAGE", "method": method, **result})
    return output


def print_table(rows, averages):
    """Print per-image and average results in a compact terminal table."""
    print("\nPer-image RGB reconstruction metrics")
    print(f"{'Image':<8} {'Method':<10} {'MAE':>10} {'MSE':>12} {'RMSE':>10} {'PSNR (dB)':>12}")
    for row in rows:
        print(f"{row['image']:<8} {row['method']:<10} {row['mae']:10.3f} "
              f"{row['mse']:12.3f} {row['rgb_rmse']:10.3f} {row['psnr']:12.3f}")
    print("\nAverage RGB reconstruction metrics")
    print(f"{'Method':<10} {'MAE':>10} {'MSE':>12} {'RMSE':>10} {'PSNR (dB)':>12}")
    for method in METHODS:
        result = averages[method]
        print(f"{method:<10} {result['mae']:10.3f} {result['mse']:12.3f} "
              f"{result['rgb_rmse']:10.3f} {result['psnr']:12.3f}")


def evaluate_split(split="test"):
    """Evaluate only saved outputs for a split; no inference or model fitting occurs."""
    processed_dir = Path("outputs/processed") / split
    rows = []
    for npz_path in sorted(processed_dir.glob("*.npz")):
        name = npz_path.stem
        original = original_test_rgb(name, split)
        predictions = method_images(name, split)
        for method, image in predictions.items():
            if image.shape != original.shape:
                raise ValueError(f"Shape mismatch for {name} {method}: {image.shape} vs {original.shape}")
            rows.append({"image": name, "method": method, **metrics(image, original)})
        make_comparison(name, original, predictions["grayscale"], predictions["svr"],
                        predictions["svr_mrf"], split)

    averages = {
        method: {metric: float(np.mean([row[metric] for row in rows if row["method"] == method]))
        for metric in ("mae", "mse", "rgb_rmse", "psnr")}
        for method in METHODS
    }
    print_table(rows, averages)
    output = write_results(rows, averages, split)
    print(f"\nSaved numerical results to {output}")
    return rows, averages


if __name__ == "__main__":
    evaluate_split()
