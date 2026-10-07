"""Isolated MRF gamma sensitivity experiment for the saved test SVR outputs."""

import csv
from pathlib import Path

import numpy as np

from src.config import ICM_ITERATIONS
from src.evaluation import load_rgb, metrics, original_test_rgb
from src.mrf import refine_chrominance, save_rgb, values_to_image, yuv_to_rgb


GAMMAS = (0.1, 0.5, 1.0, 2.0)


def gamma_directory(gamma):
    """Use a filesystem-friendly, unambiguous directory name for a gamma value."""
    return Path("outputs/gamma_sensitivity") / f"gamma_{str(gamma).replace('.', '_')}"


def average(records):
    """Average the recorded RGB reconstruction metrics across test images."""
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in ("mae", "mse", "rgb_rmse", "psnr")
    }


def run_gamma(gamma, split="test"):
    """Refine the saved SVR predictions at one gamma and save isolated artifacts."""
    root = gamma_directory(gamma)
    mrf_dir = root / "mrf_results" / split
    final_dir = root / "final" / split
    records = []

    for npz_path in sorted((Path("outputs/processed") / split).glob("*.npz")):
        name = npz_path.stem
        y = np.load(npz_path)["y"]
        segments = np.load(Path("outputs/segmentation") / split / f"{name}_segments.npy")
        svr = np.load(Path("outputs/svr_results") / split / f"{name}_predicted_uv.npz")
        segment_ids = svr["segment_ids"]
        if set(segment_ids.tolist()) != set(np.unique(segments).tolist()):
            raise ValueError(f"Segment mismatch for {name}")

        refined_u = refine_chrominance(svr["predicted_u"], segment_ids, segments,
                                       gamma=gamma, iterations=ICM_ITERATIONS)
        refined_v = refine_chrominance(svr["predicted_v"], segment_ids, segments,
                                       gamma=gamma, iterations=ICM_ITERATIONS)
        u_image = values_to_image(refined_u, segment_ids, segments)
        v_image = values_to_image(refined_v, segment_ids, segments)
        rgb = yuv_to_rgb(y, u_image, v_image)
        original = original_test_rgb(name, split)
        if rgb.shape != original.shape or not np.isfinite(rgb).all():
            raise ValueError(f"Invalid RGB output for {name}, gamma={gamma}")

        mrf_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            mrf_dir / f"{name}_refined_uv.npz",
            u=u_image,
            v=v_image,
            segment_ids=segment_ids,
            refined_u=refined_u,
            refined_v=refined_v,
            gamma=gamma,
            iterations=ICM_ITERATIONS,
        )
        save_rgb(final_dir / f"{name}_final.png", rgb)
        records.append({"image": name, **metrics(rgb, original)})

    return records, average(records)


def svr_baseline(split="test"):
    """Measure the unchanged saved SVR-only images with the same ground truth."""
    records = []
    for npz_path in sorted((Path("outputs/processed") / split).glob("*.npz")):
        name = npz_path.stem
        svr = load_rgb(Path("outputs/svr_results") / split / f"{name}_svr.png")
        records.append(metrics(svr, original_test_rgb(name, split)))
    return average(records)


def write_summary(results, baseline, split="test"):
    """Write gamma averages and changes against the SVR-only baseline to CSV."""
    output = Path("outputs/evaluation") / f"{split}_gamma_sensitivity.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "gamma", "iterations", "mae", "mse", "rgb_rmse", "psnr",
        "svr_mae", "svr_mse", "svr_rgb_rmse", "svr_psnr",
        "mae_change_percent", "mse_change_percent", "psnr_change_db",
    )
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for gamma, result in results.items():
            writer.writerow({
                "gamma": gamma,
                "iterations": ICM_ITERATIONS,
                **result,
                "svr_mae": baseline["mae"],
                "svr_mse": baseline["mse"],
                "svr_rgb_rmse": baseline["rgb_rmse"],
                "svr_psnr": baseline["psnr"],
                "mae_change_percent": 100 * (baseline["mae"] - result["mae"]) / baseline["mae"],
                "mse_change_percent": 100 * (baseline["mse"] - result["mse"]) / baseline["mse"],
                "psnr_change_db": result["psnr"] - baseline["psnr"],
            })
    return output


def run_experiment(split="test"):
    """Run all requested gamma values and print a concise average-results table."""
    baseline = svr_baseline(split)
    results = {}
    for gamma in GAMMAS:
        _, results[gamma] = run_gamma(gamma, split)
    output = write_summary(results, baseline, split)

    print("\nGamma sensitivity: average RGB reconstruction metrics")
    print(f"{'Gamma':>6} {'MAE':>10} {'MSE':>12} {'RMSE':>10} {'PSNR (dB)':>12}")
    for gamma, result in results.items():
        print(f"{gamma:6.1f} {result['mae']:10.3f} {result['mse']:12.3f} "
              f"{result['rgb_rmse']:10.3f} {result['psnr']:12.3f}")
    print(f"SVR baseline: MAE={baseline['mae']:.3f}, MSE={baseline['mse']:.3f}, "
          f"RMSE={baseline['rgb_rmse']:.3f}, PSNR={baseline['psnr']:.3f} dB")
    print(f"Best MAE gamma: {min(results, key=lambda value: results[value]['mae']):.1f}")
    print(f"Best PSNR gamma: {max(results, key=lambda value: results[value]['psnr']):.1f}")
    print(f"Saved gamma sensitivity summary to {output}")
    return results, baseline


if __name__ == "__main__":
    run_experiment()
