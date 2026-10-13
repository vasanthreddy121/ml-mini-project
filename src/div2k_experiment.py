"""Isolated DIV2K training experiment for the existing YUV/SLIC/SVR pipeline."""

import csv
import json
import shutil
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src import colorize, evaluation, features, mrf, preprocessing, segmentation, train
from src.config import (DIV2K_DIR, DIV2K_MAX_TRAIN_SAMPLES, DIV2K_MODEL_DIR,
                        DIV2K_OUTPUT_DIR, DIV2K_SPLIT_SEED,
                        DIV2K_VALIDATION_FRACTION, MIXED_MODEL_DIR,
                        MIXED_OUTPUT_DIR, SVR_C, SVR_EPSILON,
                        SVR_KERNEL)


DATASET_DIR = Path(DIV2K_DIR)
OUTPUT_ROOT = Path(DIV2K_OUTPUT_DIR)
MODEL_DIR = Path(DIV2K_MODEL_DIR)
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
MIXED_MODEL_ROOT = Path(MIXED_MODEL_DIR)
MIXED_OUTPUT_ROOT = Path(MIXED_OUTPUT_DIR)


def discover_images(dataset_dir=DATASET_DIR, limit=None, seed=DIV2K_SPLIT_SEED):
    """Return valid DIV2K PNG source images, optionally a deterministic subset."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"DIV2K directory does not exist: {dataset_dir}")
    paths = sorted(path for path in dataset_dir.iterdir()
                   if path.is_file() and path.suffix.lower() == ".png")
    if not paths:
        raise FileNotFoundError(f"No PNG images found in: {dataset_dir}")
    if limit is not None:
        if limit <= 1:
            raise ValueError("DIV2K limit must be at least 2 images.")
        if limit > len(paths):
            raise ValueError(f"Requested {limit} images, but only {len(paths)} valid PNG images exist.")
        indexes = np.random.default_rng(seed).choice(len(paths), limit, replace=False)
        paths = [paths[index] for index in sorted(indexes)]
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError(f"Invalid RGB DIV2K image: {path}")
    return paths


def make_split_manifest(paths, seed=DIV2K_SPLIT_SEED,
                        validation_fraction=DIV2K_VALIDATION_FRACTION):
    """Split at source-image level, so no image contributes to both sets."""
    if not 0 < validation_fraction < 1:
        raise ValueError("DIV2K validation fraction must be between 0 and 1.")
    if len(paths) < 2:
        raise ValueError("At least two source images are required for a split.")
    names = [Path(path).name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("DIV2K source image names must be unique.")
    shuffled = np.random.default_rng(seed).permutation(names).tolist()
    validation_count = max(1, int(round(len(shuffled) * validation_fraction)))
    validation_count = min(validation_count, len(shuffled) - 1)
    manifest = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "train": sorted(shuffled[validation_count:]),
        "validation": sorted(shuffled[:validation_count]),
    }
    if set(manifest["train"]) & set(manifest["validation"]):
        raise AssertionError("DIV2K source split overlap detected.")
    return manifest


def _load_manifest():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing DIV2K manifest; run 'div2k-prepare' first: {MANIFEST_PATH}")
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if set(manifest["train"]) & set(manifest["validation"]):
        raise ValueError("DIV2K manifest has overlapping source-image splits.")
    return manifest


def _estimate_storage(image_count):
    # Conservative, uncompressed estimate for resized YUV, labels, previews and features.
    estimated_bytes = image_count * 3 * 1024 * 1024
    free_bytes = shutil.disk_usage(OUTPUT_ROOT.parent if OUTPUT_ROOT.parent.exists() else ".").free
    print(f"Estimated DIV2K intermediates: up to {estimated_bytes / 1024 ** 3:.2f} GiB; "
          f"free space: {free_bytes / 1024 ** 3:.2f} GiB")
    if estimated_bytes > free_bytes * 0.8:
        raise OSError("Insufficient disk space for the requested DIV2K experiment.")


def prepare(limit=50, seed=DIV2K_SPLIT_SEED,
            validation_fraction=DIV2K_VALIDATION_FRACTION):
    """Discover, source-split, and preprocess DIV2K into isolated directories."""
    paths = discover_images(limit=limit, seed=seed)
    _estimate_storage(len(paths))
    manifest = make_split_manifest(paths, seed, validation_fraction)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
    for split in ("train", "validation"):
        print(f"Preprocessing DIV2K {split}: {len(manifest[split])} source images")
        for name in manifest[split]:
            if not preprocessing.preprocess_image(str(DATASET_DIR / name), split, output_root=str(OUTPUT_ROOT)):
                raise ValueError(f"Could not preprocess DIV2K image: {name}")
    return manifest


def _paths_for(split):
    manifest = _load_manifest()
    if split not in manifest:
        raise ValueError(f"Unknown DIV2K split: {split}")
    return [OUTPUT_ROOT / "processed" / split / f"{Path(name).stem}.npz"
            for name in manifest[split]]


def segment():
    """Apply the existing SLIC implementation to isolated DIV2K preprocessing."""
    for split in ("train", "validation"):
        output_dir = OUTPUT_ROOT / "segmentation" / split
        for npz_path in _paths_for(split):
            if not npz_path.is_file():
                raise FileNotFoundError(f"Missing DIV2K preprocessing: {npz_path}")
            segmentation.process_image(str(npz_path), str(output_dir))


def extract_features():
    """Extract the existing 103-dimensional FFT/luminance feature definition."""
    for split in ("train", "validation"):
        output_dir = OUTPUT_ROOT / "features" / split
        for npz_path in _paths_for(split):
            segmentation_path = OUTPUT_ROOT / "segmentation" / split / f"{npz_path.stem}_segments.npy"
            if not segmentation_path.is_file():
                raise FileNotFoundError(f"Missing DIV2K segmentation: {segmentation_path}")
            features.process_image(str(npz_path), str(segmentation_path),
                                   str(output_dir / f"{npz_path.stem}_features.npz"))


def load_split(split):
    """Load DIV2K feature rows and mean U/V targets via the baseline target helper."""
    feature_blocks, u_blocks, v_blocks = [], [], []
    for npz_path in _paths_for(split):
        feature_path = OUTPUT_ROOT / "features" / split / f"{npz_path.stem}_features.npz"
        segment_path = OUTPUT_ROOT / "segmentation" / split / f"{npz_path.stem}_segments.npy"
        if not feature_path.is_file() or not segment_path.is_file():
            raise FileNotFoundError(f"Missing DIV2K features or labels for {npz_path.stem}")
        feature_data = np.load(feature_path)
        processed = np.load(npz_path)
        segment_ids = feature_data["segment_ids"]
        segments = np.load(segment_path)
        feature_blocks.append(feature_data["features"])
        u_blocks.append(train.superpixel_means(processed["u"], segments, segment_ids))
        v_blocks.append(train.superpixel_means(processed["v"], segments, segment_ids))
    if not feature_blocks:
        raise FileNotFoundError(f"No usable DIV2K {split} feature files found.")
    return np.vstack(feature_blocks), np.concatenate(u_blocks), np.concatenate(v_blocks)


def rbf_memory_gib(sample_count):
    """Lower-bound memory for one dense float64 RBF kernel matrix."""
    return sample_count * sample_count * 8 / 1024 ** 3


def fit_candidate_models(features_array, target_u, target_v, model_dir=MODEL_DIR,
                         c=SVR_C, epsilon=SVR_EPSILON):
    """Fit unchanged scaler and RBF-SVR models, saving only to the candidate dir."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_array)
    settings = dict(C=c, epsilon=epsilon, kernel=SVR_KERNEL)
    model_u, model_v = SVR(**settings), SVR(**settings)
    model_u.fit(scaled, target_u)
    model_v.fit(scaled, target_v)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "feature_scaler.joblib")
    joblib.dump(model_u, model_dir / "svr_u.joblib")
    joblib.dump(model_v, model_dir / "svr_v.joblib")
    return scaler, model_u, model_v


def train_candidate(max_train_samples=DIV2K_MAX_TRAIN_SAMPLES, seed=DIV2K_SPLIT_SEED):
    """Train candidate SVRs with deterministic row sampling when needed."""
    if max_train_samples <= 0:
        raise ValueError("DIV2K maximum training samples must be positive.")
    features_array, target_u, target_v = load_split("train")
    original_count = len(features_array)
    print(f"DIV2K training rows: {original_count}; dense RBF lower bound: {rbf_memory_gib(original_count):.2f} GiB")
    if original_count > max_train_samples:
        indexes = np.random.default_rng(seed).choice(original_count, max_train_samples, replace=False)
        indexes.sort()
        features_array, target_u, target_v = features_array[indexes], target_u[indexes], target_v[indexes]
        print(f"Using deterministic RBF-SVR sample: {len(features_array)} rows; "
              f"lower bound: {rbf_memory_gib(len(features_array)):.2f} GiB")
    started = time.perf_counter()
    scaler, model_u, model_v = fit_candidate_models(features_array, target_u, target_v)
    elapsed = time.perf_counter() - started
    validation_features, validation_u, validation_v = load_split("validation")
    predicted_u = model_u.predict(scaler.transform(validation_features))
    predicted_v = model_v.predict(scaler.transform(validation_features))
    metrics = {
        "u_mae": float(mean_absolute_error(validation_u, predicted_u)),
        "v_mae": float(mean_absolute_error(validation_v, predicted_v)),
        "u_mse": float(mean_squared_error(validation_u, predicted_u)),
        "v_mse": float(mean_squared_error(validation_v, predicted_v)),
        "training_rows_available": original_count,
        "training_rows_used": len(features_array),
        "validation_rows": len(validation_features),
        "training_seconds": elapsed,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
    print("DIV2K validation superpixel metrics: " + ", ".join(
        f"{name}={value:.3f}" for name, value in metrics.items() if name.endswith(("mae", "mse"))))
    print(f"Candidate models saved to {MODEL_DIR}/ in {elapsed:.1f} seconds")
    return metrics


def _save_candidate_test_result(name, y, segments, segment_ids, predicted_u, predicted_v,
                                output_root=OUTPUT_ROOT):
    svr_dir = Path(output_root) / "test" / "svr_results"
    final_dir = Path(output_root) / "test" / "final"
    svr_u = colorize.values_to_image(predicted_u, segment_ids, segments)
    svr_v = colorize.values_to_image(predicted_v, segment_ids, segments)
    colorize.save_rgb(svr_dir / f"{name}_svr.png", colorize.yuv_to_rgb(y, svr_u, svr_v))
    refined_u = mrf.refine_chrominance(predicted_u, segment_ids, segments)
    refined_v = mrf.refine_chrominance(predicted_v, segment_ids, segments)
    final_rgb = mrf.yuv_to_rgb(y, mrf.values_to_image(refined_u, segment_ids, segments),
                               mrf.values_to_image(refined_v, segment_ids, segments))
    mrf.save_rgb(final_dir / f"{name}_final.png", final_rgb)
    return final_rgb


def evaluate_candidate(model_dir=MODEL_DIR, output_root=OUTPUT_ROOT,
                       candidate_label="div2k_svr_mrf"):
    """Evaluate candidate and baseline final outputs against untouched original tests."""
    model_dir, output_root = Path(model_dir), Path(output_root)
    scaler, model_u, model_v = colorize.load_models(model_dir)
    rows = []
    processed_dir = Path("outputs/processed/test")
    for npz_path in sorted(processed_dir.glob("*.npz")):
        name = npz_path.stem
        segments_path = Path("outputs/segmentation/test") / f"{name}_segments.npy"
        features_path = Path("outputs/features/test") / f"{name}_features.npz"
        candidate_rgb, _, _, predicted_u, predicted_v, segment_ids = colorize.colorize_image(
            npz_path, segments_path, features_path, scaler, model_u, model_v)
        data, segments = np.load(npz_path), np.load(segments_path)
        candidate_final = _save_candidate_test_result(name, data["y"], segments, segment_ids,
                                                      predicted_u, predicted_v, output_root)
        original = evaluation.original_test_rgb(name, "test")
        baseline = evaluation.method_images(name, "test")["svr_mrf"]
        for method, image in (("baseline_svr_mrf", baseline), (candidate_label, candidate_final)):
            if image.shape != original.shape:
                raise ValueError(f"Test output shape mismatch for {name} {method}.")
            rows.append({"image": name, "method": method, **evaluation.metrics(image, original)})
    averages = {method: {metric: float(np.mean([row[metric] for row in rows if row["method"] == method]))
                         for metric in ("mae", "mse", "rgb_rmse", "psnr")}
                for method in ("baseline_svr_mrf", candidate_label)}
    output = output_root / "evaluation" / "test_comparison_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=("image", "method", "mae", "mse", "rgb_rmse", "psnr"))
        writer.writeheader()
        writer.writerows(rows)
        for method, metrics in averages.items():
            writer.writerow({"image": "AVERAGE", "method": method, **metrics})
    for method, metrics in averages.items():
        print(f"{method}: " + ", ".join(f"{key}={value:.3f}" for key, value in metrics.items()))
    print(f"Saved untouched-test comparison to {output}")
    return rows, averages


def _sample_training_rows(features_array, target_u, target_v, max_train_samples, seed):
    """Apply the same deterministic cap used by candidate training."""
    if len(features_array) <= max_train_samples:
        return features_array, target_u, target_v
    indexes = np.random.default_rng(seed).choice(len(features_array), max_train_samples, replace=False)
    indexes.sort()
    return features_array[indexes], target_u[indexes], target_v[indexes]


def mixed_source_manifest():
    """Return provenance for sources allowed in mixed training and validation."""
    div2k_manifest = _load_manifest()
    original_train = sorted(path.name for path in Path("dataset/train").iterdir()
                            if path.is_file() and path.suffix.lower() in preprocessing.SUPPORTED_EXTENSIONS)
    original_validation = sorted(path.name for path in Path("dataset/validation").iterdir()
                                 if path.is_file() and path.suffix.lower() in preprocessing.SUPPORTED_EXTENSIONS)
    result = {
        "original_train": original_train,
        "div2k_train": list(div2k_manifest["train"]),
        "original_validation": original_validation,
    }
    source_ids = ([f"original:train/{name}" for name in result["original_train"]] +
                  [f"div2k:{name}" for name in result["div2k_train"]])
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate mixed-training source identifier.")
    return result


def load_mixed_training_data():
    """Concatenate original-train and DIV2K-train rows with provenance labels."""
    provenance = mixed_source_manifest()
    original_x, original_u, original_v = train.load_split("train")
    div2k_x, div2k_u, div2k_v = load_split("train")
    features_array = np.vstack((original_x, div2k_x))
    target_u = np.concatenate((original_u, div2k_u))
    target_v = np.concatenate((original_v, div2k_v))
    origins = np.concatenate((np.full(len(original_x), "original", dtype="U8"),
                              np.full(len(div2k_x), "div2k", dtype="U8")))
    if len(features_array) != len(target_u) or len(target_u) != len(origins):
        raise AssertionError("Mixed feature, target, and provenance lengths differ.")
    return features_array, target_u, target_v, origins, provenance


def stratified_sample_rows(features_array, target_u, target_v, origins,
                           max_train_samples, seed):
    """Sample capped mixed rows without replacement while retaining both sources."""
    if max_train_samples <= 0:
        raise ValueError("Maximum training samples must be positive.")
    if len(features_array) != len(origins):
        raise ValueError("Mixed provenance does not match feature rows.")
    if len(features_array) <= max_train_samples:
        indexes = np.arange(len(features_array))
    else:
        rng = np.random.default_rng(seed)
        selected = []
        for origin in sorted(np.unique(origins)):
            origin_indexes = np.flatnonzero(origins == origin)
            count = max(1, round(max_train_samples * len(origin_indexes) / len(origins)))
            count = min(count, len(origin_indexes))
            selected.append(rng.choice(origin_indexes, count, replace=False))
        indexes = np.concatenate(selected)
        # Rounding can differ by one or two rows; fill or trim deterministically.
        if len(indexes) < max_train_samples:
            remaining = np.setdiff1d(np.arange(len(features_array)), indexes, assume_unique=False)
            indexes = np.concatenate((indexes, rng.choice(remaining, max_train_samples - len(indexes), replace=False)))
        elif len(indexes) > max_train_samples:
            indexes = rng.choice(indexes, max_train_samples, replace=False)
        indexes.sort()
    if len(indexes) != len(np.unique(indexes)):
        raise AssertionError("Mixed sampling selected duplicate rows.")
    return (features_array[indexes], target_u[indexes], target_v[indexes], origins[indexes], indexes)


def tune_mixed_training(max_train_samples=DIV2K_MAX_TRAIN_SAMPLES,
                        seed=DIV2K_SPLIT_SEED):
    """Select mixed-data SVR settings on original validation, then test once."""
    mixed_x, mixed_u, mixed_v, origins, provenance = load_mixed_training_data()
    sampled_x, sampled_u, sampled_v, sampled_origins, indexes = stratified_sample_rows(
        mixed_x, mixed_u, mixed_v, origins, max_train_samples, seed)
    validation_x, validation_u, validation_v = train.load_split("validation")
    configurations = (
        {"name": "c0.125_e0.0625", "c": 0.125, "epsilon": 0.0625},
        {"name": "c0.5_e0.25", "c": 0.5, "epsilon": 0.25},
    )
    MIXED_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    composition = {
        **provenance,
        "original_rows_available": int((origins == "original").sum()),
        "div2k_rows_available": int((origins == "div2k").sum()),
        "original_rows_used": int((sampled_origins == "original").sum()),
        "div2k_rows_used": int((sampled_origins == "div2k").sum()),
        "rows_used": len(sampled_x),
        "validation_rows": len(validation_x),
        "sample_seed": seed,
        "sample_indexes_unique": bool(len(indexes) == len(np.unique(indexes))),
    }
    with (MIXED_OUTPUT_ROOT / "data_composition.json").open("w", encoding="utf-8") as file:
        json.dump(composition, file, indent=2, sort_keys=True)
    print(f"Mixed RBF-SVR: {len(sampled_x)} rows (original={composition['original_rows_used']}, "
          f"DIV2K={composition['div2k_rows_used']}); "
          f"dense-kernel lower bound: {rbf_memory_gib(len(sampled_x)):.2f} GiB")
    rows = []
    for configuration in configurations:
        started = time.perf_counter()
        model_dir = MIXED_MODEL_ROOT / configuration["name"]
        scaler, model_u, model_v = fit_candidate_models(
            sampled_x, sampled_u, sampled_v, model_dir,
            c=configuration["c"], epsilon=configuration["epsilon"])
        predicted_u = model_u.predict(scaler.transform(validation_x))
        predicted_v = model_v.predict(scaler.transform(validation_x))
        row = {
            **configuration,
            "u_mae": float(mean_absolute_error(validation_u, predicted_u)),
            "v_mae": float(mean_absolute_error(validation_v, predicted_v)),
            "u_mse": float(mean_squared_error(validation_u, predicted_u)),
            "v_mse": float(mean_squared_error(validation_v, predicted_v)),
            "mean_uv_mae": float((mean_absolute_error(validation_u, predicted_u) +
                                  mean_absolute_error(validation_v, predicted_v)) / 2),
            "training_seconds": time.perf_counter() - started,
            "training_rows_used": len(sampled_x),
            "validation_rows": len(validation_x),
        }
        rows.append(row)
        print(f"{row['name']}: original-validation mean_uv_mae={row['mean_uv_mae']:.3f}, "
              f"u_mae={row['u_mae']:.3f}, v_mae={row['v_mae']:.3f}, "
              f"seconds={row['training_seconds']:.1f}")
    selected = min(rows, key=lambda row: (row["mean_uv_mae"], row["name"]))
    with (MIXED_OUTPUT_ROOT / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (MIXED_OUTPUT_ROOT / "selected_config.json").open("w", encoding="utf-8") as file:
        json.dump(selected, file, indent=2, sort_keys=True)
    print(f"Selected on original validation only: {selected['name']}")
    return rows, selected, composition


def evaluate_mixed_selected():
    """Evaluate the selected mixed model with baseline and DIV2K-only references."""
    selected_path = MIXED_OUTPUT_ROOT / "selected_config.json"
    if not selected_path.is_file():
        raise FileNotFoundError("Run 'mixed-tune' before evaluating the selected mixed model.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    model_dir = MIXED_MODEL_ROOT / selected["name"]
    _, mixed_averages = evaluate_candidate(
        model_dir=model_dir, output_root=MIXED_OUTPUT_ROOT / "selected_test",
        candidate_label="mixed_div2k_svr_mrf")
    tuned_path = OUTPUT_ROOT / "hyperparameter_search" / "selected_test" / "evaluation" / "test_comparison_metrics.csv"
    if not tuned_path.is_file():
        raise FileNotFoundError("Missing DIV2K-only tuned comparison; run 'div2k-tune' first.")
    with tuned_path.open(newline="", encoding="utf-8") as file:
        tuned_rows = list(csv.DictReader(file))
    reference = next(row for row in tuned_rows
                     if row["image"] == "AVERAGE" and row["method"] == "div2k_tuned_svr_mrf")
    combined = {
        "baseline_svr_mrf": mixed_averages["baseline_svr_mrf"],
        "div2k_tuned_svr_mrf": {key: float(reference[key]) for key in ("mae", "mse", "rgb_rmse", "psnr")},
        "mixed_div2k_svr_mrf": mixed_averages["mixed_div2k_svr_mrf"],
    }
    output = MIXED_OUTPUT_ROOT / "three_way_test_metrics.json"
    output.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding="utf-8")
    for method, metrics in combined.items():
        print(method + ": " + ", ".join(f"{key}={value:.3f}" for key, value in metrics.items()))
    return combined


def tune_hyperparameters(max_train_samples=DIV2K_MAX_TRAIN_SAMPLES,
                         seed=DIV2K_SPLIT_SEED):
    """Select a small RBF-SVR grid using DIV2K validation only, then test once."""
    if max_train_samples <= 0:
        raise ValueError("DIV2K maximum training samples must be positive.")
    configurations = (
        {"name": "c0.125_e0.0625", "c": 0.125, "epsilon": 0.0625},
        {"name": "c0.5_e0.0625", "c": 0.5, "epsilon": 0.0625},
        {"name": "c1.0_e0.0625", "c": 1.0, "epsilon": 0.0625},
        {"name": "c0.5_e0.25", "c": 0.5, "epsilon": 0.25},
    )
    train_x, train_u, train_v = load_split("train")
    validation_x, validation_u, validation_v = load_split("validation")
    sampled_x, sampled_u, sampled_v = _sample_training_rows(
        train_x, train_u, train_v, max_train_samples, seed)
    search_root = OUTPUT_ROOT / "hyperparameter_search"
    models_root = MODEL_DIR / "hyperparameter_search"
    search_root.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"RBF-SVR grid: {len(sampled_x)} training rows, "
          f"{rbf_memory_gib(len(sampled_x)):.2f} GiB dense-kernel lower bound per fit")
    for configuration in configurations:
        started = time.perf_counter()
        model_dir = models_root / configuration["name"]
        scaler, model_u, model_v = fit_candidate_models(
            sampled_x, sampled_u, sampled_v, model_dir,
            c=configuration["c"], epsilon=configuration["epsilon"])
        predicted_u = model_u.predict(scaler.transform(validation_x))
        predicted_v = model_v.predict(scaler.transform(validation_x))
        row = {
            **configuration,
            "u_mae": float(mean_absolute_error(validation_u, predicted_u)),
            "v_mae": float(mean_absolute_error(validation_v, predicted_v)),
            "u_mse": float(mean_squared_error(validation_u, predicted_u)),
            "v_mse": float(mean_squared_error(validation_v, predicted_v)),
            "mean_uv_mae": float((mean_absolute_error(validation_u, predicted_u) +
                                  mean_absolute_error(validation_v, predicted_v)) / 2),
            "training_seconds": time.perf_counter() - started,
            "training_rows_used": len(sampled_x),
            "validation_rows": len(validation_x),
        }
        rows.append(row)
        print(f"{row['name']}: mean_uv_mae={row['mean_uv_mae']:.3f}, "
              f"u_mae={row['u_mae']:.3f}, v_mae={row['v_mae']:.3f}, "
              f"seconds={row['training_seconds']:.1f}")
    selected = min(rows, key=lambda row: (row["mean_uv_mae"], row["name"]))
    with (search_root / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (search_root / "selected_config.json").open("w", encoding="utf-8") as file:
        json.dump(selected, file, indent=2, sort_keys=True)
    print(f"Selected on DIV2K validation only: {selected['name']}")
    test_root = search_root / "selected_test"
    _, test_averages = evaluate_candidate(
        model_dir=models_root / selected["name"], output_root=test_root,
        candidate_label="div2k_tuned_svr_mrf")
    return rows, selected, test_averages
