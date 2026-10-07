"""Train the two chrominance SVR regressors."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.config import MODEL_DIR, SVR_C, SVR_EPSILON, SVR_KERNEL


def superpixel_means(channel, segments, segment_ids):
    """Return one mean target value for each feature-row superpixel."""
    return np.asarray(
        [channel[segments == segment_id].mean() for segment_id in segment_ids],
        dtype=np.float32,
    )


def load_split(split):
    """Load feature rows and their U/V superpixel targets for one split."""
    feature_dir = Path("outputs/features") / split
    processed_dir = Path("outputs/processed") / split
    segmentation_dir = Path("outputs/segmentation") / split
    feature_blocks, u_blocks, v_blocks = [], [], []

    for feature_path in sorted(feature_dir.glob("*_features.npz")):
        stem = feature_path.name.removesuffix("_features.npz")
        image_path = processed_dir / f"{stem}.npz"
        segment_path = segmentation_dir / f"{stem}_segments.npy"
        if not image_path.exists() or not segment_path.exists():
            print(f"WARNING: skipping {feature_path.name}; matching data is missing")
            continue

        feature_data = np.load(feature_path)
        image_data = np.load(image_path)
        segment_ids = feature_data["segment_ids"]
        segments = np.load(segment_path)
        feature_blocks.append(feature_data["features"])
        u_blocks.append(superpixel_means(image_data["u"], segments, segment_ids))
        v_blocks.append(superpixel_means(image_data["v"], segments, segment_ids))

    if not feature_blocks:
        raise FileNotFoundError(f"No usable feature files found for '{split}'.")
    return np.vstack(feature_blocks), np.concatenate(u_blocks), np.concatenate(v_blocks)


def evaluate_models(scaler, model_u, model_v, split="validation"):
    """Report superpixel-level validation errors in YUV chrominance values."""
    features, target_u, target_v = load_split(split)
    scaled = scaler.transform(features)
    predicted_u, predicted_v = model_u.predict(scaled), model_v.predict(scaled)
    metrics = {
        "u_mae": mean_absolute_error(target_u, predicted_u),
        "v_mae": mean_absolute_error(target_v, predicted_v),
        "u_mse": mean_squared_error(target_u, predicted_u),
        "v_mse": mean_squared_error(target_v, predicted_v),
    }
    print(f"{split.title()} superpixel metrics: " + ", ".join(
        f"{name}={value:.3f}" for name, value in metrics.items()))
    return metrics


def train_models(validate=True):
    """Fit scaler and separate U/V SVRs using training data only."""
    features, target_u, target_v = load_split("train")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    settings = dict(C=SVR_C, epsilon=SVR_EPSILON, kernel=SVR_KERNEL)
    model_u, model_v = SVR(**settings), SVR(**settings)
    model_u.fit(scaled_features, target_u)
    model_v.fit(scaled_features, target_v)

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, model_dir / "feature_scaler.joblib")
    joblib.dump(model_u, model_dir / "svr_u.joblib")
    joblib.dump(model_v, model_dir / "svr_v.joblib")
    print(f"Trained on {len(features)} superpixels; saved models to {model_dir}/")

    metrics = None
    if validate and (Path("outputs/features/validation").exists()):
        metrics = evaluate_models(scaler, model_u, model_v)
    return scaler, model_u, model_v, metrics


if __name__ == "__main__":
    train_models()
