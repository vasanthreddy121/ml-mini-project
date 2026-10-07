"""Test-time SVR-only colorization using saved superpixel features."""

from pathlib import Path

import cv2
import joblib
import numpy as np

def yuv_to_rgb(y, u, v):
    """Convert OpenCV-style uint8 YUV channel arrays back to RGB."""
    yuv = np.dstack((y, np.clip(u, 0, 255), np.clip(v, 0, 255))).astype(np.uint8)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def load_models(model_dir="models"):
    model_dir = Path(model_dir)
    return (joblib.load(model_dir / "feature_scaler.joblib"),
            joblib.load(model_dir / "svr_u.joblib"),
            joblib.load(model_dir / "svr_v.joblib"))


def values_to_image(values, segment_ids, segments):
    """Assign one predicted chrominance value to every pixel in a superpixel."""
    image = np.empty(segments.shape, dtype=np.float32)
    for value, segment_id in zip(values, segment_ids):
        image[segments == segment_id] = value
    return image


def colorize_image(npz_path, segments_path, features_path, scaler, model_u, model_v):
    """Predict U/V from saved features and combine them with the input Y channel.

    The preprocessed archive provides only Y. Ground-truth U and V are never read
    or used during this test-time inference step.
    """
    data = np.load(npz_path)
    y = data["y"]
    segments = np.load(segments_path)
    feature_data = np.load(features_path)
    features = feature_data["features"]
    segment_ids = feature_data["segment_ids"]
    if features.ndim != 2 or len(features) != len(segment_ids):
        raise ValueError(f"Invalid feature structure: {features_path}")
    if set(segment_ids.tolist()) != set(np.unique(segments).tolist()):
        raise ValueError(f"Feature segment IDs do not match labels: {features_path}")
    transformed = scaler.transform(features)
    predicted_u = model_u.predict(transformed)
    predicted_v = model_v.predict(transformed)
    u_image = values_to_image(predicted_u, segment_ids, segments)
    v_image = values_to_image(predicted_v, segment_ids, segments)
    return yuv_to_rgb(y, u_image, v_image), u_image, v_image, predicted_u, predicted_v, segment_ids


def save_rgb(path, image_rgb):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def colorize_split(split="test", model_dir="models"):
    """Write SVR-only colorizations for one split; MRF is not applied."""
    scaler, model_u, model_v = load_models(model_dir)
    processed_dir = Path("outputs/processed") / split
    output_dir = Path("outputs/svr_results") / split
    for npz_path in sorted(processed_dir.glob("*.npz")):
        segments_path = Path("outputs/segmentation") / split / f"{npz_path.stem}_segments.npy"
        features_path = Path("outputs/features") / split / f"{npz_path.stem}_features.npz"
        if not segments_path.exists() or not features_path.exists():
            print(f"WARNING: missing labels or features for {npz_path.name}")
            continue
        svr_rgb, u_image, v_image, predicted_u, predicted_v, segment_ids = colorize_image(
            npz_path, segments_path, features_path, scaler, model_u, model_v
        )
        save_rgb(output_dir / f"{npz_path.stem}_svr.png", svr_rgb)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / f"{npz_path.stem}_predicted_uv.npz",
            u=u_image,
            v=v_image,
            segment_ids=segment_ids,
            predicted_u=predicted_u,
            predicted_v=predicted_v,
        )
        print(f"Colorized {split}/{npz_path.name}")


if __name__ == "__main__":
    colorize_split()
