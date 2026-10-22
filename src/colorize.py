"""Test-time SVR-only colorization using saved superpixel features."""

from pathlib import Path

import cv2
import joblib
import numpy as np

from src import features, mrf, segmentation
from src.config import COMPACTNESS, ICM_ITERATIONS, MRF_GAMMA
from src.preprocessing import SUPPORTED_EXTENSIONS, resize_image, rgb_to_yuv

def yuv_to_rgb(y, u, v):
    """Convert OpenCV-style uint8 YUV channel arrays back to RGB."""
    yuv = np.dstack((y, np.clip(u, 0, 255), np.clip(v, 0, 255))).astype(np.uint8)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def load_models(model_dir="models"):
    model_dir = Path(model_dir)
    required = ("feature_scaler.joblib", "svr_u.joblib", "svr_v.joblib")
    missing = [str(model_dir / name) for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing trained model file(s): " + ", ".join(missing))
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
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not save image: {path}")


def load_live_luminance(input_path):
    """Load a runtime image and return the resized uint8 Y channel.

    Both single-channel files and RGB/BGR files are supported.  For colour
    files (including visually grayscale RGB files), Y is calculated through
    the same OpenCV RGB-to-YUV conversion used by training preprocessing.
    """
    input_path = Path(input_path).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image does not exist or is not a file: {input_path}")
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(extension.upper().lstrip(".") for extension in SUPPORTED_EXTENSIONS)
        raise ValueError(f"Unsupported image format '{input_path.suffix}'. Supported formats: {supported}.")

    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError(f"Could not decode image (it may be corrupt): {input_path}")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.integer):
            max_value = np.iinfo(image.dtype).max
            image = cv2.convertScaleAbs(image, alpha=255.0 / max_value)
        else:
            image = np.clip(image * 255 if image.max() <= 1 else image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        y = resize_image(image)
    elif image.ndim == 3 and image.shape[2] in (3, 4):
        conversion = cv2.COLOR_BGR2RGB if image.shape[2] == 3 else cv2.COLOR_BGRA2RGB
        image_rgb = resize_image(cv2.cvtColor(image, conversion))
        y, _, _ = rgb_to_yuv(image_rgb)
    else:
        raise ValueError(f"Unsupported image dimensions {image.shape}; expected grayscale, BGR, or BGRA image.")

    if y.ndim != 2 or y.size == 0:
        raise ValueError("Input image did not produce a valid 2D luminance channel.")
    return y


def _validate_live_result(y, feature_matrix, scaler, predicted_u, predicted_v, image_rgb):
    """Fail early if runtime data is incompatible with the trained pipeline."""
    if y.ndim != 2 or y.size == 0:
        raise ValueError("Runtime luminance must be a non-empty 2D image.")
    expected_features = getattr(scaler, "n_features_in_", None)
    if expected_features is None and hasattr(scaler, "mean_"):
        expected_features = len(scaler.mean_)
    if feature_matrix.ndim != 2 or feature_matrix.shape[0] == 0:
        raise ValueError("Feature extraction produced no superpixel features.")
    if expected_features is not None and feature_matrix.shape[1] != expected_features:
        raise ValueError(
            f"Feature count mismatch: extracted {feature_matrix.shape[1]}, "
            f"but the trained scaler expects {expected_features}."
        )
    if not np.isfinite(predicted_u).all() or not np.isfinite(predicted_v).all():
        raise ValueError("SVR produced non-finite chrominance predictions.")
    if image_rgb.shape != (*y.shape, 3) or image_rgb.dtype != np.uint8:
        raise ValueError(f"Invalid RGB output shape or type: {image_rgb.shape}, {image_rgb.dtype}.")


def _validate_superpixel_arrays(u_values, v_values, segment_ids, segments, stage):
    """Validate finite per-superpixel chrominance values before saving them."""
    if u_values.ndim != 1 or v_values.ndim != 1 or segment_ids.ndim != 1:
        raise ValueError(f"{stage} chrominance values and segment IDs must be one-dimensional.")
    if len(u_values) != len(v_values) or len(u_values) != len(segment_ids):
        raise ValueError(f"{stage} chrominance values do not match the segment ID count.")
    if set(segment_ids.tolist()) != set(np.unique(segments).tolist()):
        raise ValueError(f"{stage} segment IDs do not match the SLIC labels.")
    if not np.isfinite(u_values).all() or not np.isfinite(v_values).all():
        raise ValueError(f"{stage} chrominance values contain non-finite values.")


def _print_chrominance_stats(stage, u_values, v_values):
    """Print compact diagnostics for raw superpixel chrominance predictions."""
    for channel, values in (("U", u_values), ("V", v_values)):
        print(
            f"{stage} {channel}: min={values.min():.3f}, max={values.max():.3f}, "
            f"mean={values.mean():.3f}, std={values.std():.3f}"
        )


def colorize_live_image(input_path, model_dir="models", output_root="outputs/live",
                        compactness=None, gamma=None):
    """Colorize a new grayscale image using saved models and configured MRF/ICM.

    This inference-only path never reads or requires ground-truth U/V values.
    It returns paths for the input preview, SLIC view, SVR output, and refined
    estimated colourization.
    """
    if compactness is None:
        compactness = COMPACTNESS
    if gamma is None:
        gamma = MRF_GAMMA

    input_path = Path(input_path).expanduser()
    y = load_live_luminance(input_path)
    output_dir = Path(output_root) / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating SLIC superpixels (compactness={compactness})...")
    segments = segmentation.segment_image(y, compactness=compactness)
    slic_path = output_dir / "slic_superpixels.png"
    segmentation.save_segmentation_visualization(y, segments, str(slic_path))

    print("Extracting FFT and luminance features...")
    feature_matrix, segment_ids = features.extract_fft_features(y, segments)
    scaler, model_u, model_v = load_models(model_dir)
    _validate_live_result(y, feature_matrix, scaler, np.array([0.0]), np.array([0.0]),
                          cv2.cvtColor(y, cv2.COLOR_GRAY2RGB))

    print("Predicting U/V chrominance with the saved SVR models...")
    scaled_features = scaler.transform(feature_matrix)
    predicted_u = model_u.predict(scaled_features)
    predicted_v = model_v.predict(scaled_features)
    _validate_superpixel_arrays(predicted_u, predicted_v, segment_ids, segments, "SVR")
    _print_chrominance_stats("SVR", predicted_u, predicted_v)
    svr_u = values_to_image(predicted_u, segment_ids, segments)
    svr_v = values_to_image(predicted_v, segment_ids, segments)
    svr_rgb = yuv_to_rgb(y, svr_u, svr_v)
    _validate_live_result(y, feature_matrix, scaler, predicted_u, predicted_v, svr_rgb)

    print(f"Refining chrominance with MRF/ICM ({ICM_ITERATIONS} iterations, gamma={gamma})...")
    refined_u = mrf.refine_chrominance(predicted_u, segment_ids, segments, gamma=gamma)
    refined_v = mrf.refine_chrominance(predicted_v, segment_ids, segments, gamma=gamma)
    _validate_superpixel_arrays(refined_u, refined_v, segment_ids, segments, "MRF")
    _print_chrominance_stats("MRF", refined_u, refined_v)
    final_rgb = mrf.yuv_to_rgb(
        y,
        mrf.values_to_image(refined_u, segment_ids, segments),
        mrf.values_to_image(refined_v, segment_ids, segments),
    )
    _validate_live_result(y, feature_matrix, scaler, refined_u, refined_v, final_rgb)

    preview_path = output_dir / "input_preview.png"
    svr_path = output_dir / "svr_only_colorization.png"
    final_path = output_dir / "mrf_refined_colorization.png"
    predicted_uv_path = output_dir / "predicted_uv.npz"
    refined_uv_path = output_dir / "refined_uv.npz"
    save_rgb(preview_path, cv2.cvtColor(y, cv2.COLOR_GRAY2RGB))
    save_rgb(svr_path, svr_rgb)
    mrf.save_rgb(final_path, final_rgb)
    np.savez_compressed(predicted_uv_path, u=predicted_u, v=predicted_v,
                        segment_ids=segment_ids)
    np.savez_compressed(refined_uv_path, u=refined_u, v=refined_v,
                        segment_ids=segment_ids, compactness=compactness, gamma=gamma)
    for path in (preview_path, slic_path, svr_path, final_path,
                 predicted_uv_path, refined_uv_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise OSError(f"Expected output was not saved: {path}")

    return {
        "input_preview": preview_path,
        "slic_visualization": slic_path,
        "svr_only": svr_path,
        "mrf_refined": final_path,
        "predicted_uv": predicted_uv_path,
        "refined_uv": refined_uv_path,
    }


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
