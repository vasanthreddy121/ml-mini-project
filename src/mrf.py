"""Small continuous MRF/ICM smoother for SVR superpixel chrominance predictions."""

from pathlib import Path

import cv2
import numpy as np

from src.config import ICM_ITERATIONS, MRF_GAMMA


def superpixel_adjacency(segments):
    """Find labels that share a horizontal or vertical pixel boundary."""
    labels = np.unique(segments)
    neighbours = {int(label): set() for label in labels}
    for first, second in ((segments[:, :-1], segments[:, 1:]),
                          (segments[:-1, :], segments[1:, :])):
        changed = first != second
        for left, right in zip(first[changed], second[changed]):
            left, right = int(left), int(right)
            neighbours[left].add(right)
            neighbours[right].add(left)
    return neighbours


def refine_chrominance(predicted, segment_ids, segments, gamma=MRF_GAMMA,
                       iterations=ICM_ITERATIONS):
    """Minimize E=sum_i(x_i-p_i)^2 + gamma*sum_(i,j)(x_i-x_j)^2 by ICM.

    Each coordinate update has the closed form
    x_i = (p_i + gamma * sum(neighbour values)) / (1 + gamma * degree).
    """
    original = {int(segment): float(value) for segment, value in zip(segment_ids, predicted)}
    values = original.copy()
    adjacency = superpixel_adjacency(segments)
    for _ in range(iterations):
        for segment_id in segment_ids:
            segment_id = int(segment_id)
            neighbours = [n for n in adjacency[segment_id] if n in values]
            if neighbours:
                values[segment_id] = (original[segment_id] + gamma * sum(values[n] for n in neighbours)) / (1 + gamma * len(neighbours))
    return np.asarray([values[int(segment)] for segment in segment_ids], dtype=np.float32)


def values_to_image(values, segment_ids, segments):
    """Broadcast one chrominance value to every pixel in its superpixel."""
    result = np.empty(segments.shape, dtype=np.float32)
    for value, segment_id in zip(values, segment_ids):
        result[segments == segment_id] = value
    return result


def yuv_to_rgb(y, u, v):
    """Convert channels using the same OpenCV YUV convention as preprocessing."""
    yuv = np.dstack((y, np.clip(u, 0, 255), np.clip(v, 0, 255))).astype(np.uint8)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def save_rgb(path, image_rgb):
    """Save an RGB image through OpenCV, which writes BGR arrays."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def refine_image(npz_path, segments_path, predictions_path):
    """Refine saved SVR U/V superpixel predictions without reading true U or V."""
    y = np.load(npz_path)["y"]
    segments = np.load(segments_path)
    predictions = np.load(predictions_path)
    segment_ids = predictions["segment_ids"]
    predicted_u = predictions["predicted_u"]
    predicted_v = predictions["predicted_v"]

    if set(segment_ids.tolist()) != set(np.unique(segments).tolist()):
        raise ValueError(f"Prediction segment IDs do not match labels: {predictions_path}")
    if len(segment_ids) != len(predicted_u) or len(segment_ids) != len(predicted_v):
        raise ValueError(f"Invalid SVR prediction structure: {predictions_path}")

    refined_u = refine_chrominance(predicted_u, segment_ids, segments)
    refined_v = refine_chrominance(predicted_v, segment_ids, segments)
    u_image = values_to_image(refined_u, segment_ids, segments)
    v_image = values_to_image(refined_v, segment_ids, segments)
    return yuv_to_rgb(y, u_image, v_image), u_image, v_image, refined_u, refined_v, segment_ids


def refine_split(split="test"):
    """Run the configured MRF/ICM refinement for the saved SVR results of a split."""
    processed_dir = Path("outputs/processed") / split
    mrf_dir = Path("outputs/mrf_results") / split
    final_dir = Path("outputs/final") / split

    for npz_path in sorted(processed_dir.glob("*.npz")):
        name = npz_path.stem
        segments_path = Path("outputs/segmentation") / split / f"{name}_segments.npy"
        predictions_path = Path("outputs/svr_results") / split / f"{name}_predicted_uv.npz"
        if not segments_path.exists() or not predictions_path.exists():
            print(f"WARNING: missing labels or SVR predictions for {npz_path.name}")
            continue

        rgb, u_image, v_image, refined_u, refined_v, segment_ids = refine_image(
            npz_path, segments_path, predictions_path
        )
        mrf_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            mrf_dir / f"{name}_refined_uv.npz",
            u=u_image,
            v=v_image,
            segment_ids=segment_ids,
            refined_u=refined_u,
            refined_v=refined_v,
            gamma=MRF_GAMMA,
            iterations=ICM_ITERATIONS,
        )
        save_rgb(final_dir / f"{name}_final.png", rgb)
        print(f"Refined {split}/{npz_path.name} with {ICM_ITERATIONS} ICM iterations")


if __name__ == "__main__":
    refine_split()
