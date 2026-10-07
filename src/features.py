import os
import numpy as np

from src.config import PATCH_SIZE


def get_centroid(segment_mask):
    """
    Calculate the centroid (row, column) of a superpixel.
    """
    coordinates = np.argwhere(segment_mask)

    if len(coordinates) == 0:
        return None

    row_mean, col_mean = coordinates.mean(axis=0)

    return int(round(row_mean)), int(round(col_mean))


def extract_patch(image, center_row, center_col, patch_size=PATCH_SIZE):
    """
    Extract a square patch centered around the given point.

    If the patch reaches the image boundary, reflection padding is used.
    """

    if image.ndim != 2:
        raise ValueError("Feature extraction expects a 2D Y channel.")
    if patch_size <= 0 or patch_size % 2:
        raise ValueError("PATCH_SIZE must be a positive even number.")

    half = patch_size // 2

    padded_image = np.pad(
        image,
        ((half, half), (half, half)),
        mode="reflect"
    )

    center_row += half
    center_col += half

    patch = padded_image[
        center_row - half:center_row + half,
        center_col - half:center_col + half
    ]

    return patch


def extract_fft_features(y_channel, segments):
    """
    Extract FFT-based features for every superpixel.

    For each superpixel:
        1. Find centroid
        2. Extract 10x10 luminance patch
        3. Compute 2D FFT
        4. Use FFT magnitude as features
        5. Add local luminance statistics

    Returns:
        features      -> feature matrix
        segment_ids   -> corresponding superpixel IDs
    """

    features = []
    segment_ids = []

    unique_segments = np.unique(segments)

    for segment_id in unique_segments:

        mask = segments == segment_id

        centroid = get_centroid(mask)

        if centroid is None:
            continue

        row, col = centroid

        patch = extract_patch(
            y_channel,
            row,
            col
        )

        # 2D Fast Fourier Transform
        fft_result = np.fft.fft2(patch)

        # Shift low frequencies to the center
        fft_shifted = np.fft.fftshift(fft_result)

        # Magnitude of FFT
        fft_magnitude = np.abs(fft_shifted)

        # Normalize FFT magnitude
        fft_magnitude = np.log1p(fft_magnitude)

        # Flatten into feature vector
        fft_features = fft_magnitude.flatten()

        # Additional local luminance statistics
        mean_intensity = np.mean(patch)
        std_intensity = np.std(patch)
        contrast = np.max(patch) - np.min(patch)

        local_features = np.array([
            mean_intensity,
            std_intensity,
            contrast
        ])

        # Combine FFT + local statistics
        feature_vector = np.concatenate(
            [fft_features, local_features]
        )

        features.append(feature_vector)
        segment_ids.append(segment_id)

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(segment_ids, dtype=np.int32),
    )


def process_image(npz_path, segmentation_path, output_path):

    data = np.load(npz_path)

    y_channel = data["y"]

    segments = np.load(segmentation_path)

    features, segment_ids = extract_fft_features(
        y_channel,
        segments
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    np.savez_compressed(
        output_path,
        features=features,
        segment_ids=segment_ids
    )

    print(
        f"{os.path.basename(npz_path)}: "
        f"{features.shape[0]} superpixels, "
        f"{features.shape[1]} features"
    )


def process_dataset():

    for split in ["train", "validation", "test"]:

        processed_dir = os.path.join(
            "outputs",
            "processed",
            split
        )

        segmentation_dir = os.path.join(
            "outputs",
            "segmentation",
            split
        )

        feature_dir = os.path.join(
            "outputs",
            "features",
            split
        )

        if not os.path.exists(processed_dir):
            continue

        os.makedirs(
            feature_dir,
            exist_ok=True
        )

        files = [
            f for f in os.listdir(processed_dir)
            if f.endswith(".npz")
        ]

        print(
            f"\nExtracting FFT features "
            f"from {split} dataset..."
        )

        for filename in sorted(files):

            npz_path = os.path.join(
                processed_dir,
                filename
            )

            base_name = os.path.splitext(filename)[0]

            segmentation_path = os.path.join(
                segmentation_dir,
                f"{base_name}_segments.npy"
            )

            output_path = os.path.join(
                feature_dir,
                f"{base_name}_features.npz"
            )

            if not os.path.exists(segmentation_path):

                print(
                    f"WARNING: Missing segmentation "
                    f"for {filename}"
                )

                continue

            process_image(
                npz_path,
                segmentation_path,
                output_path
            )


if __name__ == "__main__":
    process_dataset()
