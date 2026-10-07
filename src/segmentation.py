import os
import cv2
import numpy as np

from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float

from src.config import N_SEGMENTS, COMPACTNESS


def segment_image(y_channel):
    """
    Apply SLIC superpixel segmentation to the luminance channel.
    """

    # Convert grayscale image to floating-point format
    image_float = img_as_float(y_channel)

    segments = slic(
    image_float,
    n_segments=N_SEGMENTS,
    compactness=COMPACTNESS,
    start_label=0,
    channel_axis=None
)
    return segments


def save_segmentation_visualization(
    y_channel,
    segments,
    output_path
):
    """
    Save an image showing SLIC superpixel boundaries.
    """

    # Normalize grayscale image to 0-1
    normalized = cv2.normalize(
        y_channel,
        None,
        0,
        1,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_32F
    )

    # Mark superpixel boundaries
    visualization = mark_boundaries(
        normalized,
        segments,
        color=(1, 1, 0)
    )

    # Convert to uint8
    visualization = (
        visualization * 255
    ).astype(np.uint8)

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )


def process_image(npz_path, output_dir):
    """
    Segment one preprocessed image.
    """

    data = np.load(npz_path)

    y_channel = data["y"]

    segments = segment_image(y_channel)

    filename = os.path.splitext(
        os.path.basename(npz_path)
    )[0]

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # Save segmentation labels
    np.save(
        os.path.join(
            output_dir,
            f"{filename}_segments.npy"
        ),
        segments
    )

    # Save visualization
    visualization_path = os.path.join(
        output_dir,
        f"{filename}_slic.png"
    )

    save_segmentation_visualization(
        y_channel,
        segments,
        visualization_path
    )

    number_of_segments = len(
        np.unique(segments)
    )

    print(
        f"{filename}: "
        f"{number_of_segments} superpixels"
    )


def process_dataset():
    """
    Apply SLIC to all preprocessed images.
    """

    for split in [
        "train",
        "validation",
        "test"
    ]:

        input_dir = os.path.join(
            "outputs",
            "processed",
            split
        )

        output_dir = os.path.join(
            "outputs",
            "segmentation",
            split
        )

        if not os.path.exists(input_dir):
            continue

        files = [
            f for f in os.listdir(input_dir)
            if f.endswith(".npz")
        ]

        print(
            f"\nProcessing {split} segmentation..."
        )

        for filename in sorted(files):

            process_image(
                os.path.join(
                    input_dir,
                    filename
                ),
                output_dir
            )


if __name__ == "__main__":
    process_dataset()