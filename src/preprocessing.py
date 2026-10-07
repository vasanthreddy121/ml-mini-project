import os
import cv2
import numpy as np

from src.config import IMAGE_WIDTH


SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def resize_image(image):
    """Resize image to fixed width while preserving aspect ratio."""
    height, width = image.shape[:2]

    new_height = int(height * IMAGE_WIDTH / width)

    return cv2.resize(
        image,
        (IMAGE_WIDTH, new_height),
        interpolation=cv2.INTER_AREA
    )


def rgb_to_yuv(image_rgb):
    """Convert RGB image to YUV and return Y, U, V channels."""
    image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)

    y = image_yuv[:, :, 0]
    u = image_yuv[:, :, 1]
    v = image_yuv[:, :, 2]

    return y, u, v


def preprocess_image(image_path, split):
    """Preprocess one image."""

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print(f"WARNING: Could not read {image_path}")
        return False

    # OpenCV loads BGR → convert to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize
    image_rgb = resize_image(image_rgb)

    # RGB → YUV
    y, u, v = rgb_to_yuv(image_rgb)

    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    # Save grayscale Y channel
    grayscale_dir = os.path.join(
        "outputs",
        "grayscale",
        split
    )

    os.makedirs(grayscale_dir, exist_ok=True)

    grayscale_path = os.path.join(
        grayscale_dir,
        f"{filename}_gray.png"
    )

    cv2.imwrite(grayscale_path, y)

    # Save processed Y, U, V channels
    processed_dir = os.path.join(
        "outputs",
        "processed",
        split
    )

    os.makedirs(processed_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(
            processed_dir,
            f"{filename}.npz"
        ),
        y=y,
        u=u,
        v=v
    )

    return True


def process_dataset():
    """Process train, validation and test datasets."""

    splits = ["train", "validation", "test"]

    total = 0
    successful = 0

    for split in splits:

        input_dir = os.path.join(
            "dataset",
            split
        )

        if not os.path.exists(input_dir):
            print(f"WARNING: {input_dir} does not exist")
            continue

        files = sorted(os.listdir(input_dir))

        print(f"\nProcessing {split} dataset...")

        for filename in files:

            if not filename.lower().endswith(
                SUPPORTED_EXTENSIONS
            ):
                continue

            total += 1

            image_path = os.path.join(
                input_dir,
                filename
            )

            if preprocess_image(
                image_path,
                split
            ):
                successful += 1
                print(f"  ✓ {filename}")

    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"Images found     : {total}")
    print(f"Images processed : {successful}")
    print(f"Failed           : {total - successful}")


if __name__ == "__main__":
    process_dataset()
