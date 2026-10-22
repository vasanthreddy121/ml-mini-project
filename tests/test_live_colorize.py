"""Lightweight integration check for runtime colorization."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.colorize import colorize_live_image


class LiveColorizeTest(unittest.TestCase):
    def test_rgb_grayscale_input_writes_all_live_outputs(self):
        source = cv2.imread("dataset/test/l1.jpg", cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(source)
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_dir = Path(temporary_dir)
            input_path = temporary_dir / "runtime_gray.png"
            # Make an RGB grayscale file to exercise the RGB input path.
            self.assertTrue(cv2.imwrite(str(input_path), cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)))
            outputs = colorize_live_image(input_path, output_root=temporary_dir / "live")
            for output_name in ("input_preview", "slic_visualization", "svr_only", "mrf_refined"):
                path = outputs[output_name]
                saved = cv2.imread(str(path), cv2.IMREAD_COLOR)
                self.assertIsNotNone(saved)
                self.assertGreater(saved.size, 0)
            result = cv2.imread(str(outputs["mrf_refined"]), cv2.IMREAD_COLOR)
            self.assertEqual(result.shape[2], 3)
            for output_name in ("predicted_uv", "refined_uv"):
                with np.load(outputs[output_name]) as diagnostics:
                    self.assertEqual(diagnostics["u"].ndim, 1)
                    self.assertEqual(diagnostics["v"].ndim, 1)
                    self.assertEqual(len(diagnostics["u"]), len(diagnostics["segment_ids"]))
                    self.assertTrue(np.isfinite(diagnostics["u"]).all())
                    self.assertTrue(np.isfinite(diagnostics["v"]).all())

    def test_values_to_image_alignment_and_no_transposition(self):
        """Verify values_to_image maps segment IDs exactly to mask without transposition."""
        from src.colorize import values_to_image
        # Create non-symmetric grid: 3 rows x 5 columns
        segments = np.zeros((30, 50), dtype=int)
        segments[:15, :25] = 10
        segments[:15, 25:] = 20
        segments[15:, :25] = 30
        segments[15:, 25:] = 40
        segment_ids = np.array([10, 20, 30, 40])
        values = np.array([111.0, 122.0, 133.0, 144.0], dtype=np.float32)

        mapped = values_to_image(values, segment_ids, segments)
        self.assertEqual(mapped.shape, (30, 50))
        self.assertTrue(np.all(mapped[:15, :25] == 111.0))
        self.assertTrue(np.all(mapped[:15, 25:] == 122.0))
        self.assertTrue(np.all(mapped[15:, :25] == 133.0))
        self.assertTrue(np.all(mapped[15:, 25:] == 144.0))

    def test_yuv_rgb_channel_order_and_clipping(self):
        """Verify YUV-to-RGB conversion preserves channel order and clips without overflow."""
        from src.colorize import yuv_to_rgb
        from src.preprocessing import rgb_to_yuv
        y = np.full((10, 10), 128, dtype=np.uint8)
        # Test extreme out-of-range values
        u = np.full((10, 10), 300.0, dtype=np.float32)  # Should clip to 255
        v = np.full((10, 10), -50.0, dtype=np.float32)  # Should clip to 0
        rgb = yuv_to_rgb(y, u, v)
        self.assertEqual(rgb.shape, (10, 10, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        # Verify roundtrip on known color
        ref_rgb = np.array([[[180, 140, 100]]], dtype=np.uint8)
        y_val, u_val, v_val = rgb_to_yuv(ref_rgb)
        recon_rgb = yuv_to_rgb(y_val, u_val, v_val)
        self.assertLessEqual(np.max(np.abs(ref_rgb.astype(int) - recon_rgb.astype(int))), 1)

    def test_live_colorize_with_custom_compactness_and_gamma(self):
        """Verify custom compactness and gamma are applied and saved in diagnostics."""
        source = cv2.imread("dataset/test/l1.jpg", cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(source)
        with tempfile.TemporaryDirectory() as temporary_dir:
            temp_dir = Path(temporary_dir)
            input_path = temp_dir / "custom_test.png"
            self.assertTrue(cv2.imwrite(str(input_path), source))
            outputs = colorize_live_image(
                input_path,
                model_dir="models",
                output_root=temp_dir / "custom_out",
                compactness=0.2,
                gamma=0.1,
            )
            self.assertTrue(Path(outputs["mrf_refined"]).is_file())
            with np.load(outputs["refined_uv"]) as diag:
                self.assertAlmostEqual(float(diag["compactness"]), 0.2)
                self.assertAlmostEqual(float(diag["gamma"]), 0.1)

    def test_live_colorize_with_smooth_chroma_flag(self):
        """Verify smooth_chroma bilateral edge-preserving smoothing flag functions correctly."""
        source = cv2.imread("dataset/test/l1.jpg", cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(source)
        with tempfile.TemporaryDirectory() as temporary_dir:
            temp_dir = Path(temporary_dir)
            input_path = temp_dir / "smooth_test.png"
            self.assertTrue(cv2.imwrite(str(input_path), source))
            outputs = colorize_live_image(
                input_path,
                model_dir="models",
                output_root=temp_dir / "smooth_out",
                compactness=0.1,
                gamma=0.1,
                smooth_chroma=True,
            )
            self.assertTrue(Path(outputs["mrf_refined"]).is_file())
            result = cv2.imread(str(outputs["mrf_refined"]), cv2.IMREAD_COLOR)
            self.assertIsNotNone(result)
            self.assertEqual(result.shape[2], 3)
            with np.load(outputs["refined_uv"]) as diag:
                self.assertTrue(bool(diag["smooth_chroma"]))


if __name__ == "__main__":
    unittest.main()
