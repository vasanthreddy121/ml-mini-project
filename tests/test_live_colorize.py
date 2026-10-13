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


if __name__ == "__main__":
    unittest.main()
