"""Regression tests for sprint improvement models."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src import colorize


class SprintModelsTest(unittest.TestCase):
    def test_nature_aware_svr_loads_and_predicts(self):
        model_dir = Path("models_sprint_svr_nature")
        self.assertTrue((model_dir / "feature_scaler.joblib").is_file())
        self.assertTrue((model_dir / "svr_u.joblib").is_file())
        self.assertTrue((model_dir / "svr_v.joblib").is_file())

        scaler, model_u, model_v = colorize.load_models(str(model_dir))
        dummy_feat = np.ones((5, 103), dtype=np.float32)
        scaled = scaler.transform(dummy_feat)
        pred_u = model_u.predict(scaled)
        pred_v = model_v.predict(scaled)

        self.assertEqual(pred_u.shape, (5,))
        self.assertEqual(pred_v.shape, (5,))
        self.assertTrue(np.isfinite(pred_u).all())
        self.assertTrue(np.isfinite(pred_v).all())

    def test_nature_aware_extratrees_loads_and_predicts(self):
        model_dir = Path("models_sprint_et_nature")
        self.assertTrue((model_dir / "feature_scaler.joblib").is_file())
        self.assertTrue((model_dir / "svr_u.joblib").is_file())
        self.assertTrue((model_dir / "svr_v.joblib").is_file())

        scaler, model_u, model_v = colorize.load_models(str(model_dir))
        dummy_feat = np.ones((5, 103), dtype=np.float32)
        scaled = scaler.transform(dummy_feat)
        pred_u = model_u.predict(scaled)
        pred_v = model_v.predict(scaled)

        self.assertEqual(pred_u.shape, (5,))
        self.assertEqual(pred_v.shape, (5,))
        self.assertTrue(np.isfinite(pred_u).all())
        self.assertTrue(np.isfinite(pred_v).all())

    def test_live_colorize_with_sprint_models(self):
        source = cv2.imread("dataset/test/l1.jpg", cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(source)
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            in_file = temp_path / "test_in.png"
            cv2.imwrite(str(in_file), source)

            # Test SVR
            out_svr = colorize.colorize_live_image(
                str(in_file),
                model_dir="models_sprint_svr_nature",
                output_root=str(temp_path / "svr_out"),
                compactness=0.2,
                gamma=0.2,
            )
            self.assertTrue(Path(out_svr["mrf_refined"]).is_file())

            # Test ExtraTrees
            out_et = colorize.colorize_live_image(
                str(in_file),
                model_dir="models_sprint_et_nature",
                output_root=str(temp_path / "et_out"),
                compactness=0.2,
                gamma=0.2,
            )
            self.assertTrue(Path(out_et["mrf_refined"]).is_file())


if __name__ == "__main__":
    unittest.main()

