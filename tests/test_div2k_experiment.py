"""Focused checks for the isolated DIV2K candidate workflow."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src import colorize, div2k_experiment
import main as cli


class Div2kExperimentTest(unittest.TestCase):
    def test_discovery_validation_and_source_split_isolation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            directory = Path(temporary_dir)
            for name in ("0001.png", "0002.png", "0003.png", "0004.png"):
                image = np.full((12, 16, 3), 100, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(directory / name), image))
            paths = div2k_experiment.discover_images(directory)
            manifest = div2k_experiment.make_split_manifest(paths, seed=7, validation_fraction=0.25)
            self.assertEqual(len(paths), 4)
            self.assertFalse(set(manifest["train"]) & set(manifest["validation"]))
            self.assertEqual(set(manifest["train"]) | set(manifest["validation"]),
                             {path.name for path in paths})
            (directory / "broken.png").write_text("not an image", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid RGB"):
                div2k_experiment.discover_images(directory)

    def test_candidate_models_save_and_load(self):
        rng = np.random.default_rng(5)
        feature_matrix = rng.normal(size=(12, 103)).astype(np.float32)
        target_u = rng.uniform(100, 150, size=12).astype(np.float32)
        target_v = rng.uniform(100, 150, size=12).astype(np.float32)
        with tempfile.TemporaryDirectory() as temporary_dir:
            model_dir = Path(temporary_dir) / "models_div2k"
            div2k_experiment.fit_candidate_models(feature_matrix, target_u, target_v, model_dir)
            scaler, model_u, model_v = colorize.load_models(model_dir)
            predictions = model_u.predict(scaler.transform(feature_matrix))
            self.assertEqual(predictions.shape, (12,))
            self.assertTrue(np.isfinite(predictions).all())
            self.assertTrue((model_dir / "svr_v.joblib").is_file())

    def test_cli_lists_div2k_workflow(self):
        result = subprocess.run([sys.executable, "main.py", "--help"], text=True,
                                capture_output=True, check=True)
        self.assertIn("div2k-prepare", result.stdout)
        self.assertIn("div2k-tune", result.stdout)
        self.assertIn("mixed-tune", result.stdout)
        self.assertIn("--div2k-limit", result.stdout)

    def test_cli_routes_div2k_prepare_options(self):
        with patch.object(sys, "argv", ["main.py", "div2k-prepare", "--div2k-limit", "2",
                                         "--div2k-seed", "3", "--div2k-validation-fraction", "0.5"]), \
                patch.object(cli.div2k_experiment, "prepare") as prepare:
            cli.main()
        prepare.assert_called_once_with(limit=2, seed=3, validation_fraction=0.5)

    def test_stratified_mixed_sampling_is_unique_and_keeps_both_origins(self):
        features = np.arange(12 * 103, dtype=np.float32).reshape(12, 103)
        u_values = np.arange(12, dtype=np.float32)
        v_values = u_values + 1
        origins = np.array(["original"] * 4 + ["div2k"] * 8)
        _, sampled_u, _, sampled_origins, indexes = div2k_experiment.stratified_sample_rows(
            features, u_values, v_values, origins, max_train_samples=8, seed=9)
        self.assertEqual(len(indexes), len(set(indexes.tolist())))
        self.assertEqual(len(sampled_u), 8)
        self.assertIn("original", sampled_origins)
        self.assertIn("div2k", sampled_origins)


if __name__ == "__main__":
    unittest.main()
