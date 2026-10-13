"""Command-line entry point for the automatic image colorization pipeline."""

import argparse

from src import colorize, div2k_experiment, evaluation, features, mrf, preprocessing, segmentation, train
from src.config import (DIV2K_MAX_TRAIN_SAMPLES, DIV2K_PILOT_IMAGES,
                        DIV2K_SPLIT_SEED, DIV2K_VALIDATION_FRACTION)


def main():
    parser = argparse.ArgumentParser(description="Automatic image colorization pipeline")
    parser.add_argument("command", nargs="?", default="all",
                        choices=["preprocess", "segment", "features", "train", "colorize", "mrf", "evaluate", "all",
                                 "div2k-prepare", "div2k-segment", "div2k-features", "div2k-train", "div2k-tune", "div2k-evaluate", "div2k-all", "mixed-tune", "mixed-evaluate"])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"],
                        help="split for colorize, MRF, and evaluation (default: test)")
    parser.add_argument("--input", help="new grayscale image for the colorize command")
    parser.add_argument("--model-dir", default="models",
                        help="model directory for live colorize (default: models)")
    parser.add_argument("--div2k-limit", type=int, default=DIV2K_PILOT_IMAGES,
                        help=f"number of DIV2K source images (default: {DIV2K_PILOT_IMAGES})")
    parser.add_argument("--div2k-seed", type=int, default=DIV2K_SPLIT_SEED,
                        help=f"deterministic DIV2K split seed (default: {DIV2K_SPLIT_SEED})")
    parser.add_argument("--div2k-validation-fraction", type=float, default=DIV2K_VALIDATION_FRACTION,
                        help=f"DIV2K validation fraction (default: {DIV2K_VALIDATION_FRACTION})")
    parser.add_argument("--div2k-max-train-samples", type=int, default=DIV2K_MAX_TRAIN_SAMPLES,
                        help=f"maximum RBF-SVR training superpixels (default: {DIV2K_MAX_TRAIN_SAMPLES})")
    args = parser.parse_args()

    if args.input and args.command != "colorize":
        parser.error("--input is only valid with the colorize command.")

    if args.command in ("preprocess", "all"):
        preprocessing.process_dataset()
    if args.command in ("segment", "all"):
        segmentation.process_dataset()
    if args.command in ("features", "all"):
        features.process_dataset()
    if args.command in ("train", "all"):
        train.train_models()
    if args.command == "colorize" and args.input:
        try:
            outputs = colorize.colorize_live_image(args.input, model_dir=args.model_dir)
        except (FileNotFoundError, OSError, ValueError) as error:
            parser.error(str(error))
        print("\nLive colorization complete (estimated colours; not ground truth):")
        for label, path in outputs.items():
            print(f"  {label}: {path}")
    elif args.command in ("colorize", "all"):
        colorize.colorize_split(args.split)
    if args.command in ("mrf", "all"):
        mrf.refine_split(args.split)
    if args.command in ("evaluate", "all"):
        evaluation.evaluate_split(args.split)
    div2k_options = dict(limit=args.div2k_limit, seed=args.div2k_seed,
                         validation_fraction=args.div2k_validation_fraction)
    if args.command in ("div2k-prepare", "div2k-all"):
        div2k_experiment.prepare(**div2k_options)
    if args.command in ("div2k-segment", "div2k-all"):
        div2k_experiment.segment()
    if args.command in ("div2k-features", "div2k-all"):
        div2k_experiment.extract_features()
    if args.command in ("div2k-train", "div2k-all"):
        div2k_experiment.train_candidate(max_train_samples=args.div2k_max_train_samples,
                                         seed=args.div2k_seed)
    if args.command == "div2k-tune":
        div2k_experiment.tune_hyperparameters(max_train_samples=args.div2k_max_train_samples,
                                               seed=args.div2k_seed)
    if args.command == "mixed-tune":
        div2k_experiment.tune_mixed_training(max_train_samples=args.div2k_max_train_samples,
                                              seed=args.div2k_seed)
    if args.command == "mixed-evaluate":
        div2k_experiment.evaluate_mixed_selected()
    if args.command in ("div2k-evaluate", "div2k-all"):
        div2k_experiment.evaluate_candidate()


if __name__ == "__main__":
    main()
