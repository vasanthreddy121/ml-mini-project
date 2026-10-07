"""Command-line entry point for the automatic image colorization pipeline."""

import argparse

from src import colorize, evaluation, features, mrf, preprocessing, segmentation, train


def main():
    parser = argparse.ArgumentParser(description="Automatic image colorization pipeline")
    parser.add_argument("command", nargs="?", default="all",
                        choices=["preprocess", "segment", "features", "train", "colorize", "mrf", "evaluate", "all"])
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"],
                        help="split for colorize, MRF, and evaluation (default: test)")
    args = parser.parse_args()

    if args.command in ("preprocess", "all"):
        preprocessing.process_dataset()
    if args.command in ("segment", "all"):
        segmentation.process_dataset()
    if args.command in ("features", "all"):
        features.process_dataset()
    if args.command in ("train", "all"):
        train.train_models()
    if args.command in ("colorize", "all"):
        colorize.colorize_split(args.split)
    if args.command in ("mrf", "all"):
        mrf.refine_split(args.split)
    if args.command in ("evaluate", "all"):
        evaluation.evaluate_split(args.split)


if __name__ == "__main__":
    main()
