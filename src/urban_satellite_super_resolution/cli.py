"""Command-line interface for the urban satellite SR research framework."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="urban-sr")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("prepare-data", help="Generate a local synthetic sample")
    sample.add_argument("--output", type=Path, default=Path("data/synthetic"))
    infer = commands.add_parser("infer", help="Run MC-dropout multi-task inference")
    infer.add_argument("--input", type=Path, required=True); infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--checkpoint", type=Path); infer.add_argument("--passes", type=int, default=20); infer.add_argument("--change-input", type=Path)
    evaluate = commands.add_parser("evaluate", help="Evaluate prediction and reference arrays")
    evaluate.add_argument("--prediction", type=Path, required=True); evaluate.add_argument("--target", type=Path, required=True)
    train = commands.add_parser("train", help="Train the baseline on paired data")
    train.add_argument("--config", type=Path, default=Path("configs/urban_satellite_super_resolution.yaml"))
    report = commands.add_parser("generate-report", help="Print or copy an inference report")
    report.add_argument("--input", type=Path, required=True); report.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        from .data.synthetic import generate_sample
        print(generate_sample(args.output))
    elif args.command == "infer":
        from .inference.pipeline import infer_scene
        print(infer_scene(args.input, args.output, checkpoint=args.checkpoint, passes=args.passes, change_input=args.change_input)["report"])
    elif args.command == "evaluate":
        import numpy as np
        from .evaluation.metrics import image_metrics
        print(image_metrics(np.load(args.prediction), np.load(args.target)))
    elif args.command == "generate-report":
        text = args.input.read_text(encoding="utf-8")
        if args.output: args.output.write_text(text, encoding="utf-8")
        print(text)
    elif args.command == "train":
        from .training.runner import train_from_config
        print(train_from_config(args.config))


if __name__ == "__main__":
    main()
