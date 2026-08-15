"""Command-line entry point for BitWAM workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

COMMANDS = ("train", "screen", "evaluate", "export", "benchmark", "summarize")


def build_parser() -> argparse.ArgumentParser:
    """Build the stable BitWAM command surface without loading model weights."""
    parser = argparse.ArgumentParser(
        prog="bitwam",
        description="Train, evaluate, export, benchmark, and summarize BitWAM policies.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse a BitWAM command and dispatch to the workflow runner."""
    args = build_parser().parse_args(argv)
    if not args.config.is_file():
        raise SystemExit(f"Configuration file does not exist: {args.config}")

    from lerobot_policy_bitwam.workflows import run_command

    return run_command(args.command, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
