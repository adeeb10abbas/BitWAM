"""CPU-only tests for the package and command surface."""

from lerobot_policy_bitwam import __version__
from lerobot_policy_bitwam.cli import COMMANDS, build_parser


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_cli_exposes_all_commands_without_loading_weights() -> None:
    parser = build_parser()
    for command in COMMANDS:
        args = parser.parse_args([command, "--config", "configs/smoke.yaml"])
        assert args.command == command
