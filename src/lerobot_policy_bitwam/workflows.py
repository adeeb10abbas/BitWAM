"""Workflow dispatch used by the public CLI."""

from pathlib import Path


def run_command(command: str, config_path: Path) -> int:
    """Fail clearly until the phase-specific workflows are implemented."""
    raise NotImplementedError(
        f"The {command!r} workflow is not implemented yet; config was {config_path}."
    )
