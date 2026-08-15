"""Materialize an unchanged BitWAM wrapper checkpoint for the BF16 control."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from lerobot_policy_bitwam.modeling_bitwam import BitWAMPolicy


def _processor_source_directory(source_checkpoint: str, source_revision: str | None) -> Path:
    source_path = Path(source_checkpoint)
    if source_path.is_dir():
        return source_path
    return Path(
        snapshot_download(
            repo_id=source_checkpoint,
            revision=source_revision,
            allow_patterns="policy_*",
        )
    )


def _copy_policy_processors(
    source_checkpoint: str, output_checkpoint: Path, source_revision: str | None
) -> None:
    source_directory = _processor_source_directory(source_checkpoint, source_revision)
    processor_files = sorted(source_directory.glob("policy_*"))
    if not processor_files:
        raise FileNotFoundError(f"No policy processor artifacts found in {source_checkpoint!r}")
    for processor_file in processor_files:
        shutil.copy2(processor_file, output_checkpoint / processor_file.name)


def materialize_bf16_control(
    source_checkpoint: str,
    output_checkpoint: Path,
    *,
    source_revision: str | None = None,
    world_loss_weight: float = 0.1,
) -> None:
    """Save the native wrapper without an optimizer update or quantization change."""
    policy = BitWAMPolicy.from_source_checkpoint(
        source_checkpoint,
        revision=source_revision,
        config_overrides={
            "quantization_scope": "none",
            "qat_recovery": "none",
            "inference_backend": "native",
            "world_loss_weight": world_loss_weight,
        },
    )
    if policy.quantization_report.ternary_parameter_count:
        raise RuntimeError("The BF16 control unexpectedly contains ternary parameters.")
    policy.save_pretrained(output_checkpoint)
    _copy_policy_processors(source_checkpoint, output_checkpoint, source_revision)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--source-revision")
    parser.add_argument("--world-loss-weight", type=float, default=0.1)
    args = parser.parse_args()
    materialize_bf16_control(
        args.source_checkpoint,
        args.output_checkpoint,
        source_revision=args.source_revision,
        world_loss_weight=args.world_loss_weight,
    )


if __name__ == "__main__":
    main()
