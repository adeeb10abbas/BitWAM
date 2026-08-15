"""Export a BitVLA packed artifact and measure dense versus direct-load memory.

Run in the BitVLA environment, not the lightweight local test environment:

    PYTHONPATH=src:<BitVLA>/openvla-oft/bitvla:<BitVLA>/openvla-oft \\
      python scripts/validate_packed_artifact.py \\
      --checkpoint <checkpoint> --artifact <new-artifact-dir>
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from lerobot_policy_bitwam.bitvla_packed_checkpoint import (
    build_bitvla_topology,
    export_packed_checkpoint,
    load_packed_checkpoint,
    move_packed_bitvla_to_device,
)


def _rss_kib() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _artifact_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _dense_model(checkpoint: Path) -> torch.nn.Module:
    # Importing this only here keeps unit tests free of BitVLA's runtime stack.
    from experiments.robot import bitnet_utils

    def no_checkpoint_mutation(*args: object, **kwargs: object) -> None:
        return None

    bitnet_utils.update_auto_map = no_checkpoint_mutation
    bitnet_utils.check_model_logic_mismatch = no_checkpoint_mutation

    return bitnet_utils.get_bitnet_vla(
        SimpleNamespace(
            pretrained_checkpoint=str(checkpoint),
            load_in_8bit=False,
            load_in_4bit=False,
        )
    )


def _emit(payload: dict[str, object], output_json: Path | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(rendered, encoding="utf-8")
    print("BITWAM_PACKED_ARTIFACT_VALIDATION=" + json.dumps(payload, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--reuse-artifact",
        action="store_true",
        help="Skip dense export and measure only a first direct packed load.",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="Measure the upstream dense loader without writing an artifact.",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    artifact = args.artifact.resolve()
    if args.reuse_artifact and args.dense_only:
        raise ValueError("--reuse-artifact and --dense-only are mutually exclusive")
    if artifact.exists() and not args.reuse_artifact and not args.dense_only:
        raise FileExistsError(f"Refusing to overwrite existing artifact directory: {artifact}")
    if args.reuse_artifact and not artifact.is_dir():
        raise FileNotFoundError(f"Packed artifact does not exist: {artifact}")
    if not torch.cuda.is_available():
        raise RuntimeError("This validation requires CUDA")

    dense_metrics = None
    export_metrics = None
    if not args.reuse_artifact:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dense_started = time.perf_counter()
        dense_model = _dense_model(checkpoint)
        torch.cuda.synchronize()
        dense_metrics = {
            "load_seconds": time.perf_counter() - dense_started,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
            "max_rss_kib": _rss_kib(),
        }
        if args.dense_only:
            _emit(
                {
                    "checkpoint": str(checkpoint),
                    "artifact": None,
                    "dense": dense_metrics,
                    "export": None,
                    "packed_direct_load": None,
                },
                args.output_json,
            )
            return

        export_started = time.perf_counter()
        manifest = export_packed_checkpoint(
            dense_model,
            artifact,
            source_metadata={
                "checkpoint": str(checkpoint),
                "model_config": "config.json",
                "model_safetensors": "model-00001-of-00002.safetensors,model-00002-of-00002.safetensors",
            },
        )
        torch.cuda.synchronize()
        export_metrics = {
            "seconds": time.perf_counter() - export_started,
            "cuda_allocated_bytes_after_pack": torch.cuda.memory_allocated(),
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
            "max_rss_kib": _rss_kib(),
            "artifact_bytes": _artifact_bytes(artifact),
            "packed_layers": len(manifest.packed_layers),
            "packing": manifest.packing,
        }
        del dense_model
        torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    packed_started = time.perf_counter()
    packed_model = build_bitvla_topology(checkpoint)
    load_packed_checkpoint(packed_model, artifact)
    packed_model = move_packed_bitvla_to_device(packed_model, args.device)
    first_packed_layer = next(
        candidate
        for candidate in packed_model.modules()
        if candidate.__class__.__name__ == "BitLinear" and bool(getattr(candidate, "enable_qlora", False))
    )
    layer_input = torch.randn(
        1,
        first_packed_layer.orig_shape[1],
        device=args.device,
        dtype=torch.bfloat16,
    )
    layer_output = first_packed_layer(layer_input)
    if not torch.isfinite(layer_output).all():
        raise RuntimeError("Direct-loaded packed BitLinear produced non-finite output")
    torch.cuda.synchronize()
    packed_metrics = {
        "load_seconds": time.perf_counter() - packed_started,
        "cuda_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
        "max_rss_kib": _rss_kib(),
        "meta_parameters_after_load": sum(parameter.is_meta for parameter in packed_model.parameters()),
        "first_packed_bitlinear_forward": {
            "input_features": int(first_packed_layer.orig_shape[1]),
            "output_features": int(first_packed_layer.orig_shape[0]),
            "finite": True,
        },
    }
    _emit(
        {
            "checkpoint": str(checkpoint),
            "artifact": str(artifact),
            "dense": dense_metrics,
            "export": export_metrics,
            "packed_direct_load": packed_metrics,
        },
        args.output_json,
    )


if __name__ == "__main__":
    main()
