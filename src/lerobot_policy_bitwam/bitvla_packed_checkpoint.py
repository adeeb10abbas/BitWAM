"""Versioned, self-validating artifacts for BitVLA packed inference weights.

The artifact is intentionally a directory rather than an opaque replacement for a
Hugging Face checkpoint.  It contains a compact tensor archive and a JSON
manifest.  The archive only has ``q_weight`` buffers for packed ``BitLinear``
layers; their BF16 master ``weight`` parameters are omitted.  Non-packed
parameters and buffers remain in the archive as the deployed fallbacks.

This module is deliberately independent of the runtime kernels.  A caller can
construct the normal model topology, load this artifact, and then choose any
packed inference backend without first reading the dense source checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot_policy_bitwam.bitvla_packing import PackingReport, _matches_scope, pack_bitlinear_weights

PACKED_CHECKPOINT_FORMAT = "bitwam.bitvla.packed"
PACKED_CHECKPOINT_SCHEMA_VERSION = 1
_MANIFEST_NAME = "manifest.json"
_TENSORS_NAME = "tensors.pt"


class PackedCheckpointError(ValueError):
    """Raised when a packed checkpoint is incomplete, incompatible, or corrupt."""


@dataclass(frozen=True)
class PackedCheckpointManifest:
    """Validated metadata for a :mod:`bitvla_packed_checkpoint` artifact."""

    payload: dict[str, Any]

    @property
    def source_metadata(self) -> dict[str, Any]:
        return dict(self.payload["source_metadata"])

    @property
    def packing(self) -> dict[str, Any]:
        return dict(self.payload["packing"])

    @property
    def packed_layers(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(layer) for layer in self.payload["packed_layers"])

    def to_dict(self) -> dict[str, Any]:
        """Return a deep-copyable JSON-compatible manifest payload."""
        return json.loads(json.dumps(self.payload))


def _json_value(value: Any, *, name: str) -> Any:
    """Reject metadata that cannot be represented unambiguously in the manifest."""
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON serializable") from error
    return json.loads(encoded)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    """Hash raw tensor bytes plus its shape and dtype, independent of device."""
    detached = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("utf-8"))
    digest.update(json.dumps(list(detached.shape)).encode("utf-8"))
    # Tensor.view(dtype) does not permit a zero-dimensional input when the
    # element size changes; BitVLA's per-channel scales include scalar buffers.
    digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "nbytes": tensor.numel() * tensor.element_size(),
        "sha256": _sha256_tensor(tensor),
    }


def _artifact_paths(path: str | Path) -> tuple[Path, Path, Path]:
    root = Path(path)
    return root, root / _MANIFEST_NAME, root / _TENSORS_NAME


def _state_key(module_name: str, name: str) -> str:
    return f"{module_name}.{name}" if module_name else name


def _is_packable(candidate: nn.Module, scope: str) -> bool:
    return (
        candidate.__class__.__name__ == "BitLinear"
        and callable(getattr(candidate, "quantize_weights", None))
        and isinstance(getattr(candidate, "weight", None), torch.Tensor)
        and int(getattr(candidate, "weight_bits", 1)) == 1
        and _matches_scope(candidate, scope)
    )


def _packed_layer_specs(
    module: nn.Module,
    original_shapes: Mapping[str, tuple[int, ...]],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for name, candidate in module.named_modules():
        if candidate.__class__.__name__ != "BitLinear":
            continue
        packed = getattr(candidate, "q_weight", None)
        if not isinstance(packed, torch.Tensor):
            continue
        shape = original_shapes.get(name)
        if shape is None:
            shape_value = getattr(candidate, "orig_shape", None)
            if not isinstance(shape_value, (tuple, list)) or len(shape_value) != 2:
                raise PackedCheckpointError(
                    f"Packed BitLinear '{name}' has no original shape; pack from a dense model "
                    "or set its orig_shape before exporting"
                )
            shape = tuple(int(value) for value in shape_value)
        if len(shape) != 2 or min(shape) < 1:
            raise PackedCheckpointError(f"Packed BitLinear '{name}' has invalid original shape {shape}")
        expected_bytes = (shape[0] * shape[1] + 3) // 4
        if packed.dtype != torch.uint8 or packed.numel() != expected_bytes:
            raise PackedCheckpointError(
                f"Packed BitLinear '{name}' has invalid q_weight: expected {expected_bytes} uint8 bytes"
            )
        specs.append(
            {
                "module": name,
                "weight_shape": list(shape),
                "packed_weight": _state_key(name, "q_weight"),
                "packed_nbytes": expected_bytes,
            }
        )
    if not specs:
        raise PackedCheckpointError("No packed BitLinear layers were found for serialization")
    return specs


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_torch_save(path: Path, value: Any) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_manifest(path: str | Path) -> PackedCheckpointManifest:
    root, manifest_path, _ = _artifact_paths(path)
    if not root.is_dir() or not manifest_path.is_file():
        raise PackedCheckpointError(f"Packed artifact is missing {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PackedCheckpointError(f"Unable to read packed manifest {manifest_path}") from error
    if not isinstance(payload, dict):
        raise PackedCheckpointError("Packed manifest must be a JSON object")
    required = {
        "format",
        "schema_version",
        "tensors_file",
        "tensors_sha256",
        "tensors",
        "source_metadata",
        "packing",
        "packed_layers",
    }
    missing = required.difference(payload)
    if missing:
        raise PackedCheckpointError(f"Packed manifest is missing fields: {sorted(missing)}")
    if payload["format"] != PACKED_CHECKPOINT_FORMAT:
        raise PackedCheckpointError(f"Unsupported packed artifact format: {payload['format']!r}")
    if payload["schema_version"] != PACKED_CHECKPOINT_SCHEMA_VERSION:
        raise PackedCheckpointError(
            f"Unsupported packed artifact schema version: {payload['schema_version']!r}"
        )
    if payload["tensors_file"] != _TENSORS_NAME:
        raise PackedCheckpointError("Packed manifest references an unexpected tensor archive name")
    if not isinstance(payload["tensors"], dict) or not isinstance(payload["packed_layers"], list):
        raise PackedCheckpointError("Packed manifest has invalid tensor or layer metadata")
    if not isinstance(payload["source_metadata"], dict) or not isinstance(payload["packing"], dict):
        raise PackedCheckpointError("Packed manifest has invalid source or packing metadata")
    return PackedCheckpointManifest(payload)


def read_packed_checkpoint_manifest(path: str | Path) -> PackedCheckpointManifest:
    """Read and validate artifact metadata without loading any tensors."""
    return _read_manifest(path)


def export_packed_checkpoint(
    module: nn.Module,
    path: str | Path,
    *,
    scope: str = "all",
    source_metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> PackedCheckpointManifest:
    """Pack eligible weights and atomically export a direct-load deployment artifact.

    The provided module is converted in-place.  Once this returns, its eligible
    ``BitLinear.weight`` parameters have been replaced by packed ``q_weight``
    buffers, just as they are for the inference runtime.
    """
    root, manifest_path, tensors_path = _artifact_paths(path)
    if root.exists() and not root.is_dir():
        raise PackedCheckpointError(f"Packed artifact path is not a directory: {root}")
    if not overwrite and (manifest_path.exists() or tensors_path.exists()):
        raise FileExistsError(f"Packed artifact already exists at {root}")
    root.mkdir(parents=True, exist_ok=True)

    original_shapes = {
        name: tuple(int(value) for value in candidate.weight.shape)
        for name, candidate in module.named_modules()
        if _is_packable(candidate, scope)
    }
    packing: PackingReport | None = None
    if original_shapes:
        packing = pack_bitlinear_weights(module, scope=scope)
    specs = _packed_layer_specs(module, original_shapes)
    state = module.state_dict()
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise PackedCheckpointError("Only tensor-valued module state can be serialized")
    tensors = {name: value.detach().cpu().contiguous() for name, value in state.items()}
    tensor_manifest = {name: _tensor_metadata(value) for name, value in tensors.items()}

    packed_keys = {spec["packed_weight"] for spec in specs}
    if not packed_keys.issubset(tensors):
        raise PackedCheckpointError("Packed q_weight buffers were absent from the exported state")
    _atomic_torch_save(tensors_path, tensors)
    manifest = {
        "format": PACKED_CHECKPOINT_FORMAT,
        "schema_version": PACKED_CHECKPOINT_SCHEMA_VERSION,
        "tensors_file": _TENSORS_NAME,
        "tensors_sha256": _sha256_file(tensors_path),
        "tensors": tensor_manifest,
        "source_metadata": _json_value(dict(source_metadata or {}), name="source_metadata"),
        "packing": packing.to_dict() if packing is not None else {"scope": scope, "reused": True},
        "packed_layers": specs,
    }
    _atomic_write_json(manifest_path, manifest)
    return PackedCheckpointManifest(manifest)


def _load_tensor_archive(path: str | Path, manifest: PackedCheckpointManifest) -> dict[str, torch.Tensor]:
    root, _, tensors_path = _artifact_paths(path)
    if not tensors_path.is_file():
        raise PackedCheckpointError(f"Packed artifact is missing {tensors_path}")
    if _sha256_file(tensors_path) != manifest.payload["tensors_sha256"]:
        raise PackedCheckpointError("Packed tensor archive SHA-256 does not match its manifest")
    try:
        try:
            archive = torch.load(tensors_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older PyTorch.
            archive = torch.load(tensors_path, map_location="cpu")
    except (OSError, RuntimeError, ValueError) as error:
        raise PackedCheckpointError(f"Unable to load packed tensor archive {tensors_path}") from error
    if not isinstance(archive, dict) or any(
        not isinstance(value, torch.Tensor) for value in archive.values()
    ):
        raise PackedCheckpointError("Packed tensor archive must be a tensor dictionary")
    expected = manifest.payload["tensors"]
    if set(archive) != set(expected):
        raise PackedCheckpointError("Packed tensor archive keys do not match its manifest")
    for name, tensor in archive.items():
        metadata = expected[name]
        if (
            str(tensor.dtype) != metadata.get("dtype")
            or list(tensor.shape) != metadata.get("shape")
            or tensor.numel() != metadata.get("numel")
            or tensor.numel() * tensor.element_size() != metadata.get("nbytes")
            or _sha256_tensor(tensor) != metadata.get("sha256")
        ):
            raise PackedCheckpointError(f"Packed tensor '{name}' does not match its manifest")
    for spec in manifest.packed_layers:
        key = spec["packed_weight"]
        shape = spec["weight_shape"]
        if key not in archive or archive[key].dtype != torch.uint8:
            raise PackedCheckpointError(f"Packed layer '{spec['module']}' has no uint8 q_weight")
        if archive[key].numel() != (int(shape[0]) * int(shape[1]) + 3) // 4:
            raise PackedCheckpointError(f"Packed layer '{spec['module']}' has invalid q_weight size")
    return archive


def _validate_source_metadata(
    manifest: PackedCheckpointManifest,
    expected_source_metadata: Mapping[str, Any] | None,
) -> None:
    if expected_source_metadata is None:
        return
    expected = _json_value(dict(expected_source_metadata), name="expected_source_metadata")
    actual = manifest.source_metadata
    mismatched = {
        key: (value, actual.get(key)) for key, value in expected.items() if actual.get(key) != value
    }
    if mismatched:
        raise PackedCheckpointError(f"Packed artifact source metadata mismatch: {mismatched}")


def _module_device(module: nn.Module) -> torch.device:
    for tensor in tuple(module.parameters()) + tuple(module.buffers()):
        return tensor.device
    return torch.device("cpu")


def _has_meta_tensors(module: nn.Module) -> bool:
    return any(tensor.is_meta for tensor in tuple(module.parameters()) + tuple(module.buffers()))


def _prepare_target_for_packed_state(
    module: nn.Module,
    archive: Mapping[str, torch.Tensor],
    manifest: PackedCheckpointManifest,
) -> None:
    """Validate all layer shapes before replacing any dense target parameters."""
    targets: list[tuple[nn.Module, dict[str, Any], torch.Tensor]] = []
    for spec in manifest.packed_layers:
        name = str(spec["module"])
        try:
            target = module.get_submodule(name) if name else module
        except AttributeError as error:
            raise PackedCheckpointError(f"Target model is missing packed module '{name}'") from error
        if target.__class__.__name__ != "BitLinear":
            raise PackedCheckpointError(f"Target module '{name}' is not a BitLinear")
        expected_shape = tuple(int(value) for value in spec["weight_shape"])
        dense_weight = getattr(target, "weight", None)
        current_shape = tuple(dense_weight.shape) if isinstance(dense_weight, torch.Tensor) else tuple(
            getattr(target, "orig_shape", ())
        )
        if current_shape != expected_shape:
            raise PackedCheckpointError(
                f"Packed layer '{name}' shape mismatch: artifact {expected_shape}, target {current_shape}"
            )
        targets.append((target, spec, archive[str(spec["packed_weight"]) ]))

    for target, spec, q_weight in targets:
        # Register a correctly shaped placeholder so strict load_state_dict can
        # load q_weight without ever allocating the dense master tensor again.
        target.register_parameter("weight", None)
        target.register_buffer("q_weight", torch.empty_like(q_weight, device=_module_device(target)))
        scale_key = _state_key(str(spec["module"]), "w_step")
        if scale_key in archive and not isinstance(getattr(target, "w_step", None), torch.Tensor):
            target.register_buffer(
                "w_step",
                torch.empty_like(archive[scale_key], device=_module_device(target)),
            )
        target.orig_shape = tuple(int(value) for value in spec["weight_shape"])
        # BitVLA keeps these inference controls as plain Python attributes, so
        # state_dict cannot restore them.  They are determined exactly by the
        # manifest's validated dense matrix shape.
        target.n_elems = target.orig_shape[0] * target.orig_shape[1]
        target.enable_qlora = True


def load_packed_checkpoint(
    module: nn.Module,
    path: str | Path,
    *,
    expected_source_metadata: Mapping[str, Any] | None = None,
    assign: bool | None = None,
) -> PackedCheckpointManifest:
    """Load a packed artifact into an already-constructed model topology.

    Integrity, per-tensor hashes, source metadata, target layer names, and
    matrix shapes are all checked before replacing a dense ``BitLinear.weight``
    parameter with its compact ``q_weight`` buffer.  A topology constructed on
    PyTorch's ``meta`` device automatically uses ``assign=True`` so no random
    dense weights are materialized before the packed archive is assigned.
    """
    manifest = _read_manifest(path)
    _validate_source_metadata(manifest, expected_source_metadata)
    archive = _load_tensor_archive(path, manifest)
    _prepare_target_for_packed_state(module, archive, manifest)
    should_assign = _has_meta_tensors(module) if assign is None else assign
    try:
        result = module.load_state_dict(archive, strict=True, assign=should_assign)
    except TypeError as error:  # pragma: no cover - PyTorch before the assign argument.
        if should_assign:
            raise PackedCheckpointError(
                "This PyTorch version cannot load a packed artifact into a meta-device topology"
            ) from error
        result = module.load_state_dict(archive, strict=True)
    if result.missing_keys or result.unexpected_keys:  # Defensive for future PyTorch changes.
        raise PackedCheckpointError(
            f"Packed load was not strict: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return manifest


def build_bitvla_topology(
    source_checkpoint: str | Path,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Build BitVLA's model topology on ``meta`` without reading safetensors.

    ``source_checkpoint`` is used only for ``config.json`` and trusted dynamic
    model Python files.  It must be the same BitVLA release used to create the
    packed artifact, but this function deliberately never calls
    ``from_pretrained`` and therefore never reads ``model*.safetensors``.
    """
    try:
        from accelerate import init_empty_weights
    except ImportError as error:  # Keep unit-test imports independent of BitVLA's optional stack.
        raise PackedCheckpointError(
            "Building a BitVLA topology requires the BitVLA transformers and accelerate environment"
        ) from error

    source_path = Path(source_checkpoint).expanduser().resolve()
    source = str(source_path)
    config_path = source_path / "configuration_bit_vla.py"
    model_path = source_path / "bitvla_for_action_prediction.py"
    if not config_path.is_file() or not model_path.is_file():
        raise PackedCheckpointError(
            "BitVLA topology source needs configuration_bit_vla.py and "
            f"bitvla_for_action_prediction.py in {source}"
        )
    previous_config_module = sys.modules.get("configuration_bit_vla")
    checkpoint_path_was_present = source in sys.path
    try:
        # BitVLA's model source uses ``from configuration_bit_vla import ...``
        # rather than a relative import.  Loading the two source files explicitly
        # avoids AutoModel's dynamic-module cache, which cannot resolve that
        # absolute local import for ``from_config``.
        config_spec = importlib.util.spec_from_file_location("configuration_bit_vla", config_path)
        if config_spec is None or config_spec.loader is None:
            raise PackedCheckpointError(f"Unable to import BitVLA config from {config_path}")
        config_module = importlib.util.module_from_spec(config_spec)
        sys.modules["configuration_bit_vla"] = config_module
        config_spec.loader.exec_module(config_module)
        model_spec = importlib.util.spec_from_file_location("bitwam_packed_bitvla_model", model_path)
        if model_spec is None or model_spec.loader is None:
            raise PackedCheckpointError(f"Unable to import BitVLA model from {model_path}")
        if not checkpoint_path_was_present:
            sys.path.insert(0, source)
        model_module = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_module)
        config = config_module.Bitvla_Config.from_pretrained(source)
        config.torch_dtype = torch_dtype
        with init_empty_weights(include_buffers=True):
            model = model_module.BitVLAForActionPrediction(config)
    except (AttributeError, ImportError, OSError, RuntimeError, ValueError) as error:
        raise PackedCheckpointError(
            f"Unable to build a meta-device BitVLA topology from {source}"
        ) from error
    finally:
        if not checkpoint_path_was_present:
            sys.path.remove(source)
        if previous_config_module is None:
            sys.modules.pop("configuration_bit_vla", None)
        else:
            sys.modules["configuration_bit_vla"] = previous_config_module
    if not _has_meta_tensors(model):
        raise PackedCheckpointError("BitVLA topology construction unexpectedly materialized model tensors")
    return model


def move_packed_bitvla_to_device(module: nn.Module, device: torch.device | str) -> nn.Module:
    """Materialize BitVLA's non-persistent helpers, then move packed state to ``device``.

    A meta-device topology intentionally does not serialize helper buffers such
    as SigLIP position IDs and BitNet RoPE caches.  They are deterministic and
    small compared with model weights, but must be rebuilt before ``Module.to``
    can move the assigned packed parameters.  Unknown meta buffers fail closed
    instead of being silently initialized with incorrect values.
    """
    destination = torch.device(device)
    unresolved: list[str] = []
    for prefix, candidate in module.named_modules():
        for name, buffer in tuple(candidate._buffers.items()):
            if buffer is None or not buffer.is_meta:
                continue
            qualified_name = _state_key(prefix, name)
            if name == "position_ids":
                candidate._buffers[name] = torch.arange(
                    buffer.shape[-1], device=destination, dtype=buffer.dtype
                ).expand(tuple(buffer.shape))
                continue
            if name == "inv_freq" and callable(getattr(candidate, "_set_cos_sin_cache", None)):
                base = float(candidate.base)
                dimension = int(candidate.dim)
                candidate._buffers[name] = 1.0 / (
                    base
                    ** (
                        torch.arange(0, dimension, 2, device=destination, dtype=torch.float32)
                        / dimension
                    )
                )
                candidate._set_cos_sin_cache(
                    seq_len=int(candidate.max_position_embeddings),
                    device=destination,
                    dtype=torch.get_default_dtype(),
                )
                continue
            if name in {"cos_cached", "sin_cached"} and not candidate._buffers[name].is_meta:
                continue
            unresolved.append(qualified_name)
    if unresolved:
        raise PackedCheckpointError(
            "Cannot move packed BitVLA topology with unknown meta helper buffers: "
            + ", ".join(unresolved)
        )
    return module.to(destination)


def load_or_export_packed_checkpoint(
    path: str | Path,
    *,
    build_model: Callable[[], nn.Module],
    load_dense_weights: Callable[[nn.Module], None],
    scope: str = "all",
    source_metadata: Mapping[str, Any] | None = None,
) -> tuple[nn.Module, PackedCheckpointManifest, bool]:
    """Use an existing artifact or create it once from a dense checkpoint.

    Returns ``(model, manifest, created)``.  Crucially, when a valid artifact
    already exists, ``load_dense_weights`` is never called.
    """
    root, manifest_path, tensors_path = _artifact_paths(path)
    if manifest_path.is_file() or tensors_path.is_file():
        if not (manifest_path.is_file() and tensors_path.is_file()):
            raise PackedCheckpointError(f"Packed artifact at {root} is incomplete")
        model = build_model()
        return model, load_packed_checkpoint(
            model,
            root,
            expected_source_metadata=source_metadata,
        ), False

    model = build_model()
    load_dense_weights(model)
    manifest = export_packed_checkpoint(
        model,
        root,
        scope=scope,
        source_metadata=source_metadata,
    )
    return model, manifest, True
