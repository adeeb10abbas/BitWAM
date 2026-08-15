from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from lerobot_policy_bitwam.bitvla_packed_checkpoint import (
    PACKED_CHECKPOINT_SCHEMA_VERSION,
    PackedCheckpointError,
    export_packed_checkpoint,
    load_or_export_packed_checkpoint,
    load_packed_checkpoint,
    move_packed_bitvla_to_device,
    read_packed_checkpoint_manifest,
)


class BitLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, *, weight_bits: int = 1) -> None:
        super().__init__(in_features, out_features, bias=False, dtype=torch.bfloat16)
        self.weight_bits = weight_bits

    def quantize_weights(self) -> None:
        values = self.weight.detach().flatten()
        codes = values.sign().to(torch.int8).add(1).to(torch.uint8)
        padding = (-codes.numel()) % 4
        codes = torch.nn.functional.pad(codes, (0, padding)).view(-1, 4)
        packed = codes[:, 0] | codes[:, 1] << 2 | codes[:, 2] << 4 | codes[:, 3] << 6
        self.register_buffer("q_weight", packed)
        self.register_buffer("w_step", torch.tensor(0.25, dtype=torch.float32))
        self.register_parameter("weight", None)


class FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ternary = BitLinear(5, 3)
        self.fallback = nn.Linear(5, 3, bias=False, dtype=torch.bfloat16)


class MetaRotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = 10_000
        self.dim = 8
        self.max_position_embeddings = 4
        self.register_buffer("inv_freq", torch.empty(4, device="meta"), persistent=False)
        self.register_buffer("cos_cached", torch.empty(4, 8, device="meta"), persistent=False)
        self.register_buffer("sin_cached", torch.empty(4, 8, device="meta"), persistent=False)

    def _set_cos_sin_cache(self, *, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        position = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(position, self.inv_freq)
        embeddings = torch.cat((freqs, freqs), dim=-1)
        self._buffers["cos_cached"] = embeddings.cos().to(dtype)
        self._buffers["sin_cached"] = embeddings.sin().to(dtype)


class MetaHelperModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "position_ids",
            torch.empty(1, 3, dtype=torch.long, device="meta"),
            persistent=False,
        )
        self.rotary = MetaRotary()


def _dense_policy() -> FakePolicy:
    torch.manual_seed(17)
    return FakePolicy()


def test_export_and_direct_load_reconstructs_packed_and_fallback_state(tmp_path: Path) -> None:
    source = _dense_policy()
    original_fallback = source.fallback.weight.detach().clone()
    artifact = tmp_path / "policy.bitwam-packed"

    manifest = export_packed_checkpoint(
        source,
        artifact,
        source_metadata={"source_revision": "abc123", "checkpoint": "released"},
    )

    assert source.ternary.weight is None
    assert source.ternary.q_weight.dtype == torch.uint8
    assert manifest.payload["schema_version"] == PACKED_CHECKPOINT_SCHEMA_VERSION
    assert manifest.packed_layers[0]["weight_shape"] == [3, 5]
    assert read_packed_checkpoint_manifest(artifact).source_metadata["source_revision"] == "abc123"

    loaded = _dense_policy()
    load_packed_checkpoint(loaded, artifact, expected_source_metadata={"source_revision": "abc123"})

    assert loaded.ternary.weight is None
    assert loaded.ternary.q_weight.dtype == torch.uint8
    assert tuple(loaded.ternary.orig_shape) == (3, 5)
    assert loaded.ternary.n_elems == 15
    assert loaded.ternary.enable_qlora is True
    assert torch.equal(loaded.ternary.q_weight, source.ternary.q_weight)
    assert torch.equal(loaded.fallback.weight, original_fallback)


def test_load_or_export_skips_dense_loader_when_artifact_exists(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.bitwam-packed"
    calls = 0

    def load_dense(model: nn.Module) -> None:
        nonlocal calls
        calls += 1
        reference = _dense_policy()
        model.load_state_dict(reference.state_dict())

    _, first_manifest, created = load_or_export_packed_checkpoint(
        artifact,
        build_model=FakePolicy,
        load_dense_weights=load_dense,
        source_metadata={"checkpoint_sha256": "a" * 64},
    )
    assert created is True
    assert calls == 1
    assert first_manifest.source_metadata["checkpoint_sha256"] == "a" * 64

    def fail_if_dense_load(_: nn.Module) -> None:
        raise AssertionError("existing packed artifact must not load dense weights")

    loaded, _, created = load_or_export_packed_checkpoint(
        artifact,
        build_model=FakePolicy,
        load_dense_weights=fail_if_dense_load,
        source_metadata={"checkpoint_sha256": "a" * 64},
    )
    assert created is False
    assert loaded.ternary.weight is None
    assert isinstance(loaded.ternary.q_weight, torch.Tensor)


def test_load_rejects_shape_mismatch_before_replacing_dense_target(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.bitwam-packed"
    export_packed_checkpoint(_dense_policy(), artifact)
    target = FakePolicy()
    target.ternary = BitLinear(4, 3)

    with pytest.raises(PackedCheckpointError, match="shape mismatch"):
        load_packed_checkpoint(target, artifact)

    assert target.ternary.weight is not None


def test_load_assigns_into_meta_topology_without_materializing_dense_weights(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.bitwam-packed"
    export_packed_checkpoint(_dense_policy(), artifact)
    target = _dense_policy().to(device="meta")

    load_packed_checkpoint(target, artifact)

    assert target.ternary.weight is None
    assert target.ternary.q_weight.device.type == "cpu"
    assert target.fallback.weight.device.type == "cpu"


def test_load_registers_runtime_scale_buffers_created_by_packing(tmp_path: Path) -> None:
    artifact = tmp_path / "scaled-policy.bitwam-packed"
    export_packed_checkpoint(FakePolicy(), artifact)
    target = FakePolicy()

    load_packed_checkpoint(target, artifact)

    assert target.ternary.weight is None
    assert target.ternary.w_step.item() == pytest.approx(0.25)


def test_move_packed_bitvla_to_device_rebuilds_known_meta_helpers() -> None:
    helpers = MetaHelperModule()

    moved = move_packed_bitvla_to_device(helpers, "cpu")

    assert moved.position_ids.tolist() == [[0, 1, 2]]
    assert not any(buffer.is_meta for buffer in moved.buffers())
    assert moved.rotary.cos_cached.shape == (4, 8)


def test_load_rejects_manifest_or_tensor_tampering(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.bitwam-packed"
    export_packed_checkpoint(_dense_policy(), artifact)
    tensors_path = artifact / "tensors.pt"
    with tensors_path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(PackedCheckpointError, match="SHA-256"):
        load_packed_checkpoint(FakePolicy(), artifact)

    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackedCheckpointError, match="schema version"):
        read_packed_checkpoint_manifest(artifact)


def test_load_rejects_tensor_change_even_if_archive_hash_is_rewritten(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.bitwam-packed"
    export_packed_checkpoint(_dense_policy(), artifact)
    tensors_path = artifact / "tensors.pt"
    try:
        archive = torch.load(tensors_path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older PyTorch compatibility.
        archive = torch.load(tensors_path, map_location="cpu")
    archive["ternary.q_weight"][0] ^= 1
    torch.save(archive, tensors_path)

    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tensors_sha256"] = hashlib.sha256(tensors_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(PackedCheckpointError, match="does not match its manifest"):
        load_packed_checkpoint(FakePolicy(), artifact)


def test_existing_artifact_rejects_source_metadata_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "policy.bitwam-packed"
    export_packed_checkpoint(_dense_policy(), artifact, source_metadata={"revision": "one"})

    with pytest.raises(PackedCheckpointError, match="source metadata mismatch"):
        load_packed_checkpoint(FakePolicy(), artifact, expected_source_metadata={"revision": "two"})


def test_export_supports_scalar_buffers(tmp_path: Path) -> None:
    policy = _dense_policy()
    policy.register_buffer("scalar_scale", torch.tensor(0.25, dtype=torch.float32))
    artifact = tmp_path / "policy.bitwam-packed"

    export_packed_checkpoint(policy, artifact)
    loaded = _dense_policy()
    loaded.register_buffer("scalar_scale", torch.tensor(0.0, dtype=torch.float32))
    load_packed_checkpoint(loaded, artifact)

    assert loaded.scalar_scale.item() == pytest.approx(0.25)
