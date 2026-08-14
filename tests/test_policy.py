"""CPU parity and registration tests for the BF16 BitWAM wrapper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.factory import get_policy_class
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from torch import Tensor, nn

from lerobot_policy_bitwam import BitWAMConfig
from lerobot_policy_bitwam.modeling_bitwam import BitWAMPolicy


class _LanguageLayer(nn.Module):
    def forward(self, hidden: Tensor) -> tuple[Tensor]:
        return (hidden,)


class _LanguageModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_LanguageLayer()])
        self.hidden_size = hidden_size


class _Backbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.model = SimpleNamespace(language_model=_LanguageModel(hidden_size))

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def forward(self, input_ids: Tensor, **_: object) -> SimpleNamespace:
        batch, length = input_ids.shape
        values = torch.arange(batch * length * self.config.hidden_size, device=input_ids.device)
        hidden = values.view(batch, length, self.config.hidden_size).float() + self.anchor
        self.model.language_model.layers[-1](hidden)
        return SimpleNamespace()


class _QwenInterface(nn.Module):
    def __init__(self, config: VLAJEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.model = _Backbone(config.action_hidden_size)

    @staticmethod
    def _get_torch_dtype(_: str) -> torch.dtype:
        return torch.float32

    def expand_tokenizer(self) -> tuple[list[str], list[int], int]:
        count = self.config.chunk_size * self.config.num_action_tokens_per_timestep
        return [f"<a{i}>" for i in range(count)], list(range(1000, 1000 + count)), 2000

    def build_inputs(self, images, instructions, action_prompt, embodied_prompt) -> dict[str, Tensor]:
        del instructions, action_prompt, embodied_prompt
        embodied = [2000] * self.config.num_embodied_action_tokens_per_instruction
        return {"input_ids": torch.tensor([[10, *embodied, 11]] * len(images), device=self.model.device)}

    @staticmethod
    def to_pixel_values(images: Tensor) -> Tensor:
        return images.float()


@pytest.fixture
def patch_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    from lerobot.policies.vla_jepa import modeling_vla_jepa

    monkeypatch.setattr(modeling_vla_jepa, "Qwen3VLInterface", _QwenInterface)


def _features() -> tuple[dict, dict]:
    inputs = {
        f"{OBS_IMAGES}.camera": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(4,)),
    }
    outputs = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(3,))}
    return inputs, outputs


def _vla_config(device: str = "cpu") -> VLAJEPAConfig:
    inputs, outputs = _features()
    return VLAJEPAConfig(
        input_features=inputs,
        output_features=outputs,
        device=device,
        enable_world_model=False,
        torch_dtype="float32",
        chunk_size=4,
        n_action_steps=2,
        num_video_frames=2,
        jepa_tubelet_size=1,
        action_hidden_size=16,
        action_model_type="DiT-test",
        action_num_layers=1,
        num_embodied_action_tokens_per_instruction=3,
        num_inference_timesteps=2,
        repeated_diffusion_steps=1,
    )


def _batch(device: str = "cpu") -> dict:
    return {
        f"{OBS_IMAGES}.camera": torch.rand(2, 3, 8, 8, device=device),
        OBS_STATE: torch.randn(2, 4, device=device),
        ACTION: torch.randn(2, 4, 3, device=device),
        "task": ["pick up the cube"] * 2,
    }


def test_policy_is_registered() -> None:
    assert "bitwam" in PreTrainedConfig.get_known_choices()
    assert get_policy_class("bitwam") is BitWAMPolicy


def test_config_copies_upstream_behavior() -> None:
    config = BitWAMConfig.from_vla_jepa(_vla_config(), source_checkpoint="lerobot/VLA-JEPA-LIBERO")
    assert config.type == "bitwam"
    assert config.quantization_scope == "none"
    assert config.inference_backend == "native"
    assert config.world_loss_weight == config.world_model_loss_weight


def test_quantization_disabled_matches_upstream_actions(patch_qwen: None) -> None:
    torch.manual_seed(7)
    upstream = VLAJEPAPolicy(_vla_config())
    config = BitWAMConfig.from_vla_jepa(upstream.config, source_checkpoint="test")
    wrapped = BitWAMPolicy(config)
    wrapped.upstream.load_state_dict(upstream.state_dict())
    batch = _batch()

    torch.manual_seed(99)
    expected = upstream.predict_action_chunk(batch)
    torch.manual_seed(99)
    actual = wrapped.predict_action_chunk(batch)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cpu_forward_backward_delegates_to_upstream(patch_qwen: None) -> None:
    config = BitWAMConfig.from_vla_jepa(_vla_config(), source_checkpoint="test")
    policy = BitWAMPolicy(config)
    loss, logs = policy(_batch())
    loss.backward()
    assert torch.isfinite(loss)
    assert set(logs) == {"action_loss", "wm_loss", "loss"}
    assert any(parameter.grad is not None for parameter in policy.parameters() if parameter.requires_grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device is not available on this host")
def test_cuda_forward_backward_smoke(patch_qwen: None) -> None:
    config = BitWAMConfig.from_vla_jepa(_vla_config("cuda"), source_checkpoint="test")
    policy = BitWAMPolicy(config).cuda().to(dtype=torch.bfloat16)
    batch = _batch("cuda")
    loss, _ = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
