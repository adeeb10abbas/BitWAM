"""LeRobot configuration for the BitWAM policy plugin."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig

QuantizationScope = Literal[
    "none",
    "qwen",
    "qwen_dit",
    "qwen_attention",
    "qwen_mlp",
    "qwen_middle_half",
]
InferenceBackend = Literal["native", "reference", "triton"]
QATRecovery = Literal["none", "qwen_edges", "dit_tail4"]


@PreTrainedConfig.register_subclass("bitwam")
@dataclass
class BitWAMConfig(VLAJEPAConfig):
    """VLA-JEPA configuration extended with BitWAM deployment controls."""

    source_checkpoint: str = "lerobot/VLA-JEPA-LIBERO"
    source_revision: str | None = None
    world_loss_weight: float | None = None
    quantization_scope: QuantizationScope = "none"
    inference_backend: InferenceBackend = "native"
    qat_recovery: QATRecovery = "none"
    representation_distillation_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.world_loss_weight is not None:
            self.world_model_loss_weight = self.world_loss_weight
        super().__post_init__()
        self.world_loss_weight = self.world_model_loss_weight
        if self.quantization_scope not in {
            "none",
            "qwen",
            "qwen_dit",
            "qwen_attention",
            "qwen_mlp",
            "qwen_middle_half",
        }:
            raise ValueError(f"Unknown quantization scope: {self.quantization_scope}")
        if self.inference_backend not in {"native", "reference", "triton"}:
            raise ValueError(f"Unknown inference backend: {self.inference_backend}")
        if self.quantization_scope == "none" and self.inference_backend != "native":
            raise ValueError("Quantization-disabled policies must use the native backend.")
        if self.qat_recovery not in {"none", "qwen_edges", "dit_tail4"}:
            raise ValueError(f"Unknown QAT recovery: {self.qat_recovery}")
        if self.qat_recovery == "qwen_edges" and self.quantization_scope != "qwen":
            raise ValueError("qwen_edges recovery requires quantization_scope='qwen'.")
        if self.qat_recovery == "dit_tail4" and self.quantization_scope != "qwen_dit":
            raise ValueError("dit_tail4 recovery requires quantization_scope='qwen_dit'.")
        if self.representation_distillation_weight < 0:
            raise ValueError("representation_distillation_weight must be non-negative.")
        if self.representation_distillation_weight and self.quantization_scope == "none":
            raise ValueError("Representation distillation requires an enabled quantization scope.")

    @classmethod
    def from_vla_jepa(
        cls,
        config: VLAJEPAConfig,
        *,
        source_checkpoint: str,
        source_revision: str | None = None,
        **overrides: Any,
    ) -> BitWAMConfig:
        """Copy an upstream checkpoint config without changing VLA-JEPA behavior."""
        values = {field.name: getattr(config, field.name) for field in fields(VLAJEPAConfig) if field.init}
        values.update(
            source_checkpoint=source_checkpoint,
            source_revision=source_revision,
            world_loss_weight=config.world_model_loss_weight,
        )
        values.update(overrides)
        return cls(**values)
