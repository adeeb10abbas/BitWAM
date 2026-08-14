"""Thin LeRobot policy wrapper around upstream VLA-JEPA."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy
from torch import Tensor, nn

from lerobot_policy_bitwam.configuration_bitwam import BitWAMConfig


class BitWAMPolicy(PreTrainedPolicy):
    """Registered BitWAM policy that preserves upstream BF16 behavior."""

    config_class = BitWAMConfig
    name = "bitwam"

    def __init__(
        self,
        config: BitWAMConfig,
        *,
        upstream_policy: VLAJEPAPolicy | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        config.validate_features()
        if upstream_policy is None:
            upstream_policy = VLAJEPAPolicy(config, **kwargs)
        self.upstream = upstream_policy
        self.upstream.config = config
        from lerobot_policy_bitwam.quantization import convert_for_qat

        self.quantization_report = convert_for_qat(self, config.quantization_scope)
        self.reset()

    @property
    def model(self) -> nn.Module:
        """Expose the native model tree used by conversion and export code."""
        return self.upstream.model

    @classmethod
    def from_source_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        revision: str | None = None,
        config_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> BitWAMPolicy:
        """Load an upstream VLA-JEPA checkpoint and wrap it without changing weights."""
        upstream = VLAJEPAPolicy.from_pretrained(checkpoint, revision=revision, **kwargs)
        config = BitWAMConfig.from_vla_jepa(
            upstream.config,
            source_checkpoint=str(checkpoint),
            source_revision=revision,
            **(config_overrides or {}),
        )
        return cls(config, upstream_policy=upstream)

    def reset(self) -> None:
        self.upstream.reset()

    def get_optim_params(self):
        return self.upstream.get_optim_params()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        return self.upstream.forward(batch)

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        return self.upstream.predict_action_chunk(batch, **kwargs)

    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        return self.upstream.select_action(batch, **kwargs)
