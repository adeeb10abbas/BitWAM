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
    # Keep each repeated transformer unit independently shardable on the two local GPUs.
    _fsdp_wrap_modules = [
        "Qwen3VLTextDecoderLayer",
        "Qwen3VLVisionBlock",
        "BasicTransformerBlock",
        "ACBlock",
    ]

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

        self.quantization_report = convert_for_qat(
            self,
            config.quantization_scope,
            recovery=config.qat_recovery,
        )
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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: BitWAMConfig | None = None,
        revision: str | None = None,
        **kwargs: Any,
    ) -> BitWAMPolicy:
        """Load either a native VLA-JEPA source or an exported BitWAM checkpoint."""
        is_native_source = config is not None and str(pretrained_name_or_path) == str(
            config.source_checkpoint
        )
        if is_native_source:
            upstream = VLAJEPAPolicy.from_pretrained(
                pretrained_name_or_path,
                config=config,
                revision=revision or config.source_revision,
                **kwargs,
            )
            policy = cls(config, upstream_policy=upstream)
            policy.eval()
            return policy
        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            revision=revision,
            **kwargs,
        )

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
