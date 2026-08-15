"""Thin LeRobot policy wrapper around upstream VLA-JEPA."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.vla_jepa.modeling_vla_jepa import VLAJEPAPolicy, _get_autocast_context
from torch import Tensor, nn

from lerobot_policy_bitwam.configuration_bitwam import BitWAMConfig


def _representation_alignment_loss(student: Tensor, teacher: Tensor) -> Tensor:
    """Match pre-RMS tokens without letting their large native scale dominate QAT."""
    student = student.float()
    teacher = teacher.float()
    student_rms = student.square().mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    teacher_rms = teacher.square().mean(dim=-1, keepdim=True).clamp_min(1e-8).sqrt()
    direction_loss = F.mse_loss(student / student_rms, teacher / teacher_rms)
    scale_loss = F.mse_loss(student_rms.log(), teacher_rms.log())
    return direction_loss + 0.1 * scale_loss


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
        teacher_qwen = None
        if config.representation_distillation_weight:
            # Keep the fixed BF16 source teacher outside the module tree: it must not
            # enter optimizer groups or double the size of exported checkpoints.
            teacher_qwen = copy.deepcopy(self.upstream.model.qwen).requires_grad_(False).eval()
        object.__setattr__(self, "_representation_teacher_qwen", teacher_qwen)
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
        if not self.config.representation_distillation_weight:
            return self.upstream.forward(batch)

        inputs = self.upstream._prepare_model_inputs(batch, training=True)
        model = self.upstream.model
        embodied_tokens, action_tokens = model._encode_qwen(
            inputs["images"],
            inputs["instructions"],
            need_action_tokens=model.config.enable_world_model,
        )
        if model.config.enable_world_model and "videos" in inputs:
            wm_loss = model._world_model_loss(inputs["videos"], action_tokens)
        else:
            wm_loss = torch.zeros((), device=embodied_tokens.device)

        if "actions" in inputs:
            action_loss = model._action_loss(
                embodied_tokens,
                inputs["actions"],
                inputs.get("state"),
                inputs.get("action_is_pad"),
            )
        else:
            action_loss = torch.zeros((), device=embodied_tokens.device)

        with torch.no_grad():
            teacher_embodied, teacher_actions = self._encode_teacher(
                inputs["images"],
                inputs["instructions"],
                need_action_tokens=model.config.enable_world_model,
            )
        representation_loss = _representation_alignment_loss(embodied_tokens, teacher_embodied)
        if action_tokens is not None and teacher_actions is not None:
            representation_loss = 0.5 * (
                representation_loss + _representation_alignment_loss(action_tokens, teacher_actions)
            )

        weighted_wm_loss = wm_loss * model.config.world_model_loss_weight
        weighted_representation_loss = (
            representation_loss * self.config.representation_distillation_weight
        )
        total_loss = action_loss + weighted_wm_loss + weighted_representation_loss
        logs = {
            "action_loss": action_loss.detach().item(),
            "wm_loss": weighted_wm_loss.detach().item(),
            "representation_loss": representation_loss.detach().item(),
            "loss": total_loss.detach().item(),
        }
        return total_loss, logs

    def _encode_teacher(
        self,
        images: list[list[Tensor]],
        instructions: list[str],
        *,
        need_action_tokens: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Gather source-BF16 tokens using the same prompts as the ternary student."""
        teacher = self._representation_teacher_qwen
        if teacher is None:
            raise RuntimeError("Representation teacher was not initialized.")
        device = next(self.upstream.parameters()).device
        if next(teacher.parameters()).device != device:
            teacher.to(device)
        teacher.eval()

        model = self.upstream.model
        qwen_inputs = teacher.build_inputs(
            images=images,
            instructions=instructions,
            action_prompt=model.replace_prompt,
            embodied_prompt=model.embodied_replace_prompt,
        )
        input_ids = qwen_inputs["input_ids"]
        embodied_idx = (input_ids == model.embodied_action_token_id).nonzero(as_tuple=True)
        action_idx = None
        if need_action_tokens:
            action_ids = model._action_token_ids_t.to(input_ids.device)
            action_idx = torch.isin(input_ids, action_ids).nonzero(as_tuple=True)

        captured: list[Tensor] = []

        def capture_last_layer(_module, _inputs, output) -> None:
            captured.append(output[0] if isinstance(output, tuple) else output)

        last_layer = teacher.model.model.language_model.layers[-1]
        handle = last_layer.register_forward_hook(capture_last_layer)
        try:
            with _get_autocast_context(device.type, torch.bfloat16):
                teacher.model(
                    **qwen_inputs,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True,
                )
        finally:
            handle.remove()

        last_hidden = captured[0]
        batch_size, _, hidden_size = last_hidden.shape
        embodied_tokens = last_hidden[embodied_idx[0], embodied_idx[1], :].view(
            batch_size, -1, hidden_size
        )
        action_tokens = (
            last_hidden[action_idx[0], action_idx[1], :].view(batch_size, -1, hidden_size)
            if action_idx is not None
            else None
        )
        return embodied_tokens, action_tokens

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        return self.upstream.predict_action_chunk(batch, **kwargs)

    def select_action(self, batch: dict[str, Tensor], **kwargs: Any) -> Tensor:
        return self.upstream.select_action(batch, **kwargs)
