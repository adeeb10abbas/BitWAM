"""VLA-JEPA processor reuse for BitWAM."""

from typing import Any

from lerobot.policies.vla_jepa.processor_vla_jepa import make_vla_jepa_pre_post_processors

from lerobot_policy_bitwam.configuration_bitwam import BitWAMConfig


def make_bitwam_pre_post_processors(
    config: BitWAMConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
):
    """Build the same preprocessing and postprocessing pipelines as VLA-JEPA."""
    return make_vla_jepa_pre_post_processors(config, dataset_stats=dataset_stats)
