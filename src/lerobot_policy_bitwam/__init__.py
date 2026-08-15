"""BitWAM policy package with lazy framework-specific imports."""

__version__ = "0.1.0"

__all__ = [
    "BitWAMConfig",
    "BitWAMPolicy",
    "QuantizationReport",
    "TernaryLinear",
    "__version__",
    "convert_for_qat",
]


def __getattr__(name: str):
    """Keep the model implementation lazy during plugin discovery."""
    if name == "BitWAMConfig":
        from lerobot_policy_bitwam.configuration_bitwam import BitWAMConfig

        return BitWAMConfig
    if name == "BitWAMPolicy":
        from lerobot_policy_bitwam.modeling_bitwam import BitWAMPolicy

        return BitWAMPolicy
    if name in {"QuantizationReport", "TernaryLinear", "convert_for_qat"}:
        from lerobot_policy_bitwam import quantization

        return getattr(quantization, name)
    raise AttributeError(name)
