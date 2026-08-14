"""BitWAM LeRobot policy plugin."""

from lerobot_policy_bitwam.configuration_bitwam import BitWAMConfig

__version__ = "0.1.0"

__all__ = ["BitWAMConfig", "BitWAMPolicy", "__version__"]


def __getattr__(name: str):
    """Keep the model implementation lazy during plugin discovery."""
    if name == "BitWAMPolicy":
        from lerobot_policy_bitwam.modeling_bitwam import BitWAMPolicy

        return BitWAMPolicy
    raise AttributeError(name)
