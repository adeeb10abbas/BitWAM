"""
1-bit VLA Research Package

A research package for exploring Vision-Language-Action models with 
1.58-bit quantization, building on BitNet principles for efficient 
robotics policies.

Key Features:
- BitNet-based VLA models with 1.58-bit weights
- BitACT policies for continuous control
- Training utilities and benchmarks

Author: Ali Adeeb
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Ali Adeeb"

# Core model imports
from .models.vla_bitnet import VLABitNet, BitLinear
from .models.vision_encoder import VisionEncoder  
from .models.language_encoder import LanguageEncoder
from .models.action_decoder import ActionDecoder

# Policy imports
from .policies.bitact_policy import BitACTPolicy, BitACTConfig

# Training utilities
from .training.trainer import VLATrainer
from .training.bitnet_optimizer import BitNetOptimizer

# Utilities
from .utils.quantization import (
    absmean_quantize_weights, 
    absmax_quantize_activations
)
from .utils.data_loading import create_sample_data
from .utils.model_analysis import print_model_info, analyze_quantization

__all__ = [
    # Models
    "VLABitNet",
    "BitLinear", 
    "VisionEncoder",
    "LanguageEncoder",
    "ActionDecoder",
    
    # Policies
    "BitACTPolicy",
    "BitACTConfig",
    
    # Training
    "VLATrainer",
    "BitNetOptimizer",
    
    # Utils
    "absmean_quantize_weights",
    "absmax_quantize_activations", 
    "create_sample_data",
    "print_model_info",
    "analyze_quantization",
] 