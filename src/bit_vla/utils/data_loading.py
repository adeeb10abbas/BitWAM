"""
Data loading utilities for 1-bit VLA research.

Functions for creating sample data and loading datasets for testing.
"""

import torch
from typing import Tuple, Dict


def create_sample_data(
    batch_size: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    max_seq_len: int = 32,
    vocab_size: int = 1000,
    action_dim: int = 7
) -> Dict[str, torch.Tensor]:
    """
    Create sample data for testing VLA models.
    
    Args:
        batch_size: Number of samples in batch
        image_size: Size of input images (height, width)
        max_seq_len: Maximum sequence length for text
        vocab_size: Size of vocabulary for token IDs
        action_dim: Dimensionality of action space
        
    Returns:
        Dictionary containing sample data tensors
        
    Example:
        >>> data = create_sample_data(batch_size=2)
        >>> print(data['images'].shape)  # torch.Size([2, 3, 224, 224])
        >>> print(data['token_ids'].shape)  # torch.Size([2, 32])
    """
    # Create sample RGB images
    images = torch.randn(batch_size, 3, image_size[0], image_size[1])
    
    # Create sample token IDs for language instructions
    token_ids = torch.randint(
        0, vocab_size, (batch_size, max_seq_len), dtype=torch.long
    )
    
    # Create attention mask (1 for real tokens, 0 for padding)
    # Simulate variable length sequences
    attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.bool)
    for i in range(batch_size):
        # Random sequence length between 10 and max_seq_len
        seq_len = torch.randint(10, max_seq_len, (1,)).item()
        attention_mask[i, seq_len:] = 0
    
    # Create sample actions (continuous control)
    actions = torch.randn(batch_size, action_dim)
    
    return {
        "images": images,
        "token_ids": token_ids,
        "attention_mask": attention_mask,
        "actions": actions,
    }


def create_pusht_sample_data(batch_size: int = 4) -> Dict[str, torch.Tensor]:
    """
    Create sample data matching PushT environment format.
    
    Args:
        batch_size: Number of samples in batch
        
    Returns:
        Dictionary with PushT-formatted sample data
    """
    # PushT uses 96x96 images and 2D actions
    return {
        "observation.image": torch.randn(batch_size, 3, 96, 96),
        "observation.state": torch.randn(batch_size, 2),  # 2D state
        "action": torch.randn(batch_size, 8, 2),  # 8 action steps, 2D
    }


def create_sequence_data(
    batch_size: int = 4,
    sequence_length: int = 50,
    action_dim: int = 7
) -> Dict[str, torch.Tensor]:
    """
    Create sequential data for temporal modeling.
    
    Args:
        batch_size: Number of sequences
        sequence_length: Length of each sequence
        action_dim: Action dimensionality
        
    Returns:
        Dictionary with sequential data
    """
    return {
        "observations": torch.randn(batch_size, sequence_length, 3, 64, 64),
        "states": torch.randn(batch_size, sequence_length, 8),
        "actions": torch.randn(batch_size, sequence_length, action_dim),
        "rewards": torch.randn(batch_size, sequence_length, 1),
        "dones": torch.randint(0, 2, (batch_size, sequence_length, 1)),
    }


def normalize_data(data: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Normalize data to zero mean and unit variance.
    
    Args:
        data: Input tensor to normalize
        dim: Dimension along which to compute statistics
        
    Returns:
        Normalized tensor
    """
    mean = data.mean(dim=dim, keepdim=True)
    std = data.std(dim=dim, keepdim=True)
    return (data - mean) / (std + 1e-8)


def add_noise(
    data: torch.Tensor, 
    noise_level: float = 0.1
) -> torch.Tensor:
    """
    Add Gaussian noise to data for robustness testing.
    
    Args:
        data: Input tensor
        noise_level: Standard deviation of noise relative to data std
        
    Returns:
        Noisy tensor
    """
    noise = torch.randn_like(data) * data.std() * noise_level
    return data + noise


def create_multimodal_batch(
    batch_size: int = 4,
    include_language: bool = True,
    include_proprioception: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Create a comprehensive multimodal batch for VLA testing.
    
    Args:
        batch_size: Number of samples
        include_language: Whether to include language instructions
        include_proprioception: Whether to include proprioceptive state
        
    Returns:
        Dictionary with multimodal data
    """
    batch = {
        "rgb": torch.randn(batch_size, 3, 224, 224),
        "depth": torch.randn(batch_size, 1, 224, 224),
        "actions": torch.randn(batch_size, 7),
    }
    
    if include_language:
        batch.update({
            "instruction_tokens": torch.randint(0, 1000, (batch_size, 32)),
            "instruction_mask": torch.ones(batch_size, 32, dtype=torch.bool),
        })
    
    if include_proprioception:
        batch.update({
            "joint_positions": torch.randn(batch_size, 7),
            "joint_velocities": torch.randn(batch_size, 7),
            "end_effector_pose": torch.randn(batch_size, 6),
        })
    
    return batch 