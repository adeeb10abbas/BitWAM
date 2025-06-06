"""Tests for bit_vla models."""

import pytest
import torch
import torch.nn as nn
from bit_vla.models import (
    BitLinear,
    VisionEncoder,
    LanguageEncoder,
    ActionDecoder,
    VLABitNet,
)


class TestBitLinear:
    """Test BitLinear layer."""
    
    def test_init(self):
        """Test BitLinear initialization."""
        layer = BitLinear(128, 64)
        assert layer.in_features == 128
        assert layer.out_features == 64
        assert hasattr(layer, 'weight')
        assert hasattr(layer, 'bias')
    
    def test_forward(self):
        """Test BitLinear forward pass."""
        layer = BitLinear(128, 64)
        x = torch.randn(32, 128)
        y = layer(x)
        assert y.shape == (32, 64)
    
    def test_quantization(self):
        """Test weight quantization."""
        layer = BitLinear(128, 64)
        # Weights should be quantized during forward pass
        x = torch.randn(32, 128)
        _ = layer(x)
        # Check that weights are in expected range
        assert layer.weight.abs().max() <= 1.1  # Allow small numerical error


class TestVisionEncoder:
    """Test vision encoder."""
    
    def test_init(self):
        """Test VisionEncoder initialization."""
        encoder = VisionEncoder(hidden_dim=256)
        assert isinstance(encoder, nn.Module)
    
    def test_forward(self):
        """Test VisionEncoder forward pass."""
        encoder = VisionEncoder(hidden_dim=256)
        # Test with RGB images
        images = torch.randn(2, 3, 224, 224)
        features = encoder(images)
        assert features.shape == (2, 256)


class TestLanguageEncoder:
    """Test language encoder."""
    
    def test_init(self):
        """Test LanguageEncoder initialization."""
        encoder = LanguageEncoder(vocab_size=1000, hidden_dim=256)
        assert isinstance(encoder, nn.Module)
    
    def test_forward(self):
        """Test LanguageEncoder forward pass."""
        encoder = LanguageEncoder(vocab_size=1000, hidden_dim=256, max_length=50)
        tokens = torch.randint(0, 1000, (2, 20))
        features = encoder(tokens)
        assert features.shape == (2, 256)


class TestActionDecoder:
    """Test action decoder."""
    
    def test_init(self):
        """Test ActionDecoder initialization."""
        decoder = ActionDecoder(hidden_dim=256, action_dim=7)
        assert isinstance(decoder, nn.Module)
    
    def test_forward(self):
        """Test ActionDecoder forward pass."""
        decoder = ActionDecoder(hidden_dim=256, action_dim=7)
        features = torch.randn(2, 256)
        actions = decoder(features)
        assert actions.shape == (2, 7)


class TestVLABitNet:
    """Test complete VLA model."""
    
    def test_init(self):
        """Test VLABitNet initialization."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        assert isinstance(model, nn.Module)
        assert hasattr(model, 'vision_encoder')
        assert hasattr(model, 'language_encoder')
        assert hasattr(model, 'action_decoder')
    
    def test_forward(self):
        """Test VLABitNet forward pass."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        images = torch.randn(2, 3, 224, 224)
        tokens = torch.randint(0, 1000, (2, 20))
        
        actions = model(images, tokens)
        assert actions.shape == (2, 7)
    
    def test_quantization_summary(self):
        """Test quantization summary method."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        summary = model.get_quantization_summary()
        
        assert 'total_parameters' in summary
        assert 'quantized_parameters' in summary
        assert 'quantized_ratio' in summary
        assert isinstance(summary['total_parameters'], int)
        assert 0 <= summary['quantized_ratio'] <= 100
    
    def test_device_placement(self):
        """Test model device placement."""
        if torch.cuda.is_available():
            model = VLABitNet(hidden_dim=256, action_dim=7)
            model = model.cuda()
            
            images = torch.randn(2, 3, 224, 224).cuda()
            tokens = torch.randint(0, 1000, (2, 20)).cuda()
            
            actions = model(images, tokens)
            assert actions.device.type == 'cuda'


class TestModelIntegration:
    """Integration tests for models."""
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        images = torch.randn(2, 3, 224, 224)
        tokens = torch.randint(0, 1000, (2, 20))
        
        # Forward pass
        actions = model(images, tokens)
        loss = actions.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_parameter_counts(self):
        """Test parameter counting is reasonable."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be reasonable number of parameters
        assert 100_000 < total_params < 10_000_000
        
        # Should have some quantized parameters
        summary = model.get_quantization_summary()
        assert summary['quantized_parameters'] > 0 