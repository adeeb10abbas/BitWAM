"""Tests for bit_vla utils."""

import torch
import torch.nn as nn
from bit_vla.utils import (
    quantize_weight_absmean,
    dequantize_weight,
    print_model_info,
    get_model_size,
    create_sample_data,
)
from bit_vla.models import BitLinear, VLABitNet


class TestQuantization:
    """Test quantization utilities."""
    
    def test_quantize_weight_absmean(self):
        """Test absmean weight quantization."""
        weight = torch.randn(64, 128)
        quantized, scale = quantize_weight_absmean(weight)
        
        # Check output shape
        assert quantized.shape == weight.shape
        assert isinstance(scale, torch.Tensor)
        
        # Check quantization values
        unique_vals = torch.unique(quantized)
        assert len(unique_vals) <= 3  # Should be {-1, 0, 1}
        assert quantized.abs().max() <= 1.0
    
    def test_dequantize_weight(self):
        """Test weight dequantization."""
        weight = torch.randn(64, 128)
        quantized, scale = quantize_weight_absmean(weight)
        dequantized = dequantize_weight(quantized, scale)
        
        # Check shape preserved
        assert dequantized.shape == weight.shape
        
        # Should be approximately reconstructed
        # (won't be exact due to quantization loss)
        relative_error = (weight - dequantized).norm() / weight.norm()
        assert relative_error < 1.0  # Some reconstruction error is expected
    
    def test_quantization_identity(self):
        """Test quantization of special weights."""
        # Test zero weight
        zero_weight = torch.zeros(32, 64)
        q_zero, s_zero = quantize_weight_absmean(zero_weight)
        assert torch.allclose(q_zero, zero_weight)
        
        # Test constant weight
        const_weight = torch.ones(32, 64)
        q_const, s_const = quantize_weight_absmean(const_weight)
        assert torch.allclose(q_const.abs(), torch.ones_like(q_const))


class TestModelAnalysis:
    """Test model analysis utilities."""
    
    def test_print_model_info(self):
        """Test model info printing."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        # Should not raise an exception
        print_model_info(model, "Test Model")
    
    def test_get_model_size(self):
        """Test model size calculation."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        size_mb = get_model_size(model)
        
        assert isinstance(size_mb, float)
        assert size_mb > 0
        assert size_mb < 1000  # Should be reasonable size
    
    def test_quantization_summary(self):
        """Test quantization summary for different models."""
        # Test BitLinear layer
        layer = BitLinear(128, 64)
        summary = layer.get_quantization_summary()
        
        assert 'total_parameters' in summary
        assert 'quantized_parameters' in summary
        assert 'quantized_ratio' in summary
        
        # BitLinear should have high quantization ratio
        assert summary['quantized_ratio'] > 50
        
        # Test full model
        model = VLABitNet(hidden_dim=256, action_dim=7)
        model_summary = model.get_quantization_summary()
        assert model_summary['quantized_ratio'] > 0


class TestDataLoading:
    """Test data loading utilities."""
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        data = create_sample_data(batch_size=4)
        
        assert 'images' in data
        assert 'token_ids' in data
        
        images = data['images']
        tokens = data['token_ids']
        
        # Check shapes
        assert images.shape[0] == 4  # batch_size
        assert images.shape[1] == 3  # RGB channels
        assert len(tokens.shape) == 2  # batch_size x sequence_length
        assert tokens.shape[0] == 4  # batch_size
    
    def test_sample_data_types(self):
        """Test sample data types."""
        data = create_sample_data(batch_size=2)
        
        assert isinstance(data['images'], torch.Tensor)
        assert isinstance(data['token_ids'], torch.Tensor)
        
        # Check data types
        assert data['images'].dtype == torch.float32
        assert data['token_ids'].dtype == torch.long
    
    def test_sample_data_ranges(self):
        """Test sample data value ranges."""
        data = create_sample_data(batch_size=2)
        
        images = data['images']
        tokens = data['token_ids']
        
        # Images should be in reasonable range (0-1 or normalized)
        assert images.min() >= -5.0  # Allow for normalization
        assert images.max() <= 5.0
        
        # Tokens should be valid indices
        assert tokens.min() >= 0
        assert tokens.max() < 10000  # Assuming reasonable vocab size


class TestModelCompatibility:
    """Test model compatibility with utilities."""
    
    def test_bitlinear_with_utils(self):
        """Test BitLinear compatibility with utils."""
        layer = BitLinear(128, 64)
        
        # Test that we can get model info
        print_model_info(layer, "BitLinear Layer")
        
        # Test that we can get size
        size = get_model_size(layer)
        assert size > 0
    
    def test_full_model_with_utils(self):
        """Test full model compatibility."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        
        # Test with sample data
        data = create_sample_data(batch_size=2)
        
        with torch.no_grad():
            actions = model(data['images'], data['token_ids'])
        
        assert actions.shape == (2, 7)
    
    def test_model_device_compatibility(self):
        """Test device compatibility."""
        if torch.cuda.is_available():
            model = VLABitNet(hidden_dim=256, action_dim=7).cuda()
            data = create_sample_data(batch_size=2)
            
            # Move data to GPU
            images = data['images'].cuda()
            tokens = data['token_ids'].cuda()
            
            with torch.no_grad():
                actions = model(images, tokens)
            
            assert actions.device.type == 'cuda' 