"""Tests for bit_vla training utilities."""

import torch
import torch.nn as nn
from bit_vla.training import BitNetOptimizer
from bit_vla.models import VLABitNet, BitLinear


class TestBitNetOptimizer:
    """Test BitNet optimizer."""
    
    def test_init(self):
        """Test BitNetOptimizer initialization."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        optimizer = BitNetOptimizer(model.parameters())
        
        assert hasattr(optimizer, 'param_groups')
        assert len(optimizer.param_groups) == 2  # BitLinear and FP16 groups
    
    def test_parameter_grouping(self):
        """Test parameter grouping for different layer types."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        optimizer = BitNetOptimizer(model.parameters())
        
        # Should have different learning rates for different groups
        lrs = [group['lr'] for group in optimizer.param_groups]
        assert len(set(lrs)) > 1  # Should have different learning rates
    
    def test_optimization_step(self):
        """Test optimization step."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        optimizer = BitNetOptimizer(model.parameters())
        
        # Forward pass
        images = torch.randn(2, 3, 224, 224)
        tokens = torch.randint(0, 1000, (2, 20))
        actions = model(images, tokens)
        loss = actions.sum()
        
        # Backward pass
        loss.backward()
        
        # Optimization step
        optimizer.step()
        optimizer.zero_grad()
        
        # Should complete without error
        assert True
    
    def test_stage_transition(self):
        """Test training stage transition."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        optimizer = BitNetOptimizer(model.parameters())
        
        # Get initial learning rates
        initial_lrs = [group['lr'] for group in optimizer.param_groups]
        
        # Transition to stage 2
        optimizer.transition_to_stage2()
        
        # Learning rates should change
        new_lrs = [group['lr'] for group in optimizer.param_groups]
        assert initial_lrs != new_lrs
    
    def test_learning_rate_schedule(self):
        """Test learning rate scheduling."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        optimizer = BitNetOptimizer(model.parameters())
        
        # Test warmup
        for step in range(100):
            lr = optimizer.get_lr_scale(step, warmup_steps=50)
            if step < 50:
                assert lr <= 1.0  # Should be ramping up
            else:
                assert lr == 1.0  # Should be at full scale


class TestTrainingCompatibility:
    """Test training compatibility with models."""
    
    def test_bitlinear_training(self):
        """Test training BitLinear layers."""
        layer = BitLinear(128, 64)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)
        
        # Training step
        x = torch.randn(32, 128)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Should complete without error
        assert True
    
    def test_full_model_training(self):
        """Test training full VLA model."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        optimizer = BitNetOptimizer(model.parameters())
        
        # Training loop
        for _ in range(3):
            images = torch.randn(2, 3, 224, 224)
            tokens = torch.randint(0, 1000, (2, 20))
            
            # Forward pass
            actions = model(images, tokens)
            loss = actions.sum()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # Should complete without error
        assert True
    
    def test_gradient_clipping(self):
        """Test gradient clipping compatibility."""
        model = VLABitNet(hidden_dim=256, action_dim=7)
        optimizer = BitNetOptimizer(model.parameters())
        
        # Forward pass with large loss
        images = torch.randn(2, 3, 224, 224)
        tokens = torch.randint(0, 1000, (2, 20))
        actions = model(images, tokens)
        loss = actions.sum() * 1000  # Large loss for large gradients
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check gradients are clipped
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        # Should be approximately clipped (allowing for numerical precision)
        assert total_norm <= 1.1
    
    def test_mixed_precision_compatibility(self):
        """Test mixed precision training compatibility."""
        if torch.cuda.is_available():
            model = VLABitNet(hidden_dim=256, action_dim=7).cuda()
            optimizer = BitNetOptimizer(model.parameters())
            scaler = torch.cuda.amp.GradScaler()
            
            images = torch.randn(2, 3, 224, 224).cuda()
            tokens = torch.randint(0, 1000, (2, 20)).cuda()
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                actions = model(images, tokens)
                loss = actions.sum()
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Should complete without error
            assert True


class TestMemoryEfficiency:
    """Test memory efficiency of training."""
    
    def test_memory_usage(self):
        """Test memory usage during training."""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        model = VLABitNet(hidden_dim=256, action_dim=7)
        optimizer = BitNetOptimizer(model.parameters())
        
        # Get initial memory if on GPU
        if torch.cuda.is_available():
            model = model.cuda()
            initial_memory = torch.cuda.memory_allocated()
        
        # Training step
        batch_size = 16  # Larger batch for memory test
        images = torch.randn(batch_size, 3, 224, 224)
        tokens = torch.randint(0, 1000, (batch_size, 20))
        
        if torch.cuda.is_available():
            images = images.cuda()
            tokens = tokens.cuda()
        
        actions = model(images, tokens)
        loss = actions.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Should complete without memory errors
        assert True
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            # Memory usage should be reasonable
            assert final_memory < initial_memory + 500 * 1024 * 1024  # 500MB limit 