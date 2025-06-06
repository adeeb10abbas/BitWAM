"""Tests for bit_vla policies."""

import torch
from bit_vla.policies import BitACTPolicy, BitACTConfig


class TestBitACTConfig:
    """Test BitACT configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BitACTConfig()
        assert config.hidden_dim > 0
        assert config.action_dim > 0
        assert config.chunk_size > 0
        assert config.n_layer > 0
        assert config.n_head > 0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BitACTConfig(
            hidden_dim=512,
            action_dim=14,
            chunk_size=50,
            n_layer=8,
            n_head=16
        )
        assert config.hidden_dim == 512
        assert config.action_dim == 14
        assert config.chunk_size == 50
        assert config.n_layer == 8
        assert config.n_head == 16


class TestBitACTPolicy:
    """Test BitACT policy."""
    
    def test_init(self):
        """Test BitACTPolicy initialization."""
        config = BitACTConfig(hidden_dim=256, action_dim=7)
        policy = BitACTPolicy(config)
        assert isinstance(policy, torch.nn.Module)
        assert hasattr(policy, 'config')
    
    def test_forward(self):
        """Test BitACTPolicy forward pass."""
        config = BitACTConfig(hidden_dim=256, action_dim=7, chunk_size=10)
        policy = BitACTPolicy(config)
        
        # Create sample observation
        obs = torch.randn(2, 256)  # batch_size=2, obs_dim=256
        
        # Forward pass
        actions = policy(obs)
        assert actions.shape == (2, config.chunk_size, config.action_dim)
    
    def test_predict(self):
        """Test action prediction method."""
        config = BitACTConfig(hidden_dim=256, action_dim=7, chunk_size=10)
        policy = BitACTPolicy(config)
        
        obs = torch.randn(1, 256)
        action = policy.predict(obs)
        assert action.shape == (config.action_dim,)
    
    def test_predict_chunk(self):
        """Test chunk prediction method."""
        config = BitACTConfig(hidden_dim=256, action_dim=7, chunk_size=10)
        policy = BitACTPolicy(config)
        
        obs = torch.randn(1, 256)
        chunk = policy.predict_chunk(obs)
        assert chunk.shape == (config.chunk_size, config.action_dim)
    
    def test_quantization_summary(self):
        """Test quantization summary for policy."""
        config = BitACTConfig(hidden_dim=256, action_dim=7)
        policy = BitACTPolicy(config)
        
        summary = policy.get_quantization_summary()
        assert 'total_parameters' in summary
        assert 'quantized_parameters' in summary
        assert 'quantized_ratio' in summary
    
    def test_device_placement(self):
        """Test policy device placement."""
        if torch.cuda.is_available():
            config = BitACTConfig(hidden_dim=256, action_dim=7)
            policy = BitACTPolicy(config).cuda()
            
            obs = torch.randn(2, 256).cuda()
            actions = policy(obs)
            assert actions.device.type == 'cuda'


class TestPolicyIntegration:
    """Integration tests for policies."""
    
    def test_gradient_flow(self):
        """Test gradient flow through policy."""
        config = BitACTConfig(hidden_dim=256, action_dim=7, chunk_size=10)
        policy = BitACTPolicy(config)
        
        obs = torch.randn(2, 256, requires_grad=True)
        actions = policy(obs)
        loss = actions.sum()
        loss.backward()
        
        # Check gradients exist
        for name, param in policy.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_training_mode(self):
        """Test training vs eval mode."""
        config = BitACTConfig(hidden_dim=256, action_dim=7)
        policy = BitACTPolicy(config)
        
        # Training mode
        policy.train()
        obs = torch.randn(2, 256)
        actions_train = policy(obs)
        
        # Eval mode
        policy.eval()
        with torch.no_grad():
            actions_eval = policy(obs)
        
        # Shapes should be the same
        assert actions_train.shape == actions_eval.shape
    
    def test_deterministic_inference(self):
        """Test deterministic inference in eval mode."""
        config = BitACTConfig(hidden_dim=256, action_dim=7)
        policy = BitACTPolicy(config)
        policy.eval()
        
        obs = torch.randn(1, 256)
        
        with torch.no_grad():
            actions1 = policy.predict(obs)
            actions2 = policy.predict(obs)
        
        # Should be deterministic in eval mode
        torch.testing.assert_close(actions1, actions2) 