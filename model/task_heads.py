import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class DeepMLP(nn.Module):
    """
    Multi-layer perceptron with configurable depth for task-specific heads.
    
    Features:
    - Configurable number of hidden layers
    - Batch normalization for training stability
    - GELU activation for better gradient flow
    - Dropout for regularization
    """
    
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024, depth: int = 3, dropout: float = 0.3):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output dimension (number of classes)
            hidden: Hidden layer dimension
            depth: Number of layers (including output layer)
            dropout: Dropout probability
        """
        super().__init__()
        
        if depth < 1:
            raise ValueError(f"Depth must be at least 1, got {depth}")
        
        layers = []
        dim = in_dim
        
        # Hidden layers with BatchNorm, GELU, and Dropout
        for _ in range(depth - 1):
            layers.extend([
                nn.Linear(dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            dim = hidden
        
        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Store config for debugging/analysis
        self.config = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'hidden': hidden,
            'depth': depth,
            'dropout': dropout
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, in_dim)
            
        Returns:
            Output tensor of shape (B, out_dim)
        """
        return self.mlp(x)


class SEMLP(nn.Module):
    """
    Squeeze-and-Excitation MLP combining channel attention with deep MLP.
    
    The SE module applies learned channel-wise attention weights to the input features
    before passing them through the MLP, allowing the model to focus on the most
    relevant features for each task.
    """
    
    def __init__(self, in_dim: int, out_dim: int, reduction: int = 4, **mlp_kwargs):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output dimension (number of classes)
            reduction: SE reduction ratio (in_dim // reduction for bottleneck)
            **mlp_kwargs: Additional arguments passed to DeepMLP
        """
        super().__init__()
        
        if in_dim // reduction < 1:
            raise ValueError(f"Reduction ratio {reduction} is too large for input dim {in_dim}")
        
        # Squeeze-and-Excitation module
        self.se = nn.Sequential(
            nn.Linear(in_dim, in_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // reduction, in_dim),
            nn.Sigmoid()
        )
        
        # Deep MLP for actual classification
        self.mlp = DeepMLP(in_dim, out_dim, **mlp_kwargs)
        
        # Store config for debugging/analysis
        self.config = {
            'in_dim': in_dim,
            'out_dim': out_dim,
            'reduction': reduction,
            'mlp_config': self.mlp.config
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with SE attention.
        
        Args:
            x: Input tensor of shape (B, in_dim)
            
        Returns:
            Output tensor of shape (B, out_dim)
        """
        # Compute channel attention weights
        attention_weights = self.se(x)  # (B, in_dim)
        
        # Apply attention to input features
        attended_features = x * attention_weights  # (B, in_dim)
        
        # Pass through MLP
        return self.mlp(attended_features)


def build_task_head(head_type: str, in_dim: int, out_dim: int, **kwargs) -> nn.Module:
    """
    Factory function for creating task-specific heads.
    
    Args:
        head_type: Type of head ('linear', 'deep_mlp', 'se_mlp')
        in_dim: Input feature dimension
        out_dim: Output dimension (number of classes)
        **kwargs: Additional arguments passed to the head constructor
        
    Returns:
        The constructed head module
        
    Raises:
        ValueError: If head_type is not recognized
    """
    head_type = head_type.lower()
    
    if head_type == 'linear':
        # Simple linear head (backward compatibility)
        return nn.Linear(in_dim, out_dim)
    elif head_type == 'deep_mlp':
        return DeepMLP(in_dim, out_dim, **kwargs)
    elif head_type == 'se_mlp':
        return SEMLP(in_dim, out_dim, **kwargs)
    else:
        raise ValueError(f'Unknown head_type: {head_type}. Supported types: linear, deep_mlp, se_mlp')


def get_head_info(head: nn.Module) -> Dict[str, Any]:
    """
    Get information about a task head for debugging/analysis.
    
    Args:
        head: The head module
        
    Returns:
        Dictionary containing head information
    """
    info = {
        'type': head.__class__.__name__,
        'parameters': sum(p.numel() for p in head.parameters()),
        'trainable_parameters': sum(p.numel() for p in head.parameters() if p.requires_grad)
    }
    
    # Add config if available
    if hasattr(head, 'config'):
        info['config'] = head.config
    
    return info


def test_task_heads():
    """Test function to verify task heads work correctly."""
    batch_size = 4
    in_dim = 1024
    out_dim = 2
    
    # Test input
    x = torch.randn(batch_size, in_dim)
    
    print("Testing task heads...")
    
    # Test linear head
    linear_head = build_task_head('linear', in_dim, out_dim)
    linear_out = linear_head(x)
    print(f"Linear head output shape: {linear_out.shape}")
    print(f"Linear head parameters: {sum(p.numel() for p in linear_head.parameters())}")
    
    # Test DeepMLP
    deep_mlp = build_task_head('deep_mlp', in_dim, out_dim, hidden=512, depth=3, dropout=0.2)
    deep_out = deep_mlp(x)
    print(f"DeepMLP output shape: {deep_out.shape}")
    print(f"DeepMLP parameters: {sum(p.numel() for p in deep_mlp.parameters())}")
    
    # Test SEMLP
    se_mlp = build_task_head('se_mlp', in_dim, out_dim, reduction=8, hidden=512, depth=2)
    se_out = se_mlp(x)
    print(f"SEMLP output shape: {se_out.shape}")
    print(f"SEMLP parameters: {sum(p.numel() for p in se_mlp.parameters())}")
    
    # Test head info
    print("\nHead information:")
    for name, head in [('Linear', linear_head), ('DeepMLP', deep_mlp), ('SEMLP', se_mlp)]:
        info = get_head_info(head)
        print(f"{name}: {info}")
    
    print("All task heads work correctly!")


if __name__ == "__main__":
    test_task_heads() 