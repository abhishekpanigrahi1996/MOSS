"""
Model building blocks for transformer architecture.

This module implements foundational transformer components optimized for
continuous data (like GMM points) rather than discrete tokens.
"""

import math
from typing import Optional, Callable, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from torch.nn.utils.parametrizations import orthogonal


class LayerNorm(nn.Module):
    """Layer normalization with optional bias parameter."""
    
    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5) -> None:
        """
        Initialize layer normalization.
        
        Args:
            dim: Feature dimension to normalize
            bias: Whether to use bias parameter
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization."""
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for transformer blocks."""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        dropout: float = 0.0,
        bias: bool = True,
        flash_attn: bool = True
    ) -> None:
        """
        Initialize multi-head attention.
        
        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in projections
            flash_attn: Whether to use flash attention when available
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Projections for query, key, value
        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)
        
        # Output projection
        self.output = nn.Linear(dim, dim, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Flash attention availability
        self.flash_attn = flash_attn and hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply attention
        if self.flash_attn:
            # Use PyTorch's optimized attention implementation
            context = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            # Manual attention implementation
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
                
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            
            context = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.output_dropout(self.output(context))
        
        return output


class FeedForward(nn.Module):
    """Feed-forward network used in transformer blocks."""
    
    def __init__(
        self, 
        dim: int, 
        expansion_factor: int = 4, 
        dropout: float = 0.0,
        activation: str = "gelu",
        bias: bool = True
    ) -> None:
        """
        Initialize feed-forward network.
        
        Args:
            dim: Input and output dimension
            expansion_factor: Hidden dimension multiplier
            dropout: Dropout probability
            activation: Activation function name
            bias: Whether to use bias in linear layers
        """
        super().__init__()
        hidden_dim = dim * expansion_factor
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=bias),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=bias),
            nn.Dropout(dropout)
        )
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish()
        }
        if name not in activations:
            raise ValueError(f"Activation {name} not supported. Options: {list(activations.keys())}")
        return activations[name]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward network to input."""
        return self.net(x)


class DummyFlowPredictor(nn.Module):
    """
    A simple flow predictor that returns a constant value of 1.
    Useful for testing the flow speed integration without complex logic.
    """
    
    def __init__(self, per_layer=False, num_layers=None):
        """
        Initialize dummy flow predictor.
        
        Args:
            per_layer: Whether to output per-layer flow speeds
            num_layers: Number of layers (required if per_layer=True)
        """
        super().__init__()
        self.per_layer = per_layer
        self.num_layers = num_layers
        
        if per_layer and num_layers is None:
            raise ValueError("num_layers must be specified when per_layer=True")
            
        # Register a dummy parameter so the module has parameters for device detection
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
    
    def forward(self, targets=None, inputs=None):
        """
        Return constant flow speed of 1.
        
        Args:
            targets: Ignored, here for interface compatibility
            inputs: Used only for batch size and device detection
            
        Returns:
            Tensor of ones, shape depends on per_layer setting
        """
        if inputs is not None:
            batch_size = inputs.size(0)
        elif targets is not None and isinstance(targets, torch.Tensor):
            batch_size = targets.size(0)
        else:
            batch_size = 1
            
        device = self.dummy_param.device
            
        if self.per_layer and self.num_layers is not None:
            # Return [batch_size, num_layers] tensor of ones
            return torch.ones(batch_size, self.num_layers, device=device)
        else:
            # Return [batch_size] tensor of ones
            return torch.ones(batch_size, device=device)


class TransformerBlock(nn.Module):
    """Full transformer block combining attention and feed-forward layers."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        ff_expansion: int = 4,
        bias: bool = True,
        activation: str = "gelu",
        norm_eps: float = 1e-5,
        use_flash_attn: bool = True,
        use_random_features: bool = False,
        num_random_features: int = 64,
        num_repeats: int = 1,
        random_feature_eps: float = 1e-6
    ) -> None:
        """
        Initialize transformer block.
        
        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate for all dropout layers
            ff_expansion: Feed-forward expansion factor
            bias: Whether to use bias in layers
            activation: Activation function for feed-forward
            norm_eps: Epsilon for layer normalization
            use_flash_attn: Whether to use flash attention (only for standard attention)
            use_random_features: Whether to use random feature attention
            num_random_features: Number of random features per head (m)
            num_repeats: Number of different random feature matrices to use
            random_feature_eps: Epsilon for numerical stability in random feature attention
        """
        super().__init__()
        
        self.use_random_features = use_random_features
        
        # First sub-block: attention with residual connection
        self.norm1 = LayerNorm(dim, bias=bias, eps=norm_eps)
        
        # Choose attention mechanism based on configuration
        if use_random_features:
            self.attention = MultiHeadRandomFeatureAttention(
                dim=dim,
                num_heads=num_heads,
                num_features=num_random_features,
                num_repeats=num_repeats,
                dropout=dropout,
                bias=bias,
                eps=random_feature_eps
            )
        else:
            self.attention = MultiHeadAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                flash_attn=use_flash_attn
            )
        
        # Second sub-block: feed-forward with residual connection
        self.norm2 = LayerNorm(dim, bias=bias, eps=norm_eps)
        self.feedforward = FeedForward(
            dim=dim,
            expansion_factor=ff_expansion,
            dropout=dropout,
            activation=activation,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor, flow_speed=None, repeat_idx=None) -> torch.Tensor:
        """
        Apply transformer block to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            flow_speed: Flow speed for controlling update magnitude
                      Can be a scalar, tensor of shape [batch_size], 
                      or tensor of shape [batch_size, num_layers]
            repeat_idx: Index of the random matrix to use (for random features mode)
                        If None, defaults to 0
            
        Returns:
            Processed tensor [batch_size, seq_len, dim]
        """
        # Default flow speed is 1.0 (normal residual update)
        if flow_speed is None:
            flow_speed = torch.ones(x.size(0), device=x.device, requires_grad=False)
        
        # Reshape flow_speed to broadcast properly with attention/FF outputs
        flow_speed = torch.reshape(flow_speed, (x.size(0), 1, 1))
        
        # Normalized input for attention
        norm_x = self.norm1(x)
        
        # Process attention based on configuration
        if self.use_random_features:
            # Use random feature attention with specified repeat index
            attention_output = self.attention(norm_x, repeat_idx=0 if repeat_idx is None else repeat_idx)
        else:
            # Use standard attention
            attention_output = self.attention(norm_x)
        
        # Apply attention with flow speed scaling
        x = x + flow_speed * attention_output
        
        # Feed-forward with flow speed scaling
        x = x + flow_speed * self.feedforward(self.norm2(x))
        
        return x


class OrthogonalEncDec(nn.Module):
    """
    Orthogonal encoder-decoder pair for mapping between input and latent spaces.
    
    This module implements an orthogonal encoding-decoding pair that:
    1. Uses orthogonal matrices for encoding and decoding
    2. Preserves distances in the latent space
    3. Handles null-space components appropriately
    """
    
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initialize orthogonal encoder-decoder.
        
        Args:
            input_dim: Dimension of input/output space
            latent_dim: Dimension of latent space
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Orthogonal projection matrix for encoding/decoding
        self.projection = nn.Parameter(torch.empty(latent_dim, input_dim))
        nn.init.orthogonal_(self.projection)
        # Apply orthogonal parametrization with Cayley map
        orthogonal(self, "projection", orthogonal_map="cayley")

        # Null-space projection matrix
        self.null_projection = nn.Parameter(torch.empty(input_dim, latent_dim))
        nn.init.orthogonal_(self.null_projection)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent space.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Encoded tensor [batch_size, seq_len, latent_dim]
        """
        encoded = x @ self.projection.T  # [batch_size, seq_len, latent_dim]
        return encoded

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to output space.
        
        Args:
            z: Latent tensor [batch_size, seq_len, latent_dim]
            
        Returns:
            Decoded tensor [batch_size, seq_len, input_dim]
        """
        # Project the latent vector onto input space
        main_projection = z @ self.projection  # [batch_size, seq_len, input_dim]
        
        # Handle null-space component
        z_proj = main_projection @ self.projection.T  # [batch_size, seq_len, latent_dim]
        z_perp = z - z_proj  # [batch_size, seq_len, latent_dim]
        
        # Add null-space contribution
        null_component = z_perp @ self.null_projection.T  # [batch_size, seq_len, input_dim]
        
        return main_projection + null_component

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply encoder and decoder in sequence (for testing).
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Reconstructed tensor [batch_size, seq_len, input_dim]
        """
        return self.decode(self.encode(x))


class MultiHeadRandomFeatureAttention(nn.Module):
    """Multi-head random feature attention for linear time complexity."""
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int, 
        num_features: int = 64,
        num_repeats: int = 1,
        dropout: float = 0.0,
        bias: bool = True,
        eps: float = 1e-6
    ) -> None:
        """
        Initialize multi-head random feature attention.
        
        Args:
            dim: Model dimension
            num_heads: Number of attention heads
            num_features: Number of random features per head
            num_repeats: Number of different random feature matrices to use
            dropout: Dropout probability
            bias: Whether to use bias in projections
            eps: Small constant for numerical stability
        """
        super().__init__()
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_features = num_features
        self.num_repeats = num_repeats
        self.eps = eps
        
        # Projections for query, key, value
        self.query = nn.Linear(dim, dim, bias=bias)
        self.key = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)
        
        # Output projection
        self.output = nn.Linear(dim, dim, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Sample multiple random projection matrices for different repeats
        # Each repeat has its own W ∈ R^{h×d×m}
        W = torch.randn(num_repeats, num_heads, self.head_dim, num_features)
        self.register_buffer('W', W)

    def compute_key_summary(self, K, V, repeat_idx=0):
        """
        Compute key summaries for efficient attention.
        
        Args:
            K: Key tensor [batch_size, num_heads, seq_len, head_dim]
            V: Value tensor [batch_size, num_heads, seq_len, head_dim]
            repeat_idx: Index for which random feature matrix to use
            
        Returns:
            N: Numerator tensor [batch_size, num_heads, num_features, head_dim]
            D: Denominator tensor [batch_size, num_heads, num_features]
        """
        B, h, L, d = K.shape
        idx = min(repeat_idx, self.num_repeats - 1)
        
        # Select the appropriate random feature matrix
        W = self.W[idx]  # [h, d, m]
        
        # 1) logits over (L,m)
        logits = torch.matmul(K, W)                                   # [B, h, L, m]
        logits = logits - 0.5 * (K*K).sum(dim=-1, keepdim=True)
        
        # 2) single softmax over each flattened L×m block
        flat = logits.view(B, h, L*self.num_features)                 # [B, h, L*m]
        alpha = torch.softmax(flat, dim=2).view(B, h, L, self.num_features)  # [B, h, L, m]
        alpha = self.attn_dropout(alpha)
        
        # 3) summaries via batched matmul
        #    N[b,h,i,:] = ∑_k α[b,h,k,i] · V[b,h,k,:]
        #    D[b,h,i]   = ∑_k α[b,h,k,i]
        N = torch.matmul(alpha.transpose(-2, -1), V)                  # [B, h, m, d]
        D = alpha.sum(dim=2)                                          # [B, h, m]
        
        return N, D

    def feature_map(self, Q, repeat_idx=0):
        """
        Apply feature map to queries.
        
        Args:
            Q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            repeat_idx: Index for which random feature matrix to use
            
        Returns:
            Feature map output [batch_size, num_heads, seq_len, num_features]
        """
        idx = min(repeat_idx, self.num_repeats - 1)
        W = self.W[idx]  # [h, d, m]
        
        # softmax over features m
        scores = torch.matmul(Q, W)                                   # [B, h, Lq, m]
        return torch.softmax(scores, dim=-1)

    def forward(self, x, mask=None, repeat_idx=0):
        """
        Apply multi-head random feature attention to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            mask: Optional attention mask (not used in random feature attention)
            repeat_idx: Index for which random feature matrix to use
            
        Returns:
            Output tensor [batch_size, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply random feature attention
        N, D = self.compute_key_summary(k, v, repeat_idx)             # [B, h, m, d], [B, h, m]
        φ = self.feature_map(q, repeat_idx)                           # [B, h, Lq, m]
        
        # numerator & denominator
        num = torch.matmul(φ, N)                                      # [B, h, Lq, d]
        den = torch.matmul(φ, D.unsqueeze(-1)).squeeze(-1)            # [B, h, Lq]
        
        # final normalize
        out = num / (den.unsqueeze(-1) + self.eps)                    # [B, h, Lq, d]
        
        # Reshape and apply output projection
        context = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.output_dropout(self.output(context))
        
        return output