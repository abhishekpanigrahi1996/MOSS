"""
Transformer model implementation for Gaussian Mixture Model data.

This module implements a transformer encoder optimized for processing
continuous point data from Gaussian Mixture Models.
"""

import math
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

import torch
import torch.nn as nn

from .blocks import LayerNorm, TransformerBlock, OrthogonalEncDec
from .monotonic_flow import MonotonicFlowPredictor

logger = logging.getLogger(__name__)


class GMMTransformer(nn.Module):
    """
    Transformer model for processing point clouds from Gaussian Mixture Models.
    
    This model processes continuous data (point coordinates) and learns representations
    that capture the underlying structure of the GMM distribution.
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        activation: str = "gelu",
        ff_expansion: int = 4,
        bias: bool = True,
        norm_eps: float = 1e-5,
        use_flash_attn: bool = True,
        use_random_features: bool = False,
        num_random_features: int = 64,
        num_repeats: int = 1,
        random_feature_eps: float = 1e-6,
        use_flow_predictor: bool = False,
        flow_predictor_type: str = "dummy",
        flow_predictor_per_layer: bool = False,
        flow_distribution_mode: str = "direct",
        layer_repeat_mode: str = "none",
        repeat_factor: int = 1,
        layer_groups: Optional[List[int]] = None,
        group_repeat_factors: Optional[List[int]] = None,
        load_pretrained_flow: bool = False,
        pretrained_flow_path: Optional[str] = None,
        freeze_flow_weights: bool = False,
        **kwargs
    ) -> None:
        """
        Initialize transformer model.
        
        Args:
            hidden_dim: Dimension of model embeddings
            num_layers: Number of unique transformer layer definitions
            num_heads: Number of attention heads
            dropout: Dropout probability (applied to dropout layers)
            activation: Activation function for feed-forward networks
            ff_expansion: Expansion factor for feed-forward networks
            bias: Whether to use bias in layers
            norm_eps: Epsilon for layer normalization
            use_flash_attn: Whether to use flash attention when available
            use_random_features: Whether to use random feature attention instead of standard attention
            num_random_features: Number of random features per head for random feature attention
            num_repeats: Number of different random feature matrices to use
            random_feature_eps: Epsilon for numerical stability in random feature attention
            use_flow_predictor: Whether to use flow speed prediction
            flow_predictor_type: Type of flow predictor ("dummy", "mlp", etc.)
            flow_predictor_per_layer: Whether to use per-layer flow speeds
            flow_distribution_mode: Mode for distributing flow speed across repetitions ("direct", "fractional")
            layer_repeat_mode: Mode for layer repetition ("none", "cycle", "layerwise", "grouped")
            repeat_factor: Number of times to repeat layers in "cycle" or "layerwise" mode
            layer_groups: List of layer group sizes for "grouped" mode
            group_repeat_factors: List of repeat factors for each group in "grouped" mode
            load_pretrained_flow: Whether to load a pre-trained flow predictor
            pretrained_flow_path: Path to pre-trained model containing flow predictor
            freeze_flow_weights: Whether to freeze flow predictor weights during training
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.layer_repeat_mode = layer_repeat_mode
        self.repeat_factor = repeat_factor
        self.layer_groups = layer_groups
        self.group_repeat_factors = group_repeat_factors
        self.flow_distribution_mode = flow_distribution_mode
        self.use_random_features = use_random_features
        self.num_random_features = num_random_features
        self.num_repeats = num_repeats
        
        # Validate flow distribution mode
        if flow_distribution_mode not in ["direct", "fractional"]:
            raise ValueError(f"Unknown flow_distribution_mode: {flow_distribution_mode}")
        
        # Log flow distribution mode
        logger.info(f"Using flow distribution mode: {flow_distribution_mode}")
        
        # Log attention type
        if use_random_features:
            logger.info(f"Using random feature attention with {num_random_features} features and {num_repeats} repeats")
        else:
            logger.info(f"Using standard attention with flash attention: {use_flash_attn}")
        
        # Calculate the effective number of layers after repetition
        self.effective_num_layers = self._calculate_effective_layers()
        
        # Create blocks (same creation for all modes)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                ff_expansion=ff_expansion,
                bias=bias,
                activation=activation,
                norm_eps=norm_eps,
                use_flash_attn=use_flash_attn,
                use_random_features=use_random_features,
                num_random_features=num_random_features,
                num_repeats=num_repeats,
                random_feature_eps=random_feature_eps
            ) for _ in range(num_layers)
        ])
        
        # For grouped mode, create a mapping of groups to block indices
        if layer_repeat_mode == "grouped":
            if layer_groups is None or group_repeat_factors is None:
                raise ValueError("layer_groups and group_repeat_factors must be provided for grouped mode")
                
            if len(layer_groups) != len(group_repeat_factors):
                raise ValueError("layer_groups and group_repeat_factors must have the same length")
                
            # Verify that total layers in groups matches num_layers
            if sum(layer_groups) != num_layers:
                raise ValueError(f"Sum of layer_groups {sum(layer_groups)} must match num_layers {num_layers}")
                
            # Create a list of block indices for each group
            self.group_indices = []
            start_idx = 0
            for group_size in layer_groups:
                self.group_indices.append(list(range(start_idx, start_idx + group_size)))
                start_idx += group_size
                
            # Log the group indices for debugging
            logger.info(f"Created layer groups with indices: {self.group_indices}")
                
        # Log info about the repetition mode
        if layer_repeat_mode != "none":
            logger.info(f"Using {layer_repeat_mode} repetition mode with {self.effective_num_layers} effective layers")
            if layer_repeat_mode == "grouped":
                logger.info(f"Group sizes: {layer_groups}, repeat factors: {group_repeat_factors}")
            else:
                logger.info(f"Unique layers: {num_layers}, repeat factor: {repeat_factor}")
            
            # Log flow distribution mode when repetition is used
            logger.info(f"Flow distribution mode: {flow_distribution_mode}")
        
        # No final layer normalization
        
        # Flow predictor (optional)
        self.use_flow_predictor = use_flow_predictor
        if use_flow_predictor:
            if flow_predictor_type == "dummy":
                from .blocks import DummyFlowPredictor
                self.flow_predictor = DummyFlowPredictor(
                    per_layer=flow_predictor_per_layer,
                    num_layers=num_layers if flow_predictor_per_layer else None
                )
                logger.info(f"Initialized dummy flow predictor (per_layer={flow_predictor_per_layer})")
            elif flow_predictor_type == "linear":
                from .flow_speed import LinearFlowPredictor
                
                # Get flow predictor parameters from kwargs
                flow_min_value = kwargs.get('flow_min_value', 0.0)
                flow_max_value = kwargs.get('flow_max_value', 1.0)
                flow_min_snr = kwargs.get('flow_min_snr', -20.0)
                flow_max_snr = kwargs.get('flow_max_snr', 0.0)
                
                self.flow_predictor = LinearFlowPredictor(
                    per_layer=flow_predictor_per_layer,
                    num_layers=num_layers if flow_predictor_per_layer else None,
                    min_snr=flow_min_snr,
                    max_snr=flow_max_snr,
                    min_flow=flow_min_value,
                    max_flow=flow_max_value
                )
                
                logger.info(f"Initialized linear flow predictor (per_layer={flow_predictor_per_layer}, "
                           f"SNR_range=[{flow_min_snr}, {flow_max_snr}], flow_range=[{flow_min_value}, {flow_max_value}])")
            elif flow_predictor_type == "monotonic":
                # Get flow predictor parameters from kwargs
                num_basis = kwargs.get('flow_num_basis', 8)
                flow_min_value = kwargs.get('flow_min_value', 0.0)
                flow_max_value = kwargs.get('flow_max_value', 1.0)
                flow_min_snr = kwargs.get('flow_min_snr', -20.0)
                flow_max_snr = kwargs.get('flow_max_snr', 0.0)
                
                self.flow_predictor = MonotonicFlowPredictor(
                    per_layer=flow_predictor_per_layer,
                    num_layers=num_layers if flow_predictor_per_layer else None,
                    num_basis=num_basis,
                    min_snr=flow_min_snr,
                    max_snr=flow_max_snr,
                    min_flow=flow_min_value,
                    max_flow=flow_max_value
                )
                
                logger.info(f"Initialized monotonic flow predictor (per_layer={flow_predictor_per_layer}, "
                           f"num_basis={num_basis}, flow_range=[{flow_min_value}, {flow_max_value}])")
            else:
                logger.warning(f"Unknown flow predictor type: {flow_predictor_type}, using dummy")
                from .blocks import DummyFlowPredictor
                self.flow_predictor = DummyFlowPredictor(
                    per_layer=flow_predictor_per_layer,
                    num_layers=num_layers if flow_predictor_per_layer else None
                )
            
            # Load pre-trained flow predictor if requested
            if load_pretrained_flow and pretrained_flow_path:
                try:
                    # Load pre-trained model
                    pretrained = torch.load(pretrained_flow_path)
                    
                    # Get the model state dictionary
                    if 'model_state_dict' in pretrained:
                        model_state = pretrained['model_state_dict']
                    else:
                        model_state = pretrained
                    
                    # Try different key patterns to find flow predictor state
                    flow_state = {}
                    
                    # Pattern 1: Direct flow_predictor prefix
                    flow_state = {k: v for k, v in model_state.items() if k.startswith('flow_predictor.')}
                    # Remove 'flow_predictor.' prefix if found
                    if flow_state:
                        flow_state = {k.replace('flow_predictor.', ''): v for k, v in flow_state.items()}
                    
                    # Pattern 2: Nested under transformer
                    if not flow_state:
                        flow_state = {k: v for k, v in model_state.items() if k.startswith('transformer.flow_predictor.')}
                        # Remove 'transformer.flow_predictor.' prefix if found
                        if flow_state:
                            flow_state = {k.replace('transformer.flow_predictor.', ''): v for k, v in flow_state.items()}
                    
                    # Pattern 3: Nested under model
                    if not flow_state:
                        flow_state = {k: v for k, v in model_state.items() if k.startswith('model.flow_predictor.')}
                        # Remove 'model.flow_predictor.' prefix if found
                        if flow_state:
                            flow_state = {k.replace('model.flow_predictor.', ''): v for k, v in flow_state.items()}
                    
                    if not flow_state:
                        logger.warning(f"No flow predictor state found in {pretrained_flow_path}. Available keys in model_state_dict: {list(model_state.keys())}")
                    else:
                        # Load into current model
                        missing_keys, unexpected_keys = self.flow_predictor.load_state_dict(flow_state, strict=False)
                        
                        # Verify loading was successful
                        if missing_keys:
                            logger.warning(f"Missing keys when loading flow predictor: {missing_keys}")
                        if unexpected_keys:
                            logger.warning(f"Unexpected keys when loading flow predictor: {unexpected_keys}")
                            
                        # Verify that at least some parameters were loaded
                        loaded_params = set(flow_state.keys())
                        expected_params = set(self.flow_predictor.state_dict().keys())
                        if not loaded_params.intersection(expected_params):
                            logger.error("No matching parameters were loaded for flow predictor")
                        else:
                            logger.info(f"Successfully loaded flow predictor with {len(loaded_params.intersection(expected_params))} parameters")
                        
                        # Freeze weights if requested
                        if freeze_flow_weights:
                            for param in self.flow_predictor.parameters():
                                param.requires_grad = False
                            logger.info("Froze flow predictor weights")
                except Exception as e:
                    logger.error(f"Error loading pre-trained flow predictor: {e}")
        
        
        
        # Initialize weights
        self._init_weights()
        # Speed-conditioned scaling parameter
        self.log_gamma = nn.Parameter(torch.zeros(hidden_dim))
        
        # Log model size
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(f"GMMTransformer created with {num_params:,} parameters")
    
    def _calculate_effective_layers(self) -> int:
        """Calculate the effective number of layers after applying repetition."""
        if self.layer_repeat_mode == "none":
            return self.num_layers
        elif self.layer_repeat_mode in ["cycle", "layerwise"]:
            return self.num_layers * self.repeat_factor
        elif self.layer_repeat_mode == "grouped":
            if self.layer_groups is None or self.group_repeat_factors is None:
                raise ValueError("layer_groups and group_repeat_factors must be provided for grouped mode")
            return sum(size * rep for size, rep in zip(self.layer_groups, self.group_repeat_factors))
        else:
            raise ValueError(f"Unknown layer_repeat_mode: {self.layer_repeat_mode}")
    
    def _compute_fractional_flow(
        self, 
        original_flow: torch.Tensor,  # [batch_size]
        repeat_index: int,            # Current repetition index
        repeat_factor: int            # Total number of repetitions
    ) -> torch.Tensor:
        """
        Compute flow speed for a specific repetition based on the fractional distribution.
        
        Args:
            original_flow: Original flow speed tensor [batch_size]
            repeat_index: Current repetition index (0-based)
            repeat_factor: Total number of repetitions
            
        Returns:
            Flow speed tensor for this repetition [batch_size]
        """
        if self.flow_distribution_mode != "fractional":
            # In direct mode, just use the original flow
            return original_flow
        
        # Calculate fractional repeat count T
        T = repeat_factor * original_flow  # [batch_size]
        
        # Start with zero flow
        current_flow = torch.zeros_like(original_flow)
        
        # Case 1: repeat_index < floor(T) - full strength (1.0)
        full_mask = (repeat_index < torch.floor(T))
        current_flow = torch.where(full_mask, torch.ones_like(current_flow), current_flow)
        
        # Case 2: repeat_index == floor(T) - partial strength (fractional part)
        partial_mask = (repeat_index == torch.floor(T))
        partial_strength = T - torch.floor(T)
        current_flow = torch.where(partial_mask, partial_strength, current_flow)
        
        return current_flow
    
    def _init_weights(self) -> None:
        """Initialize model weights for stable training."""
        def _init_weight(m):
            if isinstance(m, nn.Linear):
                # Initialize linear layers with small normal distribution
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        
        self.apply(_init_weight)
        
        # Special initialization for residual projections
        for name, p in self.named_parameters():
            if "output.weight" in name:
                # Smaller init for output projections for better training dynamics
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.num_layers))
    
    def compute_flow_speed(self, targets=None, x=None, flow_speed=None) -> torch.Tensor:
        """
        Compute flow speed based on targets and input tensor.
        
        Args:
            targets: Optional targets for flow speed prediction
            x: Input tensor [batch_size, seq_len, hidden_dim]
            flow_speed: Optional explicit flow speed parameter
            
        Returns:
            Flow speed tensor [batch_size, num_layers] if per-layer flow is used,
            or [batch_size] otherwise
        """
        batch_size = x.size(0) if x is not None else (targets['snr_db'].size(0) if targets is not None and isinstance(targets, dict) and 'snr_db' in targets else 1)
        
        # Use flow predictor if available and no explicit flow_speed provided
        if flow_speed is None and self.use_flow_predictor and hasattr(self, 'flow_predictor'):
            flow_speed = self.flow_predictor(targets, x)
        elif flow_speed is None:
            # Default flow speed is 1.0 (standard residual update)
            device = x.device if x is not None else (
                targets['snr_db'].device if targets is not None and isinstance(targets, dict) and 'snr_db' in targets
                else self.log_gamma.device
            )
            flow_speed = torch.ones(batch_size, device=device)
        
        # Determine if we're using per-layer flow
        per_layer_flow = hasattr(self, 'flow_predictor') and getattr(self.flow_predictor, 'per_layer', False)
        
        # For non-per-layer case, reshape to per-layer format for uniform handling
        if not per_layer_flow:
            flow_speed = flow_speed.unsqueeze(1).expand(batch_size, self.num_layers)
        elif flow_speed.dim() == 1:
            # For per-layer case with scalar input, expand to proper shape
            flow_speed = flow_speed.unsqueeze(1).expand(batch_size, self.num_layers)
        
        return flow_speed
    
    def forward(self, x: torch.Tensor, targets=None, flow_speed=None) -> torch.Tensor:
        """
        Process input through transformer model.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
               where seq_len is the number of points
            targets: Optional targets for flow speed prediction
            flow_speed: Optional explicit flow speed parameter
            
        Returns:
            Contextualized embeddings [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute flow speed
        flow_speed = self.compute_flow_speed(targets, x, flow_speed)
        
        # Define epsilon for fractional calculations to avoid applying blocks with near-zero flow
        epsilon = 1e-7
        
        # Process through transformer blocks based on repetition mode
        if self.layer_repeat_mode == "none":
            # Standard processing - each block used once
            for i, block in enumerate(self.blocks):
                repeat_idx = 0 if not self.use_random_features else None
                x = block(x, flow_speed=flow_speed[:, i], repeat_idx=repeat_idx)
                
        elif self.layer_repeat_mode == "cycle":
            # Cycle mode - repeat the entire sequence of blocks
            for cycle in range(self.repeat_factor):
                for i, block in enumerate(self.blocks):
                    # Compute flow speed for this repetition
                    current_flow = self._compute_fractional_flow(
                        flow_speed[:, i], cycle, self.repeat_factor
                    )
                    
                    # Only apply the block if any flow is non-zero (optimization)
                    if current_flow.max() > epsilon:
                        # Use cycle index as repeat_idx when using random features
                        repeat_idx = cycle if self.use_random_features else None 
                        x = block(x, flow_speed=current_flow, repeat_idx=repeat_idx)
                    
        elif self.layer_repeat_mode == "layerwise":
            # Layerwise mode - repeat each block before moving to the next
            for i, block in enumerate(self.blocks):
                for j in range(self.repeat_factor):
                    # Compute flow speed for this repetition
                    current_flow = self._compute_fractional_flow(
                        flow_speed[:, i], j, self.repeat_factor
                    )
                    
                    # Only apply the block if any flow is non-zero (optimization)
                    if current_flow.max() > epsilon:
                        # Use j as repeat_idx when using random features
                        repeat_idx = j if self.use_random_features else None
                        x = block(x, flow_speed=current_flow, repeat_idx=repeat_idx)
                    
        elif self.layer_repeat_mode == "grouped":
            # Grouped mode - repeat each group according to its factor
            for group_indices, group_repeat_factor in zip(self.group_indices, self.group_repeat_factors):
                # Process each group
                for rep in range(group_repeat_factor):
                    # Process each block in the group
                    for block_idx in group_indices:
                        # Compute flow speed for this repetition
                        current_flow = self._compute_fractional_flow(
                            flow_speed[:, block_idx], rep, group_repeat_factor
                        )
                        
                        # Only apply the block if any flow is non-zero (optimization)
                        if current_flow.max() > epsilon:
                            # Use rep as repeat_idx when using random features
                            repeat_idx = rep if self.use_random_features else None
                            x = self.blocks[block_idx](x, flow_speed=current_flow, repeat_idx=repeat_idx)
        
        return x
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        config = {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "layer_repeat_mode": self.layer_repeat_mode,
            "model_type": "GMMTransformer",
            "use_flow_predictor": self.use_flow_predictor,
            "flow_distribution_mode": self.flow_distribution_mode
        }
        
        # Add attention configuration
        config["use_random_features"] = self.use_random_features
        if self.use_random_features:
            config["num_random_features"] = self.num_random_features
            config["num_repeats"] = self.num_repeats
        
        # Add flow speed configuration if enabled
        if self.use_flow_predictor and hasattr(self, 'flow_predictor'):
            config["flow_predictor_type"] = getattr(self.flow_predictor, 'predictor_type', 'dummy')
            config["flow_predictor_per_layer"] = getattr(self.flow_predictor, 'per_layer', False)
        
        # Add layer repetition configuration
        if self.layer_repeat_mode != "none":
            config["effective_num_layers"] = self.effective_num_layers
            
        if self.layer_repeat_mode in ["cycle", "layerwise"]:
            config["repeat_factor"] = self.repeat_factor
            
        if self.layer_repeat_mode == "grouped":
            config["layer_groups"] = self.layer_groups
            config["group_repeat_factors"] = self.group_repeat_factors
            
        return config




class ClusterPredictionModel(nn.Module):
    """
    Extension of GMMTransformer for predicting cluster centers.
    
    This model includes:
    1. Orthogonal encoder for raw coordinates to model dimension
    2. Transformer blocks for contextual processing
    3. Orthogonal decoder to map back to coordinate space
    """
    
    def __init__(
        self,
        transformer: Optional[GMMTransformer] = None,
        input_dim: int = 2,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        norm_eps: float = 1e-5,
        use_orthogonal_encdec: bool = True,
        **transformer_kwargs
    ) -> None:
        """
        Initialize cluster prediction model.
        
        Args:
            transformer: Optional pre-configured transformer model
            input_dim: Input dimension for points
            hidden_dim: Hidden dimension for the model
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability for transformer layers
            bias: Whether to use bias in input projection and output head
            norm_eps: Epsilon for layer normalization
            use_orthogonal_encdec: Whether to use orthogonal encoder-decoder architecture
            **transformer_kwargs: Additional arguments for the transformer
        """
        super().__init__()
        
        # Store configuration parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.model_dropout = dropout
        self.model_bias = bias
        self.use_orthogonal_encdec = use_orthogonal_encdec
        
        # Create encoder/decoder components
        if use_orthogonal_encdec:
            # Orthogonal encoder-decoder
            self.encdec = OrthogonalEncDec(input_dim=input_dim, latent_dim=hidden_dim)
        else:
            # Standard linear encoder and decoder with separate norm
            self.encoder = nn.Linear(input_dim, hidden_dim, bias=bias)
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=bias)
            self.final_norm = LayerNorm(hidden_dim, bias=bias, eps=norm_eps)
        
        # Create transformer if not provided
        if transformer is None:
            self.transformer = GMMTransformer(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                bias=bias,
                **transformer_kwargs
            )
        else:
            self.transformer = transformer

    def _normalize_data(self, x: torch.Tensor, scale_factor: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize input data by centering and scaling.
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            scale_factor: Optional scaling factor to divide the scalar by
            
        Returns:
            Tuple of (normalized_data, mean, scalar)
            - normalized_data: Centered and scaled data [batch_size, seq_len, dim]
            - mean: Mean of the data [batch_size, 1, dim]
            - scalar: Scaling factor [batch_size, 1, 1]
        """
        batch_size, seq_len, dim = x.shape
        
        # Calculate mean and center the data
        mu = torch.mean(x, dim=1, keepdim=True)  # [batch_size, 1, dim]
        x_hat = x - mu
        
        # Calculate covariance matrix [batch_size, dim, dim]
        x_hat_t = x_hat.transpose(1, 2)  # [batch_size, dim, seq_len]
        cov = torch.bmm(x_hat_t, x_hat) / seq_len  # [batch_size, dim, dim]
        
        # Calculate scalar^2 as tr(cov)
        scalar_squared = torch.diagonal(cov, dim1=1, dim2=2).sum(dim=1)  # [batch_size]
        if scale_factor is not None:
            scalar_squared = scalar_squared / scale_factor
        scalar = torch.sqrt(scalar_squared).view(batch_size, 1, 1)  # [batch_size, 1, 1]
        
        # Normalize x_hat by dividing by the scalar
        x_normalized = x_hat / (scalar + 1e-8)  # Add epsilon for numerical stability
        
        return x_normalized, mu, scalar

    def forward(self, x: torch.Tensor, targets=None, flow_speed=None) -> torch.Tensor:
        """
        Process input and predict cluster centers.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            targets: Optional targets for flow speed prediction
            flow_speed: Optional explicit flow speed parameter (in [0,1])
            
        Returns:
            Predictions tensor: [batch_size, seq_len, input_dim]
        """
        # Verify input dimension matches expected dimension
        batch_size, seq_len, input_dim = x.shape
        if not torch.jit.is_tracing() and input_dim != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {input_dim}")
        
        # Normalize input data with scaling by input_dim
        x_normalized, mu, scalar = self._normalize_data(x, scale_factor=input_dim)
            
        if self.use_orthogonal_encdec:
            # Apply orthogonal encoding
            encoded = self.encdec.encode(x_normalized)
            
            # Process through transformer with speed conditioning
            hidden = self.transformer(encoded, targets=targets, flow_speed=flow_speed)
            
            # Normalize hidden representation with scaling by hidden_dim
            hidden_normalized, hidden_mu, hidden_scalar = self._normalize_data(hidden, scale_factor=self.input_dim)
            
            # Decode with orthogonal decoder
            predictions_normalized = self.encdec.decode(hidden_normalized)
        else:
            # Standard processing without speed conditioning
            encoded = self.encoder(x_normalized)
            hidden = self.transformer(encoded, targets=targets, flow_speed=flow_speed)
            normed = self.final_norm(hidden)
            predictions_normalized = self.decoder(normed)
        
        # Rescale predictions by multiplying by scalar and adding back the mean
        predictions_centered = predictions_normalized * scalar
        predictions = predictions_centered + mu
        
        return predictions
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        # Get transformer config
        config = self.transformer.get_config()
        
        # Add model-specific config
        config.update({
            "model_type": "ClusterPredictionModel",
            "input_dim": self.input_dim,
            "dropout": self.model_dropout,
            "bias": self.model_bias,
            "norm_eps": getattr(self.final_norm, 'eps', 1e-5),
            "use_orthogonal_encdec": self.use_orthogonal_encdec
        })
        
        return config