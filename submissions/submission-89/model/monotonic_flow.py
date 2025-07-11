"""
Monotonic flow speed components for transformer models.

This module implements flow speed predictors using monotonic basis functions
for controlling information flow through transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MonotonicBasis(nn.Module):
    """
    Monotonic basis function for flow speed prediction.
    
    Uses a weighted sum of sigmoid basis functions to ensure a monotonic mapping
    from inputs to flow speeds. Guarantees monotonicity by construction.
    """
    
    def __init__(self, num_basis: int = 8):
        """
        Initialize monotonic basis function.
        
        Args:
            num_basis: Number of basis functions
        """
        super().__init__()
        self.num_basis = num_basis
        
        # Initialize logits for weights (p_i)
        self.weight_logits = nn.Parameter(torch.zeros(num_basis))
        
        # Initialize raw thresholds (θ_i)
        # Start with evenly spaced values with offset: θ_i = 1/(2n) + i/n
        initial_thetas = torch.tensor([1/(2*num_basis) + i/num_basis for i in range(num_basis)])
        self.theta_raw = nn.Parameter(torch.logit((initial_thetas + 5/num_basis) / (1 + 10/num_basis)))
        logger.info(f"Num basis: {num_basis}")
        logger.info(f"Initial thetas: {initial_thetas}")
        logger.info(f"Initial theta_raw: {self.theta_raw}")
        
        logger.info(f"Initialized MonotonicBasis with {num_basis} basis functions")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply monotonic basis function transformation.
        
        Args:
            x: Input tensor in [0, 1]
            
        Returns:
            Monotonically transformed output in [0, 1]
        """
        # Convert logits to probabilities using softmax
        weights = F.softmax(self.weight_logits, dim=0)
        
        # Convert raw thresholds to [0, 1] using sigmoid
        thetas = torch.sigmoid(self.theta_raw) * (1 + 10/self.num_basis) - 5/self.num_basis
        
        # Compute basis functions
        # Shape: [batch_size, num_basis]
        basis = torch.sigmoid((x.unsqueeze(-1) - thetas) * self.num_basis)
        
        # Weighted sum
        # Shape: [batch_size]
        return torch.sum(weights * basis, dim=-1)


class MonotonicFlowPredictor(nn.Module):
    """
    Flow predictor that uses monotonic basis functions to map SNR values to flow speeds.
    
    This predictor extracts SNR values from the targets dictionary and maps them
    to flow speeds using a collection of monotonic basis functions (one per layer).
    """
    
    def __init__(
        self,
        per_layer: bool = False,
        num_layers: Optional[int] = None,
        num_basis: int = 8,
        min_snr: float = 0.0,
        max_snr: float = 20.0,
        min_flow: float = 0.0,
        max_flow: float = 1.0
    ):
        """
        Initialize monotonic flow predictor.
        
        Args:
            per_layer: Whether to output per-layer flow speeds
            num_layers: Number of layers (required if per_layer=True)
            num_basis: Number of basis functions
            min_snr: Minimum expected SNR value (domain left boundary)
            max_snr: Maximum expected SNR value (domain right boundary)
            min_flow: Minimum flow speed value (range bottom boundary, at max_snr)
            max_flow: Maximum flow speed value (range top boundary, at min_snr)
        """
        super().__init__()
        self.per_layer = per_layer
        self.num_layers = num_layers
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.min_flow = min_flow
        self.max_flow = max_flow
        
        if per_layer and num_layers is None:
            raise ValueError("num_layers must be specified when per_layer=True")
            
        # Create basis functions
        if per_layer:
            self.basis_functions = nn.ModuleList([
                MonotonicBasis(num_basis=num_basis)
                for _ in range(num_layers)
            ])
        else:
            self.basis_function = MonotonicBasis(num_basis=num_basis)
            
        logger.info(
            f"Initialized MonotonicFlowPredictor (per_layer={per_layer}, "
            f"num_layers={num_layers if per_layer else 'N/A'}, "
            f"SNR range=[{min_snr}, {max_snr}], "
            f"flow range=[{min_flow}, {max_flow}])"
        )
    
    def forward(self, targets: Optional[Dict[str, Any]] = None, inputs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predict flow speeds based on SNR values in targets.
        
        Args:
            targets: Dictionary containing 'snr_db' key with SNR values
            inputs: Input tensor (used only for batch size)
            
        Returns:
            Flow speeds tensor, shape depends on per_layer setting
        """
        # Get device and batch size
        device = next(self.parameters()).device
        if inputs is not None:
            batch_size = inputs.size(0)
        elif targets is not None and isinstance(targets, dict) and 'snr_db' in targets:
            batch_size = targets['snr_db'].size(0)
        else:
            batch_size = 1
            
        # Handle missing SNR values
        if targets is None or not isinstance(targets, dict) or 'snr_db' not in targets:
            logger.warning("No SNR values found in targets, using default flow=1.0")
            if self.per_layer:
                return torch.ones(batch_size, self.num_layers, device=device)
            else:
                return torch.ones(batch_size, device=device)
        
        # Get SNR values and normalize to [0, 1]
        snr_values = targets['snr_db'].to(device)
        snr_normalized = (snr_values - self.min_snr) / (self.max_snr - self.min_snr)
        # snr_normalized = torch.clamp(snr_normalized, 0.0, 1.0)
        
        # Apply basis functions
        if self.per_layer:
            flows = []
            for i in range(self.num_layers):
                # Apply basis function and invert (1 - f(x)) for countermonotonicity
                layer_flow = 1.0 - self.basis_functions[i](snr_normalized)
                flows.append(layer_flow.unsqueeze(1))
            flow = torch.cat(flows, dim=1)
        else:
            # Apply single basis function and invert
            flow = 1.0 - self.basis_function(snr_normalized)
            
        # Scale from [0,1] to [min_flow, max_flow]
        flow = self.min_flow + flow * (self.max_flow - self.min_flow)
            
        return flow 