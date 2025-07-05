"""
Flow speed components for transformer models.

This module implements flow speed predictors and related utilities for controlling
information flow through transformer layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class MonotonicSpline(nn.Module):
    """
    Monotonic spline for flow speed prediction.
    
    Uses piecewise-linear interpolation to ensure a monotonic mapping from inputs to flow speeds.
    Guarantees monotonicity by construction.
    """
    
    def __init__(self, num_bins=8, left=0.0, right=25.0):
        """
        Initialize monotonic spline.
        
        Args:
            num_bins: Number of bins for the spline
            left: Left boundary of the spline domain
            right: Right boundary of the spline domain
        """
        super().__init__()
        self.left = float(left)
        self.right = float(right)
        self.num_bins = int(num_bins)

        # Learn non-negative height increments Δh_i
        # Initialize close to zero for better behavior
        self.delta_h = nn.Parameter(torch.zeros(num_bins))
        
        # Fixed, uniform knots t_i
        knots = torch.linspace(left, right, num_bins + 1)
        self.register_buffer("knots", knots)  # [num_bins+1]
        
        logger.info(f"Initialized MonotonicSpline with {num_bins} bins over range [{left}, {right}]")

    def forward(self, x):
        """
        Apply monotonic spline transformation.
        
        Args:
            x: Input tensor
            
        Returns:
            Monotonically transformed output in range [0, 1]
        """
        # Store original shape for reshaping at the end
        original_shape = x.shape
        
        # Flatten input to [B]
        x = x.view(-1)

        # 1) Build cumulative heights h_i (h_0=0, h_end=1)
        deltas = F.softplus(self.delta_h)  # [num_bins] ≥0
        h = torch.cat([torch.zeros(1, device=deltas.device),
                      torch.cumsum(deltas, dim=0)], dim=0)  # [num_bins+1]
        h = h / (h[-1] + 1e-6)  # Normalize so h[0]=0, h[-1]=1
        
        # 2) Clamp input and find bin indices
        x_clamped = torch.clamp(x, self.left, self.right)
        idx = torch.searchsorted(self.knots, x_clamped, right=False)
        idx = idx.clamp(1, self.num_bins)  # [B], in 1..num_bins
        i0, i1 = idx - 1, idx  # Left and right knot indices

        # 3) Gather knot positions & heights
        t0, t1 = self.knots[i0], self.knots[i1]  # [B]
        y0, y1 = h[i0], h[i1]  # [B]

        # 4) Linear interpolation within each bin
        slope = (y1 - y0) / (t1 - t0 + 1e-8)
        y = y0 + slope * (x_clamped - t0)

        # 5) Constant values outside domain
        y = torch.where(x < self.left, 0.0, y)
        y = torch.where(x > self.right, 1.0, y)
        
        # Ensure output is strictly within [0,1] range
        y = torch.clamp(y, 0.0, 1.0)
        
        # Reshape output back to original shape
        y = y.view(original_shape)
        return y  # Shape same as input, guaranteed monotonic, in [0,1]


class MonotonicFlowPredictor(nn.Module):
    """
    Flow predictor that uses monotonic splines to map SNR values to flow speeds.
    
    This predictor extracts SNR values from the targets dictionary and maps them
    to flow speeds using a collection of monotonic splines (one per layer).
    """
    
    def __init__(
        self,
        per_layer: bool = False,
        num_layers: Optional[int] = None,
        num_bins: int = 8, 
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
            num_bins: Number of bins for the splines
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
        
        # Check if per_layer is enabled
        if per_layer and num_layers is None:
            raise ValueError("num_layers must be specified when per_layer=True")
            
        # Create splines
        if per_layer:
            # Create a separate spline for each layer
            self.splines = nn.ModuleList([
                MonotonicSpline(num_bins=num_bins, left=min_snr, right=max_snr)
                for _ in range(num_layers)
            ])
        else:
            # Single spline for global flow
            self.spline = MonotonicSpline(num_bins=num_bins, left=min_snr, right=max_snr)
        
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
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Determine batch size
        if inputs is not None:
            batch_size = inputs.size(0)
        elif targets is not None and isinstance(targets, dict) and 'snr_db' in targets:
            batch_size = targets['snr_db'].size(0)
        else:
            batch_size = 1
            
        # Check if targets contain the required SNR values
        if targets is None or not isinstance(targets, dict) or 'snr_db' not in targets:
            logger.warning("No SNR values found in targets, using default flow=1.0")
            if self.per_layer:
                return torch.ones(batch_size, self.num_layers, device=device)
            else:
                return torch.ones(batch_size, device=device)
        
        # Get SNR values from targets and ensure they're on the correct device
        snr_values = targets['snr_db'].to(device)  # Move to the same device as the model
        
        # Use negative SNR values directly since we've already specified the appropriate
        # range for negative SNR values in the constructor
        neg_snr = -snr_values  # Higher SNR -> lower flow speed
            
        # Apply splines to map SNR to flow speed
        if self.per_layer:
            # Apply separate spline to each layer's flow
            flows = []
            for i in range(self.num_layers):
                layer_flow = self.splines[i](neg_snr)
                flows.append(layer_flow.unsqueeze(1))
            flow = torch.cat(flows, dim=1)
        else:
            # Apply single spline to global flow
            flow = self.spline(neg_snr)
            
        # Scale from [0,1] to [min_flow, max_flow]
        flow = self.min_flow + flow * (self.max_flow - self.min_flow)
            
        return flow


class LinearFlowPredictor(nn.Module):
    """
    Flow predictor that uses a simple linear mapping from SNR values to flow speeds.
    
    This predictor extracts SNR values from the targets dictionary and maps them
    to flow speeds using a linear interpolation between min_flow and max_flow.
    For SNR values:
    - SNR <= min_snr: flow = max_flow
    - SNR >= max_snr: flow = min_flow
    - Between min_snr and max_snr: linear interpolation
    """
    
    def __init__(
        self,
        per_layer: bool = False,
        num_layers: Optional[int] = None,
        min_snr: float = 0.0,      # Minimum expected SNR value in dB
        max_snr: float = 20.0,     # Maximum expected SNR value in dB  
        min_flow: float = 0.0,     # Minimum flow speed (at max_snr)
        max_flow: float = 1.0,     # Maximum flow speed (at min_snr)
        predictor_type: str = "linear",  # Identifier for this predictor type
        **kwargs
    ):
        """
        Initialize linear flow predictor.
        
        Args:
            per_layer: Whether to output per-layer flow speeds
            num_layers: Number of layers (required if per_layer=True)
            min_snr: Minimum expected SNR value (in dB)
            max_snr: Maximum expected SNR value (in dB)
            min_flow: Minimum flow speed value (at max_snr)
            max_flow: Maximum flow speed value (at min_snr)
            predictor_type: Type identifier for this predictor
        """
        super().__init__()
        self.per_layer = per_layer
        self.num_layers = num_layers
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.min_flow = min_flow
        self.max_flow = max_flow
        self.predictor_type = predictor_type
        
        # Check if per_layer is enabled
        if per_layer and num_layers is None:
            raise ValueError("num_layers must be specified when per_layer=True")
            
        # Register parameters for serialization/device tracking
        self.register_buffer("_dummy", torch.zeros(1))
        
        logger.info(
            f"Initialized LinearFlowPredictor (per_layer={per_layer}, "
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
        # Get device from model parameters
        device = self._dummy.device
        
        # Determine batch size
        if inputs is not None:
            batch_size = inputs.size(0)
        elif targets is not None and isinstance(targets, dict) and 'snr_db' in targets:
            batch_size = targets['snr_db'].size(0)
        else:
            batch_size = 1
            
        # Check if targets contain the required SNR values
        if targets is None or not isinstance(targets, dict) or 'snr_db' not in targets:
            logger.warning("No SNR values found in targets, using default flow=1.0")
            if self.per_layer:
                return torch.ones(batch_size, self.num_layers, device=device)
            else:
                return torch.ones(batch_size, device=device)
        
        # Get SNR values from targets and ensure they're on the correct device
        snr_values = targets['snr_db'].to(device)
        
        # Clamp SNR values to the specified range
        snr_clamped = torch.clamp(snr_values, self.min_snr, self.max_snr)
        
        # Linear mapping from SNR to flow speed:
        # - At min_snr: flow = max_flow
        # - At max_snr: flow = min_flow
        # - Linear interpolation in between
        snr_range = self.max_snr - self.min_snr
        flow_range = self.max_flow - self.min_flow
        
        # Calculate normalized position in the range [0, 1]
        norm_pos = (self.max_snr - snr_clamped) / snr_range if snr_range != 0 else torch.zeros_like(snr_clamped)
        
        # Map to flow speed
        flow = self.min_flow + norm_pos * flow_range
        
        # Handle per-layer flow if needed
        if self.per_layer:
            # Expand flow to have a separate value for each layer
            flow = flow.unsqueeze(1).expand(-1, self.num_layers)
        
        return flow