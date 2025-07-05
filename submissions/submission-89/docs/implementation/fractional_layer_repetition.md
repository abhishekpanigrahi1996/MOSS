# Fractional Flow Distribution within Layer Repetition

## Overview

This document describes a new way to distribute flow speed within repeated transformer layers in our GMM model. The method provides a way to smoothly vary the effective number of active transformer layers based on both the layer repeat count and the flow speed parameter, while maintaining a fixed computational structure.

## Current Implementation

Currently, the system has several layer repetition modes:

1. **None**: Each unique layer is applied exactly once (standard transformer)
2. **Cycle**: The entire sequence of layers is repeated multiple times
3. **Layerwise**: Each layer is repeated multiple times before moving to the next layer
4. **Grouped**: Layers are grouped, with each group having its own repeat factor

In all modes, when `flow_speed` is provided, it directly controls the magnitude of the residual update in each transformer block:

```python
# Inside TransformerBlock.forward()
x = x + flow_speed * self.attention(self.norm1(x))
x = x + flow_speed * self.feedforward(self.norm2(x))
```

## Proposed Fractional Flow Distribution

We propose a new way to distribute flow speed across layer repetitions within the existing repetition modes:

1. For each layer type, it is still repeated according to its repetition mode (cycle, layerwise, etc.)
2. For each layer, multiply the repeat count by the flow speed to get a "fractional repeat count" T
3. The first `floor(T)` repetitions are applied with `flow_speed=1.0` (fully active)
4. The next repetition is applied with `flow_speed = T - floor(T)` (partially active)
5. All remaining repetitions are applied with `flow_speed = 0.0` (inactive)

This creates a smooth interpolation between discrete layer counts, allowing for partial layer application, while keeping the computational structure fixed.

## Implementation Details

### Changes Needed

1. Add a new `flow_distribution_mode` parameter with a "fractional" option
2. Modify the forward method to implement the fractional flow distribution algorithm within the existing repetition modes

### Mathematical Formulation

For a single layer with repeat factor R and input flow speed s:
- Calculate T = R * s
- Apply the layer R times with the following flow speeds:
  - For repetitions 0 ≤ j < floor(T): flow_speed = 1.0
  - For repetition j = floor(T): flow_speed = T - floor(T)
  - For repetitions j > floor(T): flow_speed = 0.0

This allows us to have a smooth transition between, for example, 2 and 3 effective layers as the flow speed increases, while always performing the same computational work.

### Advantages

1. **Fixed Computational Structure**: Always performs the same number of layer applications, making it easier to train
2. **Smoother Transitions**: Enables a continuous spectrum of model depths
3. **Interpolation Capability**: Creates a natural interpolation between different model configurations
4. **Controlled Information Flow**: Combines the benefits of layer repetition with the precision of flow speed control

## Implementation Plan

The changes will be made to:

1. **GMMTransformer.__init__()**: Add support for the new "fractional" mode
2. **GMMTransformer._calculate_effective_layers()**: Handle the new mode for effective layer calculation
3. **GMMTransformer.forward()**: Implement the fractional repetition algorithm

The implementation will support both global flow speed and per-layer flow speeds.

### Flow Distribution Mode Configuration

We'll add a new configuration parameter to the flow prediction system to control how flow speed is distributed across layer repetitions:

1. **Current "Direct" Distribution**:
   - Flow speed is applied directly to the residual connections in each transformer block
   - Same flow speed is applied to all repetitions of a layer

2. **New "Fractional" Distribution**:
   - Flow speed determines how many repetitions are active
   - Flow is redistributed across repetitions: some get full strength (1.0), one gets partial strength, rest get zero

This will be configured as part of the flow predictor configuration:
```python
# In flow predictor configuration parameters
flow_distribution_mode: str = "direct"  # Options: "direct", "fractional"
```

Important: This is orthogonal to whether we're using a single global flow speed or per-layer flow speeds. The flow speed can still be determined in either of these ways:
- `use_flow_predictor=True/False`: Whether to use flow prediction at all
- `flow_predictor_per_layer=True/False`: Whether to predict one flow speed per layer

When `flow_distribution_mode="fractional"`, the system will use the layer's flow speed (whether global or per-layer) to calculate which repetitions should be active, regardless of which repetition mode is being used (cycle, layerwise, etc.).

For example, with `layer_repeat_mode="layerwise"`, `flow_predictor_per_layer=True`, and `flow_distribution_mode="fractional"`:
- Flow predictor determines a specific flow speed for each layer based on SNR
- Each layer is repeated `repeat_factor` times following the layerwise pattern
- The effective flow_speed for each repetition is calculated based on the fractional allocation
- This keeps the computational structure the same while smoothly varying the effective depth

The implementation will work the same way across all repetition modes by using a unified helper function to compute the current flow for each repetition.

```python
# Example usage in any repetition mode
# 1. Get original flow speed for this layer
original_flow = flow_speed[:, layer_idx]

# 2. Compute the fractional flow distribution for this repetition
current_flow = self._compute_fractional_flow(original_flow, repeat_idx, total_repeats)

# 3. Apply the block with the computed flow
x = block(x, current_flow)
```

For example, in "layerwise" mode with repeat_factor=3 and flow_speed=0.7:
- The first repetition (idx=0) gets flow=1.0 because 0 < floor(3*0.7)=2
- The second repetition (idx=1) gets flow=1.0 because 1 < floor(3*0.7)=2
- The third repetition (idx=2) gets flow=0.1 because 2 = floor(3*0.7) and 3*0.7-2=0.1

While in "cycle" mode with the same settings:
- The first complete cycle (idx=0) gets flow=1.0
- The second complete cycle (idx=1) gets flow=1.0
- The third complete cycle (idx=2) gets flow=0.1

The implementation will preserve all existing functionality while adding the new flow distribution capability.

## Implementation Note

The `flow_distribution_mode` parameter should be passed to the model via the existing kwargs mechanism:

```python
# When creating a model
model = GMMTransformer(
    hidden_dim=256,
    num_layers=6,
    use_flow_predictor=True,
    flow_predictor_type="linear",
    flow_distribution_mode="fractional",  # Add this parameter
    # other parameters...
)
```

This keeps the implementation clean and allows the parameter to be passed along with other flow-related configurations.

## Benefits of Fractional Flow Distribution

This approach offers several advantages:

1. **Smooth Variation of Model Depth**: By enabling partial layer activation, we can smoothly interpolate between discrete numbers of layers
2. **Computational Efficiency**: The computational structure remains constant, making it easier to train and deploy
3. **Compatibility**: Works with existing repetition modes and both global/per-layer flow speeds
4. **Numerically Stable**: Fully active layers use flow_speed=1.0, which is well-conditioned for training

## Testing and Validation

To test this implementation, we should verify:

1. With flow_speed=0.0, no updates should occur in any repetition
2. With flow_speed=1.0, all repetitions should be fully active (as in the current system)
3. Intermediate flow_speed values should activate the appropriate number of repetitions with the correct strengths
4. The system should work correctly with both global and per-layer flow speeds

## Detailed Implementation Changes

### Changes to `transformer.py`

To implement this feature, we'll need to make the following changes to the transformer.py file:

```python
def __init__(
    self,
    # existing parameters...
    **kwargs
) -> None:
    """
    Initialize transformer model.
    """
    super().__init__()
    
    # Store configuration parameters
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.layer_repeat_mode = layer_repeat_mode
    self.repeat_factor = repeat_factor
    self.layer_groups = layer_groups
    self.group_repeat_factors = group_repeat_factors
    
    # Flow configuration parameters
    # These are passed via kwargs and will be available if flow prediction is enabled
    self.flow_distribution_mode = kwargs.get('flow_distribution_mode', 'direct')
    
    # Validate the flow distribution mode
    if self.flow_distribution_mode not in ["direct", "fractional"]:
        raise ValueError(f"Unknown flow_distribution_mode: {self.flow_distribution_mode}")
    
    # Continue with the rest of the initialization...
```

First, add a helper method to compute the current flow based on repetition index and flow speed:

```python
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
```

Then modify the forward method to use this helper function:

```python
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
    
    # Use flow predictor if available and no explicit flow_speed provided
    if flow_speed is None and self.use_flow_predictor and hasattr(self, 'flow_predictor'):
        flow_speed = self.flow_predictor(targets, x)
    elif flow_speed is None:
        # Default flow speed is 1.0 (standard residual update)
        flow_speed = torch.ones(batch_size, device=x.device)
    
    # Determine if we're using per-layer flow
    per_layer_flow = hasattr(self, 'flow_predictor') and getattr(self.flow_predictor, 'per_layer', False)
    
    # For non-per-layer case, reshape to per-layer format for uniform handling
    if not per_layer_flow:
        flow_speed = flow_speed.unsqueeze(1).expand(batch_size, self.num_layers)
    elif flow_speed.dim() == 1:
        # For per-layer case with scalar input, expand to proper shape
        flow_speed = flow_speed.unsqueeze(1).expand(batch_size, self.num_layers)
    
    # Define epsilon for fractional calculations
    epsilon = 1e-7
    
    # Process through transformer blocks based on repetition mode
    if self.layer_repeat_mode == "none":
        # Standard processing - each block used once
        for i, block in enumerate(self.blocks):
            x = block(x, flow_speed[:, i])
            
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
                    x = block(x, current_flow)
                    
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
                    x = block(x, current_flow)
                    
    elif self.layer_repeat_mode == "grouped":
        # Grouped mode - repeat each group according to its factor
        for group_indices, group_repeat_factor in zip(self.group_indices, self.group_repeat_factors):
            for rep in range(group_repeat_factor):
                # Process each block in the group
                for block_idx in group_indices:
                    # Compute flow speed for this repetition
                    current_flow = self._compute_fractional_flow(
                        flow_speed[:, block_idx], rep, group_repeat_factor
                    )
                    
                    # Only apply the block if any flow is non-zero (optimization)
                    if current_flow.max() > epsilon:
                        x = self.blocks[block_idx](x, current_flow)
    
    return x
```

Finally, update the get_config method to include the new parameter:

```python
def get_config(self) -> Dict[str, Any]:
    """Get model configuration for serialization."""
    config = {
        "hidden_dim": self.hidden_dim,
        "num_layers": self.num_layers,
        "num_heads": self.num_heads,
        "layer_repeat_mode": self.layer_repeat_mode,
        "model_type": "GMMTransformer",
        "use_flow_predictor": self.use_flow_predictor
    }
    
    # Add flow speed configuration if enabled
    if self.use_flow_predictor and hasattr(self, 'flow_predictor'):
        config["flow_predictor_type"] = getattr(self.flow_predictor, 'predictor_type', 'dummy')
        config["flow_predictor_per_layer"] = getattr(self.flow_predictor, 'per_layer', False)
        config["flow_distribution_mode"] = self.flow_distribution_mode
    
    # Add existing configuration
    # [...]
    
    return config
```