# GMM Loss Functions

This package provides various loss functions for training and evaluating Gaussian Mixture Models (GMMs) or any model that predicts points or cluster centers.

> **Note:** For comprehensive documentation, please see the [Loss Functions Documentation](../docs/losses.md) in the unified project documentation.

## Features

- **Unified Interface**: All losses follow a consistent interface for labeled data
- **Multiple Loss Types**: 
  - MSE (Mean Squared Error) 
  - Energy Distance
  - Wasserstein Distance (with exact and regularized implementations)
- **Flexible Weighting**: Support for both uniform and non-uniform weights
- **Hardware Support**: 
  - CPU and GPU acceleration
  - JAX, POT, and SciPy implementations for Wasserstein distance

## Package Structure

- `__init__.py`: Exports public functions and handles imports
- `mse.py`: MSE loss implementation
- `energy.py`: Energy distance loss implementation
- `wasserstein.py`: Unified interface for all Wasserstein distance implementations
- `utils.py`: Common utility functions for loss calculations
- `wasserstein_impl/`: Backend implementations for Wasserstein distance
  - `pot.py`: Python Optimal Transport implementation
  - `scipy.py`: SciPy-based implementation
  - `jax.py`: JAX-based implementation
  - `base.py`: Base classes and factory functions
  - `backends.py`: Backend availability detection

## Installation Requirements

- PyTorch: Required for all losses
- NumPy: Required for all losses
- POT (Python Optimal Transport): Optional, for Wasserstein distance
- JAX + OTT-JAX: Optional, for accelerated Wasserstein distance
- SciPy: Optional, for exact Wasserstein distance with uniform weights

```bash
# First install base dependencies
pip install torch numpy

# For Wasserstein distance
pip install scipy pot

# Optional: install JAX for improved GPU performance
pip install jax jaxlib
```

## Unified Interface

All losses in this package follow a consistent interface for labeled data:

```python
loss = loss_function(
    predictions,  # [batch_size, seq_len, dim]
    labels,       # [batch_size, seq_len]
    positions,    # [batch_size, n_positions, dim]
    ...           # Loss-specific parameters
)
```

Where:
- `predictions`: Predicted points with shape `[batch_size, seq_len, dim]`
- `labels`: Target labels (integers) with shape `[batch_size, seq_len]`
- `positions`: Target positions for each label with shape `[batch_size, n_positions, dim]`

## Loss Functions

### MSE Loss

Mean Squared Error loss between predicted points and target positions.

```python
from gmm.losses import mse_loss, MSELoss

# Functional API
loss = mse_loss(predictions, labels, positions, reduction='mean')

# Module API
loss_fn = MSELoss(reduction='mean')
loss = loss_fn(predictions, labels, positions)
```

Parameters:
- `reduction`: How to reduce batch-wise losses ('mean', 'sum', 'none')

### Energy Distance Loss

Energy distance between predicted point distribution and target positions.

```python
from gmm.losses import energy_loss, EnergyLoss

# Functional API
loss = energy_loss(
    predictions, 
    labels, 
    positions, 
    p=2.0,            # Power parameter for distance
    squared=True,     # Return squared distances
    reduction='mean'  # Reduction method
)

# Module API
loss_fn = EnergyLoss(p=2.0, squared=True, reduction='mean')
loss = loss_fn(predictions, labels, positions)
```

Parameters:
- `p`: Power parameter for distance calculation (p=2 for Euclidean)
- `squared`: If True, return squared energy distance
- `reduction`: How to reduce batch-wise losses ('mean', 'sum', 'none')

### Wasserstein Loss

Wasserstein distance between predicted point distribution and target positions.

```python
from gmm.losses import wasserstein_loss, WassersteinLoss

# Functional API
loss = wasserstein_loss(
    predictions, 
    labels, 
    positions, 
    implementation='auto',    # Implementation to use
    algorithm='auto',         # Algorithm to use
    epsilon=0.01,             # Regularization parameter
    max_iterations=10000,     # Max iterations for Sinkhorn
    reduction='mean'          # Reduction method
)

# Module API
loss_fn = WassersteinLoss(
    implementation='auto',
    algorithm='auto',
    epsilon=0.01,
    max_iterations=10000,
    reduction='mean'
)
loss = loss_fn(predictions, labels, positions)
```

Parameters:
- `implementation`: Backend implementation ('auto', 'pot', 'scipy', 'jax')
- `algorithm`: OT algorithm ('auto', 'exact', 'sinkhorn')
- `epsilon`: Regularization parameter for Sinkhorn algorithm
- `max_iterations`: Maximum iterations for Sinkhorn algorithm
- `reduction`: How to reduce batch-wise losses ('mean', 'sum', 'none')

## Advanced Usage

### Working with Weights

The Energy and Wasserstein losses support weighted distributions. Weights are derived automatically from label frequencies:

```python
# Example with 3 clusters where cluster 0 has more points than others
labels = torch.tensor([0, 0, 0, 1, 1, 2])
positions = torch.randn(3, 2)  # 3 positions in 2D

# The weights will be computed as [0.5, 0.33, 0.17] automatically
loss = wasserstein_loss(predictions, labels, positions)
```

### Direct Point Comparison 

All loss functions also support direct point-to-point comparison:

```python
from gmm.losses import mse_loss_direct, energy_loss_direct, wasserstein_loss_direct

# Direct comparison between point sets
# x, y: tensors of shape [batch_size, n_points, dim]
mse = mse_loss_direct(x, y)
energy = energy_loss_direct(x, y)
wasserstein = wasserstein_loss_direct(x, y)
```

### Specialized Wasserstein Loss Implementations

For more control over Wasserstein distance calculation:

```python
from gmm.losses import ExactWassersteinLoss, RegularizedWassersteinLoss

# Exact Wasserstein implementation with specific backend
exact_loss = ExactWassersteinLoss(backend="pot")

# Regularized Wasserstein (Sinkhorn algorithm)
reg_loss = RegularizedWassersteinLoss(
    backend="jax", 
    epsilon=0.01, 
    max_iterations=1000,
    reduction="mean"
)

# Compute losses
loss1 = exact_loss(x, y)
loss2 = reg_loss(x, y, x_weights, y_weights)
```

## Implementation Details and Performance

### MSE Loss

- Fastest computation, ideal for quick training iterations
- Easy to interpret
- Limited for comparing distributions with different shapes

### Energy Distance

- Good balance of speed and statistical accuracy
- No parameters to tune
- Works well for comparing distributions with different shapes
- Supports weighted distributions
- Always differentiable

### Wasserstein Distance

This package provides three Wasserstein distance implementations:

| Implementation | Advantages | Limitations |
|----------------|------------|-------------|
| POT            | Supports non-uniform weights, different point counts | Slower for large point sets |
| JAX            | Fast GPU acceleration, good gradients | Requires JAX installation |
| SciPy          | No additional dependencies | Only supports exact algorithm with uniform weights |

Algorithm selection:
- **Exact Algorithm**: Use for small point clouds (< 50 points)
  - More accurate
  - Slower for large point sets
  - Not always differentiable
  
- **Sinkhorn Algorithm**: Use for larger point clouds
  - Faster for large point sets
  - Always differentiable
  - Requires tuning epsilon parameter (higher = faster but less accurate)

The `algorithm="auto"` option automatically selects the best algorithm based on your data size.

## Performance Considerations

- For small datasets, MSE is fastest, followed by Energy Distance and Wasserstein
- For large datasets, Energy Distance may outperform regularized Wasserstein
- On GPU, JAX-based Wasserstein implementation provides best performance
- Adjust epsilon and max_iterations for regularized Wasserstein to balance accuracy and speed

## Testing

The package includes comprehensive test suites:

- `tests/test_unified_losses.py`: Tests the unified loss interface
- `tests/test_unified_wasserstein.py`: Tests specific Wasserstein implementations
- `tests/test_losses.py`: Basic tests for loss functions

Run tests with pytest:

```bash
pytest -xvs losses/tests/
```