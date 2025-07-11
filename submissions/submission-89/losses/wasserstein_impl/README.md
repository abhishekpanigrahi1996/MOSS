# Wasserstein Distance Implementations

This package provides multiple implementations of the Wasserstein distance (Earth Mover's Distance) for PyTorch, with automatic differentiation support.

> **Note:** For comprehensive documentation, please see the [Loss Functions Documentation](../../docs/losses.md) in the unified project documentation.

## Package Structure

The package is organized into the following structure:

```
wasserstein_impl/
├── __init__.py            # Main package interface
├── backends/              # Backend-specific implementation details
│   ├── __init__.py        # Backend availability detection
│   ├── jax_backend.py     # JAX backend implementation
│   ├── pot_backend.py     # POT backend implementation
│   └── scipy_backend.py   # SciPy backend implementation
├── autograd/              # PyTorch autograd integration
│   ├── __init__.py
│   ├── jax_autograd.py    # JAX-to-PyTorch autograd functions
│   ├── pot_autograd.py    # POT-to-PyTorch autograd functions
│   └── scipy_autograd.py  # SciPy-to-PyTorch autograd functions
├── methods/               # Algorithm-specific implementations
│   ├── __init__.py
│   ├── exact.py           # Exact (unregularized) methods
│   └── regularized.py     # Regularized methods (Sinkhorn)
└── utils.py               # Utility functions
```

## Usage

### Simple Interface

The package provides a simple interface via the `wasserstein_distance` function:

```python
import torch
from losses.wasserstein_impl import wasserstein_distance

# Create two batches of point clouds
x = torch.randn(2, 100, 2)  # [batch_size, n_points_x, dim]
y = torch.randn(2, 80, 2)   # [batch_size, n_points_y, dim]

# Optional weights
x_weights = torch.ones(2, 100) / 100  # [batch_size, n_points_x]
y_weights = torch.ones(2, 80) / 80    # [batch_size, n_points_y]

# Compute distances
distances = wasserstein_distance(
    x, y, x_weights, y_weights,
    method="regularized",            # "exact" or "regularized"
    implementation="auto",           # "auto", "jax", "pot", or "scipy"
    epsilon=0.1,                     # Regularization parameter (for regularized)
    max_iterations=1000              # Max iterations (for regularized)
)

print(distances)  # [batch_size]
```

### PyTorch Modules

For integration into PyTorch models, you can use the method-specific implementations:

```python
import torch
from losses.wasserstein_impl.methods.regularized import get_regularized_implementation
from losses.wasserstein_impl.methods.exact import get_exact_implementation

# Create point clouds
x = torch.randn(2, 100, 2)  # [batch_size, n_points_x, dim]
y = torch.randn(2, 80, 2)   # [batch_size, n_points_y, dim]

# Get a regularized implementation
sinkhorn = get_regularized_implementation(
    implementation="auto",  # "auto", "jax", or "pot"
    epsilon=0.1,
    max_iterations=1000
)

# Or get an exact implementation
exact = get_exact_implementation(
    implementation="auto"  # "auto", "jax", "pot", or "scipy"
)

# Compute distances
sinkhorn_distances = sinkhorn(x, y)
exact_distances = exact(x, y)

print(sinkhorn_distances)  # [batch_size]
print(exact_distances)     # [batch_size]
```

### Advanced Usage with Specific Implementations

You can also use specific implementation classes directly:

```python
import torch
from losses.wasserstein_impl.methods.regularized import JaxRegularizedWasserstein, PotRegularizedWasserstein
from losses.wasserstein_impl.methods.exact import JaxExactWasserstein, PotExactWasserstein, ScipyExactWasserstein

# Create your own instances
jax_sinkhorn = JaxRegularizedWasserstein(epsilon=0.1, max_iterations=1000)
pot_sinkhorn = PotRegularizedWasserstein(epsilon=0.1, max_iterations=1000)

jax_exact = JaxExactWasserstein()
pot_exact = PotExactWasserstein()
scipy_exact = ScipyExactWasserstein()

# Use them as PyTorch modules
# ...
```

## Implementation Details

### Available Backends

- **JAX**: Uses OTT-JAX library for fast GPU-accelerated Wasserstein distance computation.
- **POT**: Uses Python Optimal Transport library, which supports more features but may be slower.
- **SciPy**: Uses SciPy's linear sum assignment for exact Wasserstein distance (CPU only).

### Method Types

- **Exact**: Computes the exact (unregularized) Wasserstein distance.
  - Advantages: Exact solution, no hyperparameters to tune
  - Disadvantages: Slower for large point clouds, may have optimization challenges
  
- **Regularized**: Computes the entropy-regularized Wasserstein distance (Sinkhorn algorithm).
  - Advantages: Faster, more stable gradients
  - Disadvantages: Approximate solution, requires tuning regularization parameter
  
### Implementation Features

|                        | JAX          | POT         | SciPy       |
|------------------------|--------------|-------------|-------------|
| **Exact**              | ✅           | ✅          | ✅          |
| **Regularized**        | ✅           | ✅          | ❌          |
| **Different # points** | ✅           | ✅          | ❌          |
| **Non-uniform weights**| ✅           | ✅          | ❌          |
| **GPU acceleration**   | ✅           | ❌          | ❌          |
| **PyTorch integration**| ✅           | ✅          | ✅          |

## Usage Recommendations

1. **General Use**: Use the POT implementation as the default option. It works well in most cases and provides a good balance of stability and speed.

2. **GPU Computing**: 
   - For smaller batches: POT implementation
   - For larger batches and when differentiability is required: JAX implementation

3. **Exact Solutions**:
   - When point counts are equal: SciPy implementation
   - When point counts differ: POT implementation

4. **Differentiability**:
   - Both POT and JAX support gradients, but with different magnitudes
   - POT gradients may be more numerically stable for optimization

## Implementation Notes

- The JAX implementation has a fallback mechanism when the exact solver fails, automatically switching to the Sinkhorn algorithm.
- The POT implementation includes optimizations for vectorized processing of batches and handles special cases like identical distributions.
- For very large point clouds, the Sinkhorn algorithm is generally preferred as it scales better than exact solutions.

## References

- [POT: Python Optimal Transport](https://pythonot.github.io/)
- [OTT: Optimal Transport Tools in JAX](https://ott-jax.readthedocs.io/en/latest/)
- [Wasserstein Distance (Earth Mover's Distance)](https://en.wikipedia.org/wiki/Wasserstein_metric) 