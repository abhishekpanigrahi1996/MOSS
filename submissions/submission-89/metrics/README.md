# Metrics Package

This package provides tools for creating, tracking, and evaluating metrics for the GMM transformer models.

## Components

- **Factory**: Creates loss functions from string/dictionary configurations
- **Configuration**: Handles parsing of different configuration formats
- **Registry**: Provides a registry for loss function factories
- **Tracker**: Accumulates and computes metrics over batches
- **K-means**: Implements baseline comparison with K-means clustering
- **Utils**: Common utility functions for metrics computation

## Usage

### Creating Loss Functions

```python
from metrics import create_loss_from_config

# Create loss function from simple string
mse_loss = create_loss_from_config('mse')

# Create from extended string format
wasserstein_loss = create_loss_from_config('wasserstein_exact_jax')

# Create from detailed configuration
wasserstein_loss = create_loss_from_config({
    'type': 'wasserstein',
    'algorithm': 'sinkhorn',
    'backend': 'jax',
    'epsilon': 0.01,
    'max_iterations': 100
})
```

### Tracking Metrics

```python
from metrics import MetricsTracker

# Create tracker with metric functions
tracker = MetricsTracker(
    metric_fns={'mse': mse_loss},
    compare_with_kmeans=True
)

# Update with batches
for batch in data_loader:
    # Forward pass
    predictions = model(batch['points'])
    
    # Update metrics
    tracker.update(
        predictions=predictions,
        targets=batch['targets'],
        points=batch['points']
    )

# Compute final metrics
metrics = tracker.compute()
print(metrics)
```

### Computing One-off Metrics

```python
from metrics import compute_metrics, create_metric_functions

# Create metric functions
metric_fns = create_metric_functions(['mse', 'wasserstein'])

# Compute metrics for a single batch
results = compute_metrics(
    predictions=model_predictions,
    targets=ground_truth,
    points=input_points,
    metric_fns=metric_fns,
    compare_with_kmeans=True
)
```

## Configuration

The metrics package supports various configuration formats for maximum flexibility:

- Simple string: `'mse'`
- Extended string: `'wasserstein_exact_jax'`
- Dictionary: `{'type': 'wasserstein', 'algorithm': 'exact', 'backend': 'jax'}`