# Data State Management in GMM-v2

This document explains how state management and restoration work in the GMM-v2 data loaders and generators, providing examples and guidelines for ensuring reproducibility during training, validation, and when resuming from checkpoints.

## Overview

The GMM-v2 framework provides a comprehensive system for managing the state of data generators and loaders, enabling reproducible data generation and the ability to resume training from exact points. This is particularly important for:

1. **Reproducible experiments** - Being able to generate identical data given the same seed
2. **Resumable training** - Continuing training from exactly where it left off
3. **Validation consistency** - Ensuring validation is performed on consistent data distributions

## State Management API

The data loader offers a clear API for managing state with two key parameters:

- `state_path`: Specifies the path for saving or loading state
- `resume`: Controls whether to load state from the file (`True`) or initialize fresh with the seed (`False`, default)

### Behavior

When both `state_path` and `seed` parameters are provided, the behavior depends on the `resume` parameter:

- **If `resume=True`**:
  - Load and restore state from the file, ignoring the provided seed
  - Continue generating data from the exact point where the state was saved
  - This is useful for resuming training from checkpoints

- **If `resume=False` (default)**:
  - Initialize the loader using the seed
  - The `state_path` is only used for saving state, not loading initially
  - This is useful for reproducible experiments that should always start the same way

## Key Components

### 1. RandomState

The core of state management is the `RandomState` class in `data/utils/random_utils.py`, which provides a reliable wrapper around numpy's random number generators:

```python
class RandomState:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self._initial_state = self._copy_bit_generator_state(self.rng.bit_generator.state)
        
    def get_state(self) -> Dict[str, Any]:
        bg_state = self._copy_bit_generator_state(self.rng.bit_generator.state)
        return {
            "bit_generator_state": bg_state,
            "seed": self.seed
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        if "bit_generator_state" in state:
            bg_state = self._copy_bit_generator_state(state["bit_generator_state"])
            self.rng.bit_generator.state = bg_state
```

This class ensures that random number generation can be serialized and restored precisely.

### 2. ParameterGenerator

The `ParameterGenerator` class in `data/core/parameter_generator.py` handles generation of GMM parameters with state management:

```python
def get_state(self) -> Dict[str, Any]:
    return {
        "random_state": self.random_state.get_state()
    }

def set_state(self, state: Dict[str, Any]) -> None:
    if "random_state" in state:
        self.random_state.set_state(state["random_state"])
```

### 3. DataGenerator

The `DataGenerator` class in `data/core/data_generator.py` manages the generation of GMM data using parameters from the ParameterGenerator:

```python
def get_state(self) -> Dict[str, Any]:
    state = {
        "random_state": self.random_state.get_state(),
        "param_generator_state": self.param_generator.get_state()
    }
    
    # Also save cached data if available
    if hasattr(self, '_cached_data'):
        state["cached_data"] = {...}
    
    return state

def set_state(self, state: Dict[str, Any]) -> None:
    if "random_state" in state:
        self.random_state.set_state(state["random_state"])
    
    if "param_generator_state" in state:
        self.param_generator.set_state(state["param_generator_state"])
        
    # Restore cached data if available
    if "cached_data" in state:
        self._cached_data = np.array(state["cached_data"]["data"])
        self._cached_labels = np.array(state["cached_data"]["labels"])
        self._cached_batch_size = state["cached_data"]["batch_size"]
```

### 4. GMMDataLoader

The `GMMDataLoader` class in `data/loaders/data_loader.py` provides an iterator interface for training, with state management:

```python
def save_state(self, path: Optional[str] = None) -> Dict[str, Any]:
    save_path = path or self.state_path
    if save_path is None:
        save_path = f"gmm_loader_state_{self.loader_id}.json"
        
    return save_generator_state(
        self.param_generator,
        self.data_generator,
        save_path
    )
```

### 5. State Serialization Utilities

The factory functions in `data/loaders/factory.py` handle saving and loading states to/from files:

```python
def save_generator_state(param_generator, data_generator, file_path):
    state = {
        "param_generator_state": param_generator.get_state(),
        "data_generator_state": data_generator.get_state()
    }
    
    with open(file_path, 'w') as f:
        json.dump(state, f, cls=NumpyEncoder)
        
    return state

def load_generator_state(file_path):
    with open(file_path, 'r') as f:
        state = json.load(f)
    
    # Process state to ensure proper bit generator state handling
    # ...
    
    return state
```

## State Management Workflow

The typical workflow for state management follows these patterns:

### 1. Basic Reproducibility with Seeds

For simple reproducibility, use the same seed:

```python
# Create first loader with seed 42
loader1 = GMMDataLoader(
    config_dict=config_dict,
    batch_size=4,
    num_samples=16,
    seed=42
)

# Create second loader with same seed
loader2 = GMMDataLoader(
    config_dict=config_dict,
    batch_size=4,
    num_samples=16,
    seed=42
)

# Both will produce the same data sequence
```

### 2. Saving and Restoring State with the Resume Parameter

For resuming from specific points:

```python
# Create loader and save state
loader = GMMDataLoader(
    config_dict=config_dict,
    batch_size=4,
    num_samples=16,
    seed=42,
    state_path="./loader_state.json",
    resume=False  # Default - use seed, don't load state
)

# Generate some batches
for _ in range(3):
    batch = next(iter(loader))
    # Process batch...

# Save state
state = loader.save_state()

# Later, resume from this state
resumed_loader = GMMDataLoader(
    config_dict=config_dict,
    batch_size=4,
    num_samples=16,
    seed=99,  # Different seed - ignored when resume=True
    state_path="./loader_state.json",
    resume=True  # Important! This tells the loader to load state
)

# The resumed loader will continue exactly where the previous loader left off
# next(iter(resumed_loader)) will give the 4th batch, not repeat the 1st batch
next_batch = next(iter(resumed_loader))
```

### 3. Manual State Management

For more fine-grained control, you can still use the explicit state management methods:

```python
# Create a loader
loader = GMMDataLoader(
    config_dict=config_dict,
    batch_size=4,
    num_samples=16,
    seed=42
)

# Save state to a custom location
state = loader.save_state("./custom_state.json")

# Create a new loader and manually restore state
new_loader = GMMDataLoader(
    config_dict=config_dict,
    batch_size=4,
    num_samples=16
)

# Explicitly load and restore state
state = load_generator_state("./custom_state.json")
new_loader.restore_state(state)

# Now the new loader will continue where the original left off
```

### 4. Using State for Resumable Training

When implementing resumable training:

```python
# Training loop with state checkpointing
def train_with_checkpointing(model, loader, num_epochs):
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(loader):
            # Training step...
            
            # Save state periodically
            if batch_idx % 10 == 0:
                # Save loader state
                loader.save_state('./loader_state.json')
                
                # Also save model checkpoint
                torch.save({
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'batch_idx': batch_idx
                }, f'checkpoint_e{epoch}_b{batch_idx}.pt')
    
# Resume training
def resume_training(model, config_dict, checkpoint_path):
    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']
    batch_idx = checkpoint['batch_idx']
    
    # Create loader with resume=True to load state automatically
    loader = GMMDataLoader(
        config_dict=config_dict,
        batch_size=4,
        num_samples=16,
        state_path='./loader_state.json',
        resume=True  # This automatically restores the state
    )
    
    # Resume training
    train_with_checkpointing(model, loader, num_epochs)
```

## Important Considerations

### 1. State vs. Seed Initialization

- **Seed Initialization**: Initializes the random state from scratch with a given seed. Useful for starting new experiments.
- **State Restoration**: Restores the exact internal state of random generators. Required for resuming from specific points.

Using the same seed only guarantees reproducibility when starting from the beginning. To resume from a specific point, you must save and restore the exact state.

### 2. Fixed vs. Dynamic Data

The `GMMDataLoader` can operate in two modes:

- **Fixed Data Mode**: Generates all batches upfront and reuses them in each epoch
- **Dynamic Data Mode**: Generates new data for each batch on-the-fly

State management is important for both, but especially critical for dynamic data mode when resuming training.

### 3. Common Pitfalls

#### Correct Way to Resume Training

With the new `resume` parameter, state restoration is much simpler:

```python
# INCORRECT - missing resume=True
loader = GMMDataLoader(
    config_dict=config_dict,
    state_path=state_path
)
# This loader will NOT automatically load state from file

# CORRECT - use resume=True to load state
loader = GMMDataLoader(
    config_dict=config_dict,
    state_path=state_path,
    resume=True  # This signals intent to load and restore state
)
# This loader will automatically load state and continue where previous left off
```

#### Failing to Handle Batch/Epoch Counters

When resuming with `resume=True`, batch and epoch counters are automatically restored from the state file. However, for manual state management, ensure you track these:

```python
# For manual state management, track these in your training loop and checkpoints
loader.current_batch = checkpoint['batch_idx']
loader.current_epoch = checkpoint['epoch']
```

#### Not Saving State at the Right Time

Save state at stable points, usually between batches or epochs:

```python
# Save at end of epoch
for epoch in range(num_epochs):
    for batch in loader:
        # Training...
    loader.save_state()  # Save at end of epoch
```

## Integrating with ExperimentManager

The `ExperimentManager` in the GMM-v2 framework can be extended to handle data state management when resuming training. When implementing the `resume_training` method, leverage the new `resume` parameter:

1. Save data loader state along with model checkpoints
2. When resuming, use `resume=True` when initializing the data loader
3. Counters for epochs and batches will be automatically synchronized from the state file

Example implementation with the new API:

```python
def resume_training(self, num_epochs=None):
    """Resume training from the latest checkpoint."""
    # Get latest checkpoint
    checkpoint_dir = self.experiment_dir / "checkpoints"
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Verify data state file exists
    state_path = self.experiment_dir / "data_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Data state file not found: {state_path}")
    
    # Load checkpoint for model
    self.trainer._load_checkpoint(latest_checkpoint)
    
    # Create new data loader with resume=True to automatically restore state
    self.train_loader = GMMDataLoader(
        config_dict=self.config.data_config,
        batch_size=self.config.training_config.batch_size,
        num_samples=self.config.training_config.samples_per_epoch,
        state_path=str(state_path),
        resume=True  # Automatically load and restore state
    )
    
    # Continue training
    return self.run(num_epochs=num_epochs)
```

## Conclusion

Proper state management is essential for reproducible experiments and resumable training in the GMM-v2 framework. The enhanced API with the `resume` parameter makes it much clearer and simpler to distinguish between:

1. Starting fresh with a seed (reproducible from the beginning)
2. Resuming from a saved state (continuing from a specific point)

With this clear, explicit control over state management behavior, you can ensure:
- Experiments are reproducible when needed
- Training can be resumed precisely where it left off
- Code is more readable with explicit intent through the `resume` parameter