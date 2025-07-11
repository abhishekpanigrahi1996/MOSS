# Loader Components Implementation

This document provides an overview of the loader components that facilitate easy data loading and integration with machine learning frameworks.

## Factory Functions (`loaders/factory.py`)

Simplifies creation of generators and loaders from configuration dictionaries:

```python
def create_parameter_generator(config, seed=None):
    """Create a ParameterGenerator from configuration."""
    # Extract configuration parameters
    dim = config.get("dim", 2)
    cluster_config = config.get("cluster_config", {"type": "fixed", "value": 2})
    # ... extract other configs
    
    # Create and return a ParameterGenerator
    return ParameterGenerator(dim=dim, cluster_config=cluster_config, ...)

def create_data_generator(config=None, seed=None):
    """Create a DataGenerator from configuration."""
    return DataGenerator(seed=seed)

def create_data_loader(config, seed=None):
    """Create a GMMDataLoader from configuration."""
    # Create generators
    param_gen = create_parameter_generator(config.get("parameter_generator", {}))
    data_gen = create_data_generator(config.get("data_generator", {}))
    
    # Create and return a data loader
    return GMMDataLoader(
        parameter_generator=param_gen,
        data_generator=data_gen,
        batch_size=config.get("batch_size", 32),
        # ... other parameters
    )
```

## GMMDataLoader (`loaders/data_loader.py`)

Provides a PyTorch-compatible data loader for GMM data:

```python
class GMMDataLoader:
    def __init__(self, parameter_generator, data_generator, 
                 batch_size=32, fixed_dataset=False, state_path=None):
        self.parameter_generator = parameter_generator
        self.data_generator = data_generator
        self.batch_size = batch_size
        self.fixed_dataset = fixed_dataset
        
        # If fixed_dataset, generate all data upfront
        if fixed_dataset:
            self._generate_fixed_dataset()
            
        # Load state if provided
        if state_path:
            self.load_state(state_path)
            
    def __iter__(self):
        """Return iterator over batches."""
        self.current_batch = 0
        return self
        
    def __next__(self):
        """Get next batch of data."""
        if self.current_batch >= self.total_batches:
            raise StopIteration
            
        # Generate or retrieve batch
        if self.fixed_dataset:
            batch_data, batch_labels = self._get_fixed_batch(self.current_batch)
        else:
            batch_data, batch_labels = self._generate_dynamic_batch()
            
        self.current_batch += 1
        return batch_data, batch_labels
```

## GMMDataManager (`loaders/data_manager.py`)

Manages creation of multiple data loaders (e.g., for training and validation):

```python
class GMMDataManager:
    def __init__(self, config, seed=None):
        self.config = config
        self.seed = seed
        self.random_state = RandomState(seed)
        
    def create_train_loader(self):
        """Create data loader for training."""
        train_config = self.config.get("train", {})
        train_seed = self._get_seed("train_seed")
        return create_data_loader(train_config, seed=train_seed)
        
    def create_val_loader(self):
        """Create data loader for validation."""
        val_config = self.config.get("val", {})
        val_seed = self._get_seed("val_seed")
        return create_data_loader(val_config, seed=val_seed)
        
    def create_train_val_loaders(self):
        """Create both training and validation loaders."""
        return self.create_train_loader(), self.create_val_loader()
```

## Reproducibility and State Management

All loader components support saving and loading state for reproducibility:

```python
# GMMDataLoader methods for state management
def save_state(self, path):
    """Save the state of the data loader."""
    state = {
        "parameter_generator_state": self.parameter_generator.get_state(),
        "data_generator_state": self.data_generator.get_state(),
        "current_batch": self.current_batch,
        "batch_size": self.batch_size,
        "fixed_dataset": self.fixed_dataset
    }
    # Save to file
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)
        
def load_state(self, path):
    """Load the state of the data loader."""
    # Load from file
    with open(path, 'r') as f:
        state = json.load(f)
    
    # Restore state
    self.parameter_generator.set_state(state["parameter_generator_state"])
    self.data_generator.set_state(state["data_generator_state"])
    self.current_batch = state["current_batch"]
    self.batch_size = state["batch_size"]
    self.fixed_dataset = state["fixed_dataset"]
```