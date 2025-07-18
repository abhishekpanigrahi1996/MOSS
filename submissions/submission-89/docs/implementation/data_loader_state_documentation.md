# GMM Data Loader State Management

This documentation provides a comprehensive guide to understanding and working with the state management system in the GMM data loaders. Understanding these concepts is essential for creating reproducible experiments, especially when resuming training.

## Table of Contents
- [Overview](#overview)
- [Key Components](#key-components)
- [State Management Mechanisms](#state-management-mechanisms)
- [Common Use Cases](#common-use-cases)
- [Special Considerations](#special-considerations)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The GMM data loader state management system enables:
- Deterministic data generation by saving and restoring random states
- Consistent resumption of training across sessions
- Reproducible results from fixed random seeds
- Flexibility to use both fixed and dynamically generated datasets

State management is particularly important in the GMM context due to the complex, multi-level randomness involved in generating GMM parameters, noise levels, and sample data.

## Key Components

### 1. GMMDataLoader 

The main interface for loading and generating GMM data. It manages:
- Batch position (epoch and batch counters)
- Random seed and derived seeds
- Parameter and data generators
- State serialization and deserialization

```python
loader = GMMDataLoader(
    config_dict=config,
    batch_size=4,
    num_samples=16,
    base_seed=42,           # Base seed for reproducibility
    loader_id="train",      # ID for unique derived seeds
    fixed_data=False,       # Whether to use fixed data
    state_path="state.json", # Where to save/load state
    resume=True             # Whether to resume from saved state
)
```

### 2. RandomState

A wrapper for NumPy's random number generator that ensures reliable state saving and restoration:
- Uses bit generator state for reproducibility
- Performs deep copying of state to prevent shared references
- Handles serialization edge cases for various NumPy versions

### 3. ParameterGenerator

Manages the generation of GMM parameters with:
- Control of SNR, MI, or MI factor
- Consistent parameter sets based on random seed
- State tracking for reproducibility

### 4. DataGenerator

Responsible for generating batches of data using the parameters:
- Manages internal random state
- Creates samples according to GMM parameters
- Controls batch-level parameter variation

## State Management Mechanisms

### State Structure

The saved state is a nested dictionary containing:

```
state
   param_generator       # ParameterGenerator state
      random_state      # Random state info
         seed          # Original seed
         bit_generator_state  # NumPy RNG state
      cached_values     # Cached parameter values
      ...               # Other generator settings
   data_generator        # DataGenerator state
      random_state      # Random state info
      config            # Generator configuration
      ...               # Other generator settings
   loader                # DataLoader state
       current_epoch     # Current epoch counter
       current_batch     # Current batch counter
       loader_id         # Unique loader ID
       fixed_data        # Whether using fixed data
       base_seed         # Original base seed
       seed              # Derived seed value
```

### Seed Derivation

To ensure different loaders have unique but deterministic random sequences:

1. The loader uses both `base_seed` and `loader_id` to create a derived seed:
   ```python
   hashed_id = int(hashlib.md5(str(loader_id).encode()).hexdigest(), 16) % 1000000
   derived_seed = base_seed + hashed_id
   ```

2. This derived seed is used for the generator random states, ensuring:
   - Same base_seed + same loader_id = same data sequence
   - Same base_seed + different loader_id = different but deterministic sequences

### Saving State

State saving captures multiple levels:

```python
# Save state explicitly
state_path = "path/to/state.json"
loader.save_state()  # Writes to the state_path provided in constructor
```

Behind the scenes, this:
1. Gets state from parameter generator
2. Gets state from data generator
3. Adds loader state (epoch/batch position, seeds, etc.)
4. Serializes to JSON with special handling for NumPy types

### Loading State

State loading can happen in two ways:

1. **Explicit**: When `resume=True` is used in constructor and a state file exists
2. **Manual**: By calling a factory function with a loaded state

```python
# Automatic resumption
loader = GMMDataLoader(
    config_dict=config,
    state_path="path/to/state.json",
    resume=True  # This triggers state loading if file exists
)

# Manual resumption
loaded_state = load_state("path/to/state.json")
param_gen, data_gen = create_generators(config, loaded_state)
```

## Common Use Cases

### 1. Resuming Training from Checkpoint

Create a loader that continues from a saved state:

```python
train_loader = GMMDataLoader(
    config_dict=config,
    batch_size=16,
    num_samples=1000,
    state_path="output/checkpoint/loader_state.json",
    resume=True  # Critical for resumption
)
```

### 2. Creating Train/Validation Pairs

Create consistent training and validation loaders:

```python
train_loader, val_loader = GMMDataLoader.create_train_val_pair(
    config_dict=config,
    train_batch_size=16,
    val_batch_size=32,
    train_samples=1000,
    val_samples=200,
    base_seed=42,
    state_dir="output/experiment_1/",
    resume=False  # True would resume from saved states
)
```

### 3. Ensuring Deterministic Datasets with Fixed Data

For smaller datasets that should be fully deterministic:

```python
fixed_loader = GMMDataLoader(
    config_dict=config,
    batch_size=16,
    num_samples=1000,
    base_seed=42,
    fixed_data=True  # Generates all data upfront
)
```

### 4. Checkpoint/Resume for Fixed Data

Fixed data loaders maintain state for batch positioning:

```python
# Create fixed data loader
fixed_loader = GMMDataLoader(
    config_dict=config,
    batch_size=16,
    num_samples=1000,
    base_seed=42,
    fixed_data=True,
    state_path="output/fixed_state.json"
)

# Save state after training for a bit
for i, (data, targets) in enumerate(fixed_loader):
    if i == 50:  # After 50 batches
        fixed_loader.save_state()
        break

# Resume later from the same position
resumed_loader = GMMDataLoader(
    config_dict=config,
    batch_size=16,
    num_samples=1000,
    base_seed=42,  # Same seed
    fixed_data=True,
    state_path="output/fixed_state.json",
    resume=True
)
# First batch from resumed_loader will be the 51st batch
```

## Special Considerations

### MI Estimation Randomness

Mutual Information (MI) estimation uses Monte Carlo sampling that introduces some non-deterministic behavior even with fixed seeds. When comparing target values:

- Most tensor values are exactly reproducible
- MI and MI factor values may have small variations (~1-2% difference)
- Consider using approximate comparisons for MI-related values: `torch.allclose(t1, t2, rtol=1e-1, atol=1e-1)`

### Multiprocessing Consistency

To ensure consistency across processes:

- Use the **same `loader_id`** for loaders that should produce identical data
- Different `loader_id` values will produce different data sequences even with the same `base_seed`
- This is critical for distributed training scenarios

```python
# This will produce the same data in different processes
loader = GMMDataLoader(
    config_dict=config,
    base_seed=42,
    loader_id="shared_id"  # Critical for cross-process consistency
)
```

### Fixed Data vs. Dynamic Data

- **Fixed data** (`fixed_data=True`):
  - Generates all data upfront
  - Guarantees batch consistency across epochs
  - Higher memory usage
  - Best for smaller datasets or validation

- **Dynamic data** (`fixed_data=False`):
  - Generates data on-the-fly
  - Different batches in each epoch (unless explicitly restored)
  - Lower memory usage
  - Best for training with large datasets

## Examples

### Example 1: Basic Training Loop with State Saving

```python
def train_with_checkpointing(config, num_epochs=10, checkpoint_interval=2):
    # Create data loader
    loader = GMMDataLoader(
        config_dict=config,
        batch_size=16,
        num_samples=1000,
        base_seed=42,
        state_path="output/train_state.json"
    )
    
    # Create model, optimizer, etc.
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(loader):
            # Forward pass
            outputs = model(data)
            loss = calculate_loss(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Save data loader state and model checkpoint
        if (epoch + 1) % checkpoint_interval == 0:
            loader.save_state()
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch
            }, f"output/model_checkpoint_epoch_{epoch+1}.pt")
            
    return model
```

### Example 2: Resuming Training from Checkpoint

```python
def resume_training(config, checkpoint_path, model_path, additional_epochs=5):
    # Load model checkpoint
    checkpoint = torch.load(model_path)
    model = create_model()
    model.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    
    # Create data loader with resumption
    loader = GMMDataLoader(
        config_dict=config,
        batch_size=16,
        num_samples=1000,
        base_seed=42,
        state_path=checkpoint_path,
        resume=True  # Resume from saved state
    )
    
    # Continue training
    for epoch in range(start_epoch, start_epoch + additional_epochs):
        for i, (data, targets) in enumerate(loader):
            # Training steps as before
            outputs = model(data)
            loss = calculate_loss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Save checkpoints
        loader.save_state()
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, f"output/model_checkpoint_epoch_{epoch+1}.pt")
            
    return model
```

### Example 3: Creating and Using Fixed Data Loaders

```python
def run_experiment_with_fixed_data(config):
    # Create train/val pair with fixed validation data
    train_loader, val_loader = GMMDataLoader.create_train_val_pair(
        config_dict=config,
        train_batch_size=16,
        val_batch_size=8,
        train_samples=1000,
        val_samples=200,
        base_seed=42,
        fixed_data=False,  # Dynamic training data
        fixed_validation_data=True,  # Fixed validation data
        state_dir="output/experiment"
    )
    
    # Create model
    model = create_model()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Train and validate
    best_val_loss = float('inf')
    for epoch in range(10):
        # Training
        model.train()
        for data, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = calculate_loss(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation (same data every epoch due to fixed_validation_data=True)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, targets in val_loader:
                outputs = model(data)
                val_loss += calculate_loss(outputs, targets).item()
        
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
        
        # Save model if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "output/best_model.pt")
            
        # Save loader states
        train_loader.save_state()
        val_loader.save_state()
    
    return model
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Data Doesn't Match After Resumption

**Issue**: Data from resumed loader doesn't match original loader.

**Solutions**:
- Check that you're comparing the **next batch** after saving state, not the same batch
- Ensure `loader_id` is the same when resuming
- Make sure `fixed_data` setting is consistent before and after resumption
- For MI-related values, use approximate equality checks with appropriate tolerances

#### 2. Loader ID Warnings

**Warning**: `Mismatch in loader_id: stored=loader_X, current=loader_Y`

**Solutions**:
- For exact reproduction, use explicit, consistent loader_ids rather than auto-generated ones
- If the warning appears but data still matches, it may be safe to ignore
- For critical reproducibility, add logic to adopt the stored loader_id during resumption

#### 3. Multiprocessing Inconsistency

**Issue**: Data from loaders in different processes doesn't match despite same seeds.

**Solution**:
- Always use the **same explicit `loader_id`** in all processes
- Avoid dynamic ID generation based on process information

#### 4. Memory Issues with Fixed Data

**Issue**: Running out of memory with `fixed_data=True`.

**Solutions**:
- Use `fixed_data=False` for large datasets
- Reduce `num_samples` and batch sizes
- Use `fixed_validation_data=True` with `fixed_data=False` to have fixed validation data but dynamic training data

### Debugging Tips

1. **Inspect State Files**: Check JSON state files to confirm loader positions and seed values
   ```python
   import json
   with open("state.json", "r") as f:
       state = json.load(f)
       print(f"Loader state: {state.get('loader', {})}")
       print(f"Current position: Epoch {state.get('loader', {}).get('current_epoch')}, Batch {state.get('loader', {}).get('current_batch')}")
   ```

2. **Compare Generated Data**: Print sample values from different loaders to verify
   ```python
   data1, _ = next(iter(loader1))
   data2, _ = next(iter(loader2))
   print(f"Loader1 first few values: {data1[0, 0, :4].tolist()}")
   print(f"Loader2 first few values: {data2[0, 0, :4].tolist()}")
   print(f"Data is identical: {torch.allclose(data1, data2)}")
   ```

3. **Check Random State**: Directly inspect RandomState derivatives
   ```python
   # Verify that RNG state properly advances
   first_batch = next(iter(loader))
   print(loader.param_generator.random_state.rng.random())
   second_batch = next(iter(loader))
   print(loader.param_generator.random_state.rng.random())
   # Should produce different random numbers
   ```