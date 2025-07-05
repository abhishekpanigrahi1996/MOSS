# GMM-v2 Training Resumption Documentation

This document provides a comprehensive guide to the training resumption functionality in the GMM-v2 framework, explaining how it works, what it supports, and how to use it effectively.

## 1. Overview

Training resumption allows experiments to be paused and continued later, which is essential for:
- Long-running experiments that may exceed resource allocation time
- Recovery from unexpected interruptions (system crashes, power outages)
- Extending training after evaluating intermediate results
- Changing training parameters without starting from scratch

The GMM-v2 framework provides a robust system for experiment resumption through checkpoints and state management.

## 2. Core Components

The training resumption system consists of the following core components:

### 2.1 State Tracking Components

1. **Model State**: Neural network weights, biases, and internal parameters
2. **Optimizer State**: Learning rate, momentum, and adaptive parameters
3. **Scheduler State**: Learning rate schedule state and step count
4. **Data Loader State**: Random number generator state, batch position, and sequence information
5. **Training Metrics**: Loss history and validation metrics
6. **Batch and Epoch Counters**: Current position in the training process

### 2.2 Checkpoint Management

Checkpoints save the entire training state at specific points:
- Regular intervals (every N epochs)
- Best performing model (based on validation loss)
- Final model (at the end of training)
- Latest model (most recent state)

The framework ensures consistency between all state components through atomic operations and metadata tracking.

## 3. Resumption Process

### 3.1 Standard Resumption

The standard resumption process:

1. Load experimental configuration from the experiment directory
2. Find the latest checkpoint or a specified checkpoint
3. Load model weights and optimizer state
4. Restore data loader state to ensure consistent data sequences
5. Restore scheduler state to maintain the learning rate schedule
6. Continue training from the exact point where it was interrupted

```python
# Example of standard resumption
experiment = ExperimentManager.load("path/to/experiment")
experiment.resume_training()
```

### 3.2 Parameter Modification During Resumption

The framework supports changing certain parameters during resumption:

```python
# Resuming with modified parameters
experiment.resume_training(
    num_epochs=10,                  # Train for 10 additional epochs
    batch_size=64,                  # Change batch size from original
    train_samples=50000,            # Change training sample count
    reconfigure_scheduler=True      # Reconfigure learning rate schedule
)
```

#### 3.2.1 Changing Number of Epochs

- **Default behavior**: Continue for the remaining epochs in the original plan
- **Explicit epochs**: Train for the specified number of additional epochs
- **Reconfiguring scheduler**: Option to adjust learning rate schedule for the new timeline

#### 3.2.2 Changing Batch Size

When changing batch size during resumption:
- Data sequence integrity is maintained (same data points, different batching)
- Learning rate can optionally be scaled proportionally to batch size
- Scheduler is adjusted to account for the different number of steps per epoch
- Progress tracking is updated for the new batch size

#### 3.2.3 Changing Training Samples

When changing the number of training samples:
- Dataset coverage may change (using more or fewer samples)
- Epoch length changes accordingly
- Scheduler is adjusted to account for the different total steps
- Progress tracking is updated for the new epoch length

## 4. Data Loader State Management

### 4.1 Fixed vs. Dynamic Data Generation

The framework supports two data generation modes:

1. **Fixed Data Mode**:
   - Dataset is generated once at the beginning
   - Same exact data points are used in each epoch
   - State includes full dataset indexing information

2. **Dynamic Generation Mode**:
   - Data is generated on-the-fly during training
   - State includes random generator state to ensure reproducibility
   - Sequence is maintained despite dynamic generation

### 4.2 Validation Data Consistency

To ensure validation metrics are comparable across resumptions:
- Validation uses fixed data mode by default
- Validation state is saved separately from training state
- Same validation data is used before and after resumption
- Option to force new validation data if needed

### 4.3 Random State Management

The framework uses a sophisticated random state management system:
- Separate random generators for different components
- State includes full generator information
- Reproducible even with parameter changes
- Consistent across different system environments

## 5. Supported Functionality

The training resumption system supports:

### 5.1 Basic Features

- Resuming from the latest checkpoint
- Resuming from the best checkpoint
- Resuming from a specific checkpoint
- Loading a model for inference without resuming training

### 5.2 Advanced Features

- Changing batch size during resumption
- Changing training sample count
- Extending training beyond original epoch count
- Reconfiguring learning rate schedules
- Maintaining validation consistency

### 5.3 Scheduler Handling

The system correctly handles different scheduler types during resumption:
- Linear schedulers
- Cosine annealing schedulers
- Step-based schedulers
- Warmup + decay schedulers
- Custom scheduler configurations

## 6. Usage Examples

### 6.1 Basic Resumption

```python
# Load the experiment
experiment = ExperimentManager.load("path/to/experiment_dir")

# Resume training with default parameters
experiment.resume_training()
```

### 6.2 Resumption with Parameter Changes

```python
# Load the experiment
experiment = ExperimentManager.load("path/to/experiment_dir")

# Resume with modified parameters
experiment.resume_training(
    num_epochs=10,              # Train for 10 more epochs
    batch_size=64,              # Use batch size of 64
    reconfigure_scheduler=True  # Adjust scheduler for new parameters
)
```

### 6.3 Loading the Best Model

```python
# Load the experiment with the best model
experiment = ExperimentManager.load_best_model("path/to/experiment_dir")

# Use the model for inference
model_manager = experiment.get_model_manager()
predictions = model_manager.predict(test_data)
```

### 6.4 Extending Training

```python
# Load the experiment
experiment = ExperimentManager.load("path/to/experiment_dir")

# Train for 20 additional epochs, regardless of original plan
experiment.resume_training(num_epochs=20)
```

## 7. Implementation Details

### 7.1 Checkpoint File Structure

Checkpoints contain the following information:
- `model_state_dict`: Model weights and parameters
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: Scheduler state (if applicable)
- `epoch`: Current epoch number
- `loss`: Best validation loss
- `history`: Training and validation metrics history
- `config`: Experiment configuration
- `batch_size`: Batch size used for training
- `train_samples`: Number of training samples

### 7.2 Data State Files

Data loader state files contain:
- Random generator state
- Current position information
- Data sequence metadata
- Sample indices for fixed datasets
- Control parameters for GMM data generation

### 7.3 Metadata Tracking

To ensure consistency, metadata files track the relationship between:
- Checkpoint files
- Training data state files
- Validation data state files
- Configuration changes

### 7.4 Atomic Operations

The framework implements atomic operations to prevent state inconsistency:
- Checkpoint and state saving is coordinated
- Metadata updates happen in the same transaction
- Verification of state consistency during loading
- Fallback mechanisms for partial state recovery

## 8. Best Practices

### 8.1 When to Use Different Resumption Types

- **Standard Resumption**: For continuing interrupted experiments
- **Parameter Modification**: When experiment tuning is needed
- **Loading Best Model**: For inference or further fine-tuning
- **Mid-Epoch Resumption**: Automatically handled in all cases

### 8.2 Parameter Change Guidelines

When changing parameters during resumption:
- **Batch Size**: Changes primarily affect memory usage and optimization dynamics
- **Training Samples**: Changes affect total training data seen
- **Epochs**: Changes affect total training duration
- **Scheduler Reconfiguration**: Recommended when changing epochs or batch size

### 8.3 Validation Considerations

To maintain proper validation metrics:
- Always use the same validation data before and after resumption
- Don't change validation parameters during resumption
- Use fixed data mode for validation
- Save and restore validation states consistently

## 9. Troubleshooting

### 9.1 Common Issues

1. **Inconsistent Results After Resumption**:
   - Check if random state is properly restored
   - Verify data loader state is being used
   - Ensure validation data is consistent

2. **Failed Resumption**:
   - Check for missing checkpoint files
   - Verify data state files exist
   - Look for inconsistent configuration

3. **Learning Rate Issues**:
   - Check if scheduler reconfiguration is needed
   - Verify optimizer state is properly loaded
   - Consider explicit learning rate setting

### 9.2 Debugging Tools

The framework provides several tools for debugging resumption issues:
- Extensive logging during resumption process
- State validation functions
- Metadata inspection utilities
- Comparative training analysis

## 10. Limitations and Future Work

### 10.1 Current Limitations

- Learning rate schedules may need adjustment when significantly changing parameters
- Some highly customized data loaders may not fully support state saving
- Very large datasets may have increased overhead for state management

### 10.2 Future Enhancements

Planned enhancements to the resumption system:
- Distributed training resumption support
- Automatic parameter tuning during resumption
- More sophisticated scheduler reconfiguration options
- Enhanced data state compression for large datasets