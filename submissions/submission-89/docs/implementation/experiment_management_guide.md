# GMM Experiment Management Guide

This guide provides a comprehensive overview of how to manage experiments in the GMM-v2 framework, including starting, resuming, loading, and analyzing models.

## Table of Contents

1. [Setting Up a New Experiment](#setting-up-a-new-experiment)
2. [Training a Model](#training-a-model)
3. [Resuming Training](#resuming-training)
4. [Loading Trained Models](#loading-trained-models)
5. [Evaluating Models](#evaluating-models)
6. [Visualizing Model Outputs](#visualizing-model-outputs)
7. [Saving and Exporting Models](#saving-and-exporting-models)
8. [Common Workflows](#common-workflows)

## Setting Up a New Experiment

### Using Configuration Presets

The GMM-v2 framework provides a configuration system with presets for different experiment components:

```python
from config.base import ExperimentConfig
from config.presets import get_preset
from training.experiment import ExperimentManager

# Configure experiment using presets
config = ExperimentConfig(
    experiment=get_preset("experiment", "simple"),
    model=get_preset("model", "gmm_transformer_small"),
    data=get_preset("data", "gmm_1d_simple"),
    training=get_preset("training", "basic"),
    validation=get_preset("validation", "basic")
)

# Create experiment manager
experiment = ExperimentManager(config)

# Set up components (model, data loaders, trainer)
experiment.setup()
```

### Using Custom Configuration

For more control, you can create a custom configuration by modifying preset values:

```python
# Start with a preset and customize
config = ExperimentConfig(
    experiment=get_preset("experiment", "simple"),
    model=get_preset("model", "gmm_transformer_small"),
    data=get_preset("data", "gmm_1d_simple"),
    training=get_preset("training", "basic"),
    validation=get_preset("validation", "basic")
)

# Customize model configuration
config.model.transformer.hidden_dim = 128
config.model.transformer.num_layers = 4

# Customize training parameters
config.training.num_epochs = 100
config.training.batch_size = 32

# Create experiment manager
experiment = ExperimentManager(config)
experiment.setup()
```

### Loading Configuration from File

You can also load a configuration from a JSON file:

```python
from pathlib import Path
from training.experiment import ExperimentManager

# Load from configuration file
config_path = Path("path/to/config.json")
experiment = ExperimentManager(config=config_path)
experiment.setup()
```

## Training a Model

Once your experiment is set up, you can start training:

```python
# Run training for the specified number of epochs
experiment.run(num_epochs=50)
```

### Training with Explicit Output Directory

```python
# Specify output directory
experiment.run(num_epochs=50, experiment_dir="./runs/my_experiment")
```

### Monitoring Training Progress

During training, the framework will:
- Log progress to the console
- Save checkpoints at regular intervals
- Save the latest model after each epoch as `latest_model.pt`
- Save the best model based on validation loss as `best_model.pt`
- Log metrics to TensorBoard if enabled

You can check training progress in TensorBoard:

```bash
tensorboard --logdir=./runs/my_experiment/tensorboard
```

## Resuming Training

### Resume from Latest Checkpoint

```python
# Load experiment from directory
experiment = ExperimentManager.load("./runs/my_experiment")

# Resume training for additional epochs
experiment.resume_training(num_epochs=20)
```

### Resume with the Enhanced API

The enhanced model management API provides a simpler way to resume training:

```python
# Load experiment with latest model
experiment = ExperimentManager.load_latest_model("./runs/my_experiment")

# Resume training for additional epochs
experiment.resume_training(num_epochs=20)
```

### How State Restoration Works

When resuming training, the experiment manager properly handles state restoration:

1. The model state is loaded from the checkpoint
2. The data loader state is restored using the enhanced `resume=True` API
3. Training continues from where it left off

Here's how the data loader state management works internally:

```python
# This is typically handled by ExperimentManager
data_state_path = "./runs/my_experiment/data_state.json"

# Data loader with resume=True automatically restores state
data_loader = GMMDataLoader(
    config_dict=config_dict,
    batch_size=batch_size,
    num_samples=samples_per_epoch,
    state_path=data_state_path,
    resume=True  # This signals intent to load and restore state
)

# Training then continues where it left off
trainer.train(data_loader)
```

The `resume=True` parameter ensures that:

1. The data loader loads the state file at initialization
2. The random number generator state is properly restored
3. The batch and epoch counters are restored
4. The next iteration continues from where the previous training session left off

## Loading Trained Models

### Load Best Model

```python
# Load experiment with best model
best_experiment = ExperimentManager.load_best_model("./runs/my_experiment")
```

### Load Latest Model

```python
# Load experiment with latest model
latest_experiment = ExperimentManager.load_latest_model("./runs/my_experiment")
```

### Load Specific Checkpoint

```python
# Load experiment
experiment = ExperimentManager.load("./runs/my_experiment")

# Set up experiment
experiment.setup()

# Load specific checkpoint
checkpoint_path = "./runs/my_experiment/checkpoints/checkpoint_epoch_10.pt"
experiment.trainer._load_checkpoint(checkpoint_path)
```

### Using ModelManager for Direct Checkpoint Loading

```python
from utils.model_management import ModelManager

# Load model directly from checkpoint
model_manager = ModelManager.from_checkpoint("./runs/my_experiment/checkpoints/best_model.pt")

# Or load from experiment directory (best model by default)
model_manager = ModelManager.from_experiment_dir("./runs/my_experiment", load_best=True)
```

## Evaluating Models

### Evaluate Using ExperimentManager

```python
# Create a data loader for evaluation
from data.loaders.factory import create_data_loader

data_loader = create_data_loader(
    config_preset=experiment.config.data,
    batch_size=32,
    shuffle=False,
    num_samples=1000
)

# Evaluate the model
metrics = experiment.evaluate(data_loader=data_loader)
print(metrics)
```

### Evaluate Using ModelManager

```python
from utils.model_management import ModelManager

# Load model
model_manager = ModelManager.from_checkpoint("./runs/my_experiment/checkpoints/best_model.pt")

# Create data loader
data_loader = create_data_loader(...)

# Evaluate
metrics = model_manager.evaluate(data_loader=data_loader, metrics=['mse', 'wasserstein'])
print(metrics)
```

## Visualizing Model Outputs

```python
# Load model
model_manager = ModelManager.from_checkpoint("./runs/my_experiment/checkpoints/best_model.pt")

# Create data loader with a few samples
data_loader = create_data_loader(
    config_preset=experiment.config.data,
    batch_size=4,
    shuffle=True,
    num_samples=4
)

# Generate visualizations
figures = model_manager.visualize(
    data_loader=data_loader, 
    num_samples=3,
    output_dir="./visualizations"
)
```

## Saving and Exporting Models

### Saving a Model Checkpoint Explicitly

```python
# Save a checkpoint explicitly
checkpoint_path = experiment.save_checkpoint(is_best=False)
print(f"Checkpoint saved to: {checkpoint_path}")

# Save as best model
best_path = experiment.save_checkpoint(is_best=True)
print(f"Best model saved to: {best_path}")
```

### Exporting a Model

```python
# Export model weights only
export_path = "./exported_models/my_model.pt"
experiment.export_model(export_path)
print(f"Model exported to: {export_path}")

# Or using ModelManager
model_manager = experiment.get_model_manager()
model_manager.save("./exported_models/my_model_weights.pt")
```

## Common Workflows

### Complete Training Workflow

```python
from config.base import ExperimentConfig
from config.presets import get_preset
from training.experiment import ExperimentManager

# 1. Create configuration
config = ExperimentConfig(
    experiment=get_preset("experiment", "simple"),
    model=get_preset("model", "gmm_transformer_small"),
    data=get_preset("data", "gmm_1d_simple"),
    training=get_preset("training", "basic"),
    validation=get_preset("validation", "basic")
)

# 2. Customize configuration as needed
config.training.num_epochs = 50
config.training.batch_size = 32

# 3. Create and set up experiment
experiment = ExperimentManager(config)
experiment.setup()

# 4. Run training
experiment.run(num_epochs=50, experiment_dir="./runs/my_experiment")

# 5. Evaluate best model
from data.loaders.factory import create_data_loader

test_loader = create_data_loader(
    config_preset=config.data,
    batch_size=32,
    shuffle=False,
    num_samples=1000
)

best_experiment = ExperimentManager.load_best_model("./runs/my_experiment")
metrics = best_experiment.evaluate(data_loader=test_loader)
print(f"Evaluation metrics: {metrics}")

# 6. Export final model
best_experiment.export_model("./exported_models/final_model.pt")
```

### Incremental Training Workflow

```python
# 1. Initial training phase
experiment = ExperimentManager(config)
experiment.setup()
experiment.run(num_epochs=20, experiment_dir="./runs/my_experiment")

# 2. Resume training with more epochs
experiment = ExperimentManager.load_latest_model("./runs/my_experiment")
experiment.resume_training(num_epochs=20)

# 3. Final training phase with lower learning rate
experiment = ExperimentManager.load_latest_model("./runs/my_experiment")
experiment.config.training.optimizer.learning_rate = 1e-5
experiment.resume_training(num_epochs=10)

# 4. Evaluate and export final model
experiment = ExperimentManager.load_best_model("./runs/my_experiment")
metrics = experiment.evaluate(data_loader=test_loader)
experiment.export_model("./exported_models/final_model.pt")
```

### Analysis Workflow

```python
from utils.model_management import ModelManager
import matplotlib.pyplot as plt

# 1. Load multiple models
model1 = ModelManager.from_checkpoint("./runs/experiment1/checkpoints/best_model.pt")
model2 = ModelManager.from_checkpoint("./runs/experiment2/checkpoints/best_model.pt")
model3 = ModelManager.from_checkpoint("./runs/experiment3/checkpoints/best_model.pt")

# 2. Create evaluation data
test_loader = create_data_loader(...)

# 3. Evaluate all models
metrics1 = model1.evaluate(test_loader)
metrics2 = model2.evaluate(test_loader)
metrics3 = model3.evaluate(test_loader)

# 4. Compare metrics
models = ['Model 1', 'Model 2', 'Model 3']
mse_values = [metrics1['mse'], metrics2['mse'], metrics3['mse']]
wasserstein_values = [metrics1['wasserstein'], metrics2['wasserstein'], metrics3['wasserstein']]

# 5. Visualize comparison
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(models, mse_values)
plt.title('MSE Comparison')
plt.ylabel('MSE')

plt.subplot(1, 2, 2)
plt.bar(models, wasserstein_values)
plt.title('Wasserstein Distance Comparison')
plt.ylabel('Wasserstein Distance')

plt.tight_layout()
plt.savefig("./analysis/model_comparison.png")
```

This guide provides a comprehensive overview of experiment management in the GMM-v2 framework. By following these patterns, you can effectively manage the full lifecycle of your models - from training and evaluation to export and analysis.