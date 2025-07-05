# GMM Experiment Configuration System

## Experiment Presets Overview

The GMM Transformer framework uses a comprehensive experiment configuration system with predefined presets to simplify setup and ensure reproducibility. The experiment configuration integrates multiple subsystems into a unified, coherent experiment definition.

## Configuration Hierarchy

The experiment configuration system combines several levels of configuration components:

1. **ExperimentConfig**: Top-level configuration class that combines all other components
2. **Component Configs**: Specialized configurations for model, training, data, etc.
3. **Presets**: Named, predefined configurations in JSON files

The complete hierarchy includes:

- **ExperimentConfig**
  - **Metadata**: Experiment identification and tracking
  - **Paths**: File paths for outputs, checkpoints, etc.
  - **Device**: Hardware configuration (CPU/GPU/TPU)
  - **Model**: Transformer and prediction configuration
  - **Training**: Training loop, optimizer, and scheduler
  - **Data**: Data generation and loading parameters
  - **Validation**: Evaluation metrics and visualization
  - **Logging**: Console and file logging settings

## Configuration Classes

### ExperimentConfig

The `ExperimentConfig` class is the main container for all experiment settings and can be created using the `ExperimentRegistry`:

```python
from config.registry import ExperimentRegistry

# Create an experiment configuration
experiment_config = ExperimentRegistry.get_experiment_config(
    model_name="medium",
    training_preset="standard",
    data_preset="standard",
    validation_preset="standard",
    device="cuda",
    exp_name="my_experiment"
)
```

You can also use the preset system for a simplified approach:

```python
from config.presets import get_preset_config

# Create a complete preset-based configuration
experiment_config = get_preset_config(
    preset_name="standard",
    device="cuda",
    exp_name="my_experiment"
)
```

### Device Configuration

The device configuration controls hardware settings:

```python
device_config = {
    "device": "cuda",     # "cpu", "cuda", "mps", or "auto"
    "gpu_id": 0,          # GPU index for multi-GPU systems
    "mixed_precision": True,  # Whether to use mixed precision
    "seed": 42            # Random seed for reproducibility
}
```

## Available Experiment Presets

The GMM Transformer framework provides several experiment presets designed for different use cases:

| Preset Name | Description | Model | Training | Data | Validation | Use Case |
|-------------|-------------|-------|----------|------|------------|----------|
| `quick_test` | Fast experiment for debugging | tiny | quick | simple | minimal | Development debugging |
| `standard` | Default balanced configuration | medium | standard | standard | standard | General experiments |
| `high_performance` | Maximum performance | large | optimized | complex | comprehensive | Production models |

## How to Use Experiment Presets

### Command Line Usage

The simplest way to use presets is through the command line:

```bash
# Use a predefined experiment preset
python main.py --experiment-preset standard

# Mix and match component presets
python main.py \
    --model-preset medium \
    --training-preset standard \
    --data-preset complex \
    --validation-preset comprehensive
```

### Programmatic Usage

You can also use the preset system programmatically:

```python
from config.presets import get_preset_config, list_experiment_presets

# Get a complete experiment configuration
experiment_config = get_preset_config("standard")

# List available experiment presets
presets = list_experiment_presets()
print(f"Available presets: {presets}")

# Get details about a specific preset
from config.presets import get_preset_description
preset_details = get_preset_description("standard")
print(f"Standard preset components: {preset_details}")
```

## Modifying Experiment Configuration

### Overriding Specific Parameters

You can modify a preset by loading it and then changing individual parameters:

```python
from config.presets import get_preset_config

# Get the base preset
experiment_config = get_preset_config("standard")

# Override specific parameters
experiment_config.training.num_epochs = 75
experiment_config.training.learning_rate = 0.0002
experiment_config.model.transformer.dropout = 0.2
experiment_config.data.dim = 3
```

### Creating a Custom Preset

To create a new preset, add it to `config/json_defaults/experiment_simple.json`:

```json
{
  "custom_experiment": {
    "model": "medium",
    "training": "optimized",
    "data": "standard",
    "validation": "comprehensive"
  }
}
```

The system will automatically detect the new preset, making it available through the registry system.

## Component Preset Integration

Experiment presets integrate component presets. Each component has its own dedicated preset system:

| Component | JSON Location | Registry Module |
|-----------|---------------|----------------|
| Model | config/json_defaults/model/defaults.json | registry/model_configs.py |
| Training | config/json_defaults/training/defaults.json | registry/training_configs.py |
| Data | config/json_defaults/data/defaults.json | registry/data_configs.py |
| Validation | config/json_defaults/validation/defaults.json | registry/validation_configs.py |

## Experiment Execution

The experiment configuration is used to set up and run experiments:

```python
from training.experiment import ExperimentManager

# Create an experiment manager with the configuration
experiment_manager = ExperimentManager(experiment_config)

# Run the experiment
experiment_manager.run()
```

The experiment manager handles all aspects of the experiment:

1. Setting up directories
2. Creating data loaders
3. Building the model
4. Configuring training (optimizer, scheduler, loss)
5. Running the training loop
6. Performing validation
7. Logging metrics and creating visualizations
8. Saving checkpoints

## Best Practices for Experiment Configuration

1. **Start Simple**: Begin with the `quick_test` preset during initial development.

2. **Iterative Refinement**: Gradually move to more complex configurations as your model matures.

3. **Reproducibility**: Always set a fixed seed in the device configuration for reproducible results.

4. **Configuration Versioning**: Save the complete configuration JSON with your results to ensure reproducibility.

5. **Custom Presets**: Create custom presets for recurring experiment types in your research.

6. **Avoid Breaking Changes**: When adding new configuration options, ensure they have sensible defaults.

7. **Documentation**: Document your custom configurations and parameters.

8. **Configuration Inheritance**: Build on existing presets rather than starting from scratch.

## Advanced Configuration Usage

### Saving and Loading Configurations

```python
# Save your configuration to a file
import json
with open("my_experiment_config.json", "w") as f:
    json.dump(experiment_config.model_dump(), f, indent=2)

# Load configuration from a file
from config.experiment import ExperimentConfig
with open("my_experiment_config.json", "r") as f:
    config_dict = json.load(f)
    loaded_config = ExperimentConfig.model_validate(config_dict)
```

### Environment Configuration

For quick experimentation with different hardware, you can use:

```bash
# Set device before running
export DEVICE="cpu"  # or "cuda", "mps"
python main.py --experiment-preset standard
```

Or programmatically:

```python
import os
os.environ["DEVICE"] = "cuda"
```

The system will respect these environment variables when creating configurations.