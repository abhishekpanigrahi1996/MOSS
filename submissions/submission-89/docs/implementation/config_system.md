# Configuration System

The GMM Transformer framework uses a configuration system based on JSON files with a registry for accessing different presets. This document explains how the configuration system works and how to maintain it.

## Configuration Files

All configurations are stored as JSON files in the `config/json_defaults/` directory:

- `config/json_defaults/model/defaults.json`: Model architecture presets
- `config/json_defaults/training/defaults.json`: Training configuration presets
- `config/json_defaults/data/defaults.json`: Data generation presets
- `config/json_defaults/validation/defaults.json`: Validation presets
- `config/json_defaults/data/parameter_presets.json`: Parameter presets for GMM data generation
- `config/json_defaults/experiment_simple.json`: High-level experiment presets

## Configuration Registry

The configuration registry system is implemented in `config/registry/registry.py` and provides a type-safe way to access configuration presets:

- `ConfigRegistry`: A generic class that loads configurations from JSON files
- `ExperimentRegistry`: Provides static methods for accessing all configuration types

## How Configurations Are Loaded

1. When the system starts, the registry classes load configurations from their respective JSON files
2. If a JSON file is missing, the system will use built-in defaults with a warning message
3. All string presets for data generation (e.g., "few", "moderate") are resolved to their explicit dictionary form during validation

## Maintaining the Configuration System

### Adding New Presets

To add a new preset:

1. Edit the appropriate JSON file in the `config/json_defaults/` directory
2. Add a new entry with a unique name and the desired configuration parameters

### Modifying Existing Presets

To modify an existing preset:

1. Edit the appropriate JSON file in the `config/json_defaults/` directory
2. Update the parameters for the preset you want to change

### Adding New Parameter Presets

To add new parameter presets for data generation:

1. Edit `config/json_defaults/data/parameter_presets.json`
2. Add a new entry under the appropriate category (cluster_params, snr_db_params, etc.)

## Design Philosophy

The configuration system follows these principles:

1. **Explicit Configuration**: All configuration values are explicitly defined in JSON files, with fallback defaults in the code.
2. **Type Safety**: Configurations are validated using Pydantic models to ensure type safety.
3. **Preset Resolution**: String presets for data generation are automatically resolved to their explicit dictionary form.
4. **Resilience**: If a configuration file is missing, the system provides useful defaults rather than failing.

## Error Handling

If a JSON configuration file is missing or has an invalid format, the system will:

- Log a warning with details about the issue
- Fall back to built-in defaults where possible
- Raise clear errors when a requested preset name is not found in the loaded configurations

## Parameter Presets for GMM Data Generation

### Overview

The GMM Transformer framework uses string presets like "few", "moderate", etc. for data generation parameters. These presets are automatically resolved to their explicit dictionary definitions when the configuration is loaded.

### Parameter Preset Configuration

The presets are defined in `config/json_defaults/data/parameter_presets.json`. This file contains explicit definitions for:

1. Cluster parameters presets (`cluster_params`) - Controls the number of clusters in generated GMMs
2. Signal-to-noise ratio presets (`snr_db_params`) - Controls the noise level in generated data
3. Sample count distribution presets (`sample_count_distribution`) - Controls how many points are sampled from each GMM

### Available Presets

#### Cluster Parameters Presets
- `few`: 2-4 clusters (λ=2.5, truncated Poisson)
- `moderate`: 3-7 clusters (λ=5.0, truncated Poisson)
- `many`: 5-15 clusters (λ=8.0, truncated Poisson)

#### Signal-to-Noise Ratio Presets
- `easy`: 12-17 dB (mean: 14.0, std: 1.5, truncated normal)
- `moderate`: 7-12 dB (mean: 9.0, std: 1.5, truncated normal)
- `difficult`: 3-7 dB (mean: 5.0, std: 1.5, truncated normal)

#### Sample Count Distribution Presets
- `small`: 100-500 samples, log-normally distributed (mean: 5.7, sigma: 0.3)
- `medium`: 500-1500 samples, log-normally distributed (mean: 6.9, sigma: 0.3)
- `large`: 1500-5000 samples, log-normally distributed (mean: 8.0, sigma: 0.3)
- `fixed_moderate`: Exactly 1000 samples per GMM
- `fixed_large`: Exactly 3000 samples per GMM
- `variable_moderate`: 800-1500 samples, log-normally distributed (mean: 6.9, sigma: 0.3)
- `variable_large`: 2000-5000 samples, log-normally distributed (mean: 8.0, sigma: 0.3)

### Available Distribution Types

The following distribution types are supported for different parameter types:

#### Distribution Types by Parameter

| Distribution Type | cluster_params | snr_db_params | sample_count_distribution |
|-------------------|----------------|---------------|---------------------------|
| fixed | ✓ | ✓ | ✓ |
| poisson / truncated_poisson | ✓ | ✗ | ✗ |
| uniform | ✓ | ✓ | ✓ |
| normal | ✗ | ✓ | ✗ |
| truncated_normal | ✗ | ✓ | ✗ |
| truncated_lognormal | ✗ | ✗ | ✓ |
| choice | ✓ | ✓ | ✓ |

#### Distribution Type Specifications

1. **Fixed**: A single fixed value
   ```json
   {
     "type": "fixed",
     "value": 5
   }
   ```

2. **Truncated Poisson**: Poisson distribution truncated to a range
   ```json
   {
     "type": "truncated_poisson",
     "lam": 5.0,
     "min": 3,
     "max": 7
   }
   ```

3. **Truncated Normal**: Normal distribution truncated to a range
   ```json
   {
     "type": "truncated_normal",
     "mean": 12.5,
     "std": 1.5,
     "min": 10.0,
     "max": 15.0
   }
   ```

4. **Truncated Log-Normal**: Log-normal distribution truncated to a range
   ```json
   {
     "type": "truncated_lognormal",
     "mean": 6.9,
     "sigma": 0.3,
     "min": 500,
     "max": 1500
   }
   ```

5. **Choice**: Discrete choice from options with probabilities
   ```json
   {
     "type": "choice",
     "options": [500, 1000, 2000],
     "probs": [0.3, 0.5, 0.2]
   }
   ```

6. **Uniform**: Uniform distribution over a range
   ```json
   {
     "type": "uniform",
     "min": 3,
     "max": 8
   }
   ```
   
   For integer parameters (clusters, sample counts), values will be in range [min, max] inclusive.
   For continuous parameters (SNR), values will be in range [min, max].

## Experiment Presets Overview

The GMM Transformer framework uses a comprehensive experiment configuration system with predefined presets to simplify setup and ensure reproducibility. The experiment configuration integrates multiple subsystems into a unified, coherent experiment definition.

### Configuration Hierarchy

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

### Available Experiment Presets

The GMM Transformer framework provides several experiment presets designed for different use cases:

| Preset Name | Description | Model | Training | Data | Validation | Use Case |
|-------------|-------------|-------|----------|------|------------|----------|
| `quick_test` | Fast experiment for debugging | tiny | quick | simple | minimal | Development debugging |
| `standard` | Default balanced configuration | medium | standard | standard | standard | General experiments |
| `high_performance` | Maximum performance | large | optimized | complex | comprehensive | Production models |

These presets combine various component presets to create complete experiment configurations.

### How to Use Experiment Presets

You can use presets either programmatically or via command line:

```python
from config.presets import get_preset_config

# Create an experiment configuration using a preset
experiment_config = get_preset_config(
    preset_name="standard",
    device="cuda",
    exp_name="my_experiment"
)

# Override specific parameters
experiment_config.training.num_epochs = 75
experiment_config.training.learning_rate = 0.0002
```

Command line usage:
```bash
python main.py --experiment-preset standard --device cuda
```

### Component Preset Integration

Experiment presets integrate component presets. Each component has its own dedicated preset system:

| Component | JSON Location | Registry Module |
|-----------|---------------|----------------|
| Model | config/json_defaults/model/defaults.json | registry/model_configs.py |
| Training | config/json_defaults/training/defaults.json | registry/training_configs.py |
| Data | config/json_defaults/data/defaults.json | registry/data_configs.py |
| Validation | config/json_defaults/validation/defaults.json | registry/validation_configs.py |

### Creating Custom Presets

To create a new experiment preset, you can add it to `config/json_defaults/experiment_simple.json`:

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

The system will automatically detect the new preset, making it available through the preset system.

## Best Practices for Configuration

1. **Start Simple**: Begin with the `quick_test` preset during initial development.
2. **Iterative Refinement**: Gradually move to more complex configurations as your model matures.
3. **Reproducibility**: Always set a fixed seed in the device configuration for reproducible results.
4. **Configuration Versioning**: Save the complete configuration JSON with your results to ensure reproducibility.
5. **Custom Presets**: Create custom presets for recurring experiment types in your research.
6. **Avoid Breaking Changes**: When adding new configuration options, ensure they have sensible defaults.