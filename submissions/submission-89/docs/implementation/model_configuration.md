# GMM Model Configuration System

## Model Presets Overview

The GMM Transformer framework uses a configurable model system with predefined presets to simplify experiment setup. All configuration values are defined in JSON files. The model configuration is organized hierarchically through multiple configuration classes.

The framework includes two advanced features for model architecture customization:

1. **Layer Repetition**: Allows parameter-efficient deep networks by reusing layer definitions in different patterns
2. **Flow Speed Control**: Provides dynamic control over information flow through the transformer network

These features can be used independently or combined to create efficient model architectures with controlled information dynamics.

## Configuration Hierarchy

1. **TransformerConfig**: Core transformer architecture parameters (layers, dimensions, heads, etc.)
2. **ClusterPredictionConfig**: Contains a TransformerConfig plus cluster prediction options
3. **Model Config/Preset**: Named configurations in `config/json_defaults/model/defaults.json` (tiny, small, medium, large, etc.)

## Model Configuration Parameters

The transformer architecture has these configurable parameters:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `input_dim` | 2 | Input feature dimension (point coordinates) |
| `hidden_dim` | 256 | Hidden dimension size of the model |
| `num_layers` | 6 | Number of unique transformer layer definitions |
| `num_heads` | 8 | Number of attention heads |
| `dropout` | 0.1 | Dropout probability |
| `ff_expansion` | 4 | Feed-forward expansion factor |
| `activation` | "gelu" | Activation function (gelu, relu, silu, mish) |
| `bias` | true | Whether to use bias in linear layers |
| `norm_eps` | 1e-5 | Epsilon for layer normalization |
| `use_flash_attn` | true | Whether to use flash attention when available |
| `layer_repeat_mode` | "none" | Mode for layer repetition (none, cycle, layerwise, grouped) |
| `repeat_factor` | 1 | Number of times to repeat layers in "cycle" or "layerwise" mode |
| `layer_groups` | None | List of layer group sizes for "grouped" mode |
| `group_repeat_factors` | None | List of repeat factors for each group in "grouped" mode |
| `use_flow_predictor` | false | Whether to use flow speed prediction |
| `flow_predictor_type` | "dummy" | Type of flow predictor ("dummy", "mlp", etc.) |
| `flow_predictor_per_layer` | false | Whether to use per-layer flow speeds |

For the cluster prediction model, additional configuration options include:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `prediction_type` | "centers" | Type of prediction (centers or assignments) |
| `num_clusters` | None | Number of clusters for assignment prediction (required if prediction_type is "assignments") |

## Layer Repetition Modes

The GMMTransformer includes different layer repetition modes that allow for parameter-efficient deep networks by reusing layer definitions:

### None Mode

Standard processing where each layer is unique and used once:

```
Layer position:  1  2  3  4  5  6
Layer pattern:  l₁ l₂ l₃ l₄ l₅ l₆
```

### Cycle Mode

In cycle mode, a fixed sequence of unique layers is repeated multiple times:

```
# With 2 unique layers and repeat_factor 3 (6 effective layers)
Layer position:  1  2  3  4  5  6
Layer pattern:  l₁ l₂ l₁ l₂ l₁ l₂
                └─ 1 ─┘└─ 2 ─┘└─ 3 ─┘
```

### Layerwise Mode

In layerwise mode, each unique layer is repeated consecutively before moving to the next layer:

```
# With 2 unique layers and repeat_factor 3 (6 effective layers)
Layer position:  1  2  3  4  5  6
Layer pattern:  l₁ l₁ l₁ l₂ l₂ l₂
                └── l₁×3 ──┘└── l₂×3 ──┘
```

### Grouped Mode

In grouped mode, different groups of unique layers are defined, and each group is repeated according to its own repeat factor:

```
# With num_layers 2, layer_groups [1, 1], group_repeat_factors [3, 3]
Layer position:  1  2  3  4  5  6
Layer pattern:  l₁ l₁ l₁ l₂ l₂ l₂
                └─── Group 1 ───┘└─── Group 2 ───┘
```

## Flow Speed Control

The GMMTransformer also supports dynamic flow speed control, which adjusts the magnitude of updates in each transformer block:

```python
# Standard residual connection
x = x + self.attention(self.norm1(x))

# With flow speed control
x = x + flow_speed * self.attention(self.norm1(x))
```

Flow speed can be:
- Global: A single value used for all layers
- Per-layer: Different flow speeds for each layer

## Available Model Presets

| Preset Name | Description | Hidden Dim | Layers | Effective Layers | Parameter Sharing |
|-------------|-------------|------------|--------|------------------|------------------|
| `tiny` | Minimal model for quick testing | 32 | 4 | 4 | None |
| `small` | Compact model for development | 64 | 8 | 8 | None |
| `medium` | Balanced model for standard use | 128 | 16 | 16 | None |
| `large` | High capacity model | 256 | 48 | 48 | None |
| `medium_cycle` | Cycle repetition model | 128 | 2 | 16 | 8× Parameter Reuse |
| `medium_layerwise` | Layerwise repetition model | 128 | 2 | 16 | 8× Parameter Reuse |
| `medium_one_layer` | Single layer repeated model | 128 | 1 | 16 | 16× Parameter Reuse |
| `medium_flow` | Flow speed control model | 128 | 16 | 16 | None |
| `medium_layerwise_flow` | Layerwise model with flow control | 128 | 2 | 16 | 8× Parameter Reuse |

All presets include these common settings:
- `input_dim`: 2
- `dropout`: 0.0
- `activation`: "gelu"
- `use_flash_attn`: true
- `prediction_type`: "centers"

## How to Use Model Presets

### Programmatic Usage

Presets can be accessed programmatically:

```python
from config.registry import get_model_config

# Get a model configuration
model_config = get_model_config("medium")

# Get an efficient model with layer repetition
repeat_config = get_model_config("medium_layerwise")

# List available presets
from config.registry import list_model_presets
presets = list_model_presets()
print(f"Available model presets: {presets}")
```

### Command Line Usage

Models can be specified through command line arguments:

```bash
python main.py --model-preset medium

# Use parameter-efficient model
python main.py --model-preset medium_layerwise
```

## Modifying Model Configurations

### Overriding Preset Parameters

You can override specific parameters while keeping others from a preset:

```python
from config.registry import get_model_config

# Get the base preset
model_config = get_model_config("medium_cycle")

# Override specific parameters
model_config.transformer.repeat_factor = 4
model_config.transformer.layer_repeat_mode = "layerwise"

# Use the modified configuration
# ...
```

### Creating a Custom Model Configuration

For complete custom configurations:

```python
from config.model import ClusterPredictionConfig, TransformerConfig

# Create a custom transformer config
transformer_config = TransformerConfig(
    input_dim=2,
    hidden_dim=96,
    num_layers=3,
    num_heads=3,
    dropout=0.1,
    activation="gelu",
    layer_repeat_mode="cycle",
    repeat_factor=4
)

# Use it in a cluster prediction config
model_config = ClusterPredictionConfig(
    transformer=transformer_config,
    prediction_type="centers"
)
```

### Adding a New Preset

To add a new preset, you can edit the `config/json_defaults/model/defaults.json` file:

```json
{
  "custom_model": {
    "description": "Custom model configuration with specialized parameters",
    "transformer": {
      "input_dim": 2,
      "hidden_dim": 96,
      "num_layers": 3,
      "num_heads": 3,
      "dropout": 0.1,
      "activation": "gelu",
      "use_flash_attn": true,
      "layer_repeat_mode": "cycle",
      "repeat_factor": 4
    },
    "prediction_type": "centers"
  }
}
```

The system will automatically detect and make it available through the registry.

## Benefits of Layer Repetition

1. **Parameter Efficiency**: Use fewer parameters while maintaining model depth
2. **Memory Efficiency**: Smaller model size with similar effective depth
3. **Regularization**: Parameter sharing acts as a form of regularization
4. **Controlled Growth**: Experiment with different depths without scaling parameters linearly
5. **Specialization**: Different layers or groups can learn specialized functionality

## Benefits of Flow Speed Control

1. **Adaptive Processing**: Flow speed adapts to the specific characteristics of each input batch
2. **Control Over Dynamics**: Provides explicit control over information flow through the network
3. **Improved Stability**: Can help stabilize training in deeper networks
4. **Layer-Specific Adjustments**: Per-layer flow control can optimize processing at different depths