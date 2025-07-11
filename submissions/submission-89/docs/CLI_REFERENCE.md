# Enhanced Experiment Runner CLI Reference

This document outlines the command-line interface for the GMM Transformer experiment runner, which allows for detailed configuration of experiments through command-line arguments.

## Basic Usage

```bash
python scripts/runners/simple_experiment_runner.py --model-preset small --training-preset standard
```

## Core Arguments

| Argument | Description |
|----------|-------------|
| `--model-preset` | Model size preset (tiny, small, medium, large) |
| `--training-preset` | Training configuration preset (quick, standard, optimized) |
| `--data-preset` | Data generation preset (simple, standard, complex) |
| `--validation-preset` | Validation configuration preset (minimal, standard, comprehensive) |
| `--experiment-name` | Custom name for the experiment |
| `--output-dir` | Directory for experiment outputs |
| `--dry-run` | Validate configuration without running experiment |
| `--list-presets` | Display available presets and exit |

## Universal Parameter Setting

The `--set` argument allows setting any parameter in the configuration using dot notation:

```bash
python scripts/runners/simple_experiment_runner.py --set model.transformer.num_layers=12
```

Multiple parameters can be set by using `--set` multiple times:

```bash
python scripts/runners/simple_experiment_runner.py \
  --set model.transformer.hidden_dim=512 \
  --set training.optimizer.learning_rate=0.0001
```

## Transformer Architecture Shortcuts

| Shortcut | Full Parameter | Description |
|----------|----------------|-------------|
| `--hidden-dim 512` | `model.transformer.hidden_dim` | Size of hidden dimensions |
| `--num-layers 6` | `model.transformer.num_layers` | Number of transformer layers |
| `--num-heads 8` | `model.transformer.num_heads` | Number of attention heads |
| `--dropout 0.1` | `model.transformer.dropout` | Dropout probability |
| `--activation gelu` | `model.transformer.activation` | Activation function (gelu, relu, silu) |
| `--ff-expansion 4` | `model.transformer.ff_expansion` | Feed-forward expansion factor |
| `--bias true` | `model.transformer.bias` | Whether to use bias in layers |
| `--flash-attn true` | `model.transformer.use_flash_attn` | Use flash attention when available |

## Layer Repetition Shortcuts

| Shortcut | Full Parameter | Description |
|----------|----------------|-------------|
| `--layer-repeat-mode cycle` | `model.transformer.layer_repeat_mode` | Repetition mode (none, cycle, layerwise, grouped) |
| `--repeat-factor 3` | `model.transformer.repeat_factor` | Number of repetitions for cycle/layerwise modes |
| `--layer-groups 2,2,4` | `model.transformer.layer_groups` | Layer group sizes for grouped mode (comma-separated) |
| `--group-repeat-factors 2,3,1` | `model.transformer.group_repeat_factors` | Repeat factors for grouped mode (comma-separated) |

## Training Configuration Shortcuts

| Shortcut | Full Parameter | Description |
|----------|----------------|-------------|
| `--batch-size 64` | `training.batch_size` | Batch size for training |
| `--epochs 100` | `training.num_epochs` | Number of training epochs |
| `--lr 0.001` | `training.optimizer.learning_rate` | Learning rate |
| `--optimizer adam` | `training.optimizer.optimizer_type` | Optimizer type (adam, sgd, adamw) |
| `--weight-decay 0.01` | `training.optimizer.weight_decay` | Weight decay for regularization |
| `--scheduler cosine` | `training.scheduler.scheduler_type` | Learning rate scheduler (cosine, step, linear) |
| `--warmup-epochs 10` | `training.scheduler.warmup_epochs` | Number of warmup epochs |
| `--grad-clip 1.0` | `training.gradient_clipping.clip_value` | Gradient clipping value |
| `--use-amp false` | `training.use_amp` | Use automatic mixed precision |

## Loss Function Shortcuts

| Shortcut | Full Parameter | Description |
|----------|----------------|-------------|
| `--loss-type mse` | `training.loss.loss_type` | Loss function type (mse, wasserstein, energy) |
| `--wasserstein-backend jax` | `training.loss.wasserstein_backend` | Wasserstein backend (jax, pot, scipy) |
| `--wasserstein-algorithm sinkhorn` | `training.loss.wasserstein_algorithm` | Wasserstein algorithm (exact, sinkhorn) |
| `--wasserstein-epsilon 0.01` | `training.loss.wasserstein_epsilon` | Epsilon parameter for Sinkhorn algorithm |
| `--wasserstein-max-iter 100` | `training.loss.wasserstein_max_iterations` | Maximum iterations for Wasserstein calculation |

**Note:** The default Wasserstein implementation uses the SciPy backend with the exact algorithm.

## Data Generation Shortcuts

| Shortcut | Full Parameter | Description |
|----------|----------------|-------------|
| `--num-samples 10000` | `data.num_samples` | Number of data samples |
| `--num-clusters 8` | `data.num_clusters` | Number of GMM clusters/components |
| `--data-dim 2` | `data.dim` | Dimension of data points |
| `--snr 10` | `data.snr` | Signal-to-noise ratio |
| `--min-sep 1.0` | `data.min_separation` | Minimum separation between clusters |
| `--max-sep 5.0` | `data.max_separation` | Maximum separation between clusters |
| `--repeats 3` | `data.repeats` | Number of data repeats |
| `--mi-factor 0.5` | `data.mi_factor` | Mutual information factor |

## Validation and Metrics Shortcuts

| Shortcut | Full Parameter | Description |
|----------|----------------|-------------|
| `--validate-every 5` | `validation.validate_every_n_epochs` | Validation frequency (epochs) |
| `--visualize-every 10` | `validation.visualize.visualize_every_n_epochs` | Visualization frequency (epochs) |
| `--use-kmeans true` | `validation.metrics.use_kmeans` | Use K-means clustering evaluation |
| `--use-wasserstein-metric true` | `validation.metrics.use_wasserstein` | Use Wasserstein distance as metric |
| `--track-parameters true` | `validation.track_parameters` | Track model parameters |

## System and Performance Shortcuts

| Shortcut | Full Parameter | Description |
|----------|----------------|-------------|
| `--device cuda` | `device.device_type` | Device to use (cuda, cpu) |
| `--mixed-precision` | `device.use_mixed_precision` | Enable mixed precision training |
| `--seed 42` | `training.seed` | Random seed for reproducibility |
| `--deterministic true` | `training.deterministic` | Use deterministic algorithms |
| `--workers 4` | `training.num_workers` | Number of data loader workers |

## Example Commands

### Basic Experiment with Presets
```bash
python scripts/runners/simple_experiment_runner.py \
  --model-preset medium \
  --training-preset standard \
  --data-preset complex \
  --experiment-name "medium_model_test"
```

### Configuring Layer Repetition
```bash
python scripts/runners/simple_experiment_runner.py \
  --model-preset medium \
  --layer-repeat-mode cycle \
  --repeat-factor 3 \
  --experiment-name "cycle_repetition_test"
```

### Grouped Layer Repetition
```bash
python scripts/runners/simple_experiment_runner.py \
  --model-preset small \
  --layer-repeat-mode grouped \
  --layer-groups 2,2,2 \
  --group-repeat-factors 2,3,2 \
  --experiment-name "grouped_repetition_test"
```

### Custom Transformer Configuration
```bash
python scripts/runners/simple_experiment_runner.py \
  --hidden-dim 384 \
  --num-layers 8 \
  --num-heads 6 \
  --dropout 0.2 \
  --activation silu \
  --experiment-name "custom_transformer"
```

### Advanced Wasserstein Loss Configuration
```bash
python scripts/runners/simple_experiment_runner.py \
  --model-preset small \
  --loss-type wasserstein \
  --wasserstein-backend pot \
  --wasserstein-algorithm sinkhorn \
  --wasserstein-epsilon 0.005 \
  --wasserstein-max-iter 200 \
  --experiment-name "wasserstein_pot_test"
```

### Full Custom Experiment
```bash
python scripts/runners/simple_experiment_runner.py \
  --model-preset small \
  --hidden-dim 256 \
  --num-layers 6 \
  --layer-repeat-mode layerwise \
  --repeat-factor 2 \
  --batch-size 128 \
  --lr 0.0005 \
  --epochs 50 \
  --loss-type mse \
  --num-samples 20000 \
  --num-clusters 12 \
  --device cuda \
  --mixed-precision \
  --experiment-name "full_custom_experiment"
```

### Using the Universal Parameter Setter
```bash
python scripts/runners/simple_experiment_runner.py \
  --model-preset small \
  --set model.transformer.layer_repeat_mode=layerwise \
  --set model.transformer.repeat_factor=3 \
  --set model.transformer.activation=silu \
  --set training.optimizer.learning_rate=0.0003 \
  --set training.scheduler.scheduler_type=cosine \
  --set validation.metrics.wasserstein_algorithm=exact \
  --experiment-name "parameter_setter_test"
```

## Notes on Parameter Handling

- Boolean parameters accept "true"/"false" (case-insensitive)
- List parameters (like layer_groups) can be provided as comma-separated values
- Parameter types are automatically converted based on their destination type
- If invalid parameters are provided, the script will display an error and exit
- The `--dry-run` flag can be used to validate configuration without running