#!/usr/bin/env python
"""
Enhanced GMM Transformer Experiment Runner

This script provides a flexible command-line interface for running GMM Transformer
experiments with extensive parameter customization options. It supports both preset-based
configuration and fine-grained control over model, training, and data parameters.

The interface supports two ways to configure parameters:
1. Universal setting with --set param.path=value notation
2. Shortcut parameters for commonly used settings

Examples:
    # Run experiment with presets
    python scripts/runners/enhanced_experiment_runner.py \
        --model-preset medium --training-preset standard

    # Configure layer repetition
    python scripts/runners/enhanced_experiment_runner.py \
        --layer-repeat-mode cycle --repeat-factor 3

    # Set arbitrary parameters with universal setter
    python scripts/runners/enhanced_experiment_runner.py \
        --set model.transformer.hidden_dim=384 \
        --set training.optimizer.learning_rate=0.0003
        
    # Resume an experiment by name
    python scripts/runners/enhanced_experiment_runner.py \
        --resume medium_standard_cycle \
        --additional-epochs 10
        
    # List available experiments
    python scripts/runners/enhanced_experiment_runner.py --resume
    
    # Resume with scheduler reconfiguration and device change
    python scripts/runners/enhanced_experiment_runner.py \
        --resume medium_standard_cycle \
        --additional-epochs 10 \
        --reconfigure-scheduler \
        --device cuda

For a complete list of options, see the CLI_REFERENCE.md document.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

# Now import
from config import ExperimentConfig
from config.registry import (
    list_model_presets,
    list_training_presets,
    list_data_presets, 
    list_validation_presets,
    get_model_config,
    get_training_config,
    get_data_config,
    get_validation_config,
    get_model_description,
    get_training_description,
    get_data_description,
    get_validation_description
)
from training.experiment import ExperimentManager


def setup_logging(log_level="INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_value(value_str: str) -> Any:
    """
    Parse a string value into the appropriate Python type.
    
    Args:
        value_str: String representation of the value
        
    Returns:
        Parsed value with appropriate type
    """
    # Try to parse as JSON first (for lists, dicts)
    try:
        return json.loads(value_str)
    except json.JSONDecodeError:
        pass
    
    # Special handling for comma-separated lists of integers
    if "," in value_str and all(x.strip().isdigit() for x in value_str.split(",")):
        return [int(x.strip()) for x in value_str.split(",")]
    
    # Boolean values
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    
    # Numeric values
    if value_str.isdigit():
        return int(value_str)
    try:
        return float(value_str)
    except ValueError:
        pass
    
    # Default to string
    return value_str


def set_config_value(config_dict: Dict, key_path: str, value: Any) -> None:
    """
    Set a value in a nested dictionary using dot notation path.
    
    Args:
        config_dict: Configuration dictionary to modify
        key_path: Dot-separated path to the parameter
        value: Value to set
    """
    logger = logging.getLogger(__name__)
    keys = key_path.split(".")
    current = config_dict
    
    # Setting config value
    
    # transformer_kwargs has been removed from the configuration system
    
    # Navigate to the containing dictionary
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
            logger.debug(f"Created empty dict for {key}")
        elif current[key] is None:
            # Special case for any null nested objects
            current[key] = {}
            logger.debug(f"Replacing None with empty dict for {key}")
        current = current[key]
    
    # Set the final value
    final_key = keys[-1]
    current[final_key] = value
    
    # Debug: Verify the value was set correctly
    try:
        # Navigate back to the final key to check the value
        verify_current = config_dict
        for key in keys[:-1]:
            verify_current = verify_current[key]
        
        set_value = verify_current[final_key]
        logger.debug(f"Value set successfully: {key_path} = {set_value} (type: {type(set_value).__name__})")
    except Exception as e:
        logger.error(f"Failed to verify value set for {key_path}: {e}")


# Define parameter shortcuts with correct paths
# Format: "--parameter-name": "config.path.to.parameter"
PARAMETER_SHORTCUTS = {
    # Model architecture shortcuts
    "input-dim": "model.input_dim",
    "dropout": "model.transformer.dropout",
    "bias": "model.bias",
    "norm-eps": "model.norm_eps",
    "use-orthogonal-encdec": "model.use_orthogonal_encdec",
    "hidden-dim": "model.transformer.hidden_dim",
    "num-layers": "model.transformer.num_layers",
    "num-heads": "model.transformer.num_heads",
    "activation": "model.transformer.activation",
    "ff-expansion": "model.transformer.ff_expansion",
    
    # Attention configuration
    "attention-type": "model.transformer.attention_config.type",
    "flash-attn": "model.transformer.attention_config.use_flash_attn",
    "random-features": "model.transformer.attention_config.num_features",
    "random-feature-eps": "model.transformer.attention_config.feature_eps",
    
    # Layer repetition shortcuts
    "layer-repeat-mode": "model.transformer.layer_repeat_mode",
    "repeat-factor": "model.transformer.repeat_factor",
    "layer-groups": "model.transformer.layer_groups",
    "group-repeat-factors": "model.transformer.group_repeat_factors",
    
    # Flow predictor shortcuts
    "use-flow-predictor": "model.transformer.flow_config.enabled",
    "flow-predictor-type": "model.transformer.flow_config.predictor_type",
    "flow-per-layer": "model.transformer.flow_config.per_layer",
    "flow-distribution-mode": "model.transformer.flow_config.distribution_mode",
    "flow-min-value": "model.transformer.flow_config.min_value",
    "flow-max-value": "model.transformer.flow_config.max_value",
    "flow-num-basis": "model.transformer.flow_config.num_basis",
    "flow-min-snr": "model.transformer.flow_config.min_snr",
    "flow-max-snr": "model.transformer.flow_config.max_snr",
    "flow-load-pretrained": "model.transformer.flow_config.load_pretrained",
    "flow-pretrained-path": "model.transformer.flow_config.pretrained_path",
    "flow-freeze-weights": "model.transformer.flow_config.freeze_weights",
    
    # Training configuration shortcuts
    "batch-size": "training.batch_size",
    "epochs": "training.num_epochs",
    "lr": "training.optimizer.learning_rate",
    "optimizer": "training.optimizer.optimizer",
    "weight-decay": "training.optimizer.weight_decay",
    "scheduler": "training.scheduler.scheduler_type",
    "warmup-ratio": "training.scheduler.warmup_ratio",
    "grad-clip": "training.gradient_clip_val",
    "snr-power": "training.loss.snr_power",
    "loss-normalization": "training.loss.normalization",
    
    # Loss function shortcuts
    "loss-type": "training.loss.loss_type",
    "wasserstein-algorithm": "training.loss.wasserstein_algorithm",
    "wasserstein-backend": "training.loss.wasserstein_backend",
    "wasserstein-epsilon": "training.loss.wasserstein_epsilon",
    "wasserstein-max-iter": "training.loss.wasserstein_max_iter",
    "use-true-weights": "training.loss.use_true_weights",
    
    # Data generation shortcuts
    "num-samples": "training.num_train_samples",
    "num-clusters": "data.cluster_params",
    "data-dim": "data.dim",
    "snr": "data.snr_db_params",
    "random-seed": "data.random_seed",
    
    # Validation shortcuts
    "validate-every": "training.val_every",
    "visualize-every": "validation.visualize.visualize_every_n_epochs",
    "compare-with-kmeans": "validation.metrics.compare_with_kmeans",
}


def add_parameter_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for all supported shortcut parameters.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    # Add universal parameter setter
    parser.add_argument(
        "--set",
        action="append",
        metavar="PARAM.PATH=VALUE",
        dest="set_params",
        default=[],
        help="Set any parameter using dot notation (e.g., model.transformer.num_layers=12)"
    )
    
    # Add shortcut parameters
    shortcuts_group = parser.add_argument_group("Parameter Shortcuts")
    
    # Model architecture shortcuts
    shortcuts_group.add_argument("--input-dim", type=str, help="Input dimension (point coordinates)")
    shortcuts_group.add_argument("--dropout", type=str, help="Dropout probability for transformer layers")
    shortcuts_group.add_argument("--bias", type=str, help="Whether to use bias in input projection and output head (true/false)")
    shortcuts_group.add_argument("--norm-eps", type=str, help="Epsilon for layer normalization")
    shortcuts_group.add_argument("--use-orthogonal-encdec", type=str, help="Whether to use orthogonal encoder-decoder architecture instead of standard linear encoder-decoder (true/false)")
    
    # Transformer architecture shortcuts
    shortcuts_group.add_argument("--hidden-dim", type=str, help="Model hidden dimension size")
    shortcuts_group.add_argument("--num-layers", type=str, help="Number of transformer layers")
    shortcuts_group.add_argument("--num-heads", type=str, help="Number of attention heads")
    shortcuts_group.add_argument("--activation", type=str, help="Activation function (gelu, relu, silu)")
    shortcuts_group.add_argument("--ff-expansion", type=str, help="Feed-forward expansion factor")
    
    # Attention configuration shortcuts
    shortcuts_group.add_argument("--attention-type", type=str, help="Attention type (standard or random_feature)")
    shortcuts_group.add_argument("--flash-attn", type=str, help="Use flash attention when using standard attention (true/false)")
    shortcuts_group.add_argument("--random-features", type=str, help="Number of random features per head when using random feature attention")
    shortcuts_group.add_argument("--random-feature-eps", type=str, help="Epsilon for numerical stability in random feature attention")
    
    # Layer repetition shortcuts
    shortcuts_group.add_argument(
        "--layer-repeat-mode", type=str, 
        help="Layer repetition mode (none, cycle, layerwise, grouped)"
    )
    shortcuts_group.add_argument("--repeat-factor", type=str, help="Number of repetitions for cycle/layerwise modes")
    shortcuts_group.add_argument("--layer-groups", type=str, help="Layer group sizes (comma-separated)")
    shortcuts_group.add_argument("--group-repeat-factors", type=str, help="Group repeat factors (comma-separated)")
    
    # Flow predictor shortcuts
    shortcuts_group.add_argument("--use-flow-predictor", type=str, help="Enable flow predictor (true/false)")
    shortcuts_group.add_argument(
        "--flow-predictor-type", type=str, 
        help="Flow predictor type (dummy, linear, monotonic)"
    )
    shortcuts_group.add_argument("--flow-per-layer", type=str, help="Use per-layer flow speeds (true/false)")
    shortcuts_group.add_argument(
        "--flow-distribution-mode", type=str,
        help="Mode for distributing flow speed across repetitions (direct, fractional)"
    )
    shortcuts_group.add_argument("--flow-min-value", type=str, help="Minimum flow speed value (default: 0.0)")
    shortcuts_group.add_argument("--flow-max-value", type=str, help="Maximum flow speed value (default: 1.0)")
    shortcuts_group.add_argument("--flow-num-basis", type=str, help="Number of basis functions for flow predictor")
    shortcuts_group.add_argument("--flow-min-snr", type=str, help="Minimum SNR mapping value (default: 0.0) - corresponds to max_flow")
    shortcuts_group.add_argument("--flow-max-snr", type=str, help="Maximum SNR mapping value (default: 20.0) - corresponds to min_flow")
    shortcuts_group.add_argument("--flow-load-pretrained", type=str, help="Load pre-trained flow predictor (true/false)")
    shortcuts_group.add_argument("--flow-pretrained-path", type=str, help="Path to pre-trained model containing flow predictor")
    shortcuts_group.add_argument("--flow-freeze-weights", type=str, help="Freeze flow predictor weights during training (true/false)")
    
    # Training configuration shortcuts
    shortcuts_group.add_argument("--batch-size", type=str, help="Training batch size")
    shortcuts_group.add_argument("--epochs", type=str, help="Number of training epochs")
    shortcuts_group.add_argument("--lr", type=str, help="Learning rate")
    shortcuts_group.add_argument("--optimizer", type=str, help="Optimizer type (adam, adamw, sgd, etc.)")
    shortcuts_group.add_argument("--weight-decay", type=str, help="Weight decay (L2 regularization)")
    shortcuts_group.add_argument("--scheduler", type=str, help="Learning rate scheduler type")
    shortcuts_group.add_argument("--warmup-ratio", type=str, help="Ratio of total steps to use for warmup")
    shortcuts_group.add_argument("--grad-clip", type=str, help="Gradient clipping value")
    shortcuts_group.add_argument("--snr-power", type=str, help="Power to raise SNR to when using snr_power normalization (e.g., 1.0, 1.33, etc.)")
    shortcuts_group.add_argument(
        "--loss-normalization", type=str,
        help="Loss normalization type (none, snr_power, or log)"
    )
    
    # Loss function shortcuts
    shortcuts_group.add_argument("--loss-type", type=str, help="Loss function type (mse, wasserstein, energy)")
    shortcuts_group.add_argument(
        "--wasserstein-backend", type=str, 
        help="Wasserstein backend (jax, pot, scipy)"
    )
    shortcuts_group.add_argument(
        "--wasserstein-algorithm", type=str, 
        help="Wasserstein algorithm (exact, sinkhorn)"
    )
    shortcuts_group.add_argument("--wasserstein-epsilon", type=str, help="Epsilon for Sinkhorn algorithm")
    shortcuts_group.add_argument("--wasserstein-max-iter", type=str, help="Max iterations for Wasserstein")
    shortcuts_group.add_argument("--use-true-weights", type=str, help="Use true mixture weights instead of counting labels (true/false)")
    
    # Data generation shortcuts
    shortcuts_group.add_argument("--num-samples", type=str, help="Number of training samples")
    shortcuts_group.add_argument("--num-clusters", type=str, help="Number of GMM clusters")
    shortcuts_group.add_argument("--data-dim", type=str, help="Dimension of data points")
    shortcuts_group.add_argument("--snr", type=str, help="Signal-to-noise ratio")
    shortcuts_group.add_argument("--random-seed", type=str, help="Random seed for data generation")
    
    # Validation shortcuts
    shortcuts_group.add_argument("--validate-every", type=str, help="Validation frequency (epochs)")
    shortcuts_group.add_argument("--visualize-every", type=str, help="Visualization frequency (epochs)")
    shortcuts_group.add_argument("--compare-with-kmeans", type=str, help="Compare with K-means (true/false)")


def process_shortcut_args(args, set_params: List[str]) -> List[str]:
    """
    Process shortcut arguments and convert them to universal --set format.
    
    Args:
        args: Parsed command-line arguments
        set_params: Current list of set parameters
        
    Returns:
        Updated list of set parameters
    """
    args_dict = vars(args)
    logger = logging.getLogger(__name__)
    
    # Process command line arguments
    
    for arg_name, config_path in PARAMETER_SHORTCUTS.items():
        # Convert hyphenated arg names to underscore format
        arg_name_underscore = arg_name.replace("-", "_")
        arg_value = args_dict.get(arg_name_underscore)
        if arg_value is not None:
            set_params.append(f"{config_path}={arg_value}")
            logger.debug(f"Shortcut processed: --{arg_name} -> {config_path}={arg_value}")
    
    return set_params


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced GMM Transformer Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core options
    parser.add_argument(
        "--model-preset", 
        type=str, 
        default="small",
        help="Model preset to use (tiny, small, medium, large)"
    )
    
    parser.add_argument(
        "--training-preset", 
        type=str, 
        required=False,
        help="Training preset to use (quick, standard, optimized)"
    )
    
    parser.add_argument(
        "--data-preset", 
        type=str, 
        default="standard",
        help="Data preset to use (simple, standard, complex)"
    )
    
    parser.add_argument(
        "--validation-preset", 
        type=str, 
        default="standard",
        help="Validation preset to use (minimal, standard, comprehensive)"
    )
    
    # Experiment name and output
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for the experiment (used for output directory naming)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for experiment results"
    )
    
    # Add all parameter shortcuts
    add_parameter_arguments(parser)
    
    # We've standardized on dictionary-based configuration for Wasserstein loss
    # so we no longer need the --use-dict-config flag
    # (It's removed to simplify usage)
    
    # Device options
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for training (e.g., 'cpu', 'cuda', 'cuda:0', 'mps')"
    )
    
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision training"
    )
    
    # Utility options
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available presets and exit"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running the experiment"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        nargs='?',
        const='list',  # When --resume is specified without a value, use 'list'
        help="Resume experiment by name from the last saved checkpoint. Use without a value to list available experiments."
    )
    
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=None,
        help="Number of additional epochs to train when resuming an experiment"
    )
    
    parser.add_argument(
        "--reconfigure-scheduler",
        action="store_true",
        help="Reconfigure the learning rate scheduler when resuming training"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """Run an experiment with the enhanced CLI interface."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # List presets if requested
    if args.list_presets:
        print("\nAvailable model presets:")
        for preset in list_model_presets():
            desc = get_model_description(preset)
            if desc:
                print(f"  - {preset}: {desc}")
            else:
                print(f"  - {preset}")
        
        print("\nAvailable training presets:")
        for preset in list_training_presets():
            desc = get_training_description(preset)
            if desc:
                print(f"  - {preset}: {desc}")
            else:
                print(f"  - {preset}")
        
        print("\nAvailable data presets:")
        for preset in list_data_presets():
            desc = get_data_description(preset)
            if desc:
                print(f"  - {preset}: {desc}")
            else:
                print(f"  - {preset}")
        
        print("\nAvailable validation presets:")
        for preset in list_validation_presets():
            desc = get_validation_description(preset)
            if desc:
                print(f"  - {preset}: {desc}")
            else:
                print(f"  - {preset}")
        
        return 0
    
    # Handle resuming experiment by name
    if args.resume:
        output_dir = args.output_dir or "output"
        output_path = Path(output_dir)
        
        # Function to list available experiments
        def list_available_experiments():
            if output_path.exists():
                experiments = [d for d in output_path.iterdir() if d.is_dir() and (d / "config.json").exists()]
                if experiments:
                    print("\nAvailable experiments:")
                    for exp in experiments:
                        # Try to get more details from the config
                        try:
                            config_path = exp / "config.json"
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            if "metadata" in config and "experiment_name" in config["metadata"]:
                                name = config["metadata"]["experiment_name"]
                            else:
                                name = exp.name
                            
                            # Try to load timestamps and other useful info
                            checkpoint_dir = exp / "checkpoints"
                            checkpoints = []
                            if checkpoint_dir.exists():
                                checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
                            
                            # Show experiment info with checkpoint details
                            if checkpoints:
                                latest_epoch = max([int(c.stem.split("_")[-1]) for c in checkpoints if c.stem.split("_")[-1].isdigit()], default=0)
                                print(f"  - {exp.name}: '{name}', trained for {latest_epoch} epochs")
                            else:
                                print(f"  - {exp.name}: '{name}'")
                        except Exception:
                            # Fall back to simple name if error occurs
                            print(f"  - {exp.name}")
                    
                    print(f"\nUse --resume <experiment_name> to resume one of these experiments.")
                else:
                    print(f"No experiments found in {output_path}")
                return True
            else:
                print(f"Output directory {output_path} does not exist")
                return False
        
        # If just listing experiments
        if args.resume == 'list':
            list_available_experiments()
            return 0
            
        # Otherwise, try to resume the specified experiment
        experiment_name = args.resume
        experiment_path = output_path / experiment_name.lower().replace(" ", "_")
        
        # Check if the experiment directory exists
        if not experiment_path.exists():
            logger.error(f"Experiment directory not found: {experiment_path}")
            list_available_experiments()
            return 1
            
        logger.info(f"Resuming experiment from: {experiment_path}")
            
        try:
            # Load the experiment
            # Using the globally imported ExperimentManager
            experiment = ExperimentManager.load(experiment_path)
            
            # Set device if provided
            device = args.device
            if device:
                logger.info(f"Setting device to {device} for resumed experiment")
                experiment.config.device.device = device
            
            # Resume training with optional additional epochs and scheduler reconfiguration
            history = experiment.resume_training(
                num_epochs=args.additional_epochs,
                reconfigure_scheduler=args.reconfigure_scheduler
            )
            
            logger.info(f"Resumed experiment {experiment_name} completed successfully")
            return 0
        except Exception as e:
            logger.error(f"Error resuming experiment: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Check if training preset is provided for new experiments
    if not args.training_preset:
        logger.error("--training-preset is required when running a new experiment")
        print("\nERROR: --training-preset is required. Available training presets:")
        for preset in list_training_presets():
            print(f"  - {preset}")
        return 1
    
    # Get component configurations
    try:
        logger.info(f"Using model preset: {args.model_preset}")
        model_config = get_model_config(args.model_preset)
        
        logger.info(f"Using training preset: {args.training_preset}")
        training_config = get_training_config(args.training_preset)
        
        logger.info(f"Using data preset: {args.data_preset}")
        data_config = get_data_config(args.data_preset)
        
        logger.info(f"Using validation preset: {args.validation_preset}")
        validation_config = get_validation_config(args.validation_preset)
    except Exception as e:
        logger.error(f"Error loading presets: {e}")
        return 1
    
    # Process shortcut arguments into --set format
    set_params = process_shortcut_args(args, args.set_params)
    
    # Special handling for loss configuration
    if args.loss_type:
        # Check if it's a Wasserstein loss type
        if args.loss_type.startswith("wasserstein"):
            # Always use dictionary-based configuration for Wasserstein
            wasserstein_config = {
                "type": "wasserstein",
                "algorithm": args.wasserstein_algorithm or "exact",
                "backend": args.wasserstein_backend or "pot"
            }
            
            # Add optional parameters if provided
            if args.wasserstein_epsilon:
                wasserstein_config["epsilon"] = parse_value(args.wasserstein_epsilon)
            
            if args.wasserstein_max_iter:
                wasserstein_config["max_iterations"] = parse_value(args.wasserstein_max_iter)
                
            # Add use_true_weights if provided
            if hasattr(args, 'use_true_weights') and args.use_true_weights is not None:
                wasserstein_config["use_true_weights"] = parse_value(args.use_true_weights)
            
            # Convert to JSON string and add to parameters
            set_params.append(f"training.loss.loss_type={json.dumps(wasserstein_config)}")
            logger.info(f"Using Wasserstein loss with algorithm={wasserstein_config['algorithm']}, backend={wasserstein_config['backend']}")
        else:
            # For other loss types (mse, energy, etc.)
            set_params.append(f"training.loss.loss_type={args.loss_type}")
            logger.info(f"Using loss type: {args.loss_type}")
    
    # Create configuration dictionary
    config_dict = {
        "model": model_config.model_dump(),
        "training": training_config.model_dump(),
        "data": data_config.model_dump(),
        "validation": validation_config.model_dump(),
    }
    
    # Create experiment name if not provided
    experiment_name = args.experiment_name
    if experiment_name is None:
        # Auto-generate based on presets and key parameters
        parts = [args.model_preset, args.training_preset]
        
        for param in set_params:
            if param.startswith("model.transformer.layer_repeat_mode="):
                mode = param.split("=")[1]
                parts.append(mode)
        
        experiment_name = "_".join(parts)
    
    # Create metadata and other required sections
    experiment_id = experiment_name.lower().replace(" ", "_")
    config_dict["metadata"] = {
        "id": experiment_id,
        "experiment_name": experiment_name,
        "description": f"Experiment with {args.model_preset} model, {args.training_preset} training, and {args.data_preset} data"
    }
    
    config_dict["device"] = {
        "device": args.device,  # Changed from device_type to device to match DeviceConfig
        "use_mixed_precision": args.mixed_precision
    }
    
    output_dir = args.output_dir or "output"
    config_dict["paths"] = {
        "base_dir": output_dir,
        "log_dir": "logs",
        "checkpoint_dir": "checkpoints",
        "data_dir": "data"
    }
    
    # Apply all parameter overrides
    for param in set_params:
        if "=" not in param:
            logger.error(f"Invalid parameter format: {param}. Expected KEY=VALUE")
            return 1
        
        key, value_str = param.split("=", 1)
        try:
            value = parse_value(value_str)
            set_config_value(config_dict, key, value)
            logger.debug(f"Set {key} = {value}")
        except Exception as e:
            logger.error(f"Error setting parameter {key}: {e}")
            return 1
    
    # Create and validate configuration
    try:
        config = ExperimentConfig.model_validate(config_dict)
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return 1
    
    # Print configuration summary
    logger.info(f"Experiment name: {config.metadata.experiment_name}")
    logger.info(f"Model: {args.model_preset} - {config.model.transformer.hidden_dim} hidden dim, {config.model.transformer.num_layers} layers")
    
    # Print attention configuration info
    attention_type = config.model.transformer.attention_config.type
    logger.info(f"Attention type: {attention_type}")
    if attention_type == "standard":
        logger.info(f"  Flash attention: {config.model.transformer.attention_config.use_flash_attn}")
    elif attention_type == "random_feature":
        logger.info(f"  Random features: {config.model.transformer.attention_config.num_features}")
        logger.info(f"  Using repeat factor: {config.model.transformer.repeat_factor} for random matrices")
    
    # Print layer repetition info if configured
    if hasattr(config.model.transformer, "layer_repeat_mode") and config.model.transformer.layer_repeat_mode != "none":
        logger.info(f"Layer repetition: {config.model.transformer.layer_repeat_mode} mode")
        
        if config.model.transformer.layer_repeat_mode in ["cycle", "layerwise"]:
            logger.info(f"  Repeat factor: {config.model.transformer.repeat_factor}")
        elif config.model.transformer.layer_repeat_mode == "grouped" and hasattr(config.model.transformer, "layer_groups"):
            logger.info(f"  Layer groups: {config.model.transformer.layer_groups}")
            logger.info(f"  Group repeat factors: {config.model.transformer.group_repeat_factors}")
    
    # Add detailed debug logging for num_train_samples
    logger.info(f"Training: {args.training_preset} - {config.training.batch_size} batch size, {config.training.num_epochs} epochs, {config.training.num_train_samples} samples")
    
    # Parameters that were set: {len(set_params)} overrides applied
    logger.info(f"Data: {args.data_preset} - {config.data.dim} dimensions")
    logger.info(f"Output directory: {os.path.join(config.paths.base_dir, config.metadata.id)}")
    
    # If dry run, stop here
    if args.dry_run:
        logger.info("Dry run - beginning experiment setup phase for testing")
        try:
            # Create experiment manager and just do the setup phase
            experiment = ExperimentManager(config)
            experiment.setup()
            logger.info("Experiment setup completed successfully")
        except Exception as e:
            logger.error(f"Setup failed: {e}")
        return 0
    
    # Create experiment manager and run experiment
    try:
        logger.info("Creating experiment manager")
        experiment = ExperimentManager(config)
        
        logger.info("Setting up experiment")
        experiment.setup()
        
        logger.info("Running experiment")
        experiment.run()
        
        logger.info(f"Experiment {config.metadata.experiment_name} completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())