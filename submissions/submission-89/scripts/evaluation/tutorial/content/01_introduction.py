"""
GMM Model Evaluation Tutorial - Part 1: Introduction
===================================================

This is the first part of our GMM model evaluation tutorial series.
In this tutorial, we'll learn the basics of loading and inspecting
pre-trained GMM models.

Tutorial Overview:
1. Loading pre-trained models
2. Understanding model architecture
3. Loading test data
4. Running basic model inference

Let's begin!
"""

print("=" * 70)
print("GMM Model Evaluation Tutorial - Part 1: Introduction")
print("=" * 70)
print()

# Step 1: Setup environment
print("Step 1: Setting up the environment")
print("-" * 35)

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path if needed
project_root = '/mount/Storage/gmm-v4/'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✓ Device configured: {device}")
print(f"✓ PyTorch version: {torch.__version__}")
print()

# Import our tutorial utilities
print("Step 2: Importing evaluation utilities")
print("-" * 38)
from scripts.evaluation.tutorial.src.io import (
    load_model_from_experiment,
    create_data_loader
)
from scripts.evaluation.tutorial.src.eval_utils import (
    evaluate, 
    evaluate_dataset,
    run_kmeans,
    compute_metrics,
    get_flow_prediction
)
print("✓ Successfully imported all utilities")
print()

# Step 3: Understanding model loading
print("Step 3: Loading a pre-trained GMM model")
print("-" * 39)
print("\nThe load_model_from_experiment function performs these steps:")
print("  1. Finds the checkpoint file (best_model.pt or final_model.pt)")
print("  2. Loads the experiment configuration")
print("  3. Creates the model architecture based on config")
print("  4. Loads the trained weights into the model")
print()

# Define paths to experiment directories
print("Step 4: Defining experiment paths and available models")
print("-" * 53)
output_dir = Path('/mount/Storage/gmm-v4/output')
experiment_base_dir = output_dir / 'final_experiments'
print(f"✓ Experiment base directory: {experiment_base_dir}")
print()

# Define models to compare (as a dictionary for easier access by name)
print("Available pre-trained models:")
model_configs = {
    "baseline_16_layers": {
        "name": "16 Layers", 
        "path": "baseline_16_layers", 
        "linestyle": "-", 
        "color": "blue", 
        "layers": 16
    },
    "baseline_32_layers": {
        "name": "32 Layers", 
        "path": "baseline_32_layers", 
        "linestyle": "--", 
        "color": "red", 
        "layers": 32
    },
    "baseline_64_layers": {
        "name": "64 Layers", 
        "path": "baseline_64_layers", 
        "linestyle": "-.", 
        "color": "green", 
        "layers": 64
    },
    "simple_16_layers": {
        "name": "Simple 16L", 
        "path": "simple_16_layers", 
        "linestyle": ":", 
        "color": "purple", 
        "layers": 16
    },
    "hard_16_layers": {
        "name": "Hard 16L", 
        "path": "hard_16_layers", 
        "linestyle": "--", 
        "color": "orange", 
        "layers": 16
    }
}

for key, config in model_configs.items():
    print(f"  - {key}: {config['name']} ({config['layers']} layers)")
print()

# Step 5: Loading a specific model
print("Step 5: Loading a specific model")
print("-" * 33)
model_name = "baseline_64_layers"  # Change this to load different models
selected_model = model_configs[model_name]
experiment_dir = experiment_base_dir / selected_model["path"]

print(f"Selected model: {selected_model['name']}")
print(f"Experiment directory: {experiment_dir}")
print(f"Loading model...")
model, config = load_model_from_experiment(experiment_dir, load_best=False, device=device)
print(f"✓ Successfully loaded model: {selected_model['name']}")
print()

# Step 6: Creating a data loader
print("Step 6: Creating a data loader for evaluation")
print("-" * 45)
dataset_name = "high_snr_fixed"
batch_size = 16
total_samples = 512

print(f"Dataset configuration:")
print(f"  - Dataset name: {dataset_name}")
print(f"  - Batch size: {batch_size}")
print(f"  - Total samples: {total_samples}")
print()

data_loader = create_data_loader(
    dataset_name=dataset_name,
    batch_size=batch_size,
    total_samples=total_samples,
    device=device
)
print(f"✓ Data loader created successfully")
print()

# Step 7: Running inference
print("Step 7: Running basic inference on a batch of data")
print("-" * 50)
print("Getting a batch from the data loader...")

# Get a batch from the data loader
batch = next(iter(data_loader))
inputs, targets = batch
print(f"✓ Batch loaded:")
print(f"  - Input shape: {inputs.shape}")
print(f"  - Target keys: {list(targets.keys())}")
print()

# Run the model on input data
print("Running model inference...")
with torch.no_grad():
    # For models with flow prediction, we need the SNR value
    snr_values = targets['snr_db']
    
    # Check if the model has a flow predictor
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'flow_predictor'):
        print("  - Model has flow predictor, computing flow speed...")
        flow_speed = model.transformer.flow_predictor(targets={'snr_db': snr_values})
        print(f"  - Flow speed computed: {flow_speed.shape}")
        predictions = model(inputs, flow_speed=flow_speed)
    else:
        print("  - Model does not have flow predictor, running standard inference...")
        predictions = model(inputs)
    
    print(f"✓ Inference complete!")
    print(f"  - Predictions shape: {predictions.shape}")
print()

# Step 8: Analyzing model architecture
print("Step 8: Analyzing model architecture")
print("-" * 36)

def print_model_info(model):
    """Print detailed information about the model architecture"""
    print("\nModel Architecture Details:")
    print("  - Model type:", type(model).__name__)
    
    # Check if it has a transformer
    if hasattr(model, 'transformer'):
        print("  - Has transformer: Yes")
        transformer = model.transformer
        
        # Check transformer properties
        if hasattr(transformer, 'num_layers'):
            print(f"  - Transformer layers: {transformer.num_layers}")
        
        # Check flow predictor
        if hasattr(transformer, 'flow_predictor'):
            print("  - Has flow predictor: Yes")
            fp = transformer.flow_predictor
            print(f"  - Flow predictor type: {type(fp).__name__}")
        else:
            print("  - Has flow predictor: No")
    else:
        print("  - Has transformer: No")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    return model

# Run the model info function
if __name__ == "__main__":
    print_model_info(model)
    print()
    
    print("\nIntroduction complete.") 