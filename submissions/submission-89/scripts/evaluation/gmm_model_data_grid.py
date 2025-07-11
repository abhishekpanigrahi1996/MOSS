#!/usr/bin/env python
"""
GMM 2x2 Model/Data Evaluation Grid

This script:
- Loads two models (low SNR, high SNR)
- Generates two datasets (low SNR, high SNR)
- Evaluates each model on each dataset
- Visualizes all four (model, dataset) combinations in a 2x2 static grid

Usage:
    python gmm_model_data_grid.py \
        --model-low-snr <path_to_low_snr_model> \
        --model-high-snr <path_to_high_snr_model> \
        --output <output_png_path>

Optional:
    --num-points 1000
    --random-seed 42
    --no-kmeans (disable kmeans baseline)
"""
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Optional, Any

# Add project root to sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))

from config.registry import get_data_config
from utils.visualization import create_evaluation_grid, create_model_only_grid
from losses import create_wasserstein_loss


def load_model(model_dir, device='cpu'):
    from config import ExperimentConfig
    from training.experiment import ExperimentManager
    import torch

    # Find the checkpoint file
    checkpoint_path = os.path.join(model_dir, 'checkpoints', 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract configuration
    config_dict = checkpoint.get('config', {})
    if not config_dict:
        raise ValueError(f"No configuration found in checkpoint {checkpoint_path}")

    config = ExperimentConfig.model_validate(config_dict)

    # Create model
    experiment = ExperimentManager(config)
    model = experiment._create_model()

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


def generate_gmm_sample(data_config, num_points=1000, random_seed=None, device='cpu'):
    """Generate a single GMM sample (points, centers, labels)."""
    data_config.sample_count_distribution = {"type": "fixed", "value": num_points}
    if random_seed is not None:
        data_config.random_seed = random_seed
    
    generator = data_config.create_gmm_generator()
    data, targets = generator.generate_training_batch(batch_size=1, device=device)
    points = data[0]
    centers = targets['centers'][0]
    labels = targets['labels'][0]
    return points, centers, labels


def get_model_predictions(model, points, device='cpu'):
    """Run model on points and return predictions."""
    with torch.no_grad():
        points = points.unsqueeze(0).to(device)  # [1, N, D]
        preds = model(points)
        preds = preds.squeeze(0).cpu()
    return preds


def prepare_viz_dict(points, centers, labels, preds, idx):
    """Prepare dictionary for create_evaluation_grid."""
    return {
        "inputs": points.cpu(),
        "outputs": preds.cpu(),
        "targets": {"centers": centers.cpu(), "labels": labels.cpu()},
        "sample_idx": idx,
        "true_labels": labels.cpu()
    }


def main():
    # Hardcoded paths as requested
    model_low_snr_path = "output/final_experiments/hard_16_layers"
    model_high_snr_path = "output/final_experiments/simple_16_layers"
    output_path = "plots/high_low_snr/model_data_2x2_grid.png"

    parser = argparse.ArgumentParser(description="GMM 2x2 Model/Data Evaluation Grid")
    parser.add_argument('--num-points', type=int, default=1000, help='Number of points per sample')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-kmeans', action='store_true', help='Disable K-means baseline')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu)')
    args = parser.parse_args()

    # Load data configs
    data_low_preset = 'low_snr_fixed'
    data_low_preset = 'simple'
    

    data_high_preset = 'high_snr_fixed'
    data_high_preset = 'simple'

    data_low = get_data_config(data_low_preset)
    data_high = get_data_config(data_high_preset)

    data_low.snr_db_params = {
        "type": "fixed",
        "value": 5.0
    }

    data_high.snr_db_params = {
        "type": "fixed",
        "value": 15.0
    }

    # Load models
    model_low = load_model(model_low_snr_path, device=args.device)
    model_high = load_model(model_high_snr_path, device=args.device)

    # Generate datasets
    points_low, centers_low, labels_low = generate_gmm_sample(data_low, args.num_points, args.random_seed, device=args.device)
    points_high, centers_high, labels_high = generate_gmm_sample(data_high, args.num_points, args.random_seed, device=args.device)

    # Get predictions
    preds_low_on_low = get_model_predictions(model_low, points_low, device=args.device)
    preds_low_on_high = get_model_predictions(model_low, points_high, device=args.device)
    preds_high_on_low = get_model_predictions(model_high, points_low, device=args.device)
    preds_high_on_high = get_model_predictions(model_high, points_high, device=args.device)


    # Prepare visualization data (row: data, col: model)
    viz_data = [
        prepare_viz_dict(points_low, centers_low, labels_low, preds_low_on_low, idx=1),    # low data, low model
        prepare_viz_dict(points_low, centers_low, labels_low, preds_high_on_low, idx=2),   # low data, high model
        prepare_viz_dict(points_high, centers_high, labels_high, preds_low_on_high, idx=3), # high data, low model
        prepare_viz_dict(points_high, centers_high, labels_high, preds_high_on_high, idx=4),# high data, high model
    ]

    # Output directory and path with random seed in filename
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"model_data_2x2_grid_seed{args.random_seed}.png"
    output_path_with_seed = os.path.join(output_dir, output_filename)

    # Collect all points for global axis limits
    all_x = np.concatenate([data["inputs"].numpy()[:, 0] for data in viz_data])
    all_y = np.concatenate([data["inputs"].numpy()[:, 1] for data in viz_data])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    # Optionally, add a small margin
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    # Create and save 2x2 grid with axis labels instead of per-plot titles
    data_snr_labels = ["5dB", "15dB"]
    model_snr_labels = ["5dB", "15dB"]
    create_model_only_grid(
        visualization_data=viz_data,
        output_path=output_path_with_seed,
        data_snr_labels=data_snr_labels,
        model_snr_labels=model_snr_labels,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max
    )
    print(f"Saved 2x2 evaluation grid to {output_path_with_seed}")

    


if __name__ == "__main__":
    main() 