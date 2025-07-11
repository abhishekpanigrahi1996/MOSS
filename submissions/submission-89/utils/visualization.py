"""
Visualization utilities for GMM transformer models.

This module provides comprehensive functions for visualizing model predictions,
ground truth, and GMM data in various formats including single plots, grids,
and parameter sweeps.

The main visualization functions are:
- visualize_predictions: Create a 4-panel visualization of input, ground truth, predictions, and combined view
- visualize_gmm_data: Create a simple plot of GMM data with points and centers
- create_training_plots: Generate plots of training history
- create_evaluation_grid: Create a grid of visualizations for model evaluation
"""

import os
import logging
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Import to_device utility function for the deprecated create_visualization function
from metrics.utils import to_device

# Try to import scikit-learn for K-means baseline
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, K-means baseline in visualizations will be disabled")


def visualize_predictions(
    input_data: np.ndarray,
    target_data: np.ndarray,
    prediction_data: np.ndarray,
    kmeans_centers: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    colormap: str = "viridis",
    alpha: float = 0.4,  # Reduced alpha for better contrast with centers
    point_size: int = 50,  # Larger point size for better visibility
    figsize: Tuple[int, int] = (14, 6),  # Wider figure size to prevent overlap
    true_labels: Optional[np.ndarray] = None,
    only_model_panel: bool = False
) -> Figure:
    """
    Create a 2-panel visualization of GMM data, comparing K-means and model predictions.
    
    Args:
        input_data: Input data points [seq_len, dim]
        target_data: Target centers [n_clusters, dim] or [seq_len, dim]
        prediction_data: Model predictions [seq_len, dim]
        kmeans_centers: K-means centers (optional) [n_clusters, dim]
        title: Plot title
        colormap: Colormap for scatter plots
        alpha: Transparency level for points
        point_size: Size of points in scatter plots
        figsize: Figure size as (width, height) in inches
        true_labels: Ground truth cluster labels for coloring points [seq_len]
        only_model_panel: Whether to return only the model panel
        
    Returns:
        Matplotlib figure
    """
    # Create figure with 2 subplots (true centers + kmeans, true centers + predictions)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # If data is more than 2D, use first two dimensions for visualization
    if input_data.shape[-1] > 2:
        logger.warning(f"Using first 2 dimensions for visualization (original dim: {input_data.shape[-1]})")
        input_data = input_data[..., :2]
        target_data = target_data[..., :2]
        prediction_data = prediction_data[..., :2]
        if kmeans_centers is not None:
            kmeans_centers = kmeans_centers[..., :2]
    
    # Determine cluster assignments for coloring if true_labels not provided
    if true_labels is None:
        # Attempt to compute labels based on nearest target center
        if len(target_data.shape) == 2 and len(target_data) < len(input_data):
            from sklearn.metrics import pairwise_distances
            distances = pairwise_distances(input_data, target_data)
            true_labels = np.argmin(distances, axis=1)
    
    # Prepare color mapping
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    
    # Convert labels to integer indices if needed
    if true_labels is not None:
        if np.issubdtype(true_labels.dtype, np.floating):
            cluster_indices = true_labels.astype(int)
        else:
            cluster_indices = true_labels
        
        # Get a colormap with enough colors for our clusters
        n_clusters = len(np.unique(cluster_indices))
        base_colors = cm.get_cmap('tab10', 10)(range(min(n_clusters, 10)))
        if n_clusters > 10:
            # If more than 10 clusters, add colors from another colormap
            extra_colors = cm.get_cmap('Set3', n_clusters - 10)(range(n_clusters - 10))
            base_colors = np.vstack([base_colors, extra_colors])
    else:
        # Default single color
        cluster_indices = np.zeros(len(input_data), dtype=int)
        n_clusters = 1
        base_colors = np.array([[0.7, 0.7, 0.7, 1.0]])  # Gray
    
    # PANEL 1: True centers + K-means
    # Plot colored points by cluster with reduced alpha for better visibility of centers
    for cluster_idx in range(n_clusters):
        mask = cluster_indices == cluster_idx
        if np.any(mask):
            axes[0].scatter(
                input_data[mask, 0], 
                input_data[mask, 1], 
                alpha=alpha,  # More transparent
                s=point_size, 
                color=base_colors[cluster_idx % len(base_colors)]
            )
    
    # Add K-means centers with BLUE CIRCLE markers 
    if kmeans_centers is not None:
        for i in range(len(kmeans_centers)):
            # Plot blue circles
            axes[0].plot(
                kmeans_centers[i, 0],
                kmeans_centers[i, 1],
                'o',  # Circle marker
                color='blue',
                markersize=15,  # Keep original size for K-means
                markeredgewidth=2,
                markeredgecolor='black',
                label='K-means' if i == 0 else ""
            )
    
    # Add true centers with GREEN X markers - added LAST to ensure they're on top
    if len(target_data.shape) == 2 and len(target_data) < len(input_data):
        for i in range(len(target_data)):
            # Plot X symbol in green
            axes[0].plot(
                target_data[i, 0],
                target_data[i, 1],
                'X',  # X marker
                color='green',
                markersize=18,  # Larger marker
                markeredgewidth=2.5,  # Thicker edges
                markeredgecolor='black',
                label='True Centers' if i == 0 else ""
            )
    else:
        axes[0].text(0.5, 0.5, "K-means not available", 
                    ha='center', va='center', transform=axes[0].transAxes)
    
    # Set title without legend
    axes[0].set_title("True Centers vs K-means", fontsize=14)
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    # Remove axis labels to save space
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # PANEL 2: True centers + Model predictions with gray points
    # Plot all points in gray
    axes[1].scatter(
        input_data[:, 0], 
        input_data[:, 1], 
        alpha=0.2,  # Very transparent for background effect
        s=point_size, 
        color='gray',
        label='Data Points'
    )
    
    # Add model prediction centers with BLUE CIRCLE markers (same as K-means but smaller)
    if len(prediction_data.shape) == 2:
        for i in range(len(prediction_data)):
            # Plot blue circles (same as K-means but smaller)
            axes[1].plot(
                prediction_data[i, 0],
                prediction_data[i, 1],
                'o',  # Circle marker
                color='blue',
                markersize=10,  # Smaller marker for predictions
                markeredgewidth=2,
                markeredgecolor='black',
                label='Model Predictions' if i == 0 else ""
            )
    
    # Add True centers with GREEN X markers again - added LAST to be on top
    if len(target_data.shape) == 2 and len(target_data) < len(input_data):
        for i in range(len(target_data)):
            # Plot X symbol in green
            axes[1].plot(
                target_data[i, 0],
                target_data[i, 1],
                'X',  # X marker
                color='green',
                markersize=18,  # Larger marker
                markeredgewidth=2.5,  # Thicker edges
                markeredgecolor='black',
                label='True Centers' if i == 0 else ""
            )
    
    # Set title without legend
    axes[1].set_title("True Centers vs Model Predictions", fontsize=14)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    # Remove axis labels to save space
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # Set overall title if provided with larger font size
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Use a tight layout with substantial horizontal spacing to avoid panel duplication
    fig.tight_layout(pad=0.5)
    plt.subplots_adjust(wspace=0.35)  # Increased horizontal spacing between panels
    
    # Return only the model panel if requested
    if only_model_panel:
        return fig.axes[1]
    
    return fig


def visualize_gmm_data(
    points: np.ndarray, 
    centers: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    colormap: str = "viridis",
    alpha: float = 0.6,
    point_size: int = 20,
    center_size: int = 100,
    center_color: str = "red",
    center_marker: str = "X"
) -> plt.Axes:
    """
    Visualize GMM data with points, centers, and labels.
    
    Args:
        points: Point coordinates [num_points, dim]
        centers: Center coordinates [num_clusters, dim]
        labels: Point labels [num_points]
        title: Plot title
        ax: Matplotlib axes to plot on (creates new if None)
        colormap: Colormap for scatter plots
        alpha: Alpha value for point transparency
        point_size: Size of points in scatter plot
        center_size: Size of center markers
        center_color: Color of center markers
        center_marker: Marker style for centers
    
    Returns:
        Matplotlib axes with plot
    """
    # Create axes if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
        
    # If data is more than 2D, use first two dimensions
    if points.shape[1] > 2:
        logger.warning(f"Using first 2 dimensions for visualization (original dim: {points.shape[1]})")
        points = points[:, :2]
        if centers is not None:
            centers = centers[:, :2]
    
    # Plot points with labels if provided
    if labels is not None:
        scatter = ax.scatter(
            points[:, 0], 
            points[:, 1], 
            c=labels, 
            cmap=colormap,
            s=point_size, 
            alpha=alpha
        )
    else:
        scatter = ax.scatter(
            points[:, 0], 
            points[:, 1], 
            s=point_size, 
            alpha=alpha,
            c='gray'
        )
    
    # Plot centers if provided
    if centers is not None:
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            marker=center_marker,
            s=center_size,
            color=center_color,
            edgecolors='black'
        )
    
    # Set title if provided
    if title:
        ax.set_title(title)
    
    # Set equal aspect and grid
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.3)
    
    return ax


def generate_gmm_sample(data_config, num_points=1000):
    """
    Generate a single GMM sample with the given configuration.
    
    Args:
        data_config: DataConfig instance
        num_points: Number of points to generate
        
    Returns:
        points: Generated points [num_points, dim]
        centers: True centers [num_clusters, dim]
        labels: Point labels [num_points]
        params: Generation parameters
    """
    # Import here to avoid circular imports
    from config import DataConfig
    
    # Override sample count in data_config 
    data_config.sample_count_distribution = {"type": "fixed", "value": num_points}
    
    # Create GMM generator
    generator = data_config.create_gmm_generator()
    
    # Generate a single batch
    data, centers, labels, params = generator.generate_training_batch(
        batch_size=1,  # Just one sample
        device=None    # Use CPU
    )
    
    # Remove batch dimension (batch_size=1)
    points = data[0]  # Shape: [num_points, dim]
    centers = centers[0]  # Shape: [num_clusters, dim]
    labels = labels[0]  # Shape: [num_points]
    
    return points, centers, labels, params[0]


def create_gmm_grid_visualization(
    data_configs_or_params: List[Union[Dict, Any]], 
    num_points: int = 1000,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 15),
    dpi: int = 150,
    title: Optional[str] = "GMM Data Visualization Grid",
    colormap: str = "viridis",
    plot_centers: bool = True
) -> Figure:
    """
    Create a grid of GMM visualizations with different parameter configurations.
    
    Args:
        data_configs_or_params: List of DataConfigs or parameter dicts to create the grid
        num_points: Number of points per GMM sample
        output_path: If provided, save figure to this path
        figsize: Figure size as (width, height) in inches
        dpi: DPI for figure
        title: Main figure title
        colormap: Colormap for scatter plots
        plot_centers: Whether to plot the center points
        
    Returns:
        Matplotlib figure
    """
    # Import here to avoid circular imports
    from config import DataConfig
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(len(data_configs_or_params))))
    
    # Create figure
    fig, axes = plt.subplots(
        grid_size, 
        grid_size,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        sharey=True
    )
    
    # Ensure axes is 2D even for a single plot
    if grid_size == 1:
        axes = np.array([[axes]])
    elif len(axes.shape) == 1:
        axes = axes.reshape(1, -1)
    
    # Generate data and create visualizations
    for i, config_or_params in enumerate(data_configs_or_params):
        if i >= grid_size * grid_size:
            logger.warning(f"Grid size too small for {len(data_configs_or_params)} configs, showing only {grid_size*grid_size}")
            break
        
        row = i // grid_size
        col = i % grid_size
        
        # Create DataConfig if dict provided
        if isinstance(config_or_params, dict):
            data_config = DataConfig(**config_or_params)
        else:
            data_config = config_or_params
        
        # Generate data
        points, centers, labels, params = generate_gmm_sample(data_config, num_points)
        
        # Convert to numpy arrays if tensors
        if isinstance(points, torch.Tensor):
            points = points.numpy()
        if isinstance(centers, torch.Tensor):
            centers = centers.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        
        # Get the current axis
        ax = axes[row, col]
        
        # Plot points and centers
        visualize_gmm_data(
            points=points,
            centers=centers if plot_centers else None,
            labels=labels,
            ax=ax,
            colormap=colormap,
            title=f"n_clusters={params.n_clusters}\nSNR={params.snr_db:.1f}dB"
        )
    
    # Hide unused subplots
    for i in range(len(data_configs_or_params), grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        fig.delaxes(axes[row, col])
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
    
    return fig


def create_visualization_from_parameter_grid(
    rows_configs: List[Dict],
    cols_configs: List[Dict],
    data_gen_fn: Callable,
    row_param_name: str = "clusters",
    col_param_name: str = "snr_db",
    figsize: Tuple[int, int] = (15, 15),
    title: str = "Parameter Grid Visualization",
    output_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
    colormap: str = "viridis",
    plot_centers: bool = True
) -> Figure:
    """
    Create a grid visualization by varying parameters along rows and columns.
    
    Args:
        rows_configs: List of configurations to vary along rows
        cols_configs: List of configurations to vary along columns
        data_gen_fn: Function to generate data given configuration
        row_param_name: Parameter name for row labels
        col_param_name: Parameter name for column labels
        figsize: Figure size
        title: Figure title
        output_path: If provided, save the figure to this path
        dpi: DPI for saved figure
        colormap: Colormap for scatter plots
        plot_centers: Whether to plot center points
    
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, axes = plt.subplots(
        len(rows_configs),
        len(cols_configs),
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        sharey=True
    )
    
    # Ensure axes is 2D even for a single row/column
    if len(rows_configs) == 1 and len(cols_configs) == 1:
        axes = np.array([[axes]])
    elif len(rows_configs) == 1:
        axes = axes.reshape(1, -1)
    elif len(cols_configs) == 1:
        axes = axes.reshape(-1, 1)
    
    # Generate data and create visualizations for each cell in the grid
    for i, row_config in enumerate(rows_configs):
        row_value = row_config.get(row_param_name, "N/A")
        
        for j, col_config in enumerate(cols_configs):
            col_value = col_config.get(col_param_name, "N/A")
            
            # Merge configs
            cell_config = {**row_config, **col_config}
            
            # Generate data using provided function
            result = data_gen_fn(cell_config)
            
            # Assume result contains points, centers, labels
            if isinstance(result, tuple) and len(result) >= 3:
                points, centers, labels = result[:3]
            else:
                points = result
                centers = None
                labels = None
            
            # Convert to numpy if needed
            if isinstance(points, torch.Tensor):
                points = points.numpy()
            if centers is not None and isinstance(centers, torch.Tensor):
                centers = centers.numpy()
            if labels is not None and isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            
            # Get current axis
            ax = axes[i, j]
            
            # Plot data
            visualize_gmm_data(
                points=points,
                centers=centers if plot_centers else None,
                labels=labels,
                ax=ax,
                colormap=colormap,
                title=f"{row_param_name}={row_value}\n{col_param_name}={col_value}"
            )
            
            # Add row/column labels for outer cells
            if j == 0:
                ax.set_ylabel(f"{row_param_name}={row_value}")
            if i == len(rows_configs) - 1:
                ax.set_xlabel(f"{col_param_name}={col_value}")
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
    
    return fig


def create_training_plots(
    history: Dict[str, List[float]],
    output_dir: Union[str, Path],
    dpi: int = 150,
    formats: List[str] = ["png"]
) -> List[Path]:
    """
    Create plots of training history.
    
    Args:
        history: Training history dictionary
        output_dir: Directory to save plots
        dpi: DPI for output images
        formats: List of output formats ('png', 'pdf', 'svg')
        
    Returns:
        List of paths to saved plots
    """
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = []
    
    # Create loss plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = list(range(1, len(history['train_loss']) + 1))
    
    # Plot training loss
    ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    
    # Plot validation loss if available
    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save in all requested formats
    for fmt in formats:
        output_path = output_dir / f"loss_plot.{fmt}"
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        saved_paths.append(output_path)
    
    plt.close(fig)
    
    # Create learning rate plot if available
    if 'learning_rates' in history and history['learning_rates']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(epochs, history['learning_rates'], 'g-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        
        # Use log scale for y-axis
        ax.set_yscale('log')
        
        # Save in all requested formats
        for fmt in formats:
            output_path = output_dir / f"lr_plot.{fmt}"
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            saved_paths.append(output_path)
        
        plt.close(fig)
    
    # Create additional metric plots if available
    if 'val_metrics' in history and history['val_metrics']:
        # Collect metrics across epochs
        metric_values = {}
        
        for epoch_metrics in history['val_metrics']:
            for metric_name, metric_value in epoch_metrics.items():
                if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool):
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append(metric_value)
        
        # Create plot for each metric
        for metric_name, values in metric_values.items():
            if metric_name in ['loss', 'time', 'step_losses'] or len(values) < 2:
                continue
                
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metric_epochs = list(range(1, len(values) + 1))
            ax.plot(metric_epochs, values, 'g-')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.set_title(f'{metric_name.replace("_", " ").title()} vs. Epoch')
            ax.grid(True, alpha=0.3)
            
            # Save in all requested formats
            for fmt in formats:
                output_path = output_dir / f"{metric_name}_plot.{fmt}"
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
                saved_paths.append(output_path)
            
            plt.close(fig)
    
    logger.info(f"Training plots saved to {output_dir}")
    return saved_paths


def _add_metrics_summary(fig, metrics):
    """
    Add metrics summary box to the figure showing all model/k-means ratios.
    
    Args:
        fig: Matplotlib figure to add metrics to
        metrics: Dictionary of metrics
    """
    # Default title
    metrics_text = "Model/K-means Ratios:\n"
    has_ratios = False
    
    # Check for structured metrics
    if "model_metrics" in metrics and "kmeans_metrics" in metrics:
        model_metrics = metrics["model_metrics"]
        kmeans_metrics = metrics["kmeans_metrics"]
        
        for metric_name in model_metrics:
            if metric_name in kmeans_metrics and kmeans_metrics[metric_name] != 0:
                ratio = model_metrics[metric_name] / kmeans_metrics[metric_name]
                metrics_text += f"{metric_name}: {ratio:.4f}\n"
                has_ratios = True
    
    # If no structured metrics, look for ratio metrics directly
    elif any(k.endswith("_ratio") for k in metrics.keys()):
        # Find all ratio metrics without filtering
        ratio_keys = [k for k in metrics.keys() if k.endswith("_ratio") 
                     and isinstance(metrics[k], (int, float))]
        
        # Sort them for consistent display order
        ratio_keys.sort()
        
        for ratio_key in ratio_keys:
            # Format the metric name for better readability
            display_name = ratio_key.replace("_ratio", "")
            if display_name == "wasserstein_exact_scipy":
                display_name = "wasserstein"
            
            metrics_text += f"{display_name}: {metrics[ratio_key]:.4f}\n"
            has_ratios = True
    
    # Only add the text box if we have ratios to show
    if has_ratios:
        # Apply tight layout first to arrange the plots
        fig.tight_layout()
        
        # Adjust figure size to add space at the bottom for the metrics box
        fig_size = fig.get_size_inches()
        fig.set_size_inches(fig_size[0], fig_size[1] + 0.8)  # Add space at the bottom
        
        # Create a new axes for the metrics box at the bottom
        metrics_ax = fig.add_axes([0.1, 0.01, 0.8, 0.06])  # [left, bottom, width, height]
        metrics_ax.axis('off')  # Hide axes
        
        # Add metrics text centered in the new axes
        metrics_ax.text(0.5, 0.5, metrics_text, 
                        ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, pad=0.5),
                        fontsize=10,
                        transform=metrics_ax.transAxes)


def create_model_only_grid(
    visualization_data: List[Dict[str, torch.Tensor]],
    output_path: str,
    data_snr_labels: List[str] = ["5dB", "15dB"],
    model_snr_labels: List[str] = ["5dB", "15dB"],
    dpi: int = 200,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None
):
    """
    Create a 2x2 grid of 'True Centers vs Model Predictions' plots (no K-means),
    coloring points by true clusters if available, with each subplot labeled by SNR.
    All subplots are always square and the same size.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm

    S = 5  # Size of each subplot in inches
    fig, axes = plt.subplots(2, 2, figsize=(2 * S, 2 * S))
    for i, data in enumerate(visualization_data):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        input_data = data["inputs"].numpy()
        prediction_data = data["outputs"].numpy()
        true_labels = data.get("true_labels", None)
        if true_labels is not None:
            true_labels = true_labels.numpy() if hasattr(true_labels, 'numpy') else true_labels
        if input_data.shape[-1] > 2:
            input_data = input_data[..., :2]
            prediction_data = prediction_data[..., :2]
        # Color mapping by true clusters
        if true_labels is not None:
            if np.issubdtype(true_labels.dtype, np.floating):
                cluster_indices = true_labels.astype(int)
            else:
                cluster_indices = true_labels
            n_clusters = len(np.unique(cluster_indices))
            base_colors = cm.get_cmap('tab10', 10)(range(min(n_clusters, 10)))
            if n_clusters > 10:
                extra_colors = cm.get_cmap('Set3', n_clusters - 10)(range(n_clusters - 10))
                base_colors = np.vstack([base_colors, extra_colors])
        else:
            cluster_indices = np.zeros(len(input_data), dtype=int)
            n_clusters = 1
            base_colors = np.array([[0.7, 0.7, 0.7, 1.0]])
        for cluster_idx in range(n_clusters):
            mask = cluster_indices == cluster_idx
            if np.any(mask):
                ax.scatter(
                    input_data[mask, 0],
                    input_data[mask, 1],
                    alpha=0.4,
                    s=10,  # Smaller points
                    color=base_colors[cluster_idx % len(base_colors)]
                )
        if len(prediction_data.shape) == 2:
            for j in range(len(prediction_data)):
                ax.plot(
                    prediction_data[j, 0],
                    prediction_data[j, 1],
                    'o',
                    color='blue',
                    markersize=10,
                    markeredgewidth=2,
                    markeredgecolor='black',
                    label='Model Predictions' if j == 0 else ""
                )
        # Do NOT plot true centers (no green X)
        ax.set_aspect('equal', adjustable='box', anchor='C')
        ax.set_xticks([])
        ax.set_yticks([])
        if x_min is not None and x_max is not None:
            ax.set_xlim(x_min, x_max)
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        # Label each subplot with both SNRs
        # Label each subplot with both SNRs (single line)
        ax.set_title(f"Data SNR: {data_snr_labels[row]}; Model SNR: {model_snr_labels[col]}", fontsize=14)


    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.08, hspace=0.08)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def create_evaluation_grid(
    visualization_data: List[Dict[str, torch.Tensor]],
    output_dir: Union[str, Path],
    filename: str = "evaluation.png",
    compare_with_kmeans: bool = True,
    metrics: Optional[Dict[str, Any]] = None,
    show_metrics_box: bool = True,
    dpi: int = 300,
    color_by_true_cluster: bool = True,
    titles: Optional[List[str]] = None
) -> Path:
    """
    Create a grid of visualizations for model evaluation with vertical layout.
    
    Args:
        visualization_data: List of data dictionaries with inputs, outputs, targets
        output_dir: Directory to save visualizations
        filename: Filename for the saved visualization
        compare_with_kmeans: Whether to include K-means baseline
        metrics: Optional metrics to display
        show_metrics_box: Whether to show metrics summary
        dpi: DPI for saved image
        color_by_true_cluster: Whether to color points by their true cluster assignment
        titles: Optional list of subplot titles (one per grid cell)
        
    Returns:
        Path to saved visualization file
    """
    # Ensure output directory exists
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a grid layout with exactly 2 columns
    num_samples = len(visualization_data)
    
    # Fixed 2 columns, calculate rows needed
    grid_cols = 2
    grid_rows = (num_samples + grid_cols - 1) // grid_cols  # Ceiling division
    
    # Create a single figure for all samples with minimal vertical spacing
    fig, axes = plt.subplots(
        grid_rows,
        grid_cols,
        figsize=(16, 5 * grid_rows),  # Width fixed, height further reduced
        squeeze=False
    )
    
    # Ensure axes is 2D
    if len(axes.shape) == 1:
        axes = axes.reshape(1, -1)
    
    # Generate data and create visualizations
    for i, data in enumerate(visualization_data):
        if i >= num_samples:
            logger.warning(f"Only showing {num_samples} samples in visualization")
            break
            
        # Calculate grid position for this sample
        grid_row = i // grid_cols
        grid_col = i % grid_cols
        
        # Get the axes for this sample
        ax = axes[grid_row, grid_col]
            
        # Extract data (handle batch dimension properly)
        input_data = data["inputs"].numpy()
        
        # Handle target format consistently - extract first batch item if needed
        if isinstance(data["targets"], dict) and "centers" in data["targets"]:
            centers = data["targets"]["centers"]
            # Handle shape correctly - ensure we get [n_clusters, dim]
            if isinstance(centers, torch.Tensor) and centers.dim() == 3:  # [batch_size, n_clusters, dim]
                target_data = centers[0].numpy()
            else:
                target_data = centers.numpy()
                
            # Check for explicit true_labels field first (preferred source)
            true_labels = None
            if color_by_true_cluster and "true_labels" in data:
                if isinstance(data["true_labels"], torch.Tensor):
                    true_labels = data["true_labels"].numpy()
                else:
                    true_labels = data["true_labels"]
            # Fall back to labels from targets if true_labels not found
            elif color_by_true_cluster and isinstance(data["targets"], dict) and "labels" in data["targets"]:
                labels = data["targets"]["labels"]
                if isinstance(labels, torch.Tensor):
                    if labels.dim() > 1 and labels.size(0) > 1:  # [batch_size, seq_len]
                        true_labels = labels[0].numpy()
                    else:  # [seq_len]
                        true_labels = labels.numpy()
                else:
                    true_labels = labels
        else:
            target_data = data["targets"].numpy()
            true_labels = None
            
        # Get model predictions - ensure correct shape
        prediction_data = data["outputs"].numpy()
        
        # Calculate K-means baseline if requested
        kmeans_centers = None
        if compare_with_kmeans:
            try:
                from sklearn.cluster import KMeans
                n_clusters = target_data.shape[0]  # Now this is guaranteed to be correct
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(input_data)
                kmeans_centers = kmeans.cluster_centers_
            except Exception as e:
                logger.warning(f"Failed to compute K-means: {e}")
                
        # Log label information for debugging
        if true_labels is not None:
            logger.info(f"Sample {i+1}: Using true labels for coloring, shape: {true_labels.shape}, unique values: {np.unique(true_labels)}")
        else:
            logger.info(f"Sample {i+1}: No true labels found, using default coloring")
            
        # Use visualize_predictions for consistent visualization
        fig_single = visualize_predictions(
            input_data=input_data,
            target_data=target_data,
            prediction_data=prediction_data,
            kmeans_centers=kmeans_centers,
            title=None,  # We'll set the title below
            figsize=(12, 6),
            true_labels=true_labels
        )
        
        # Save the individual visualization to a temporary file
        temp_path = output_dir / f"temp_sample_{i+1}.png"
        fig_single.savefig(temp_path, dpi=dpi, bbox_inches="tight")
        
        # Load the saved image
        img = plt.imread(temp_path)
        
        # Display the image in its grid position
        ax.imshow(img)
        ax.axis('off')
        
        # Set custom title if provided
        if titles is not None and i < len(titles):
            ax.set_title(titles[i], fontsize=12)
        else:
            ax.set_title(f"Sample {data.get('sample_idx', i+1)}", fontsize=12)
        
        # Clean up
        plt.close(fig_single)
        os.remove(temp_path)
    
    # Hide any unused subplots
    for i in range(num_samples, grid_rows * grid_cols):
        grid_row = i // grid_cols
        grid_col = i % grid_cols
        axes[grid_row, grid_col].axis('off')
        axes[grid_row, grid_col].set_visible(False)
    
    # First apply tight layout with very minimal vertical padding
    fig.tight_layout(pad=0.1)
    plt.subplots_adjust(hspace=0.05)  # Significantly reduce vertical space between subplots
    
    # Add metrics summary if requested (will adjust layout internally)
    if metrics and show_metrics_box:
        _add_metrics_summary(fig, metrics)
    
    # Save visualization
    vis_path = output_dir / filename
    fig.savefig(vis_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Created visualization with {num_samples} samples in {grid_rows}x{grid_cols} grid: {vis_path}")
    
    # Return the path to the visualization
    return vis_path