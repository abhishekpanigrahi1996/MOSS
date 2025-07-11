"""
GMM Visualization Module
========================

This module combines all visualization functions for GMM data, including:
- Core visualization functions (visualize_gmm_data)
- Style utility functions (set_plotting_style, format_axis_with_grid)
- Calculation utilities (calculate_global_axis_limits)
- Helper functions (run_kmeans)

This unified module replaces the separate plots.py and plot_utils.py modules.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from typing import Tuple, Union, List, Dict, Optional
from pathlib import Path

# ---- Style Functions ----

def set_plotting_style():
    """
    Set the global matplotlib plotting style for the tutorial.
    Call this function at the beginning of any script that creates plots.
    """
    # Reset to default styling
    plt.rcdefaults()
    
    # Use common fonts that should be available everywhere
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })
    
def format_axis_with_grid(ax, x_minor_step=1, y_minor_step=0.1):
    """
    Apply consistent grid formatting to an axis.
    
    Args:
        ax: Matplotlib axis object
        x_minor_step: Step size for minor x-axis ticks
        y_minor_step: Step size for minor y-axis ticks
    """
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add minor grid
    ax.xaxis.set_minor_locator(MultipleLocator(x_minor_step))
    ax.yaxis.set_minor_locator(MultipleLocator(y_minor_step))
    ax.grid(which='minor', linestyle=':', alpha=0.4)
    
def format_legend(ax, **kwargs):
    """
    Apply consistent legend formatting.
    
    Args:
        ax: Matplotlib axis object
        **kwargs: Additional arguments to pass to legend()
    """
    legend_kwargs = {
        'frameon': True,
        'fancybox': False,
        'facecolor': 'white',
        'edgecolor': 'black',
        'loc': 'best'
    }
    
    # Update with any user-provided arguments
    legend_kwargs.update(kwargs)
    
    # Apply the legend
    ax.legend(**legend_kwargs)
    
def save_figure(fig, save_path, dpi=300, tight_layout=True):
    """
    Save a figure with consistent settings.
    
    Args:
        fig: Matplotlib figure object
        save_path: Path to save the figure
        dpi: DPI for the saved figure
        tight_layout: Whether to apply tight_layout before saving
    """
    if tight_layout:
        fig.tight_layout()
        
    fig.savefig(save_path, dpi=dpi)
    print(f"Figure saved to {save_path}")
    
def create_comparison_figure(n_plots=2, figsize=(12, 4), dpi=300):
    """
    Create a figure for comparison plots.
    
    Args:
        n_plots: Number of subplots
        figsize: Figure size
        dpi: DPI for the figure
        
    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    fig, axes = plt.subplots(1, n_plots, figsize=figsize, dpi=dpi)
    return fig, axes


# ---- Utility Functions ----

def save_animation(anim, save_path, fps=20, dpi=200):
    """
    Save matplotlib animation with optimized GIF creation.
    
    Args:
        anim: Matplotlib animation object
        save_path: Path to save the animation
        fps: Frames per second
        dpi: DPI for the animation
        
    Returns:
        Path: The actual path where the animation was saved
    """
    save_path = Path(save_path)
    
    # Try MP4 first
    if save_path.suffix.lower() == '.mp4':
        try:
            # Configure ffmpeg writer for better performance
            from matplotlib import animation
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='GMM-Tutorial'), bitrate=1800)
            
            print(f"Saving MP4 with ffmpeg (fps={fps}, dpi={dpi})...")
            anim.save(str(save_path), writer=writer, dpi=dpi, savefig_kwargs={'bbox_inches': 'tight'})
            print(f"✓ Animation saved as MP4: {save_path}")
            return save_path
        except Exception as e:
            print(f"MP4 save failed: {e}")
            print("Falling back to optimized GIF format...")
    
    # Use fast GIF creation method
    gif_path = save_path.with_suffix('.gif')
    
    try:
        # Method 1: Fast imageio-based GIF creation (fastest)
        import imageio
        import matplotlib.pyplot as plt
        from io import BytesIO
        import PIL.Image
        import numpy as np
        
        print(f"Creating optimized GIF (fps={fps}, dpi={dpi})...")
        print("Using fast imageio method...")
        
        # Generate frames quickly with reduced DPI for speed
        frames = []
        num_frames = anim.save_count if hasattr(anim, 'save_count') else 100
        
        for i in range(num_frames):
            # Update animation to frame i
            anim._draw_frame(i)
            
            # Render to buffer with optimized settings
            buf = BytesIO()
            anim._fig.savefig(
                buf, 
                format='png', 
                dpi=dpi//2,  # Half DPI for 4x speed improvement
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.1
            )
            buf.seek(0)
            
            # Convert to RGB array (faster than RGBA)
            img = PIL.Image.open(buf)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            frames.append(np.array(img))
            buf.close()
            
            # Progress indicator for long animations
            if i % max(1, num_frames//10) == 0:
                print(f"  Rendered {i+1}/{num_frames} frames...")
        
        # Create GIF with imageio (much faster than matplotlib)
        duration = 1.0 / fps
        print(f"Saving GIF with {len(frames)} frames...")
        imageio.mimsave(
            str(gif_path), 
            frames, 
            duration=duration, 
            loop=0,
            optimize=True,  # Optimize for smaller file size
            fps=fps
        )
        
        print(f"✓ Fast GIF created: {gif_path}")
        return gif_path
        
    except Exception as e:
        print(f"Fast imageio method failed: {e}")
        
        try:
            # Method 2: Matplotlib ffmpeg GIF writer (medium speed)
            print("Trying matplotlib ffmpeg GIF writer...")
            
            # Use ffmpeg for GIF creation
            from matplotlib import animation
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='GMM-Tutorial'))
            
            anim.save(str(gif_path), writer=writer, dpi=dpi)
            print(f"✓ FFmpeg GIF created: {gif_path}")
            return gif_path
            
        except Exception as e2:
            print(f"FFmpeg GIF method failed: {e2}")
            
            # Method 3: Matplotlib pillow writer (slowest but most compatible)
            print("Falling back to pillow writer (this may be slow)...")
            
            # Optimize pillow settings for speed
            anim.save(
                str(gif_path), 
                writer='pillow', 
                fps=max(1, fps//2),  # Even slower FPS for pillow
                dpi=dpi//2,  # Lower DPI for speed
                savefig_kwargs={'bbox_inches': 'tight', 'facecolor': 'white'}
            )
            print(f"✓ Pillow GIF created: {gif_path}")
            return gif_path


def _calculate_axis_limits(ax, points, predictions=None, true_centers=None, kmeans_centers=None):
    """
    Calculate and set appropriate axis limits with padding.
    
    Args:
        ax: Matplotlib axis
        points: Data points
        predictions: Optional model predictions
        true_centers: Optional true centers
        kmeans_centers: Optional KMeans centers
    """
    # Collect all data points, centers and predictions
    all_points = []
    
    # Add input points
    all_points.append(points)
    
    # Add centers if provided
    if true_centers is not None:
        all_points.append(true_centers)
    
    # Add KMeans centers if provided
    if kmeans_centers is not None:
        all_points.append(kmeans_centers)
    
    # Add predictions if provided
    if predictions is not None:
        all_points.append(predictions)
    
    # Combine all points into a single array
    all_points = np.vstack(all_points)
    
    # Calculate min and max for x and y
    x_min, y_min = np.min(all_points, axis=0)
    x_max, y_max = np.max(all_points, axis=0)
    
    # Calculate ranges with a minimum threshold to prevent zero-width issues
    min_range_size = 0.05
    x_range = max(x_max - x_min, min_range_size)
    y_range = max(y_max - y_min, min_range_size)
    
    # Find the center of the data
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    
    # Make plot square by using the larger range for both axes
    max_range = max(x_range, y_range)
    
    # Add padding (10% on each side)
    padding = 0.1
    padded_range = max_range * (1 + 2 * padding)
    half_range = padded_range / 2
    
    # Set axis limits centered on the data
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)


def calculate_global_axis_limits(points_list, centers_list=None, predictions_list=None, kmeans_centers_list=None, padding=0.1):
    """
    Calculate global axis limits that will work for all data in a slider visualization.
    
    Args:
        points_list: List of point arrays to include in limit calculation
        centers_list: Optional list of center arrays
        predictions_list: Optional list of prediction arrays
        kmeans_centers_list: Optional list of KMeans center arrays
        padding: Padding factor (0.1 = 10% padding on each side)
        
    Returns:
        xlim, ylim: Tuple of axis limits (xmin, xmax), (ymin, ymax)
    """
    # Collect all points into a single array
    all_points = []
    
    # Add all points
    for points in points_list:
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        all_points.append(points)
    
    # Add all centers if provided
    if centers_list:
        for centers in centers_list:
            if centers is not None:
                if isinstance(centers, torch.Tensor):
                    centers = centers.detach().cpu().numpy()
                all_points.append(centers)
    
    # Add all predictions if provided
    if predictions_list:
        for preds in predictions_list:
            if preds is not None:
                if isinstance(preds, torch.Tensor):
                    preds = preds.detach().cpu().numpy()
                all_points.append(preds)
    
    # Add all KMeans centers if provided
    if kmeans_centers_list:
        for kmeans in kmeans_centers_list:
            if kmeans is not None:
                if isinstance(kmeans, torch.Tensor):
                    kmeans = kmeans.detach().cpu().numpy()
                all_points.append(kmeans)
    
    # Combine all points
    combined_points = np.vstack(all_points)
    
    # Calculate min and max for x and y
    x_min, y_min = np.min(combined_points, axis=0)
    x_max, y_max = np.max(combined_points, axis=0)
    
    # Calculate ranges with a minimum threshold
    min_range_size = 0.05
    x_range = max(x_max - x_min, min_range_size)
    y_range = max(y_max - y_min, min_range_size)
    
    # Find the center of the data
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    
    # Make plot square by using the larger range for both axes
    max_range = max(x_range, y_range)
    
    # Add padding
    padded_range = max_range * (1 + 2 * padding)
    half_range = padded_range / 2
    
    # Calculate axis limits centered on the data
    xlim = (x_center - half_range, x_center + half_range)
    ylim = (y_center - half_range, y_center + half_range)
    
    return xlim, ylim


# Note: run_kmeans function has been moved to eval_utils.py


# ---- Core Visualization Functions ----

def visualize_gmm_data(
    points,
    predictions=None,
    true_centers=None,
    kmeans_centers=None,
    point_labels=None,
    title="GMM Data Visualization",
    show_legend=True,
    ax=None,
    figsize=(10, 8),
    # ------------- base design-time sizes (pts) --------------------------------
    point_size=8,
    prediction_size=12,
    center_size=22,
    kmeans_size=15,
    point_alpha=0.6,
    # ---------------------------------------------------------------------------
    save_path=None,
    dpi=300,
    calculate_axis_limits=True,
    xlim=None,
    ylim=None,
    return_artists=False,
    # ---------------------------------------------------------------------------
    size_scale: float = 1.0        # ← NEW: global zoom factor
):
    """
    Plot GMM data and optional model / clustering info.

    All marker areas, line-widths, and *every* font (title, axes labels,
    tick labels, legend text) are multiplied by `size_scale`.  Set
    `size_scale > 1` to zoom-in, `< 1` to zoom-out.
    """
    # --- helpers -------------------------------------------------------------
    _A = lambda base: (base * size_scale) ** 2   # marker *area* (scatter s)
    _L = lambda base: base * size_scale          # linear size (lw, font, pad)

    # --- figure / axis -------------------------------------------------------
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_facecolor("white")

    # convert tensors → numpy
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    # storage for interactive updates
    artists = {}

    # ----------------------------------------------------------------- points
    if point_labels is not None:
        if isinstance(point_labels, torch.Tensor):
            point_labels = point_labels.detach().cpu().numpy()

        unique_labels = np.unique(point_labels)
        n_clusters = len(unique_labels)
        
        # Color generation based on whether predictions exist
        if predictions is not None:
            # Use current code with n_clusters=6 when predictions exist
            forbidden_start, forbidden_end = 0.6, 0.75
            allowed_range = 1 - forbidden_end + forbidden_start
            step = allowed_range / 6  # Fixed to 6 clusters like in InteractiveGMMVisualizer
            hues = []
            for i in range(n_clusters):
                hue = i * step
                if hue >= forbidden_start:
                    hue += (forbidden_end - forbidden_start)
                hue = hue % 1.0
                hues.append(hue)
        else:
            # Use identical code to InteractiveGMMVisualizer when no predictions
            hues = []
            for i in range(n_clusters):
                hue = (i / 6) % 1
                hues.append(hue)
            
        # Convert HSV to RGB with specific saturation and value (only used when predictions exist)
        s = 0.8  # Saturation
        v = 0.9  # Value/Brightness
        
        
        for i, label in enumerate(unique_labels):
            mask = point_labels == label
            # Convert HSV to RGB using matplotlib's color conversion
            color = plt.cm.hsv(hues[i])
            
            # rgb_color = mpl.colors.hsv_to_rgb(hsv_color)[0,0]  # Get RGB values
            rgba = (*color[:3], point_alpha) # Add alpha

            artists[f"cluster_{label}"] = ax.scatter(
                points[mask, 0], points[mask, 1],
                color=rgba,
                s=_A(point_size),
                label=None
            )
    else:
        artists["points"] = ax.scatter(
            points[:, 0], points[:, 1],
            color="blue", alpha=point_alpha,
            s=_A(point_size),
            label="Data Points"
        )

    # ------------------------------------------------------------- predictions
    if predictions is not None:
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()

        artists["predictions"] = ax.scatter(
            predictions[:, 0], predictions[:, 1],
            color="blue", marker="o",
            s=_A(prediction_size),
            linewidths=_L(1.5),
            edgecolors="black",
            label="Model Predictions",
            zorder=5
        )

    # -------------------------------------------------------------- true cent.
    if true_centers is not None:
        if isinstance(true_centers, torch.Tensor):
            true_centers = true_centers.detach().cpu().numpy()

        artists["true_centers"] = ax.scatter(
            true_centers[:, 0], true_centers[:, 1],
            color="lime", marker="*",
            s=_A(center_size),
            linewidths=_L(2),
            edgecolors="black",
            label="True Centers",
            zorder=8
        )

    # ------------------------------------------------------------ k-means cent.
    if kmeans_centers is not None:
        if isinstance(kmeans_centers, torch.Tensor):
            kmeans_centers = kmeans_centers.detach().cpu().numpy()

        artists["kmeans_centers"] = ax.scatter(
            kmeans_centers[:, 0], kmeans_centers[:, 1],
            color="orange", marker="d",
            s=_A(kmeans_size),
            linewidths=_L(2),
            edgecolors="black",
            label="KMeans Centers",
            zorder=10
        )

    # ----------------------------------------------------------------- layout
    if title:
        ax.set_title(title, fontsize=_L(16), pad=_L(10))

    ax.set_aspect("equal", adjustable="box")

    # Fix axis limit logic - handle xlim and ylim independently
    if xlim is not None and ylim is not None:
        # Both limits provided - use them and skip automatic calculation
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    elif xlim is not None:
        # Only xlim provided
        ax.set_xlim(xlim)
        if calculate_axis_limits:
            # Calculate ylim only, excluding predictions
            _calculate_axis_limits(ax, points, None, true_centers, kmeans_centers)
    elif ylim is not None:
        # Only ylim provided  
        ax.set_ylim(ylim)
        if calculate_axis_limits:
            # Calculate xlim only, excluding predictions
            _calculate_axis_limits(ax, points, None, true_centers, kmeans_centers)
    elif calculate_axis_limits:
        # No limits provided - calculate both, excluding predictions
        _calculate_axis_limits(ax, points, None, true_centers, kmeans_centers)

    # grid
    ax.grid(True, linestyle="-", linewidth=_L(0.5),
            color="lightgray", alpha=0.7)

    # axes labels – scale if already set by caller
    ax.xaxis.label.set_fontsize(_L(12))
    ax.yaxis.label.set_fontsize(_L(12))

    # tick labels & tick lines
    ax.tick_params(axis="both", which="major",
                   labelsize=_L(10),
                   width=_L(1),
                   length=_L(4))

    # legend
    if show_legend:
        leg = ax.legend(
            loc="upper right",
            framealpha=0.7,
            edgecolor="lightgray",
            fontsize=_L(10),
            markerscale=1.0   # marker areas already scaled
        )
        if leg is not None:
            leg.get_frame().set_linewidth(_L(0.8))
            for txt in leg.get_texts():
                txt.set_fontsize(_L(10))

    # ----------------------------------------------------------------- output
    if save_path is not None and fig is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return artists if return_artists else (fig, ax)

def data_square_side_pt(ax):
    """
    Return the side length of the square in which points are drawn
    (aspect='equal', adjustable='box') in typographic points (pt).

    Works after the first draw OR inside a draw/resize callback.
    """
    fig = ax.figure
    fig.canvas.draw()                       # make sure layout is final

    # ▸ ax.patch is the full axes rectangle
    # ▸ When aspect='equal', Matplotlib centres the data square
    #   in the patch and uses the *smaller* patch dimension for the square.
    bbox_patch_px = ax.patch.get_window_extent()   # pixels
    side_px       = min(bbox_patch_px.width, bbox_patch_px.height)

    # convert pixels → points   (points = inches × 72 = pixels / dpi × 72)
    return side_px / fig.dpi * 72.0

def create_grid_figure(n_rows: int, n_cols: int, figsize: Tuple[int, int] = (15, 15), dpi: int = 150, sharex: bool = True, sharey: bool = True) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with a grid of subplots.

    Args:
        n_rows: Number of rows in the grid
        n_cols: Number of columns in the grid
        figsize: Figure size as (width, height) in inches
        dpi: DPI for the figure
        sharex: Whether to share x-axis limits across subplots
        sharey: Whether to share y-axis limits across subplots

    Returns:
        fig, axes: Matplotlib figure and 2D array of axes
    """
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=dpi, sharex=sharex, sharey=sharey)
    # Ensure axes is 2D even for a single plot
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    return fig, axes

def hide_unused_subplots(axes: np.ndarray, n_used: int) -> None:
    """
    Hide any unused subplots in a grid.

    Args:
        axes: 2D array of axes
        n_used: Number of subplots that are actually used
    """
    n_total = axes.size
    for i in range(n_used, n_total):
        row = i // axes.shape[1]
        col = i % axes.shape[1]
        axes[row, col].axis('off')
        axes[row, col].set_visible(False)

def add_metrics_to_title(ax: plt.Axes, metrics: Dict[str, float], title: str = None, fontsize: int = 10) -> None:
    """
    Add metrics to a subplot title.

    Args:
        ax: Matplotlib axis
        metrics: Dictionary of metric names and values
        title: Optional base title
        fontsize: Font size for the title
    """
    if not metrics:
        if title:
            ax.set_title(title, fontsize=fontsize)
        return

    # Format metrics as strings
    metric_strs = [f"{k}: {v:.3f}" for k, v in metrics.items()]
    metrics_text = "\n".join(metric_strs)

    # Combine with title if provided
    if title:
        full_title = f"{title}\n{metrics_text}"
    else:
        full_title = metrics_text

    ax.set_title(full_title, fontsize=fontsize)

def create_comparison_grid(
    results: Union[Dict, List[Dict]],
    layout: str = '2x2',
    show_predictions: bool = True,
    show_kmeans: bool = True,
    titles: Optional[List[str]] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
    metrics: Optional[List[str]] = None,
    return_artists: bool = False,
    shared_axis_limits: bool = False,  # NEW: default False for dataset comparison
    show_legend: bool = True,  # NEW: allow disabling legend
    size_scale: float = 1.0,  # NEW: size scaling support
    verbose: bool = True  # NEW: control warning messages
) -> Union[Tuple[plt.Figure, np.ndarray], Tuple[plt.Figure, np.ndarray, Dict]]:
    """
    Create a grid of GMM visualizations from evaluation results.

    Args:
        results: Single evaluation dict or list of dicts (one per subplot)
        layout: Grid layout as 'NxM' string (e.g., '2x2', '3x3')
        show_predictions: Whether to show model predictions
        show_kmeans: Whether to show KMeans centers
        titles: Optional list of titles (one per subplot)
        figsize: Optional (width, height) in inches
        dpi: Figure resolution
        save_path: Optional path to save figure
        metrics: Optional list of metrics to display in titles
        return_artists: Whether to return artists dict for interactive updates
        shared_axis_limits: If True, use the same axis limits for all subplots (default: False)
        show_legend: Whether to show legend in each subplot (default: True)
        size_scale: Global size scaling factor
        verbose: Whether to print warning messages (default: True)

    Returns:
        fig, axes: Matplotlib figure and array of axes
        artists: Optional dict of artists for interactive updates
    """
    # Parse layout
    n_rows, n_cols = map(int, layout.split('x'))
    n_plots = n_rows * n_cols

    # Convert single result to list
    if isinstance(results, dict):
        results = [results]

    # Ensure we have enough results
    if len(results) > n_plots:
        if verbose:
            print(f"Warning: More results ({len(results)}) than plots ({n_plots}). Only showing first {n_plots}.")
        results = results[:n_plots]

    # Create figure
    if figsize is None:
        figsize = (5 * n_cols, 5 * n_rows)
    fig, axes = create_grid_figure(
        n_rows, n_cols, figsize=figsize, dpi=dpi,
        sharex=shared_axis_limits, sharey=shared_axis_limits
    )

    # Store artists for interactive updates if requested
    artists_dict = {}

    # Compute global axis limits if requested
    xlim = ylim = None
    if shared_axis_limits and len(results) > 0:
        points_list = [r['inputs'][0] for r in results]
        centers_list = [r['targets']['centers'][0] for r in results if 'targets' in r and 'centers' in r['targets']]
        predictions_list = [r['predictions'][0] for r in results if show_predictions and 'predictions' in r]
        kmeans_centers_list = [r['kmeans_results']['centers'][0] for r in results if show_kmeans and 'kmeans_results' in r]
        xlim, ylim = calculate_global_axis_limits(points_list, centers_list, predictions_list, kmeans_centers_list)

    # Create visualizations
    for i, result in enumerate(results):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Extract data
        points = result['inputs'][0]  # Remove batch dimension
        predictions = result['predictions'][0] if show_predictions and 'predictions' in result else None
        true_centers = result['targets']['centers'][0] if 'targets' in result and 'centers' in result['targets'] else None
        kmeans_centers = result['kmeans_results']['centers'][0] if show_kmeans and 'kmeans_results' in result else None
        point_labels = result['targets']['labels'][0] if 'targets' in result and 'labels' in result['targets'] else None

        # Get title
        title = titles[i] if titles and i < len(titles) else None

        # Get metrics if requested
        metrics_dict = None
        if metrics and 'metrics' in result:
            metrics_dict = {m: result['metrics'][m].item() for m in metrics if m in result['metrics']}

        # Create visualization
        if return_artists:
            artists = visualize_gmm_data(
                points=points,
                predictions=predictions,
                true_centers=true_centers,
                kmeans_centers=kmeans_centers,
                point_labels=point_labels,
                title=title,
                ax=ax,
                xlim=xlim if shared_axis_limits else None,
                ylim=ylim if shared_axis_limits else None,
                return_artists=True,
                show_legend=show_legend,
                size_scale=size_scale
            )
            artists_dict[f'plot_{i}'] = artists
        else:
            visualize_gmm_data(
                points=points,
                predictions=predictions,
                true_centers=true_centers,
                kmeans_centers=kmeans_centers,
                point_labels=point_labels,
                title=title,
                ax=ax,
                xlim=xlim if shared_axis_limits else None,
                ylim=ylim if shared_axis_limits else None,
                show_legend=show_legend,
                size_scale=size_scale
            )

        # Add metrics to title if provided
        if metrics_dict:
            add_metrics_to_title(ax, metrics_dict, title)

    # Hide unused subplots
    hide_unused_subplots(axes, len(results))

    # Adjust layout
    fig.tight_layout()

    # Save if path provided
    if save_path:
        save_figure(fig, save_path, dpi=dpi)

    return (fig, axes, artists_dict) if return_artists else (fig, axes)