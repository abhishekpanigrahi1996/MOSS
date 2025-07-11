"""
GMM Visualization Module - Unified API
======================================

This module provides both the legacy visualization functions and the new high-level API.
It maintains backward compatibility while offering the new declarative interface.

Legacy functions (from visualization_legacy.py):
- visualize_gmm_data() - Core plotting function
- set_plotting_style() - Style configuration
- create_comparison_grid() - Grid layouts
- All other existing functions

New high-level API:
- VisualizationPipeline - Main class for declarative visualizations
- scatter_plot() - Convenience function for direct replacement
"""

# Import all legacy functions for backward compatibility
from .visualization_legacy import *

# Import the new high-level API
from .visualization_pipeline import (
    VisualizationPipeline,
    scatter_plot as scatter_plot_v2
)

# Re-export key components for convenience
__all__ = [
    # Legacy functions (imported via * from visualization_legacy)
    'visualize_gmm_data',
    'set_plotting_style', 
    'save_figure',
    'create_comparison_grid',
    'create_comparison_figure',
    'format_axis_with_grid',
    'format_legend',
    'save_animation',
    'calculate_global_axis_limits',
    'create_grid_figure',
    'hide_unused_subplots',
    'add_metrics_to_title',
    'data_square_side_pt',
    
    # New API
    'VisualizationPipeline',
    'scatter_plot_v2',
]

# Version information
__version__ = "2.0.0"
__legacy_version__ = "1.0.0"

def get_version_info():
    """Get information about available visualization APIs."""
    return {
        'current_version': __version__,
        'legacy_version': __legacy_version__,
        'features': {
            'legacy_api': 'All original functions available',
            'new_api': 'VisualizationPipeline with declarative interface',
            'single_sample_support': 'Direct replacement for visualize_gmm_data',
            'format_agnostic': 'Static, interactive, and animation from same API',
            'progressive_enhancement': 'Start simple, add complexity without API changes'
        }
    }


# Optional: provide a direct drop-in replacement function
def visualize_gmm_data_v2(
    points,
    predictions=None,
    true_centers=None,
    kmeans_centers=None,
    point_labels=None,
    title="GMM Data Visualization",
    show_legend=True,
    figsize=(10, 8),
    save_path=None,
    experiment_dir=None,
    **kwargs
):
    """
    Drop-in replacement for visualize_gmm_data using the new API.
    
    This function provides the exact same interface as the legacy visualize_gmm_data
    but uses the new pipeline internally for consistency.
    
    Args:
        Same as legacy visualize_gmm_data function
        experiment_dir: Base directory containing model experiments
        
    Returns:
        matplotlib.Figure (note: returns fig only, not (fig, ax) tuple)
    """
    # Build data dictionary
    data = {'points': points}
    show = ['points']
    
    if true_centers is not None:
        data['centers'] = true_centers
        show.append('true_centers')
        
    if point_labels is not None:
        data['labels'] = point_labels
        
    if predictions is not None:
        data['predictions'] = predictions
        show.append('predictions')
        
    if kmeans_centers is not None:
        data['kmeans_centers'] = kmeans_centers
        show.append('kmeans')
    
    # Use new API
    pipeline = VisualizationPipeline(experiment_dir=experiment_dir)
    fig = pipeline.scatter_plot(
        data=data,
        show=show,
        titles=title,
        figsize=figsize,
        save_path=save_path,
        **kwargs
    )
    
    return fig


if __name__ == "__main__":
    # Show version info and migration guide when run directly
    print("GMM Visualization Module")
    print("=" * 40)
    
    version_info = get_version_info()
    print(f"Version: {version_info['current_version']}")
    print("\nAvailable APIs:")
    for feature, description in version_info['features'].items():
        print(f"  • {feature}: {description}")
    
    print("\nTo see migration guide, run:")
    print("  from tutorial.src.visualization import migration_guide")
    print("  migration_guide()")