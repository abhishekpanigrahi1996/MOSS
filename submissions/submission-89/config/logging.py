"""
Logging configuration classes for GMM transformer framework.

This module defines the configuration options for logging,
TensorBoard integration, and visualization.
"""

from typing import Literal, Dict, Any, Optional, List, Union
from pydantic import Field, model_validator

from .base import ConfigBase


class LoggingConfig(ConfigBase):
    """
    Configuration for logging and TensorBoard integration.
    Combines general logging settings with TensorBoard functionality.
    """
    
    # General logging settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    log_to_console: bool = Field(
        default=True,
        description="Whether to log to console"
    )
    
    log_to_file: bool = Field(
        default=True,
        description="Whether to log to file"
    )
    
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Format string for log messages"
    )
    
    log_date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Format string for log timestamps"
    )
    
    # TensorBoard settings
    tensorboard_enabled: bool = Field(
        default=True,
        description="Whether to use TensorBoard for logging"
    )
    
    log_every_n_steps: int = Field(
        default=10,
        ge=1,
        description="Log metrics every N training steps"
    )
    
    log_model_graph: bool = Field(
        default=True,
        description="Whether to log the model graph"
    )
    
    log_histograms: bool = Field(
        default=False,
        description="Whether to log parameter histograms"
    )
    
    histogram_every_n_epochs: int = Field(
        default=5,
        ge=1,
        description="Log histograms every N epochs"
    )
    
    flush_secs: int = Field(
        default=120,
        ge=1,
        description="Flush TensorBoard events every N seconds"
    )


class VisualizationConfig(ConfigBase):
    """Configuration for visualizations."""
    
    enabled: bool = Field(
        default=True,
        description="Whether to create visualizations"
    )
    
    max_samples: int = Field(
        default=4,
        ge=1,
        description="Maximum number of samples to visualize"
    )
    
    visualize_every_n_epochs: int = Field(
        default=5,
        ge=1,
        description="Create visualizations every N epochs"
    )
    
    plot_dpi: int = Field(
        default=150,
        ge=72,
        description="DPI for output plots"
    )
    
    plot_formats: List[Literal["png", "pdf", "svg"]] = Field(
        default=["png"],
        description="Output formats for plots"
    )
    
    plot_with_kmeans: bool = Field(
        default=True,
        description="Whether to include K-means baseline in plots"
    )
    
    colormap: str = Field(
        default="viridis",
        description="Colormap for scatter plots"
    )
    
    plot_grid_size: Optional[int] = Field(
        default=4,
        ge=1,
        description="Number of plots in each direction for grid"
    )