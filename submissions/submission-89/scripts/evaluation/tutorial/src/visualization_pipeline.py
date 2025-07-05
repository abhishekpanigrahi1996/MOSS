"""
Visualization Pipeline - High-Level API for GMM Visualizations
==============================================================

This module provides a high-level, declarative API for creating GMM visualizations
that works seamlessly with single samples, datasets, and parameter sweeps.

Key features:
- Single sample support (direct replacement for visualize_gmm_data)
- Model evaluation integration (Phase 2)
- Dataset generation support (Phase 2)
- Parameter sweep grids (Phase 2)
- KMeans integration (Phase 2)
- Format-agnostic output (static, interactive, animation)
- Progressive enhancement from simple to complex visualizations
- Backward compatible with existing workflow
"""

import sys
import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple, Any
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.figure import Figure

import copy
from tqdm import tqdm

# Import legacy functions to build upon
from .visualization_legacy import (
    visualize_gmm_data,
    set_plotting_style,
    save_figure,
    create_comparison_grid,
    calculate_global_axis_limits
)
from .io import load_model_from_experiment, create_data_samples
from .eval_utils import evaluate, run_kmeans


class Animation:
    """
    Format-agnostic animation container for GMM visualizations.
    
    Supports multiple output formats from the same animation data:
    - MP4 video files (requires ffmpeg)
    - GIF files (uses pillow)
    - Frame sequences (individual PNG files)
    - Interactive slider widgets
    """
    
    def __init__(self, fig, anim_func, frames, fps=20, interval=50, verbose=False):
        """
        Initialize animation container.
        
        Args:
            fig: matplotlib Figure
            anim_func: Animation update function
            frames: Number of frames or frame data
            fps: Frames per second for video output
            interval: Milliseconds between frames for display
            verbose: Whether to show progress messages
        """
        self.fig = fig
        self.anim_func = anim_func
        self.frames = frames
        self.fps = fps
        self.interval = interval
        self.verbose = verbose
        self._animation = None
    
    def _create_animation(self):
        """Create the matplotlib animation object if not already created."""
        if self._animation is None:
            self._animation = animation.FuncAnimation(
                self.fig, self.anim_func, frames=len(self.frames), 
                interval=self.interval, blit=False
            )
        return self._animation
    
    def save_mp4(self, path: Union[str, Path], fps: Optional[int] = None, dpi: int = 150) -> Path:
        """
        Save animation as MP4 file with optimized performance.
        
        Args:
            path: Output file path
            fps: Frames per second (uses default if None)
            dpi: Resolution for output (reduced default for speed)
            
        Returns:
            Path to saved file
        """
        path = Path(path).with_suffix('.mp4')
        fps = fps or self.fps
        
        try:
            # Create matplotlib animation with optimized settings
            anim = self._create_animation()
            
            # Configure ffmpeg writer for better performance
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='GMM-Tutorial'), bitrate=1800)
            
            if hasattr(self, 'verbose') and self.verbose:
                print(f"Saving MP4 with ffmpeg (fps={fps}, dpi={dpi})...")
            anim.save(str(path), writer=writer, dpi=dpi, savefig_kwargs={'bbox_inches': 'tight'})
            if hasattr(self, 'verbose') and self.verbose:
                print(f"Animation saved as MP4: {path}")
            return path
            
        except Exception as e:
            # If ffmpeg fails, try to save as GIF instead
            if hasattr(self, 'verbose') and self.verbose:
                print(f"MP4 save failed: {e}")
            if hasattr(self, 'verbose') and self.verbose:
                print("Falling back to optimized GIF format...")
            gif_path = path.with_suffix('.gif')
            return self.save_gif(gif_path, fps=fps, dpi=dpi)
    
    def save_gif(self, path: Union[str, Path], fps: Optional[int] = None, dpi: int = 100, fast_mode: bool = True) -> Path:
        """
        Save animation as GIF file using the fastest available method.
        
        Args:
            path: Output file path  
            fps: Frames per second (uses default if None)
            dpi: Resolution for output
            fast_mode: If True, uses optimized settings for speed (lower DPI, faster encoding)
            
        Returns:
            Path to saved file
        """
        path = Path(path).with_suffix('.gif')
        fps = fps or max(1, self.fps // 2)  # Use slower fps for GIF
        
        if self.verbose:
            print(f"🎬 Creating optimized GIF (fps={fps}, dpi={dpi})...")
            import time
            gif_start_time = time.time()
        
        try:
            # Method 1: Fast imageio-based GIF creation (fastest)
            import imageio
            import matplotlib.pyplot as plt
            from io import BytesIO
            import PIL.Image
            
            if self.verbose:
                print("📸 Using fast imageio method...")
                render_start_time = time.time()
            
            # Generate frames quickly with reduced DPI for speed
            frames = []
            # Use tqdm for progress if verbose
            from tqdm import tqdm
            iterator = tqdm(range(len(self.frames)), desc="Rendering frames") if self.verbose else range(len(self.frames))
            for i in iterator:
                # Update plot
                self.anim_func(i)
                
                # Render to buffer with optimized settings
                buf = BytesIO()
                # Use optimized DPI for speed if fast_mode is enabled
                render_dpi = dpi // 2 if fast_mode else dpi
                self.fig.savefig(
                    buf, 
                    format='png', 
                    dpi=render_dpi,  # Half DPI for 4x speed improvement
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
            
            if self.verbose:
                render_time = time.time() - render_start_time
                print(f"✅ Frame rendering completed in {render_time:.2f}s ({render_time/len(self.frames):.3f}s per frame)")
                
            
            # Create GIF with imageio (much faster than matplotlib)
            duration = 1.0 / fps
            if self.verbose:
                print(f"💾 Encoding GIF with {len(frames)} frames...")
                encode_start_time = time.time()
            imageio.mimsave(
                str(path), 
                frames, 
                duration=duration, 
                loop=0,
                optimize=True,  # Optimize for smaller file size
                fps=fps
            )
            
            if self.verbose:
                encode_time = time.time() - encode_start_time
                total_time = time.time() - gif_start_time
                print(f"✅ GIF encoding completed in {encode_time:.2f}s")
                print(f"🎯 Total GIF creation time: {total_time:.2f}s")
                print(f"📁 Fast GIF created: {path}")
            return path
            
        except Exception as e:
            if self.verbose:
                print(f"Fast imageio method failed: {e}")
            
            try:
                # Method 2: Matplotlib ffmpeg GIF writer (medium speed)
                if self.verbose:
                    print("Trying matplotlib ffmpeg GIF writer...")
                anim = self._create_animation()
                
                # Use ffmpeg for GIF creation
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=fps, metadata=dict(artist='GMM-Tutorial'))
                
                anim.save(str(path), writer=writer, dpi=dpi)
                if hasattr(self, 'verbose') and self.verbose:
                    print(f"FFmpeg GIF created: {path}")
                return path
                
            except Exception as e2:
                if self.verbose:
                    print(f"FFmpeg GIF method failed: {e2}")
                
                try:
                    # Method 3: Matplotlib pillow writer (slowest but most compatible)
                    if self.verbose:
                        print("Falling back to pillow writer (this may be slow)...")
                    anim = self._create_animation()
                    
                    # Optimize pillow settings for speed
                    anim.save(
                        str(path), 
                        writer='pillow', 
                        fps=max(1, fps//2),  # Even slower FPS for pillow
                        dpi=dpi//2,  # Lower DPI for speed
                        savefig_kwargs={'bbox_inches': 'tight', 'facecolor': 'white'}
                    )
                    if hasattr(self, 'verbose') and self.verbose:
                        print(f"Pillow GIF created: {path}")
                    return path
                    
                except Exception as e3:
                    # If all methods fail
                    raise RuntimeError(f"All GIF creation methods failed: imageio({e}), ffmpeg({e2}), pillow({e3})")
    
    def save_frames(self, dir_path: Union[str, Path], prefix: str = "frame", 
                   format: str = "png", dpi: int = 150) -> List[Path]:
        """
        Save animation as individual frame images.
        
        Args:
            dir_path: Directory to save frames
            prefix: Filename prefix for frames
            format: Image format (png, jpg, etc.)
            dpi: Resolution for frames
            
        Returns:
            List of paths to saved frame files
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        frame_paths = []
        
        for i, frame_data in enumerate(self.frames):
            # Update the plot with frame data
            self.anim_func(i)
            
            # Save this frame
            frame_path = dir_path / f"{prefix}_{i:04d}.{format}"
            self.fig.savefig(frame_path, dpi=dpi, bbox_inches='tight')
            frame_paths.append(frame_path)
        
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Saved {len(frame_paths)} frames to: {dir_path}")
        return frame_paths
    
    def as_slider(self) -> 'InteractiveWidget':
        """
        Convert animation to interactive slider widget.
        
        Returns:
            InteractiveWidget with slider control
        """
        # Extract parameter values from frame data
        if hasattr(self.frames[0], 'keys') and 'snr_db' in self.frames[0]:
            param_values = [f['snr_db'] for f in self.frames]
            param_name = 'SNR (dB)'
        else:
            param_values = list(range(len(self.frames)))
            param_name = 'Frame'
        
        return InteractiveWidget(
            self.fig, self.anim_func, self.frames,
            param_name=param_name, 
            param_values=param_values
        )
    
    def show(self):
        """Display the animation."""
        anim = self._create_animation()
        plt.show()


class InteractiveWidget:
    """
    Interactive widget for real-time parameter exploration of GMM visualizations.
    
    Provides slider controls for parameters like SNR, flow speed, etc.
    Updates the visualization in real-time as parameters change.
    """
    
    def __init__(self, fig, update_func, frame_data, param_name="Parameter", param_values=None, default_dpi=150):
        """
        Initialize interactive widget.
        
        Args:
            fig: matplotlib Figure
            update_func: Function to update plot (takes frame index)
            frame_data: Data for each frame/parameter value
            param_name: Name for the parameter slider
            param_values: Values for parameter (if None, uses indices)
            default_dpi: Default DPI for saving snapshots
        """
        self.fig = fig
        self.update_func = update_func
        self.frame_data = frame_data
        self.param_name = param_name
        self.param_values = param_values or list(range(len(frame_data)))
        self.default_dpi = default_dpi
        
        # Store original axes position to make room for slider
        self.original_ax_pos = None
        self.slider = None
        self.slider_ax = None
        
        self._setup_slider()
    
    def _setup_slider(self):
        """Setup the parameter slider."""
        # Adjust main plot to make room for slider
        if hasattr(self.fig, 'axes') and len(self.fig.axes) > 0:
            main_ax = self.fig.axes[0]
            
            # Store original position
            self.original_ax_pos = main_ax.get_position()
            
            # Adjust main axes to leave room for slider
            main_ax.set_position([0.1, 0.15, 0.85, 0.80])
            
            # Create slider axes
            self.slider_ax = self.fig.add_axes([0.2, 0.05, 0.6, 0.03])
            
            # Create slider
            val_min = min(self.param_values)
            val_max = max(self.param_values)
            val_init = self.param_values[len(self.param_values)//2]  # Start in middle
            
            self.slider = Slider(
                ax=self.slider_ax,
                label=self.param_name,
                valmin=val_min,
                valmax=val_max,
                valinit=val_init,
                color='dodgerblue'
            )
            
            # Connect slider to update function
            self.slider.on_changed(self._on_slider_change)
            
            # Initial update
            self._on_slider_change(val_init)
    
    def _on_slider_change(self, val):
        """Handle slider value changes."""
        # Find nearest frame
        frame_idx = self._find_nearest_frame(val)
        
        # Update visualization
        self.update_func(frame_idx)
        
        # Redraw
        self.fig.canvas.draw_idle()
    
    def _find_nearest_frame(self, val):
        """Find the frame index closest to the given parameter value."""
        distances = [abs(pval - val) for pval in self.param_values]
        return distances.index(min(distances))
    
    def show(self):
        """Display the interactive widget."""
        plt.show()
    
    def save_snapshot(self, path: Union[str, Path], dpi: Optional[int] = None) -> Path:
        """
        Save current state as image.
        
        Args:
            path: Output file path
            dpi: Resolution for output (uses default_dpi if None)
            
        Returns:
            Path to saved file
        """
        path = Path(path)
        dpi = dpi or self.default_dpi
        self.fig.savefig(path, dpi=dpi, bbox_inches='tight')
        if hasattr(self, 'verbose') and self.verbose:
            if hasattr(self, 'verbose') and self.verbose:
                print(f"Interactive widget snapshot saved: {path}")
        return path


class VisualizationPipeline:
    """
    High-level visualization pipeline for GMM data and model evaluations.
    
    Designed to work seamlessly with:
    - Single samples (direct replacement for visualize_gmm_data)
    - Named datasets 
    - Model evaluations
    - Parameter sweeps
    - Multiple output formats (static, interactive, animation)
    """
    
    def __init__(
        self,
        models: Union[str, List[str], None] = None,
        datasets: Union[str, List[str], None] = None,
        experiment_dir: Union[str, Path, None] = None,
        output_dir: Union[str, Path] = 'scripts/evaluation/tutorial/output',
        device: str = 'auto',
        cache_evaluations: bool = True,
        style_preset: str = 'tutorial',
        verbose: bool = True,
        default_dpi: int = 150
    ):
        """
        Initialize the visualization pipeline.
        
        Args:
            models: Model name(s) to pre-load (None = load on-demand)
            datasets: Dataset name(s) to use (None = generate on-demand)
            experiment_dir: Base directory containing model experiments
            output_dir: Directory for saving outputs
            device: Device for computation ('auto', 'cuda', 'cpu')
            cache_evaluations: Whether to cache model evaluations
            style_preset: Plotting style preset
            verbose: Whether to print progress messages
            default_dpi: Default DPI for saved figures
        """
        self.models = models
        self.datasets = datasets
        self.experiment_dir = Path(experiment_dir) if experiment_dir is not None else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.cache_evaluations = cache_evaluations
        self.style_preset = style_preset
        self.verbose = verbose
        self.default_dpi = default_dpi
        
        # Initialize caches
        self._model_cache = {}
        self._evaluation_cache = {}
        
        # Set plotting style
        set_plotting_style()
        
        if self.verbose:
            print(f"VisualizationPipeline initialized:")
            print(f"  Device: {self.device}")
            print(f"  Experiment dir: {self.experiment_dir}")
            print(f"  Output dir: {self.output_dir}")
            print(f"  Style preset: {self.style_preset}")
    
    def scatter_plot(
        self,
        # === Data Input (flexible) ===
        data: Union[Dict, None] = None,
        models: Union[str, List[str], None] = None,
        datasets: Union[str, List[str], None] = None,
        
        # === What to Display ===
        show: List[str] = ['points', 'true_centers'],
        
        # === Parameter Control ===
        parameter_values: Union[Dict, List, float, int, None] = None,
        parameter_range: Optional[Tuple[float, float]] = None,
        parameter_steps: int = 50,
        num_samples: int = 1,  # NEW: Number of data samples to generate
        
        # === Layout ===
        layout: str = 'auto',
        titles: Union[str, List[str], None] = None,
        figsize: Optional[Tuple[int, int]] = None,
        
        # === Output Control ===
        save_path: Union[str, Path, None] = None,
        output_format: str = 'static',
        
        # === Animation/Interaction ===
        animation_parameter: str = 'snr_db',
        frames: int = 100,
        fps: int = 20,
        interactive_parameters: Optional[Dict[str, Tuple]] = None,
        
        # === Styling ===
        size_scale: float = 1.0,
        shared_axes: bool = False,
        **kwargs
    ) -> Union[Figure, Animation, InteractiveWidget]:
        """
        Create scatter plots showing GMM data with optional model outputs.
        
        This method handles both simple static plots (replacing direct visualize_gmm_data calls)
        and complex parameter sweeps that can be output as grids, animations, or interactive widgets.
        
        Args:
            data: Direct data input as dictionary with keys:
                  'points', 'centers', 'labels', 'predictions', etc.
            models: Model name(s) to evaluate
            datasets: Dataset name(s) to generate data from
            show: Elements to display ['points', 'true_centers', 'predictions', 'kmeans']
            parameter_values: Parameter variation control
            layout: Grid layout ('auto', '1x1', '2x2', etc.)
            output_format: 'static', 'interactive', or 'animation'
            save_path: Path to save output
            
        Returns:
            matplotlib.Figure, Animation, or InteractiveWidget depending on output_format
        """
        # Handle different output formats
        if output_format == 'static':
            return self._create_static_scatter_plot(
                data=data, models=models, datasets=datasets, show=show,
                parameter_values=parameter_values, layout=layout, titles=titles,
                figsize=figsize, save_path=save_path, size_scale=size_scale,
                shared_axes=shared_axes, num_samples=num_samples, **kwargs
            )
        elif output_format == 'interactive':
            kwargs['size_scale'] = size_scale
            return self._create_interactive_scatter_plot(
                data=data, models=models, datasets=datasets, show=show,
                interactive_parameters=interactive_parameters, figsize=figsize, 
                dpi=kwargs.get('dpi', None), **kwargs
            )
        elif output_format == 'animation':
            kwargs['size_scale'] = size_scale
            return self._create_animation_scatter_plot(
                data=data, models=models, datasets=datasets, show=show,
                animation_parameter=animation_parameter, parameter_range=parameter_range,
                frames=frames, fps=fps, save_path=save_path, figsize=figsize, **kwargs
            )
        else:
            raise ValueError(f"Unknown output_format: {output_format}")
    
    def _create_static_scatter_plot(
        self,
        data: Union[Dict, None],
        models: Union[str, List[str], None],
        datasets: Union[str, List[str], None], 
        show: List[str],
        parameter_values: Union[Dict, List, float, int, None],
        layout: str,
        titles: Union[str, List[str], None],
        figsize: Optional[Tuple[int, int]],
        save_path: Union[str, Path, None],
        size_scale: float,
        shared_axes: bool,
        num_samples: int = 1,
        **kwargs
    ) -> Figure:
        """Create static scatter plot(s)."""
        
        # Get evaluation results
        results = self._prepare_evaluation_results(
            data=data, models=models, datasets=datasets, 
            parameter_values=parameter_values, show=show, num_samples=num_samples
        )
        
        # Step 2: Handle single result (direct visualize_gmm_data replacement)
        if len(results) == 1:
            return self._create_single_scatter_plot(
                results[0], show=show, title=titles, figsize=figsize,
                save_path=save_path, size_scale=size_scale, **kwargs
            )
        
        # Step 3: Handle multiple results (grid layout)
        return self._create_scatter_grid(
            results=results, layout=layout, titles=titles, show=show,
            figsize=figsize, save_path=save_path, shared_axes=shared_axes,
            size_scale=size_scale, **kwargs
        )
    
    def _prepare_evaluation_results(
        self,
        data: Union[Dict, None],
        models: Union[str, List[str], None],
        datasets: Union[str, List[str], None],
        parameter_values: Union[Dict, List, float, int, None],
        show: List[str],
        num_samples: int = 1
    ) -> List[Dict]:
        """Prepare evaluation results from various input sources."""
        
        # Priority order: direct data > datasets > default data
        if data is not None:
            # Mode 1: Direct data input (single sample)
            if isinstance(parameter_values, dict) and any(isinstance(v, list) for v in parameter_values.values()):
                # Parameter sweep on single data sample
                return self._process_parameter_sweep(data, models, parameter_values, show)
            else:
                # Single evaluation
                result = self._process_single_data_input(data, models, parameter_values, show)
                return [result]
                
        elif datasets is not None:
            # Mode 2: Dataset-based input 
            return self._process_dataset_input(datasets, models, parameter_values, show, num_samples)
            
        else:
            # Mode 3: Default data generation
            print("No data or datasets specified, generating default data")
            default_data = self._generate_default_data()
            if isinstance(parameter_values, dict) and any(isinstance(v, list) for v in parameter_values.values()):
                return self._process_parameter_sweep(default_data, models, parameter_values, show)
            else:
                result = self._process_single_data_input(default_data, models, parameter_values, show)
                return [result]
    
    def _process_parameter_sweep(
        self,
        data: Dict,
        models: Union[str, List[str], None],
        parameter_values: Dict,
        show: List[str]
    ) -> List[Dict]:
        """Process parameter sweep where parameter_values contains lists."""
        results = []
        
        # Find parameters with multiple values
        param_lists = {}
        param_singles = {}
        
        for key, value in parameter_values.items():
            if isinstance(value, list):
                param_lists[key] = value
            else:
                param_singles[key] = value
        
        # Generate all combinations
        if len(param_lists) == 1:
            # Single parameter with multiple values
            param_name, param_values_list = list(param_lists.items())[0]
            for param_val in param_values_list:
                combined_params = param_singles.copy()
                combined_params[param_name] = param_val
                result = self._process_single_data_input(data, models, combined_params, show)
                results.append(result)
        else:
            # Multiple parameters with multiple values - create cartesian product
            import itertools
            
            param_names = list(param_lists.keys())
            param_value_lists = list(param_lists.values())
            
            for param_combination in itertools.product(*param_value_lists):
                combined_params = param_singles.copy()
                for i, param_name in enumerate(param_names):
                    combined_params[param_name] = param_combination[i]
                    
                result = self._process_single_data_input(data, models, combined_params, show)
                results.append(result)
        
        return results
    
    def _process_single_data_input(
        self, 
        data: Dict, 
        models: Union[str, List[str], None],
        parameter_values: Union[Dict, float, int, None],
        show: List[str]
    ) -> Dict:
        """Process a single data input and return evaluation result."""
        
        # Ensure data tensors are on correct device
        processed_data = self._ensure_tensor_device(data)
        
        # Create basic result structure
        result = {
            'inputs': processed_data['points'].unsqueeze(0),  # Add batch dim
            'targets': {
                'centers': processed_data.get('centers', torch.empty(0, 2)),
                'labels': processed_data.get('labels', torch.empty(0, dtype=torch.long))
            },
            'metadata': {
                'snr_db': processed_data.get('snr_db', parameter_values.get('snr_db', 10.0) if isinstance(parameter_values, dict) else 10.0),
                'title': processed_data.get('title', '')
            }
        }
        
        # Add batch dimension to targets if needed
        if result['targets']['centers'].numel() > 0 and result['targets']['centers'].dim() == 2:
            result['targets']['centers'] = result['targets']['centers'].unsqueeze(0)
        if result['targets']['labels'].numel() > 0 and result['targets']['labels'].dim() == 1:
            result['targets']['labels'] = result['targets']['labels'].unsqueeze(0)
        
        # Add pre-computed predictions if available
        if 'predictions' in processed_data:
            pred = processed_data['predictions']
            if pred.dim() == 2:
                pred = pred.unsqueeze(0)
            result['predictions'] = pred
        
        # Run model evaluation if needed
        if models is not None and 'predictions' in show and 'predictions' not in result:
            result = self._add_model_predictions(result, models, parameter_values)
        
        # Run KMeans if needed - but only if not already computed
        if 'kmeans' in show and 'kmeans_results' not in result:
            result = self._add_kmeans_results(result)
        
        return result
    
    def _create_single_scatter_plot(
        self,
        result: Dict,
        show: List[str],
        title: Union[str, None],
        figsize: Optional[Tuple[int, int]],
        save_path: Union[str, Path, None],
        size_scale: float,
        **kwargs
    ) -> Figure:
        """Create a single scatter plot using the legacy visualize_gmm_data function."""
        
        # Extract data for visualization
        points = result['inputs'][0]  # Remove batch dimension
        true_centers = result['targets']['centers'][0] if 'true_centers' in show and result['targets']['centers'].numel() > 0 else None
        point_labels = result['targets']['labels'][0] if result['targets']['labels'].numel() > 0 else None
        predictions = result.get('predictions', [None])[0] if 'predictions' in show and 'predictions' in result else None
        kmeans_centers = result.get('kmeans_results', {}).get('centers', [None])[0] if 'kmeans' in show and 'kmeans_results' in result else None
        
        # Determine title
        if title is None:
            title = result['metadata'].get('title', '')
            if not title:
                snr = result['metadata'].get('snr_db', '')
                if snr:
                    title = f"GMM Visualization (SNR: {snr:.1f} dB)"
                else:
                    title = "GMM Visualization"
        
        # Create the plot using legacy function
        fig, ax = visualize_gmm_data(
            points=points,
            predictions=predictions,
            true_centers=true_centers,
            kmeans_centers=kmeans_centers,
            point_labels=point_labels,
            title=title,
            figsize=figsize or (10, 8),
            size_scale=size_scale,
            **kwargs
        )
        
        # Save if requested
        if save_path:
            save_figure(fig, save_path)
        
        return fig
    
    def _create_scatter_grid(
        self,
        results: List[Dict],
        layout: str,
        titles: Union[str, List[str], None],
        show: List[str],
        figsize: Optional[Tuple[int, int]],
        save_path: Union[str, Path, None],
        shared_axes: bool,
        size_scale: float,
        **kwargs
    ) -> Figure:
        """Create a grid of scatter plots using the legacy create_comparison_grid function."""
        
        # Pass all kwargs including size_scale to create_comparison_grid
        # The legacy function now supports size_scale parameter
        grid_kwargs = kwargs.copy()
        grid_kwargs['size_scale'] = size_scale
        
        # Use the existing create_comparison_grid function
        fig, axes = create_comparison_grid(
            results=results,
            layout=layout,
            show_predictions='predictions' in show,
            show_kmeans='kmeans' in show,
            titles=titles,
            figsize=figsize,
            shared_axis_limits=shared_axes,
            **grid_kwargs
        )
        
        # Save if requested
        if save_path:
            save_figure(fig, save_path)
        
        return fig
    
    def _ensure_tensor_device(self, data: Dict) -> Dict:
        """Ensure all tensors in data dict are on the correct device."""
        processed = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                processed[key] = value.to(self.device)
            else:
                processed[key] = value
        return processed
    
    def _generate_default_data(self) -> Dict:
        """Generate default GMM data for basic visualization."""
        if self.verbose:
            print("Generating default data sample (standard dataset)...")
        
        # Use the existing data generation utility
        inputs, targets = create_data_samples(
            dataset_name='standard',
            num_samples=1,
            points_per_gmm=1000,
            device=self.device
        )
        
        if self.verbose:
            print("Default data generation completed")
        
        return {
            'points': inputs[0],
            'centers': targets['centers'][0],
            'labels': targets['labels'][0],
            'snr_db': targets.get('snr_db', [10.0])[0]
        }
    
    def _load_model(self, model_name: str):
        """Load a model by name, with caching."""
        if model_name in self._model_cache:
            return self._model_cache[model_name]
        
        if self.experiment_dir is None:
            raise ValueError("experiment_dir must be specified to load models")
            
        try:
            # Construct experiment path - fixed to point to correct location
            exp_path = self.experiment_dir / model_name
            
            if self.verbose:
                print(f"Loading model: {model_name}")
            model, config = load_model_from_experiment(
                experiment_dir=exp_path,
                load_best=True,
                device=self.device
            )
            
            # Cache the loaded model
            if self.cache_evaluations:
                self._model_cache[model_name] = (model, config)
            
            return model, config
            
        except Exception as e:
            print(f"Warning: Could not load model '{model_name}': {e}")
            return None, None
    
    def _add_model_predictions(self, result: Dict, models: Union[str, List[str]], parameter_values: Union[Dict, float, int, None]) -> Dict:
        """Add model predictions to result using actual model evaluation."""
        
        # Handle multiple models
        if isinstance(models, list):
            # For multiple models, we'll use the first one for now
            # TODO: In the future, this could return multiple prediction sets
            model_name = models[0]
            print(f"Using first model '{model_name}' from list: {models}")
        else:
            model_name = models
        
        # Load the model
        model, config = self._load_model(model_name)
        
        if model is None:
            print(f"Model '{model_name}' could not be loaded, skipping predictions")
            return result
        
        try:
            # Extract SNR value for evaluation
            snr_db = None
            if isinstance(parameter_values, dict) and 'snr_db' in parameter_values:
                snr_db = parameter_values['snr_db']
            elif 'metadata' in result and 'snr_db' in result['metadata']:
                snr_db = result['metadata']['snr_db']
            else:
                snr_db = 10.0  # Default SNR
            
            # Get model's expected dtype
            model_dtype = next(model.parameters()).dtype
            
            # Convert data to match model dtype to prevent type mismatch errors
            inputs = result['inputs'].to(dtype=model_dtype)
            
            # Convert SNR to tensor with correct dtype
            snr_tensor = torch.tensor([snr_db], device=self.device, dtype=model_dtype)
            
            # Prepare targets for evaluation and convert to correct dtype
            targets = {}
            if 'targets' in result:
                for key, value in result['targets'].items():
                    if isinstance(value, torch.Tensor) and value.numel() > 0:
                        targets[key] = value.to(dtype=model_dtype)
            
            # Get number of clusters for evaluation
            n_clusters = None
            if 'centers' in targets and targets['centers'].numel() > 0:
                n_clusters = targets['centers'].shape[1]
            
            # Run model evaluation
            eval_result = evaluate(
                model=model,
                data=inputs,
                snr=snr_tensor,
                n_clusters=n_clusters,
                targets=targets,
                device=self.device
            )
            
            # Add predictions to result
            if 'predictions' in eval_result:
                result['predictions'] = eval_result['predictions']
                
            # Also store evaluation metadata
            result['metadata']['model_name'] = model_name
            result['metadata']['snr_db'] = snr_db
            
            if self.verbose:
                if self.verbose:
                    print(f"Added model predictions from '{model_name}' at SNR {snr_db:.1f} dB")
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Model evaluation failed for '{model_name}': {e}")
        
        return result
    
    def _add_kmeans_results(self, result: Dict) -> Dict:
        """Add KMeans results to result using actual KMeans clustering."""
        
        try:
            # Determine number of clusters
            n_clusters = None
            
            # Try to get from targets first
            if 'targets' in result and 'centers' in result['targets'] and result['targets']['centers'].numel() > 0:
                n_clusters = result['targets']['centers'].shape[1]
            else:
                # Default to a reasonable number
                n_clusters = 5
                print(f"Warning: Could not determine n_clusters from targets, using default: {n_clusters}")
            
            # Run KMeans on inputs
            kmeans_results = run_kmeans(
                data=result['inputs'],
                n_clusters=n_clusters,
                device=self.device
            )
            
            result['kmeans_results'] = kmeans_results
            if self.verbose:
                if self.verbose:
                    print(f"Added KMeans clustering results with {n_clusters} clusters")
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: KMeans clustering failed: {e}")
        
        return result
    
    def _process_dataset_input(self, datasets: Union[str, List[str]], models: Union[str, List[str], None], parameter_values: Union[Dict, List, float, int, None], show: List[str], num_samples: int = 1) -> List[Dict]:
        """Process dataset-based input by generating data samples.
        
        Fixed logic:
        1. Generate num_samples different datasets
        2. For each dataset, evaluate it at all parameter values
        3. Return results in proper order for grid layout
        """
        
        results = []
        
        # Handle multiple datasets
        dataset_list = [datasets] if isinstance(datasets, str) else datasets
        
        for dataset_name in dataset_list:
            try:
                if self.verbose:
                    print(f"Generating data from dataset: {dataset_name}")
                
                # FIXED: Always generate num_samples datasets first
                # (not based on parameter_values length which was wrong)
                
                # Generate the requested number of data samples
                inputs, targets = create_data_samples(
                    dataset_name=dataset_name,
                    num_samples=num_samples,
                    points_per_gmm=1000,
                    device=self.device
                )
                
                # Handle parameter sweeps correctly
                if isinstance(parameter_values, dict) and any(isinstance(v, list) for v in parameter_values.values()):
                    # Parameter sweep: evaluate each sample at all parameter values
                    param_combinations = self._generate_parameter_combinations(parameter_values)
                    
                    for sample_idx in range(num_samples):
                        for param_combo in param_combinations:
                            # Create data dictionary for this sample
                            sample_data = {
                                'points': inputs[sample_idx],
                                'centers': targets['centers'][sample_idx],
                                'labels': targets['labels'][sample_idx],
                                'snr_db': targets['snr_db'][sample_idx].item()
                            }
                            
                            # Process this sample with this parameter combination
                            result = self._process_single_data_input(sample_data, models, param_combo, show)
                            result['metadata']['dataset_name'] = dataset_name
                            result['metadata']['sample_idx'] = sample_idx
                            result['metadata']['param_combo'] = param_combo
                            results.append(result)
                            
                else:
                    # Single parameter value or no parameters: process each sample once
                    for sample_idx in range(num_samples):
                        # Create data dictionary for this sample
                        sample_data = {
                            'points': inputs[sample_idx],
                            'centers': targets['centers'][sample_idx],
                            'labels': targets['labels'][sample_idx],
                            'snr_db': targets['snr_db'][sample_idx].item()
                        }
                        
                        # Process this sample
                        result = self._process_single_data_input(sample_data, models, parameter_values, show)
                        result['metadata']['dataset_name'] = dataset_name
                        result['metadata']['sample_idx'] = sample_idx
                        results.append(result)
                
                if self.verbose:
                    if self.verbose:
                        print(f"Generated {num_samples} samples from dataset '{dataset_name}'")
                    if isinstance(parameter_values, dict) and any(isinstance(v, list) for v in parameter_values.values()):
                        param_combinations = self._generate_parameter_combinations(parameter_values)
                        if self.verbose:
                            print(f"Evaluated each sample at {len(param_combinations)} parameter combinations")
                
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not generate data from dataset '{dataset_name}': {e}")
        
        return results
    
    def _generate_parameter_combinations(self, parameter_values: Dict) -> List[Dict]:
        """Generate all combinations of parameter values for parameter sweeps."""
        # Find parameters with multiple values
        param_lists = {}
        param_singles = {}
        
        for key, value in parameter_values.items():
            if isinstance(value, list):
                param_lists[key] = value
            else:
                param_singles[key] = value
        
        # Generate all combinations
        combinations = []
        
        if len(param_lists) == 1:
            # Single parameter with multiple values
            param_name, param_values_list = list(param_lists.items())[0]
            for param_val in param_values_list:
                combo = param_singles.copy()
                combo[param_name] = param_val
                combinations.append(combo)
        else:
            # Multiple parameters with multiple values - create cartesian product
            import itertools
            
            param_names = list(param_lists.keys())
            param_value_lists = list(param_lists.values())
            
            for param_combination in itertools.product(*param_value_lists):
                combo = param_singles.copy()
                for i, param_name in enumerate(param_names):
                    combo[param_name] = param_combination[i]
                combinations.append(combo)
        
        return combinations
    
    def _create_interactive_scatter_plot(
        self,
        data: Union[Dict, None] = None,
        models: Union[str, List[str], None] = None,
        datasets: Union[str, List[str], None] = None,
        show: List[str] = ['points', 'true_centers'],
        interactive_parameters: Optional[Dict[str, Tuple]] = None,
        figsize: Optional[Tuple[int, int]] = None,
        dpi: Optional[int] = None,
        **kwargs
    ) -> InteractiveWidget:
        """Create interactive scatter plot with parameter sliders."""
        
        # Set default interactive parameters if none provided
        if interactive_parameters is None:
            interactive_parameters = {'snr_db': (3.0, 15.0, 0.2)}
        
        # Fix parameter tuples: if only (min, max) provided, add default step
        fixed_parameters = {}
        for param_name, param_tuple in interactive_parameters.items():
            if len(param_tuple) == 2:
                # Add default step value
                min_val, max_val = param_tuple
                if param_name == 'snr_db':
                    step = 0.2
                elif param_name == 'flow_speed':
                    step = 0.05
                else:
                    step = (max_val - min_val) / 50  # Default: 50 steps
                fixed_parameters[param_name] = (min_val, max_val, step)
                if self.verbose:
                    print(f"Added default step {step} for parameter '{param_name}': ({min_val}, {max_val}) → ({min_val}, {max_val}, {step})")
            elif len(param_tuple) == 3:
                fixed_parameters[param_name] = param_tuple
            else:
                raise ValueError(f"Parameter '{param_name}' must be (min, max) or (min, max, step), got {param_tuple}")
        
        # Generate parameter sweep data
        frame_data = []
        param_values = []
        
        # Handle single parameter for now (can be extended for multiple parameters)
        param_name, (min_val, max_val, step) = list(fixed_parameters.items())[0]
        
        # Generate parameter values
        param_range = np.arange(min_val, max_val + step, step)
        
        if self.verbose:
            print(f"Generating interactive data for {param_name}: {min_val} to {max_val} (step: {step})")
            print(f"Computing {len(param_range)} parameter values...")
        
        # Generate results for each parameter value
        for param_val in tqdm(param_range, desc="Computing frames", disable=not self.verbose):
            # Create parameter dict
            param_dict = {param_name: param_val}
            
            # Get evaluation results for this parameter value
            results = self._prepare_evaluation_results(
                data=data, models=models, datasets=datasets,
                parameter_values=param_dict, show=show
            )
            
            # Store the first result (assuming single sample for interactive)
            if results:
                result = results[0]
                result['metadata']['param_name'] = param_name
                result['metadata']['param_value'] = param_val
                frame_data.append(result)
                param_values.append(param_val)
        
        if not frame_data:
            raise ValueError("No data generated for interactive visualization")
        
        # Create figure
        figsize = figsize or (10, 8)
        fig = plt.figure(figsize=figsize, facecolor='white')
        
        # Leave space for slider
        ax = fig.add_axes([0.1, 0.15, 0.85, 0.80])
        ax.set_facecolor('white')
        
        # Calculate global axis limits for consistent view
        points_list = [result['inputs'][0].cpu().numpy() for result in frame_data]
        centers_list = [result['targets']['centers'][0].cpu().numpy() 
                       for result in frame_data if result['targets']['centers'].numel() > 0]
        # Exclude predictions from axis calculation to prevent axis extension
        # predictions_list = [result['predictions'][0].cpu().numpy() 
        #                    for result in frame_data if 'predictions' in result]
        # kmeans_centers_list = [result['kmeans_results']['centers'][0].cpu().numpy()
        #                       for result in frame_data if 'kmeans_results' in result]
        
        xlim, ylim = calculate_global_axis_limits(
            points_list=points_list,
            centers_list=centers_list if centers_list else None,
            predictions_list=None,  # Exclude predictions from axis calculation
            kmeans_centers_list=None  # Exclude kmeans centers from axis calculation
        )
        
        # Define update function
        def update_plot(frame_idx):
            result = frame_data[frame_idx]
            param_val = param_values[frame_idx]
            
            # Clear and update plot
            ax.clear()
            
            # Extract data for visualization
            points = result['inputs'][0].cpu().numpy()
            true_centers = result['targets']['centers'][0].cpu().numpy() if 'true_centers' in show and result['targets']['centers'].numel() > 0 else None
            point_labels = result['targets']['labels'][0].cpu().numpy() if result['targets']['labels'].numel() > 0 else None
            predictions = result.get('predictions', [None])[0].cpu().numpy() if 'predictions' in show and 'predictions' in result else None
            kmeans_centers = result.get('kmeans_results', {}).get('centers', [None])[0].cpu().numpy() if 'kmeans' in show and 'kmeans_results' in result else None
            
            # Create visualization
            visualize_gmm_data(
                points=points,
                predictions=predictions,
                true_centers=true_centers,
                kmeans_centers=kmeans_centers,
                point_labels=point_labels,
                title=f"GMM Interactive Visualization - {param_name}: {param_val:.2f}",
                ax=ax,
                xlim=xlim,
                ylim=ylim,
                show_legend=True,
                **kwargs
            )
        
        # Create interactive widget
        param_display_name = param_name.replace('_', ' ').title()
        widget = InteractiveWidget(
            fig=fig,
            update_func=update_plot,
            frame_data=frame_data,
            param_name=param_display_name,
            param_values=param_values,
            default_dpi=dpi or self.default_dpi
        )
        
        if self.verbose:
            if self.verbose:
                print(f"Interactive widget created with {len(frame_data)} parameter values")
        return widget
    
    def _create_animation_scatter_plot(
        self,
        data: Union[Dict, None] = None,
        models: Union[str, List[str], None] = None,
        datasets: Union[str, List[str], None] = None,
        show: List[str] = ['points', 'true_centers'],
        animation_parameter: str = 'snr_db',
        parameter_range: Optional[Tuple[float, float]] = None,
        frames: int = 100,
        fps: int = 20,
        dpi: int = 100,
        save_path: Union[str, Path, None] = None,
        figsize: Optional[Tuple[int, int]] = None,
        show_animation: bool = False,
        **kwargs
    ) -> Animation:
        """Create animated scatter plot showing parameter evolution.
        
        Args:
            data: Direct data input (alternative to datasets)
            models: Model name(s) to evaluate
            datasets: Dataset name(s) to use
            show: Elements to display in plots
            animation_parameter: Parameter to animate
            parameter_range: (min, max) values for parameter
            frames: Number of animation frames
            fps: Frames per second
            save_path: Path to save animation
            figsize: Figure size
            show_animation: Whether to display the animation after creation
            **kwargs: Additional visualization arguments
            
        Returns:
            Animation object
        """
        
        # Set default parameter range if not provided
        if parameter_range is None:
            if animation_parameter == 'snr_db':
                parameter_range = (3.0, 15.0)
            elif animation_parameter == 'flow_speed':
                parameter_range = (0.0, 1.0)
            else:
                parameter_range = (0.0, 1.0)
        
        min_val, max_val = parameter_range
        param_values = np.linspace(min_val, max_val, frames)
        
        if self.verbose:
            print(f"Creating animation for {animation_parameter}: {min_val} to {max_val}")
            print(f"Pre-computing {frames} frames...")
        
        # Pre-compute all evaluation results
        frame_data = []
        
        # First, generate or get the base data once
        if data is None and datasets is not None:
            # Generate data from dataset once
            if self.verbose:
                print(f"Generating data from dataset: {datasets}")
            dataset_name = datasets if isinstance(datasets, str) else datasets[0]
            inputs, targets = create_data_samples(
                dataset_name=dataset_name,
                num_samples=1,
                points_per_gmm=1000,
                device=self.device
            )
            # Create data dictionary
            data = {
                'points': inputs[0],
                'centers': targets['centers'][0],
                'labels': targets['labels'][0],
                'snr_db': targets['snr_db'][0].item()
            }
            if self.verbose:
                print(f"Generated 1 sample from dataset '{dataset_name}'")
        elif data is None:
            # Use default data
            data = self._generate_default_data()
        
        # Compute KMeans once if needed
        kmeans_results = None
        if 'kmeans' in show:
            # Process the data once to get KMeans results
            temp_result = self._process_single_data_input(data, None, {}, ['kmeans'])
            if 'kmeans_results' in temp_result:
                kmeans_results = temp_result['kmeans_results']
        
        # Now compute frames with varying parameter, reusing data and KMeans
        # Use tqdm only if verbose, otherwise use plain loop
        iterator = tqdm(param_values, desc="Computing frames") if self.verbose else param_values
        for param_val in iterator:
            # Create parameter dict
            param_dict = {animation_parameter: param_val}
            
            # Process with model evaluation but skip KMeans
            show_without_kmeans = [s for s in show if s != 'kmeans']
            result = self._process_single_data_input(data, models, param_dict, show_without_kmeans)
            
            # Add back KMeans results if needed
            if kmeans_results is not None:
                result['kmeans_results'] = kmeans_results
                
            result['metadata']['param_name'] = animation_parameter
            result['metadata']['param_value'] = param_val
            frame_data.append(result)
        
        if not frame_data:
            raise ValueError("No data generated for animation")
        
        # Create figure
        figsize = figsize or (10, 8)
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate global axis limits for consistent view
        points_list = [result['inputs'][0].cpu().numpy() for result in frame_data]
        centers_list = [result['targets']['centers'][0].cpu().numpy() 
                       for result in frame_data if result['targets']['centers'].numel() > 0]
        # Exclude predictions from axis calculation to prevent axis extension
        # predictions_list = [result['predictions'][0].cpu().numpy() 
        #                    for result in frame_data if 'predictions' in result]
        # kmeans_centers_list = [result['kmeans_results']['centers'][0].cpu().numpy()
        #                       for result in frame_data if 'kmeans_results' in result]
        
        xlim, ylim = calculate_global_axis_limits(
            points_list=points_list,
            centers_list=centers_list if centers_list else None,
            predictions_list=None,  # Exclude predictions from axis calculation
            kmeans_centers_list=None  # Exclude kmeans centers from axis calculation
        )
        
        # Define animation update function
        def update_animation(frame_idx):
            result = frame_data[frame_idx]
            param_val = param_values[frame_idx]
            
            # Clear and update plot
            ax.clear()
            
            # Extract data for visualization
            points = result['inputs'][0].cpu().numpy()
            true_centers = result['targets']['centers'][0].cpu().numpy() if 'true_centers' in show and result['targets']['centers'].numel() > 0 else None
            point_labels = result['targets']['labels'][0].cpu().numpy() if result['targets']['labels'].numel() > 0 else None
            predictions = result.get('predictions', [None])[0].cpu().numpy() if 'predictions' in show and 'predictions' in result else None
            kmeans_centers = result.get('kmeans_results', {}).get('centers', [None])[0].cpu().numpy() if 'kmeans' in show and 'kmeans_results' in result else None
            
            # Create visualization
            param_display_name = animation_parameter.replace('_', ' ').title()
            visualize_gmm_data(
                points=points,
                predictions=predictions,
                true_centers=true_centers,
                kmeans_centers=kmeans_centers,
                point_labels=point_labels,
                title=f"GMM Animation - {param_display_name}: {param_val:.2f}",
                ax=ax,
                xlim=xlim,
                ylim=ylim,
                show_legend=True,
                **kwargs
            )
            
            return ax.collections + ax.texts
        
        # Create Animation object
        # Don't create matplotlib animation here - let our Animation wrapper handle it
        animation_obj = Animation(fig, update_animation, frame_data, fps=fps, verbose=self.verbose)
        
        # Save if requested
        if save_path:
            # Force GIF format for faster creation
            gif_path = Path(save_path).with_suffix('.gif')
            animation_obj.save_gif(gif_path, fps=fps, dpi=dpi)
        
        # Show if requested
        if show_animation:
            plt.tight_layout()
            animation_obj.show()
        else:
            # Close the figure to prevent it from displaying in Jupyter
            plt.close(animation_obj.fig)
        
        if self.verbose:
            print(f"Animation created with {len(frame_data)} frames at {fps} fps")
        return animation_obj

    # === Convenience Methods for Phase 3 ===
    
    def create_animation(
        self,
        parameter: str = 'snr_db',
        parameter_range: Optional[Tuple[float, float]] = None,
        frames: int = 100,
        fps: int = 20,
        models: Union[str, List[str], None] = None,
        datasets: Union[str, List[str], None] = None,
        data: Union[Dict, None] = None,
        show: List[str] = ['points', 'true_centers', 'predictions'],
        save_path: Union[str, Path, None] = None,
        **kwargs
    ) -> Animation:
        """
        Create an animation showing parameter evolution.
        
        High-level convenience method for creating animations as described in the API design.
        
        Args:
            parameter: Parameter to animate ('snr_db', 'flow_speed', etc.)
            parameter_range: (min, max) values for parameter
            frames: Number of animation frames
            fps: Frames per second
            models: Model name(s) to evaluate (REQUIRED for predictions)
            datasets: Dataset name(s) to use
            data: Direct data input (alternative to datasets)
            show: Elements to display (includes 'predictions' by default)
            save_path: Path to save animation
            **kwargs: Additional visualization arguments
        
        Returns:
            Animation object that can be saved as MP4, GIF, or frames
        
        Examples:
            # Basic animation with model predictions
            pipeline = VisualizationPipeline(models='baseline_16_layers')
            anim = pipeline.create_animation('snr_db', (3, 15), frames=60)
            anim.save_mp4('snr_animation.mp4')
            
            # Alternative: specify model in method call
            pipeline = VisualizationPipeline()
            anim = pipeline.create_animation(
                parameter='snr_db',
                parameter_range=(3, 15),
                frames=60,
                models='baseline_16_layers',  # Model specified here
                datasets='standard'
            )
            
            # Animation without model predictions
            pipeline = VisualizationPipeline()
            anim = pipeline.create_animation(
                parameter='snr_db',
                parameter_range=(3, 15),
                frames=60,
                datasets='standard',
                show=['points', 'true_centers']  # No predictions
            )
        """
        return self.scatter_plot(
            data=data,
            models=models or self.models,
            datasets=datasets or self.datasets,
            show=show,
            output_format='animation',
            animation_parameter=parameter,
            parameter_range=parameter_range,
            frames=frames,
            fps=fps,
            save_path=save_path,
            **kwargs
        )
    
    def create_interactive(
        self,
        parameters: Optional[Dict[str, Tuple]] = None,
        models: Union[str, List[str], None] = None,
        datasets: Union[str, List[str], None] = None,
        data: Union[Dict, None] = None,
        show: List[str] = ['points', 'true_centers', 'predictions'],
        dpi: Optional[int] = None,
        **kwargs
    ) -> InteractiveWidget:
        """
        Create an interactive widget with parameter sliders.
        
        High-level convenience method for creating interactive visualizations as described in the API design.
        
        Args:
            parameters: Dict mapping parameter names to (min, max, step) tuples
                       e.g., {'snr_db': (3, 15, 0.1), 'flow_speed': (0, 1, 0.01)}
            models: Model name(s) to evaluate (REQUIRED for predictions)
            datasets: Dataset name(s) to use
            data: Direct data input (alternative to datasets)
            show: Elements to display (includes 'predictions' by default)
            **kwargs: Additional visualization arguments
        
        Returns:
            InteractiveWidget with slider controls
        
        Examples:
            # Basic interactive widget with model predictions
            pipeline = VisualizationPipeline(models='baseline_16_layers')
            widget = pipeline.create_interactive({'snr_db': (3, 15, 0.1)})
            widget.show()
            
            # Alternative: specify model in method call
            pipeline = VisualizationPipeline()
            widget = pipeline.create_interactive(
                parameters={'snr_db': (3, 15, 0.1)},
                models='baseline_16_layers',  # Model specified here
                datasets='standard'
            )
            
            # Interactive widget without model predictions
            pipeline = VisualizationPipeline()
            widget = pipeline.create_interactive(
                parameters={'snr_db': (3, 15, 0.1)},
                datasets='standard',
                show=['points', 'true_centers']  # No predictions
            )
        """
        if parameters is None:
            parameters = {'snr_db': (3.0, 15.0, 0.2)}
            
        return self.scatter_plot(
            data=data,
            models=models or self.models,
            datasets=datasets or self.datasets,
            show=show,
            output_format='interactive',
            interactive_parameters=parameters,
            **kwargs
        )
    
    def create_comparison_animation(
        self,
        parameter: str = 'snr_db',
        parameter_range: Optional[Tuple[float, float]] = None,
        left_model: str = None,
        right_model: str = None,
        datasets: Union[str, List[str], None] = None,
        frames: int = 100,
        fps: int = 20,
        save_path: Union[str, Path, None] = None,
        **kwargs
    ) -> Animation:
        """
        Create side-by-side comparison animation.
        
        This creates an animation with two panels showing different models side-by-side.
        
        Args:
            parameter: Parameter to animate
            parameter_range: (min, max) values for parameter
            left_model: Model for left panel
            right_model: Model for right panel
            datasets: Dataset name(s) to use
            frames: Number of animation frames
            fps: Frames per second
            save_path: Path to save animation
            **kwargs: Additional visualization arguments
        
        Returns:
            Animation object with side-by-side comparison
        
        Example:
            # Compare models
            anim = pipeline.create_comparison_animation(
                parameter='flow_speed',
                parameter_range=(0, 1),
                left_model='no_flow_16_layers',
                right_model='baseline_16_layers'
            )
        """
        # This is a placeholder for future implementation
        # For now, create animation with the first model only
        model = left_model or right_model
        if model is None:
            raise ValueError("At least one of left_model or right_model must be specified")
        
        print(f"Note: Comparison animations not fully implemented yet. Using single model: {model}")
        
        return self.create_animation(
            parameter=parameter,
            parameter_range=parameter_range,
            models=model,
            datasets=datasets or self.datasets,
            frames=frames,
            fps=fps,
            save_path=save_path,
            **kwargs
        )

    # =================================================================
    # PHASE 1-2 EXTENSIONS: FLOW SUBSTITUTION SUPPORT
    # =================================================================
    
    def patch_model_flow_settings(self, model_name_or_object, **flow_kwargs):
        """
        Create a temporary model copy with modified flow settings.
        
        This enables creating variations of the same model with different flow
        configurations for comparison animations without modifying the original.
        
        Args:
            model_name_or_object: Model name (str) or actual model object
            **flow_kwargs: Flow settings to override, such as:
                - use_flow_predictor: bool - Enable/disable flow prediction
                - flow_distribution_mode: 'direct' | 'fractional' - Flow distribution type
                - repeat_factor: int - Layer repetition factor
                - flow_predictor_per_layer: bool - Per-layer vs global flow
                
        Returns:
            Patched model copy with modified flow settings
        """
        import copy
        
        # Load model if string name provided
        if isinstance(model_name_or_object, str):
            original_model, config = self._load_model(model_name_or_object)
        else:
            original_model = model_name_or_object
        
        # Create deep copy to avoid modifying original
        patched_model = copy.deepcopy(original_model)
        
        # Apply flow patches to transformer
        # Patch model with flow settings
        for attr, value in flow_kwargs.items():
            if hasattr(patched_model.transformer, attr):
                setattr(patched_model.transformer, attr, value)
        
        return patched_model
    
    def create_flow_injection_wrapper(self, base_model, manual_flow_override=True):
        """
        Create a wrapper that allows manual flow speed injection.
        
        This prepares a model for manual flow control by disabling its internal
        flow predictor (if any) so flow speeds can be provided directly.
        
        Args:
            base_model: Base model to wrap (model object or name)
            manual_flow_override: If True, disable flow predictor for manual control
            
        Returns:
            Model configured for manual flow speed injection
        """
        # Load model if string name provided
        if isinstance(base_model, str):
            model, config = self._load_model(base_model)
        else:
            model = base_model
        
        if manual_flow_override and hasattr(model.transformer, 'use_flow_predictor'):
            model.transformer.use_flow_predictor = False
        
        return model
    
    def _create_side_by_side_animation(
        self, 
        frame_data_list, 
        figsize=(16, 8),
        save_path=None,
        show_animation=True,
        fps=20,
        dpi=100,
        data=None,
        show=['points', 'true_centers', 'predictions'],
        **kwargs
    ):
        """
        Create a side-by-side animation from frame data.
        
        Args:
            frame_data_list: List of frame data dictionaries, each containing:
                - left_model: Model for left panel
                - right_model: Model for right panel  
                - left_flow: Flow speed for left model (can be None)
                - right_flow: Flow speed for right model (can be None)
                - titles: [left_title, right_title]
                - parameter_value: Value being animated (for display)
            figsize: Figure size
            save_path: Path to save animation
            show_animation: Whether to display animation
            fps: Frames per second
            data: Data to use (if None, generates default data)
            show: Elements to display in plots
            
        Returns:
            Animation object
        """
        # Use default data if none provided
        if data is None:
            data = self._generate_default_data()
        
        # Ensure data is in correct format
        data = self._ensure_tensor_device(data)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Pre-compute all predictions for smooth animation
        if self.verbose:
            print(f"\n🔄 Pre-computing predictions for {len(frame_data_list)} frames...")
            import time
            start_time = time.time()
        left_predictions = []
        right_predictions = []
        
        # Ensure data has batch dimension for model input
        points_input = data['points'].unsqueeze(0) if data['points'].dim() == 2 else data['points']
        
        # Always use tqdm for progress tracking
        iterator = tqdm(enumerate(frame_data_list), desc="Pre-computing frames", total=len(frame_data_list))
        for i, frame_data in iterator:
            # Left panel predictions
            left_model = frame_data['left_model']
            left_flow = frame_data.get('left_flow')
            
            with torch.no_grad():
                if left_flow is not None:
                    flow_tensor = torch.tensor([left_flow], device=data['points'].device, 
                                             dtype=data['points'].dtype)
                    left_pred = left_model(points_input, flow_speed=flow_tensor)
                else:
                    left_pred = left_model(points_input)
            left_predictions.append(left_pred[0])
            
            # Right panel predictions  
            right_model = frame_data['right_model']
            right_flow = frame_data.get('right_flow')
            
            with torch.no_grad():
                if right_flow is not None:
                    flow_tensor = torch.tensor([right_flow], device=data['points'].device,
                                             dtype=data['points'].dtype)
                    right_pred = right_model(points_input, flow_speed=flow_tensor)
                else:
                    right_pred = right_model(points_input)
            right_predictions.append(right_pred[0])
        
        if self.verbose:
            inference_time = time.time() - start_time
            print(f"✅ Model inference completed in {inference_time:.2f}s ({inference_time/len(frame_data_list):.3f}s per frame)")
        
        # Calculate global axis limits for consistent viewing
        # Use only input points and true centers, not predictions
        all_points = [data['points']]
        if 'centers' in data and data['centers'] is not None:
            all_centers = [data['centers']]
        else:
            all_centers = []
            
        from .visualization_legacy import calculate_global_axis_limits
        xlim, ylim = calculate_global_axis_limits(
            points_list=all_points,
            centers_list=all_centers,
            predictions_list=None,  # Exclude predictions from axis calculation
            kmeans_centers_list=None
        )
        
        # Animation update function
        def update_animation(frame_idx):
            frame_data = frame_data_list[frame_idx]
            left_pred = left_predictions[frame_idx]
            right_pred = right_predictions[frame_idx]
            
            # Clear both axes
            ax1.clear()
            ax2.clear()
            
            # Extract data correctly with proper shapes
            centers_viz = data.get('centers')
            if centers_viz is not None:
                centers_viz = centers_viz.cpu().numpy() if hasattr(centers_viz, 'cpu') else centers_viz
                # Ensure centers has correct 2D shape
                if centers_viz.ndim == 1:
                    centers_viz = None  # Skip if malformed
            else:
                centers_viz = None
                
            labels_viz = data.get('labels')
            if labels_viz is not None:
                labels_viz = labels_viz.cpu().numpy() if hasattr(labels_viz, 'cpu') else labels_viz
            
            # Plot left panel
            from .visualization_legacy import visualize_gmm_data
            visualize_gmm_data(
                points=data['points'].cpu().numpy() if hasattr(data['points'], 'cpu') else data['points'],
                predictions=left_pred.cpu().numpy() if hasattr(left_pred, 'cpu') else left_pred if 'predictions' in show else None,
                true_centers=centers_viz if 'true_centers' in show else None,
                point_labels=labels_viz if 'points' in show else None,
                title=frame_data.get('titles', ['Left', 'Right'])[0],
                ax=ax1,
                show_legend=True,
                xlim=xlim,
                ylim=ylim,
                calculate_axis_limits=False
            )
            
            # Plot right panel
            visualize_gmm_data(
                points=data['points'].cpu().numpy() if hasattr(data['points'], 'cpu') else data['points'],
                predictions=right_pred.cpu().numpy() if hasattr(right_pred, 'cpu') else right_pred if 'predictions' in show else None,
                true_centers=centers_viz if 'true_centers' in show else None,
                point_labels=labels_viz if 'points' in show else None,
                title=frame_data.get('titles', ['Left', 'Right'])[1],
                ax=ax2,
                show_legend=True,
                xlim=xlim,
                ylim=ylim,
                calculate_axis_limits=False
            )
            
            # Add parameter value indicator if available
            if 'parameter_value' in frame_data:
                param_val = frame_data['parameter_value']
                fig.suptitle(f"Parameter Value: {param_val:.3f}", fontsize=14, y=0.95)
            
            return ax1, ax2
        
        # Create animation object
        # Don't create matplotlib animation here - let our Animation wrapper handle it
        animation_obj = Animation(fig, update_animation, frame_data_list, fps=fps, verbose=self.verbose)
        
        # Save if requested
        if save_path:
            # Force GIF format for faster creation
            gif_path = Path(save_path).with_suffix('.gif')
            if self.verbose:
                print(f"\n💾 Saving animation to: {gif_path}")
            animation_obj.save_gif(gif_path, fps=fps, dpi=dpi)
        
        # Show if requested
        if show_animation:
            plt.tight_layout()
            animation_obj.show()
        else:
            # Close the figure to prevent it from displaying in Jupyter
            plt.close(fig)
        
        if self.verbose:
            print(f"\n🎉 Animation creation completed!")
            print(f"💡 Optimization tips:")
            print(f"   • Reduce frames for faster generation (current: {len(frame_data_list)})")
            print(f"   • Use lower DPI for speed (current: {dpi})")
            print(f"   • Consider MP4 format for better quality/size ratio")
        
        return animation_obj
    
    def create_flow_substitution_animation(
        self,
        base_model: str,
        flow_range: Tuple[float, float] = (0.0, 1.0),
        frames: int = 100,
        comparison_type: str = 'no_flow_vs_manual',
        data=None,
        save_path=None,
        show_animation=True,
        dpi=100,
        **kwargs
    ):
        """
        Create flow substitution animation showing same model with different flow settings.
        
        This is the main method that replaces the complex 07_flow_substitution logic
        with a simple, declarative interface.
        
        Args:
            base_model: Name of base model to use
            flow_range: Range of flow speeds to animate (min, max)
            frames: Number of animation frames
            comparison_type: Type of comparison:
                - 'no_flow_vs_manual': No flow vs manual flow injection
                - 'direct_vs_fractional': Direct vs fractional flow modes
                - 'regime_comparison': Different layer configurations
            data: Data to use (if None, generates default)
            save_path: Path to save animation
            show_animation: Whether to display animation
            **kwargs: Additional arguments passed to animation creation
            
        Returns:
            Animation object
        """
        
        # Generate flow speed values
        flow_values = np.linspace(flow_range[0], flow_range[1], frames)
        
        # Create model variants based on comparison type
        if comparison_type == 'no_flow_vs_manual':
            # Setting up no-flow vs manual flow comparison
            left_model = self.patch_model_flow_settings(
                base_model, 
                use_flow_predictor=False
            )
            right_model = self.create_flow_injection_wrapper(base_model)
            
            
            frame_data = []
            # Use tqdm for frame generation if verbose
            flow_iterator = tqdm(enumerate(flow_values), total=len(flow_values), desc="Frame setup")
            for i, flow_speed in flow_iterator:
                frame_data.append({
                    'left_model': left_model,
                    'right_model': right_model,
                    'left_flow': None,  # No flow
                    'right_flow': flow_speed,  # Manual flow
                    'titles': [f"No Flow", f"Manual Flow: {flow_speed:.2f}"],
                    'parameter_value': flow_speed
                })
                
        elif comparison_type == 'direct_vs_fractional':
            # Setting up direct vs fractional flow comparison
            # Create flow model base (disable flow predictor for manual control)
            flow_model_base = self.create_flow_injection_wrapper(base_model)
            
            left_model = self.patch_model_flow_settings(
                flow_model_base,
                flow_distribution_mode='direct'
            )
            right_model = self.patch_model_flow_settings(
                flow_model_base,
                flow_distribution_mode='fractional'
            )
            
            frame_data = []
            # Use tqdm for frame generation if verbose
            flow_iterator = tqdm(enumerate(flow_values), total=len(flow_values), desc="Frame setup")
            for i, flow_speed in flow_iterator:
                frame_data.append({
                    'left_model': left_model,
                    'right_model': right_model,
                    'left_flow': flow_speed,
                    'right_flow': flow_speed,
                    'titles': [f"Uniform Flow: {flow_speed:.2f}", f"Unit Flow Regime: {flow_speed:.2f}"],
                    'parameter_value': flow_speed
                })
                
        elif comparison_type == 'regime_comparison':
            # Setting up regime comparison
            # Extract regime settings from kwargs
            regime_settings = kwargs.get('regime_settings', {
                'left': {'repeat_factor': 16, 'flow_divisor': 1},
                'right': {'repeat_factor': 80, 'flow_divisor': 5}
            })
            
            left_settings = regime_settings['left']
            right_settings = regime_settings['right']
            
            if self.verbose:
                print(f"   Left regime: {left_settings['repeat_factor']} layers, flow÷{left_settings['flow_divisor']}")
                print(f"   Right regime: {right_settings['repeat_factor']} layers, flow÷{right_settings['flow_divisor']}")
            
            left_model = self.patch_model_flow_settings(
                base_model,
                repeat_factor=left_settings['repeat_factor']
            )
            right_model = self.patch_model_flow_settings(
                base_model,
                repeat_factor=right_settings['repeat_factor']
            )
            
            frame_data = []
            # Use tqdm for frame generation if verbose
            flow_iterator = tqdm(enumerate(flow_values), total=len(flow_values), desc="Frame setup")
            for i, flow_speed in flow_iterator:
                left_flow = flow_speed / left_settings['flow_divisor']
                right_flow = flow_speed / right_settings['flow_divisor']
                
                frame_data.append({
                    'left_model': left_model,
                    'right_model': right_model,
                    'left_flow': left_flow,
                    'right_flow': right_flow,
                    'titles': [
                        f"Regime 1 (×{left_settings['repeat_factor']//16}): {flow_speed:.2f}",
                        f"Regime 2 (×{right_settings['repeat_factor']//16}): {flow_speed:.2f}"
                    ],
                    'parameter_value': flow_speed
                })
        else:
            raise ValueError(f"Unknown comparison_type: {comparison_type}")
        
        # Create and return side-by-side animation
        return self._create_side_by_side_animation(
            frame_data, 
            data=data,
            save_path=save_path,
            show_animation=show_animation,
            **kwargs
        )
    
    def create_static_flow_comparison(
        self,
        base_model: str,
        flow_value: float = 0.2,
        comparison_type: str = 'no_flow_vs_manual',
        data=None,
        save_path=None,
        figsize=(16, 8),
        **kwargs
    ):
        """
        Create a static side-by-side comparison at a single flow value.
        
        Args:
            base_model: Name of base model to use
            flow_value: Flow speed value for comparison
            comparison_type: Type of comparison (same as create_flow_substitution_animation)
            data: Data to use (if None, generates default)
            save_path: Path to save plot
            figsize: Figure size
            
        Returns:
            matplotlib Figure
        """        
        # Create single frame data
        frame_data = [{
            'parameter_value': flow_value
        }]
        
        # Set up models based on comparison type
        if comparison_type == 'no_flow_vs_manual':
            left_model = self.patch_model_flow_settings(base_model, use_flow_predictor=False)
            right_model = self.create_flow_injection_wrapper(base_model)
            frame_data[0].update({
                'left_model': left_model,
                'right_model': right_model,
                'left_flow': None,
                'right_flow': flow_value,
                'titles': [f"No Flow", f"Manual Flow: {flow_value:.2f}"]
            })
        elif comparison_type == 'direct_vs_fractional':
            left_model = self.patch_model_flow_settings(base_model, flow_distribution_mode='direct')
            right_model = self.patch_model_flow_settings(base_model, flow_distribution_mode='fractional')
            frame_data[0].update({
                'left_model': left_model,
                'right_model': right_model,
                'left_flow': flow_value,
                'right_flow': flow_value,
                'titles': [f"Uniform Flow: {flow_value:.2f}", f"Unit Flow: {flow_value:.2f}"]
            })
        
        # Use default data if none provided
        if data is None:
            data = self._generate_default_data()
        data = self._ensure_tensor_device(data)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Get predictions for both models
        left_model = frame_data[0]['left_model']
        right_model = frame_data[0]['right_model']
        left_flow = frame_data[0].get('left_flow')
        right_flow = frame_data[0].get('right_flow')
        
        # Ensure data has batch dimension for model input
        points_input = data['points'].unsqueeze(0) if data['points'].dim() == 2 else data['points']
        
        with torch.no_grad():
            if left_flow is not None:
                flow_tensor = torch.tensor([left_flow], device=data['points'].device, dtype=data['points'].dtype)
                left_pred = left_model(points_input, flow_speed=flow_tensor)
            else:
                left_pred = left_model(points_input)
                
            if right_flow is not None:
                flow_tensor = torch.tensor([right_flow], device=data['points'].device, dtype=data['points'].dtype)
                right_pred = right_model(points_input, flow_speed=flow_tensor)
            else:
                right_pred = right_model(points_input)
        
        # Plot both panels
        from .visualization_legacy import visualize_gmm_data
        
        # Extract data correctly with proper shapes
        points_viz = left_pred[0].cpu().numpy() if hasattr(left_pred[0], 'cpu') else left_pred[0]
        centers_viz = data.get('centers')
        if centers_viz is not None:
            centers_viz = centers_viz.cpu().numpy() if hasattr(centers_viz, 'cpu') else centers_viz
            # Ensure centers has correct 2D shape
            if centers_viz.ndim == 1:
                centers_viz = None  # Skip if malformed
        else:
            centers_viz = None
            
        labels_viz = data.get('labels')
        if labels_viz is not None:
            labels_viz = labels_viz.cpu().numpy() if hasattr(labels_viz, 'cpu') else labels_viz
        
        visualize_gmm_data(
            points=data['points'].cpu().numpy() if hasattr(data['points'], 'cpu') else data['points'],
            predictions=left_pred[0].cpu().numpy() if hasattr(left_pred[0], 'cpu') else left_pred[0],
            true_centers=centers_viz,
            point_labels=labels_viz,
            title=frame_data[0]['titles'][0],
            ax=ax1,
            show_legend=True
        )
        
        visualize_gmm_data(
            points=data['points'].cpu().numpy() if hasattr(data['points'], 'cpu') else data['points'],
            predictions=right_pred[0].cpu().numpy() if hasattr(right_pred[0], 'cpu') else right_pred[0],
            true_centers=centers_viz,
            point_labels=labels_viz,
            title=frame_data[0]['titles'][1],
            ax=ax2,
            show_legend=True
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            pass  # Silently save
        
        return fig


# Convenience function for direct replacement of visualize_gmm_data
def scatter_plot(
    data: Dict,
    show: List[str] = ['points', 'true_centers'],
    title: str = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Union[str, Path] = None,
    experiment_dir: Union[str, Path, None] = None,
    **kwargs
) -> Figure:
    """
    Convenience function for direct replacement of visualize_gmm_data calls.
    
    Args:
        data: Dictionary with 'points', 'centers', 'labels', etc.
        show: What to display
        title: Plot title
        figsize: Figure size
        save_path: Where to save
        experiment_dir: Base directory containing model experiments
        **kwargs: Additional arguments
        
    Returns:
        matplotlib Figure
        
    Example:
        # Old: visualize_gmm_data(points, true_centers, point_labels)
        # New: scatter_plot({'points': points, 'centers': centers, 'labels': labels})
    """
    pipeline = VisualizationPipeline(experiment_dir=experiment_dir)
    return pipeline.scatter_plot(
        data=data,
        show=show,
        titles=title,
        figsize=figsize,
        save_path=save_path,
        **kwargs
    ) 