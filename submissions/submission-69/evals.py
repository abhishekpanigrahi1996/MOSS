from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, List, Tuple, Dict, Union

import numpy as np
from torch import Tensor
import seaborn as sns
from matplotlib import pyplot as plt

from utils import permute, brute_force_orderedness


class EvalVisualiser(ABC):
    """
    Abstract base class for evaluation visualisers.
    
    This class defines the interface that all visualisers must implement.
    Visualisers are used to collect data during training and generate
    visualisations or statistics at the end of experiments.
    """
    
    @abstractmethod
    def update(self, result: Dict[str, Any]) -> None:
        """
        Update visualiser with new result from a training run.
        
        Args:
            result: Dictionary containing training results from one run
        """
        pass
    
    @abstractmethod 
    def display(self) -> Any:
        """
        Display visualisation and return computed statistics.
        
        Returns:
            Computed statistics or visualisation data
        """
        pass


class NullVisualiser(EvalVisualiser):
    """
    Null visualiser that does nothing.
    """
    
    def update(self, result: Dict[str, Any]) -> None:
        pass
    
    def display(self) -> None:
        pass


class LineVisualiser(EvalVisualiser):
    """
    Line plot visualiser for tracking metrics over time.
    
    This visualiser collects time series data across multiple runs and
    displays line plots with mean values and confidence intervals.
    Useful for tracking training loss, accuracy, or other metrics over epochs.
    """
    
    def __init__(
        self,
        extractor: Callable[
            [Dict[str, Any]],
            Union[Tuple[np.ndarray, np.ndarray], np.ndarray]
        ],
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        only_values: bool = False,
        fname: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Initialize the line visualiser.
        
        Args:
            extractor: Function that extracts data from training results.
                       Should return (x, y) tuple or just y values.
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            only_values: If True, extractor returns only y values (x will be indices)
            fname: Optional filename to save the plot
        """
        self.all_x: List[np.ndarray] = []
        self.all_y: List[np.ndarray] = []
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.extractor = extractor
        self.only_values = only_values
        self.fname = fname
        self.show = show
    
    def update(self, result: Dict[str, Any]) -> None:
        """
        Update the visualiser with new training results.
        
        Args:
            result: Dictionary containing training results from one run
        """
        r = self.extractor(result)
        
        if self.only_values:
            # Extractor returns only y values
            self.all_y.append(r)
        else:
            # Extractor returns (x, y) tuple
            self.all_x.append(r[0])
            self.all_y.append(r[1])

    def display(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Display line plot with mean and confidence intervals.
        
        Creates a line plot showing the mean trajectory across all runs,
        with a shaded region indicating the min-max range.
        
        Returns:
            Tuple of (mean_y, max_y, min_y) arrays
        """
        # Find minimum length across all runs for consistent plotting
        min_len = min([len(ele) for ele in self.all_y])
        
        if self.only_values:
            # Use indices as x values
            x = np.arange(min_len)
        else:
            # Use x values from first run (assuming all runs have same x)
            x = self.all_x[0][:min_len]
        
        # Truncate all y arrays to minimum length
        y = np.array([ele[:min_len] for ele in self.all_y])

        # Calculate statistics across runs
        mean_y = np.mean(y, axis=0)
        max_y = np.max(y, axis=0)
        min_y = np.min(y, axis=0)

        plt.plot(x, mean_y, label=f'Mean {self.ylabel}', color='blue')
        plt.fill_between(x, max_y, min_y, color='blue', alpha=0.2, label='Range')

        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.legend()
        plt.tight_layout()
        
        if self.fname is not None:
            plt.savefig(self.fname, bbox_inches='tight')
        
        if self.show:
            plt.show()
        else:
            plt.close()

        return mean_y, max_y, min_y
    

class BoxVisualiser(EvalVisualiser):
    """
    Box plot visualiser for displaying distribution of final values.
    
    This visualiser collects scalar values from multiple runs and displays
    a box plot showing the distribution. Useful for comparing final metrics
    across different experimental conditions.
    """
    
    def __init__(
        self,
        extractor: Callable[[Dict[str, Any]], float],
        name: str = 'Values',
        ylabel: Optional[str] = None,
        fname: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Initialize the box visualiser.
        
        Args:
            extractor: Function that extracts a scalar value from training results
            name: Name for the visualiser
            ylabel: Label for y-axis
            fname: Optional filename to save the plot
            show: Whether to show the metrics and plot
        """
        self.extractor = extractor
        self.name = name
        self.ylabel = ylabel
        self.fname = fname
        self.show = show
        self.all_x: List[float] = []
    
    def update(self, result: Dict[str, Any]) -> None:
        """
        Update the visualiser with new training results.
        
        Args:
            result: Dictionary containing training results from one run
        """
        r = self.extractor(result)
        self.all_x.append(r)
    
    def display(self) -> Tuple[float, float, float, float]:
        """
        Display box plot of collected values.
        
        Creates a box plot showing the distribution of values across all runs.
        
        Returns:
            Tuple of (mean_value, std_value)
        """
        mean_x = np.mean(self.all_x).item()
        std_x = np.std(self.all_x).item()
        
        plt.figure(figsize=(3, 5))
        plt.boxplot(self.all_x)
        plt.xticks([])
        plt.ylabel(self.ylabel)
        plt.tight_layout()
        
        if self.fname is not None:
            plt.savefig(self.fname, bbox_inches='tight')
        
        if self.show:
            print(f'{self.name}: {mean_x:.3f} $\\pm$ {std_x:.3f}')
            plt.show()
        else:
            plt.close()
        
        # Store results for external access
        self.mean_x = mean_x
        self.std_x = std_x
        self.max_x = np.max(self.all_x).item()
        self.min_x = np.min(self.all_x).item()
        
        return mean_x, std_x, self.max_x, self.min_x


class WeightVisualiser(EvalVisualiser):
    """
    Weight matrix visualiser for displaying network connectivity patterns.
    
    This visualiser collects weight matrices from multiple runs and displays
    heatmaps showing various statistics (mean, std, sample) of the weights.
    Useful for analyzing how network connectivity evolves during training.
    """
    
    def __init__(
        self, 
        extractor: Callable[[Dict[str, Any]], Tensor],
        name: str = 'Weights',
        show: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the weight visualiser.
        
        Args:
            extractor: Function that extracts weight tensor from training results
            name: Name for the weight matrices (used in plot titles)
            show: List of visualisation types to show. Options: 'mean', 'std', 
                  'sample', 'abs-mean', 'triu'. If None, shows all.
        """
        self.show = show
        self.name = name
        self.all_weights = []
        self.extractor = extractor
    
    def update(self, result: Dict[str, Any]) -> None:
        """
        Update the visualiser with new training results.
        
        Args:
            result: Dictionary containing training results from one run
        """
        weight = self.extractor(result)
        self.all_weights.append(weight.detach().cpu().numpy())
    
    def heat(self, matrix: np.ndarray, title: str) -> None:
        """
        Display a heatmap of the given matrix.
        
        Args:
            matrix: 2D numpy array to visualise
            title: Title prefix for the plot
        """
        sns.heatmap(matrix, annot=True)
        plt.title(f'{title} {self.name}')
        plt.tight_layout()
        plt.show()

    def display(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Display weight matrix heatmaps for various statistics.
        
        Shows heatmaps for different weight statistics including mean,
        standard deviation, sample matrices, and specialized views.
        
        Returns:
            Tuple of (all_weights, mean_weights, std_weights)
        """
        if self.show is None:
            # Show all available visualisations by default
            self.show = ['mean', 'std', 'sample', 'abs-mean', 'triu']
        
        all_weights = np.array(self.all_weights)
        mean_weights = np.mean(all_weights, axis=0)
        std_weights = np.std(all_weights, axis=0)

        # Define all possible visualisations
        visualisation_map = {
            'mean': (mean_weights, 'Mean'),
            'std': (std_weights, 'Std'),
            'sample': (all_weights[0], 'Sample'),
            'abs-mean': (np.abs(mean_weights), 'Abs Mean'),
            'triu': (np.triu(mean_weights, 1), 'Triu Mean')
        }

        # Display requested visualisations
        for viz_type in self.show:
            if viz_type in visualisation_map:
                self.heat(*visualisation_map[viz_type])
        
        # Store sample for external access
        self.sample = all_weights[0]
        
        return all_weights, mean_weights, std_weights


class OrderednessVisualiser(EvalVisualiser):
    """
    Orderedness visualiser for analyzing topological ordering.
    
    This visualiser computes and displays the orderedness score of weight
    matrices across multiple runs. The orderedness score measures how
    "feed-forward" a network is by analyzing the triangular structure of
    weight matrices.
    """
    
    def __init__(
        self,
        extractor: Callable[[Dict[str, Any]], Tensor],
        name: str = 'Weights',
        graphs: bool = False
    ) -> None:
        """
        Initialize the orderedness visualiser.
        
        Args:
            extractor: Function that extracts weight tensor from training results
            name: Name for the weight matrices (used in plot titles)
            graphs: Whether to display individual heatmaps for each run
        """
        self.extractor = extractor
        self.name = name
        self.graphs = graphs
        self.input_size = None
        self.output_size = None
        self.all_weights = []
    
    def update(self, result: Dict[str, Any]) -> None:
        """
        Update the visualiser with new training results.
        
        Args:
            result: Dictionary containing training results from one run
        """
        weights = self.extractor(result)
        self.input_size = result['model'].input_size
        self.output_size = result['model'].output_size
        self.all_weights.append(weights.detach().cpu().numpy())

    def display(self) -> List[float]:
        """
        Display orderedness analysis results.
        
        Computes orderedness scores for all collected weight matrices,
        optionally displays individual heatmaps, and prints summary statistics.
        
        Returns:
            List of orderedness scores across all runs
        """
        scores = []
        
        for w in self.all_weights:
            square = w[:, :-self.input_size]
            orderedness, perm = brute_force_orderedness(square, self.output_size)
            
            if self.graphs:
                permuted_matrix = permute(square, perm)
                sns.heatmap(permuted_matrix, annot=True)
                plt.title(f'Orderedness of {self.name}: {orderedness:.3f}')
                plt.show()
            
            scores.append(orderedness)
        
        mean_dir = np.mean(scores)
        std_dir = np.std(scores)
        max_dir = np.max(scores)
        min_dir = np.min(scores)

        print(f'Orderedness: {mean_dir:.3f} $\\pm$ {std_dir:.3f}')

        # Store results for external access
        self.scores = scores
        self.mean_dir = mean_dir
        self.std_dir = std_dir
        self.max_dir = max_dir
        self.min_dir = min_dir

        return scores
