import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Tuple, Optional, Callable, Any, Dict, Union

from funcs import topk_mask
from inits import Size, Like, RandomNormal, Zeros, Ones


class CompleteLayer(nn.Module):
    """
    A complete perceptron layer / weight-tied RNN network.
    
    This layer represents a fully connected graph where each node can connect to
    every other node, including itself. It computes the output of the network
    in a recurrent manner.
    
    The layer consists of:
    - Values: Internal state/activations for hidden and output nodes
    - Weights: Connection weights between all nodes (input, hidden, output)
    - Scores: Importance scores for each connection (used for top-k selection)
    - Bias: Optional bias terms
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        values_init: Tuple[Callable, bool] = (RandomNormal(), True),
        weights_init: Tuple[Callable, bool] = (RandomNormal(), True),
        bias_init: Tuple[Callable, bool] = (RandomNormal(), True),
        scores_init: Tuple[Callable, bool] = (Ones(), False),
        scores_k: float = 1.,
        activation: Callable[[Tensor], Tensor] = F.sigmoid,
        use_bias: bool = False
    ) -> None:
        """
        Initialize the CompleteLayer.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units
            output_size: Number of output units
            values_init: Tuple of (initializer_function, requires_grad) for values.
                Defaults to (RandomNormal, True).
            weights_init: Tuple of (initializer_function, requires_grad) for weights.
                Defaults to (RandomNormal, True).
            bias_init: Tuple of (initializer_function, requires_grad) for bias.
                Defaults to (RandomNormal, True).
            scores_init: [DEPRECATED] Tuple of (initializer_function, requires_grad) for scores
            scores_k: [DEPRECATED] Fraction of connections to keep (0 < k <= 1)
            activation: Activation function to apply. Defaults to sigmoid.
            use_bias: Whether to use bias terms. Defaults to False.
        """
        super(CompleteLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.units = input_size + hidden_size + output_size  # Total vertices in graph

        # Initialize values (internal states for hidden and output vertices)
        self.values = nn.Parameter(
            values_init[0](torch.empty(1, hidden_size + output_size)),
            requires_grad=values_init[1]
        )
        
        # Initialize weights (connections between all vertices)
        self.weights = nn.Parameter(
            weights_init[0](torch.empty(hidden_size + output_size, self.units)),
            requires_grad=weights_init[1]
        )
        
        # [DEPRECATED] Initialize scores (importance of each connection for top-k selection)
        self.scores_k = scores_k
        self.scores = nn.Parameter(
            scores_init[0](torch.empty_like(self.weights)),
            requires_grad=scores_init[1]
        )

        self.activation = activation
        self.use_bias = use_bias
        
        # Initialize bias if requested
        if use_bias:
            self.bias = nn.Parameter(
                bias_init[0](torch.empty_like(self.values)),
                requires_grad=bias_init[1]
            )
    
    def apply(
        self,
        inp: Tensor,
        its: int,
        values: Tensor,
        weights: Tensor,
        scores: Tensor,
        bias: Optional[Tensor] = None
    ) -> Tensor:
        """
        Apply the layer computation with recurrent computation.
        
        This method performs the core computation of the layer:
        1. For each iteration:
           - Concatenate current values with input
           - Apply weights and activation
           - Update values
        2. Return output portion of final values
        
        Args:
            inp: Input tensor of shape (batch_size, input_size)
            its: Number of iterations to run
            values: Initial values for hidden and output nodes
            weights: Weight matrix for all connections
            scores: [DEPRECATED] Importance scores for top-k selection
            bias: Optional bias terms
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch = inp.size(0)
        
        # Expand values to match batch size
        values = values.expand(batch, -1)   # Broadcasting along batch dimension
        
        if self.use_bias:
            bias = bias.expand(batch, -1)   # Broadcasting along batch dimension
        
        # [DEPRECATED] Apply top-k masking to weights based on scores
        scores_mask = topk_mask(scores, self.scores_k)
        w = weights * scores_mask
        
        # Recurrent computation
        for _ in range(its):
            # Concatenate values with input (create complete node state)
            x = torch.cat((values, inp), 1)  # Shape: (batch, total_units)
            
            # Apply linear transformation and activation
            if self.use_bias:
                values = self.activation(x @ w.t() + bias)
            else:
                values = self.activation(x @ w.t())
        
        # Return only the output portion of the values
        return values[:, :self.output_size]
    
    def forward(
        self,
        inp: Tensor,
        its: int = 2,
        **kwargs: Any
    ) -> Tensor:
        """
        Forward pass through the complete layer.
        
        This is the main entry point for the layer. It sets up default parameters
        and calls the apply method to perform the actual computation.
        
        Args:
            inp: Input tensor of shape (batch_size, input_size)
            its: Number of iterations to run (default: 2)
            **kwargs: Additional arguments to override defaults (values, weights, scores, bias)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Set up default parameters
        defaults = {
            'values': self.values,
            'weights': self.weights,
            'scores': self.scores,
            'its': its
        }
        
        # Add bias if using it
        if self.use_bias:
            defaults['bias'] = self.bias
            
        # Merge with any overrides from kwargs
        return self.apply(inp, **(defaults | kwargs))
    
    def norm(self, include_values: bool = True) -> Tensor:
        """
        Compute the norm of the layer's parameters.
        
        This method calculates the combined norm of the layer's parameters,
        useful for regularization and monitoring parameter magnitudes.
        
        Args:
            include_values: Whether to include values in the norm calculation
            
        Returns:
            Scalar tensor containing the combined parameter norm
        """
        # Start with weight matrix norm
        norm = torch.linalg.matrix_norm(self.weights)
        
        # Add values norm if requested
        if include_values:
            norm += torch.linalg.vector_norm(self.values)
            
        # Add bias norm if using bias
        if self.use_bias:
            norm += torch.linalg.vector_norm(self.bias)
            
        return norm
