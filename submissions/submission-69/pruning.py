import math
import torch
from torch import Tensor
from torch.nn import Module
from typing import Optional, Dict, Any, Callable, Union

from utils import ChainableFn, filter_topk


class NoPrune(ChainableFn):
    """
    Identity pruning function that applies no pruning.
    
    This is the default pruning strategy that passes tensors through unchanged.
    Useful as a baseline or when no pruning is desired.
    """
    
    name = "NoPrune"
    
    def fn(self, arr: Tensor, **ctx: Any) -> Tensor:
        """
        Apply no pruning (identity function).
        
        Args:
            arr: Input tensor to be "pruned"
            **ctx: Additional context information (unused)
            
        Returns:
            The input tensor unchanged
        """
        return arr


class RandomPrune(ChainableFn):
    """
    Random pruning that sets a fraction of weights to zero.
    
    This pruning strategy randomly selects a percentage of weights to set to zero,
    simulating the effect of random weight dropout or random network sparsification.
    """
    
    name = "RandomPrune"
    
    def __init__(
        self,
        prev: Optional[ChainableFn] = None,
        p: float = 0.,
        persistent: bool = True
    ) -> None:
        """
        Initialize random pruning.
        
        Args:
            prev: Previous function in the chain (if any)
            p: Probability of keeping each weight (0 = prune all, 1 = keep all)
            persistent: If True, use the same mask throughout training. If False,
                        generate a new random mask each time
        """
        super().__init__(prev)
        self.p = float(p)
        self.persistent = persistent
        self.initialised = False

    def fn(self, arr: Tensor, **ctx: Any) -> Tensor:
        """
        Apply random pruning to the input tensor.
        
        Args:
            arr: Input tensor to be pruned
            **ctx: Additional context information (unused)
            
        Returns:
            Tensor with randomly selected elements set to zero
        """
        if not self.persistent or not self.initialised:
            self.initialised = True
            # Generate random mask: True means keep, False means prune
            self.mask = torch.rand(arr.shape) < self.p
        
        # Apply mask (set pruned elements to zero)
        return arr.masked_fill(self.mask.to(arr.device), 0)


class ThresholdPrune(ChainableFn):
    """
    Magnitude-based pruning that removes weights below a threshold.
    
    This pruning strategy removes weights whose absolute values are below
    a specified threshold. If no threshold is provided, it automatically
    determines threshold value to prune 50% of the weights.
    """
    
    name = "ThresholdPrune"
    
    def __init__(
        self, 
        prev: Optional[ChainableFn] = None, 
        tau: Optional[float] = None
    ) -> None:
        """
        Initialize threshold pruning.
        
        Args:
            prev: Previous function in the chain (if any)
            threshold: Magnitude threshold below which weights are pruned.
                       If None, automatically set to prune 50% of weights
        """
        super().__init__(prev)
        self.threshold = float(tau) if tau is not None else None

    def fn(self, arr: Tensor, **ctx: Any) -> Tensor:
        """
        Apply threshold-based pruning to the input tensor.
        
        Args:
            arr: Input tensor to be pruned
            **ctx: Additional context information (unused)
            
        Returns:
            Tensor with small-magnitude elements set to zero
        """
        if self.threshold is None:
            # Auto-determine threshold to prune 50% of weights
            flattened = arr.abs().flatten()
            k = len(flattened) // 2  # Remove 50% of weights
            self.threshold = torch.kthvalue(flattened, k).values.item()
        
        # Create mask for elements below threshold
        idx = arr.abs() < self.threshold
        return arr.masked_fill(idx, 0)


class TopKPrune(ChainableFn):
    """
    Top-k pruning that keeps only the top-k largest weights by magnitude.
    
    This pruning strategy keeps only the top-k weights (by absolute value)
    and sets all other weights to zero, creating a sparse network.
    """
    
    name = "TopKPrune"
    
    def __init__(
        self, 
        prev: Optional[ChainableFn] = None, 
        k: float = 1.
    ) -> None:
        """
        Initialize top-k pruning.
        
        Args:
            prev: Previous function in the chain (if any)
            k: Fraction of weights to keep (0 < k <= 1)
        """
        super().__init__(prev)
        self.k = float(k)

    def fn(self, arr: Tensor, **ctx: Any) -> Tensor:
        """
        Apply top-k pruning to the input tensor.
        
        Args:
            arr: Input tensor to be pruned
            **ctx: Additional context information (unused)
            
        Returns:
            Tensor with only the top-k largest elements preserved
        """
        return filter_topk(arr, self.k, return_mask=False)


class DynamicTopK(ChainableFn):
    """
    Dynamic top-k pruning with time-varying sparsity.
    
    This pruning strategy gradually increases sparsity over time using a
    smooth mathematical function. The sparsity level depends on training progress.
    """
    
    name = "DynamicTopK"
    
    def __init__(
        self,
        prev: Optional[ChainableFn] = None,
        k: float = 1.
    ) -> None:
        """
        Initialize dynamic top-k pruning.
        
        Args:
            prev: Previous function in the chain (if any)
            k: Final target fraction of weights to keep
        """
        super().__init__(prev)
        self.k = k
        self.eqn = lambda x: 1 - (1 - k) * (math.sin(math.pi / 2 * x) ** 4)

    def fn(self, arr: Tensor, **ctx: Any) -> Tensor:
        """
        Apply dynamic top-k pruning based on training progress.
        
        Args:
            arr: Input tensor to be pruned
            **ctx: Context dictionary that must contain 'progress' (0-1)
            
        Returns:
            Tensor with dynamically determined sparsity level
        """
        # Calculate current k value based on progress
        current_k = self.eqn(ctx['progress'])
        return filter_topk(arr, current_k, return_mask=False)


class TrilDamp(ChainableFn):
    """
    Lower triangular damping that reduces feedback connections.
    
    This function dampens the lower triangular part of weight matrices,
    encouraging feed-forward connectivity patterns by reducing recurrent
    connections.
    """
    
    name = "TrilDamp"
    
    def __init__(
        self,
        prev: Optional[ChainableFn] = None,
        diagonal: int = 0,
        f: float = 0.9
    ) -> None:
        """
        Initialize lower triangular damping.
        
        Args:
            prev: Previous function in the chain (if any)
            diagonal: Which diagonal to use as boundary (0 = main diagonal)
            f: Damping factor (fraction of each lower triangular element to remove)
        """
        super().__init__(prev)
        self.diag = diagonal
        self.f = f
    
    def fn(self, arr: Tensor, **ctx: Any) -> Tensor:
        """
        Apply lower triangular damping to the input tensor.
        
        Args:
            arr: Input tensor to be damped
            **ctx: Additional context information (unused)
            
        Returns:
            Tensor with dampened lower triangular elements
        """
        # Subtract fraction from each lower triangular element
        return arr - arr.tril(self.diag - 1) * self.f


class DynamicTrilDamp(ChainableFn):
    """
    Dynamic lower triangular damping with time-varying intensity.
    
    This function applies lower triangular damping with intensity that
    varies over time, allowing for gradual transition to feed-forward
    connectivity patterns.
    """
    
    name = "DynamicTrilDamp"
    
    def __init__(
        self,
        prev: Optional[ChainableFn] = None,
        diagonal: int = 0,
        f: float = 0.9
    ) -> None:
        """
        Initialize dynamic lower triangular damping.
        
        Args:
            prev: Previous function in the chain (if any)
            diagonal: Which diagonal to use as boundary (0 = main diagonal)
            f: Maximum damping factor
        """
        super().__init__(prev)
        self.diag = diagonal
        self.f = f
        self.eqn = lambda x: f * (math.sin(math.pi / 2 * x) ** 4)
    
    def fn(self, arr: Tensor, **ctx: Any) -> Tensor:
        """
        Apply dynamic lower triangular damping based on training progress.
        
        Args:
            arr: Input tensor to be damped
            **ctx: Context dictionary that must contain 'progress' (0-1)
            
        Returns:
            Tensor with dynamically dampened lower triangular elements
        """
        # Calculate current damping factor based on progress
        current_f = self.eqn(ctx['progress'])
        return arr - arr.tril(self.diag - 1) * current_f


class TrilPrune(ChainableFn):
    """
    Lower triangular pruning that completely removes non-topologically-
    ordered connections.
    
    This function sets all lower triangular elements to zero, enforcing
    strict topological ordering.
    """
    
    name = "TrilPrune"
    
    def __init__(
        self,
        prev: Optional[ChainableFn] = None,
        diagonal: int = 0
    ) -> None:
        """
        Initialize lower triangular pruning.
        
        Args:
            prev: Previous function in the chain (if any)
            diagonal: Which diagonal to use as boundary (0 = main diagonal)
        """
        super().__init__(prev)
        self.diag = diagonal
    
    def fn(self, arr: Tensor, **ctx: Any) -> Tensor:
        """
        Apply lower triangular pruning to the input tensor.
        
        Args:
            arr: Input tensor to be pruned
            **ctx: Additional context information (unused)
            
        Returns:
            Tensor with lower triangular elements set to zero
        """
        # Keep only upper triangular elements
        return arr.triu(self.diag)


class PruneEnsemble:
    """
    Ensemble of pruning strategies applied to different model parameters.
    
    This class allows for applying different pruning strategies to different
    named parameters in a model. Each parameter can have its own pruning
    configuration, enabling fine-grained control over network sparsity.
    """
    
    def __init__(
        self, 
        config: Optional[Dict[str, ChainableFn]], 
        requires_grad: bool = False
    ) -> None:
        """
        Initialize the pruning ensemble.
        
        Args:
            config: Dictionary mapping parameter names to pruning functions.
                   If None, no pruning is applied.
            requires_grad: Whether pruning operations should be differentiable
        """
        self.cfg = config
        self.requires_grad = requires_grad
    
    def prune(
        self,
        module: Module,
        **ctx: Any
    ) -> Module:
        """
        Apply pruning to the module's parameters.
        
        Args:
            module: PyTorch module whose parameters should be pruned
            **ctx: Context information passed to pruning functions
            
        Returns:
            The module with pruned parameters (modified in-place)
        """
        if not self.cfg:
            return module
        
        # Apply pruning to each configured parameter
        with torch.no_grad():
            for name, param in module.named_parameters():
                if name in self.cfg:
                    # Apply the pruning function for this parameter
                    pruned_param = self.cfg[name](param, **ctx)
                    param.copy_(pruned_param)
        
        return module


# Global instance for no pruning (convenience)
no_pruning = PruneEnsemble(None)
