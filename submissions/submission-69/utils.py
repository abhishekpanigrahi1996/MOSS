from torch import nn
from torch import Tensor
import itertools
import numpy as np
from typing import Optional, Tuple, Any, List, Union


class ChainableFn:
    """
    Base class for creating chainable functions that can be composed together.
    
    This class implements a pattern where functions can be chained together
    using composition, allowing for complex transformations to be built from
    simpler components. Each function can optionally take a previous function
    as input and will apply it first before applying its own transformation.
    """
    
    name: Optional[str] = None
    
    def __init__(self, prev: Optional['ChainableFn'] = None) -> None:
        """
        Initialize a chainable function.
        
        Args:
            prev: Previous function in the chain to apply first, or None
                  if this is the first function in the chain
        """
        self.prev = prev
        if prev is not None:
            self.name = f"{self.name}({prev.name})"
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Apply the function chain.
        
        If there's a previous function in the chain, apply it first,
        then apply this function to the result.
        
        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of applying the function chain
        """
        if self.prev is not None:
            # Apply the previous function first
            data = self.prev(*args, **kwargs)
            return self.fn(data, **kwargs)
        return self.fn(*args, **kwargs)

    def __str__(self) -> str:
        """
        Return string representation of the function chain.
        
        Returns:
            The name of the function chain
        """
        return self.name or "ChainableFn"

    def fn(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method to be implemented by subclasses.
        
        This method should contain the actual function logic.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            The result of applying this function
        """
        raise NotImplementedError("Subclasses must implement the fn method")


def filter_topk(
    tensor: Tensor,
    k: float,
    return_mask: bool = False
) -> Tensor:
    """
    Filter a tensor to keep only the top-k largest elements by absolute value.
    
    This function creates a sparse version of the input tensor by keeping only
    the k largest elements (by absolute value) and setting all other elements
    to zero. Optionally, it can return a binary mask instead of the filtered tensor.
    
    Args:
        tensor: Input tensor to filter
        k: Fraction of elements to keep (between 0 and 1). For example,
           k=0.1 keeps the top 10% of elements
        return_mask: If True, return a binary mask where 1 indicates kept
                    elements and 0 indicates zeroed elements. If False,
                    return the filtered tensor with non-top-k elements set to 0
        
    Returns:
        Either the filtered tensor (if return_mask=False) or a binary mask
        (if return_mask=True)
        
    Example:
        >>> x = torch.tensor([[1.0, 0.1, 0.5], [0.2, 0.8, 0.3]])
        >>> filtered = filter_topk(x, 0.5)  # Keep top 50% of elements
        >>> mask = filter_topk(x, 0.5, return_mask=True)
    """
    out = tensor.clone()
    flat = out.flatten()

    # Sort indices by absolute value in descending order
    order = flat.abs().argsort(descending=True)
    n = int(k * flat.size(0))  # Number of elements to keep

    # Zero out the smallest elements (out is mutable; flat accesses its memory)
    flat[order[n:]] = 0
    
    if return_mask:
        # Convert to binary mask: set kept elements to 1
        flat[order[:n]] = 1
    
    return out


def permute(
    x: Union[Tensor, np.ndarray],
    perm_0: List[int],
    perm_1: Optional[List[int]] = None
) -> Union[Tensor, np.ndarray]:
    """
    Permute the rows and columns of a 2D array or tensor.
    
    This function reorders the rows and columns of a 2D array according to
    the specified permutations. If only one permutation is provided, it's
    applied to both rows and columns.
    
    Args:
        x: Input 2D array or tensor to permute
        perm_0: Permutation to apply to rows (first dimension)
        perm_1: Permutation to apply to columns (second dimension).
                If None, uses the same permutation as perm_0
        
    Returns:
        The permuted array or tensor
        
    Example:
        >>> x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> perm = [2, 0, 1]  # Swap rows/columns
        >>> permuted = permute(x, perm)
    """
    if perm_1 is None:
        perm_1 = perm_0
    return x[perm_0][:, perm_1]


def brute_force_orderedness(
    adj_mat: np.ndarray,
    fixed_size: int = 0
) -> Tuple[float, Optional[List[int]]]:
    """
    Compute the orderedness score of an adjacency matrix using brute force.
    The adjacency matrix is assumed to be square.
    
    This function measures how topologically-ordered a weighted graph is
    by finding the permutation that maximizes the lower triangular sum.
    A higher orderedness score indicates a higher degree of topological order.

    Returns 0 if all weights are 0.
    
    The algorithm:
    1. Tries all possible permutations of the non-input nodes
    2. For each permutation, computes the lower triangular sum
    3. Returns the orderedness score as 1 - (max_lower_sum / total_sum)
    
    Args:
        adj_mat: Adjacency matrix of the graph (2D numpy array)
        fixed_size: Number of fixed nodes (these are kept fixed at the beginning)
        
    Returns:
        Tuple containing:
        - orderedness: Float between 0 and 1, where 1 is fully topologically ordered
        - min_perm: The permutation that achieved the maximum orderedness,
                   or None if no permutation was found
                   
    Example:
        >>> adj_mat = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        >>> orderedness, perm = brute_force_orderedness(adj_mat, 1)
        >>> print(f"Orderedness: {orderedness}")
    """
    assert adj_mat.shape[0] == adj_mat.shape[1], "Adjacency matrix must be square"
    n = adj_mat.shape[0]
    abs_mat = np.abs(adj_mat)  # Work with absolute values
    header = list(range(fixed_size))  # Input nodes stay fixed
    
    min_cost = np.inf
    min_perm = None

    # Try all permutations of non-input nodes
    for _p in itertools.permutations(range(fixed_size, n)):
        perm = header + list(_p)
        mat = permute(abs_mat, perm)
        
        # Compute lower triangular sum (excluding diagonal)
        cost = np.sum(np.tril(mat, -1))

        if cost < min_cost:
            min_cost = cost
            min_perm = perm

    # Orderedness is 1 - (lower_triangular_sum / total_sum)
    # Higher values indicate more topologically-ordered structure
    total_sum = np.sum(abs_mat)
    orderedness = 1 - (min_cost / total_sum)

    return float(orderedness), min_perm

def weights_orderedness(
    model: nn.Module,
) -> float:
    """
    Compute the orderedness score of a weight matrix using brute force.

    Args:
        model: A PyTorch model with a weights attribute

    Returns:
        Tuple containing:
        - orderedness: Float between 0 and 1, where 1 is fully topologically ordered
        - min_perm: The permutation that achieved the maximum orderedness.
    """
    square = model.weights[:, :-model.input_size].detach().cpu().numpy()
    orderedness, perm = brute_force_orderedness(square, fixed_size=model.output_size)
    return orderedness, perm
