import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Callable, Union

from pruning import no_pruning, PruneEnsemble
from utils import weights_orderedness


def train(
    model: Module,
    n_epochs: int,
    optimiser: Optimizer,
    train_dataloader: DataLoader,
    train_criterion: Callable[[Tensor, Tensor], Tensor],
    val_dataloader: Optional[DataLoader] = None,
    val_criterion: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    pruner: PruneEnsemble = no_pruning,
    early_stop: Optional[float] = None,
    trainable: bool = True,
    show_pbar: bool = True,
    leave_pbar: bool = True,
    verbose: bool = True,
    its: Optional[int] = None,
    track_orderedness: bool = False
) -> Dict[str, Any]:
    """
    Train a neural network model with optional pruning and orderedness tracking.
    
    Args:
        model: PyTorch model to train
        n_epochs: Number of training epochs
        optimiser: PyTorch optimizer for parameter updates
        train_dataloader: DataLoader for training data
        train_criterion: Loss function for training (takes y_pred, y_true)
        val_dataloader: Optional DataLoader for validation data
        val_criterion: Optional loss function for validation
        pruner: PruneEnsemble for applying pruning strategies during training
        early_stop: Early stopping threshold based on training loss
        trainable: Whether to train the model
        show_pbar: Whether to show progress bar
        leave_pbar: Whether to leave progress bar after completion
        verbose: Whether to print verbose output
        its: Optional number of iterations for models that support it
        track_orderedness: Whether to track network orderedness over time
        
    Returns:
        Dictionary containing training results:
        - 'model': The trained model
        - 'train_losses': List of training losses per step
        - 'val_losses': List of validation losses per step
        - 'val_steps': List of step numbers where validation was performed
        - 'delta_steps': List of orderedness changes per step
        - 'delta_final': Final orderedness change
    """
    
    def apply(*args: Any, **kwargs: Any) -> Tensor:
        """
        Apply the model with optional iteration count.
        
        This wrapper allows for models that support an 'its' parameter
        (like CompleteLayer) while maintaining compatibility with standard models.
        
        Args:
            *args: Positional arguments to pass to the model
            **kwargs: Keyword arguments to pass to the model
            
        Returns:
            Model output tensor
        """
        if its is None:
            return model(*args, **kwargs)
        else:
            return model(*args, **kwargs, its=its)
    
    def _orderedness() -> float:
        """
        Calculate the current orderedness of the model's weight matrix.
        
        Returns:
            Orderedness score (0-1, where 1 is fully feed-forward)
        """
        if hasattr(model, 'weights'):
            return weights_orderedness(model)[0]
        else:
            return float('nan') # not applicable

    # Initialise tracking variables
    step = 0
    train_losses = []
    val_losses = []
    val_steps = []
    delta_steps = []
    start_o = _orderedness()

    # Main training loop
    for epoch in range(n_epochs):

        train_loss = 0.0
        pbar = tqdm(enumerate(train_dataloader),
                    leave=leave_pbar,
                    disable=not show_pbar)

        # Training loop
        for batch, (x, y) in pbar:
            
            # Create context dictionary with training information
            ctx = {
                'model': model,
                'step': step,
                'epoch': epoch,
                'batch': batch,
                'total_steps': len(train_dataloader),
                'total_epochs': n_epochs,
                'progress': epoch / n_epochs,  # Training progress (0-1)
                'train': True
            }

            # Standard training step
            optimiser.zero_grad()
            y_pred = apply(x)
            loss = train_criterion(y_pred, y, **ctx)
            if trainable:
                loss.backward()
                optimiser.step()

            # Record training loss
            _loss = loss.item()
            train_losses.append(_loss)
            pbar.set_postfix({'loss': _loss})
            train_loss += _loss
            step += 1

            # Apply pruning
            pruner.prune(model, **ctx)

            # Track orderedness
            if track_orderedness:
                current_o = _orderedness()
                delta_steps.append(current_o - start_o)
        
        # Validation loop (if validation data provided)
        if val_dataloader is not None:
            val_loss = 0.0
            
            with torch.no_grad():
                for x, y in val_dataloader:
                    # Create context for validation
                    ctx = {
                        'model': model,
                        'step': step,
                        'epoch': epoch,
                        'batch': batch,
                        'train': False
                    }
                    
                    # Evaluate on validation data
                    y_pred = apply(x)
                    loss = val_criterion(y_pred, y, **ctx)
                    _loss = loss.item()
                    val_loss += _loss
                    val_losses.append(_loss)
                    val_steps.append(step - 1) # step had been incremented in training loop
        
        # Calculate average losses for this epoch
        train_loss = train_loss / len(train_dataloader)
        
        # Create description string for progress bar
        description = f'EP {epoch:4d} Trn Loss: {train_loss:.4f}'
        if val_dataloader is not None:
            val_loss = val_loss / len(val_dataloader)
            description += f' Val Loss: {val_loss:.4f}'
        pbar.set_description(description)
        
        # Check for early stopping
        if early_stop is not None and train_loss < early_stop:
            if verbose:
                print(f'Early stop. {description}')
            break
    
    final_o = _orderedness()
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_steps': val_steps,
        'delta_steps': delta_steps,
        'delta_final': final_o - start_o,
        'start_orderedness': start_o,
        'final_orderedness': final_o
    }
