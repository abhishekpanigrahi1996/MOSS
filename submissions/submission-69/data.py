import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import Literal, Dict, Any, Union


class TaskGenerator:
    """
    Generator for machine learning tasks with predefined datasets and hyperparameters.
    
    This class provides a unified interface for generating datasets and baseline models
    for specific tasks like XOR classification and sine function regression. It includes
    hyperparameters optimized for each task and methods to create appropriate baseline
    models.
    """
    
    # Task-specific hyperparameters optimized for each problem
    HYPERPARAMETERS = {
        'none': {
            'batch_size': 4,
            'baseline_lr': 0.01,
            'baseline_epochs': 10,
            'input_size': 2,
            'hidden_size': 5,
            'output_size': 1,
            'its': 1,
            'complete_lr': 0.01,
            'complete_epochs': 10,
        },
        'xor': {
            'batch_size': 4,
            'baseline_lr': 0.01,
            'baseline_epochs': 10000,
            'input_size': 2,
            'hidden_size': 5,
            'output_size': 1,
            'its': 3,
            'complete_lr': 0.01,
            'complete_epochs': 1000,
        },
        'sine': {
            'batch_size': 10,
            'baseline_lr': 0.01,
            'baseline_epochs': 1000,
            'input_size': 2,
            'hidden_size': 8,
            'output_size': 1,
            'its': 3,
            'complete_lr': 0.01,
            'complete_epochs': 40,
        }
    }

    def __init__(
        self,
        name: Literal['xor', 'sine', 'none'],
        device: Literal['cpu', 'cuda'] = 'cpu'
    ) -> None:
        """
        Initialize a task generator for a specific task.
        
        Args:
            name: Name of the task to generate ('xor', 'sine', or 'none')
            device: Device to place tensors on ('cpu' or 'cuda')
        """
        self.name = name
        self.device = device

        # Get hyperparameters for this task
        self.params = self.HYPERPARAMETERS[self.name]
        self._init_data()

    def _init_data(self) -> None:
        """
        Initialize the dataset and dataloader for the specified task.
        
        Creates task-specific input-output pairs:
        - XOR: Creates truth table for XOR function with 4 samples
        - Sine: Creates sine wave samples over [0, 3] range
        """
        if self.name in ('none', 'xor'):
            # Create XOR truth table: [[0,0], [0,1], [1,0], [1,1]]
            x = torch.tensor(
                [[0, 0], [0, 1], [1, 0], [1, 1]],
                dtype=torch.float32,
                device=self.device
            )
            # XOR outputs: [0, 1, 1, 0]
            y = torch.unsqueeze(
                torch.logical_xor(x[:, 0], x[:, 1]),
                dim=1
            ).to(torch.float32).to(self.device)
        
        elif self.name == 'sine':
            x = torch.tensor(
                np.arange(0, 3, 0.01).reshape(-1, 2),
                dtype=torch.float32
            ).to(self.device)
            y = torch.unsqueeze(
                torch.sin(x).sum(1) / 2,
                dim=1
            ).to(torch.float32).to(self.device)
        
        # Create dataset and dataloader
        self.dataset = TensorDataset(x, y)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.params['batch_size'],
            shuffle=True
        )

    def get_mlp_baseline(self) -> nn.Module:
        """
        Create a baseline MLP model appropriate for the task.
        
        Returns task-specific architectures:
        - XOR: 2-layer MLP with sigmoid activation (2→2→1)
        - Sine: 3-layer MLP with sigmoid activation (2→10→10→1)
        
        Returns:
            A PyTorch Sequential model configured for the task
        """
        if self.name in ('none', 'xor'):
            # Simple 2-layer network for XOR
            return nn.Sequential(
                nn.Linear(2, 2, bias=True),
                nn.Sigmoid(),
                nn.Linear(2, 1, bias=True),
                nn.Sigmoid()
            ).to(self.device)
    
        if self.name == 'sine':
            # Deeper network for sine function approximation
            return nn.Sequential(
                nn.Linear(2, 10, bias=True),
                nn.Sigmoid(),
                nn.Linear(10, 10, bias=True),
                nn.Sigmoid(),
                nn.Linear(10, 1, bias=True),
                nn.Sigmoid(),
            ).to(self.device)
