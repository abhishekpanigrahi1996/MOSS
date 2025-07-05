# Enhanced Model Management for GMM-v2

## Overview
The goal is to create a more intuitive, high-level API for model management that makes it easier to train, save, load, and evaluate GMM models. This will build on the existing ExperimentManager class, extending it with new capabilities while maintaining backward compatibility.

## Key Components

### 1. ModelManager Class

Create a new `ModelManager` class in `utils/model_management.py` that will handle all model-related tasks:

```python
class ModelManager:
    """
    High-level interface for model management operations.
    
    This class provides methods for saving, loading, evaluating, and
    visualizing models across different experiments.
    """
    
    def __init__(
        self, 
        experiment_dir=None,
        model=None,
        config=None
    ):
        # Initialize with an existing experiment or a new model
        pass
    
    @classmethod
    def from_experiment(cls, experiment_manager):
        # Create from an existing ExperimentManager
        pass
        
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device=None):
        # Load model directly from checkpoint
        pass
        
    @classmethod
    def from_experiment_dir(cls, experiment_dir):
        # Load model from experiment directory
        pass
    
    def save(self, path):
        # Save model to file using PyTorch's save mechanism
        pass
    
    def evaluate(self, data_loader=None, metrics=None):
        # Evaluate model on data
        pass
    
    def visualize(self, data_loader=None, num_samples=5):
        # Generate visualizations of model predictions
        pass
```

### 2. Enhanced ExperimentManager

Extend the ExperimentManager to include model management functionality:

```python
# Extend ExperimentManager with the following methods:

def get_model_manager(self):
    """Get a ModelManager for this experiment."""
    return ModelManager.from_experiment(self)

def export_model(self, path):
    """Export model."""
    manager = self.get_model_manager()
    return manager.save(path)

def save_checkpoint(self, path=None, is_best=False):
    """Explicitly save a checkpoint."""
    # Delegate to the trainer's checkpoint save functionality
    pass

@classmethod
def load_best_model(cls, experiment_dir):
    """Load the best model from an experiment directory."""
    # Find and load the best checkpoint
    pass

@classmethod
def load_latest_model(cls, experiment_dir):
    """Load the latest model from an experiment directory."""
    # Find and load the latest checkpoint
    pass

def resume_training(self, num_epochs=None):
    """Resume training from the latest checkpoint."""
    # Get latest checkpoint
    checkpoint_dir = self.experiment_dir / "checkpoints"
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Calculate total epochs
    if num_epochs is not None:
        # Continue for additional epochs
        total_epochs = self.trainer.current_epoch + num_epochs
    else:
        # Continue for the original number of epochs
        total_epochs = self.config.training.num_epochs
    
    # Resume training
    return self.run(
        num_epochs=total_epochs,
        resume_from_checkpoint=latest_checkpoint
    )
```

### 3. Enhanced Training with Automatic Checkpointing

Modify the training process to ensure the latest model is always saved:

```python
# Update GMMTrainer.train method to save the latest model after each epoch
def train(self, num_epochs=None, resume_from_checkpoint=None):
    # ... existing code ...
    
    # Training loop
    for epoch in range(start_epoch, end_epoch):
        self.current_epoch = epoch
        
        # Train for one epoch
        train_loss = self._train_epoch()
        
        # ... existing validation code ...
        
        # Always save the latest model after each epoch
        self._save_checkpoint(is_latest=True)
        
        # ... rest of existing code ...
```

```python
# Update _save_checkpoint method to handle "latest" checkpoints
def _save_checkpoint(self, is_best=False, is_final=False, is_latest=False):
    # ... existing code ...
    
    # Save checkpoint based on type
    if is_best:
        path = checkpoint_dir / "best_model.pt"
    elif is_final:
        path = checkpoint_dir / "final_model.pt"
    elif is_latest:
        path = checkpoint_dir / "latest_model.pt"
    else:
        path = checkpoint_dir / f"checkpoint_epoch_{self.current_epoch+1}.pt"
        
    torch.save(checkpoint, path)
```

## API Example Usage

Here's an example of how the enhanced API could be used:

```python
# Training and saving a model
config = ExperimentConfig(...)
experiment = ExperimentManager(config)
experiment.setup()
experiment.run(num_epochs=10)  # Will save latest_model.pt after each epoch

# Load the best model from an experiment
best_experiment = ExperimentManager.load_best_model("./runs/12345")

# Load and resume training from latest checkpoint
experiment = ExperimentManager.load_latest_model("./runs/12345")
experiment.resume_training(num_epochs=5)  # Train for 5 more epochs

# Evaluate on new data
data_loader = create_data_loader(...)
metrics = experiment.evaluate(data_loader=data_loader)

# Load a specific checkpoint and evaluate
model_manager = ModelManager.from_checkpoint("./runs/12345/checkpoints/checkpoint_epoch_5.pt")
metrics = model_manager.evaluate(data_loader)

# Visualize model outputs
model_manager.visualize(data_loader, num_samples=3)

# Export model 
model_manager.save("./exported_model.pt")
```