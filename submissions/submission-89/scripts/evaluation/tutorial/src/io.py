"""
Input/output utilities for GMM evaluation.
"""
import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union, Tuple, Any, Optional


def create_data_loader(dataset_name, batch_size=32, total_samples=512, points_per_gmm=None, device='cuda', fixed_data=True, base_seed=None, loader_id=None):
    """
    Create a data loader for evaluation.
    
    Args:
        dataset_name: Name of dataset configuration
        batch_size: Batch size for data loader
        total_samples: Total number of samples in the dataset (sum of batches lengths)
        points_per_gmm: Number of points sampled from each GMM instance/cluster
        device: Device to load data on
        fixed_data: Whether to use fixed data (recommended for evaluation)
        base_seed: Base seed for reproducibility (optional)
        loader_id: Unique loader ID for reproducibility (optional)
        
    Returns:
        GMMDataLoader: Configured data loader
    """
    from config import get_data_config
    from data.loaders import GMMDataLoader
    
    data_config = get_data_config(dataset_name)
    if points_per_gmm is not None:
        data_config.sample_count_distribution = {'type': 'fixed', 'value': points_per_gmm}
    data_loader = GMMDataLoader(
        config_dict=data_config.model_dump(),
        batch_size=batch_size,
        num_samples=total_samples,
        device=device,
        fixed_data=fixed_data,
        base_seed=base_seed,
        loader_id=loader_id or f"eval_{dataset_name}"
    )
    
    return data_loader


def create_data_samples(dataset_name, num_samples=1, points_per_gmm=100, device='cuda', base_seed=None, loader_id=None):
    """
    Create data samples directly from a dataset configuration.
    
    This function creates a data loader with batch_size and total_samples equal to the number
    of requested samples, then returns the first and only batch.
    
    Args:
        dataset_name: Name of dataset configuration
        num_samples: Number of GMM instances to generate
        points_per_gmm: Number of points sampled from each GMM instance/cluster
        device: Device to load data on
        base_seed: Base seed for reproducibility (optional)
        loader_id: Unique loader ID for reproducibility (optional)
        
    Returns:
        Dict containing:
            'points': Data points [num_samples, points_per_gmm, dim]
            'centers': Ground truth centers [num_samples, n_clusters, dim] 
            'labels': Cluster labels [num_samples, points_per_gmm]
            'snr_db': SNR values [num_samples]
    """
    # Create a data loader with batch_size = total_samples = num_samples
    # This ensures we get exactly the number of samples requested in a single batch
    data_loader = create_data_loader(
        dataset_name=dataset_name,
        batch_size=num_samples,
        total_samples=num_samples,
        points_per_gmm=points_per_gmm,
        device=device,
        fixed_data=True,  # Always use fixed data for reproducibility
        base_seed=base_seed,
        loader_id=loader_id
    )
    
    # Get the first (and only) batch
    for batch in data_loader:
        return batch  # Return the first batch
    
    # If we get here, the data loader was empty (should never happen)
    raise RuntimeError("Failed to generate data samples: data loader returned empty")


def load_model_from_experiment(
    experiment_dir, 
    load_best=False, 
    device=None
):
    """
    Load model from the last checkpoint of an experiment.
    
    Args:
        experiment_dir: Path to the experiment directory
        load_best: Whether to load the best checkpoint instead of the latest one
        device: Device to load the model on
        
    Returns:
        The loaded model and its configuration
    """
    import torch
    from pathlib import Path
    from training.experiment import ExperimentManager
    from utils.checkpointing import get_best_checkpoint, get_latest_checkpoint
    from model.factory import create_model_from_config
    from config.experiment import ExperimentConfig
    
    experiment_dir = Path(experiment_dir)
    
    try:
        # Determine device
        if device is None:
            device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)
            
        # Find checkpoint directory
        checkpoint_dir = experiment_dir / "checkpoints"
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
            
        # Find checkpoint path
        if load_best:
            # Try to find best checkpoint first as "best_model.pt"
            best_path = checkpoint_dir / "best_model.pt"
            if best_path.exists():
                checkpoint_path = best_path
            else:
                # Try to find best checkpoint by metric
                checkpoint_path = get_best_checkpoint(checkpoint_dir)
                
            if checkpoint_path is None:
                raise FileNotFoundError(f"No best checkpoint found in {checkpoint_dir}")
        else:
            # Try to find latest checkpoint first as "latest_model.pt"
            latest_path = checkpoint_dir / "latest_model.pt"
            if latest_path.exists():
                checkpoint_path = latest_path
            else:
                # Try to find latest checkpoint by modification time
                checkpoint_path = get_latest_checkpoint(checkpoint_dir)
                
            if checkpoint_path is None:
                raise FileNotFoundError(f"No latest checkpoint found in {checkpoint_dir}")
                
        print(f"Loading model from checkpoint: {checkpoint_path}")
            
        # Load config
        config_path = experiment_dir / "config.json"
        if config_path.exists():
            config = ExperimentConfig.load(config_path)
        else:
            # Try to load config from checkpoint - don't need state_dict here
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'config' in checkpoint:
                config = ExperimentConfig.model_validate(checkpoint['config'])
            else:
                raise FileNotFoundError(f"No configuration found in experiment directory or checkpoint")
            
        # Create model
        model = create_model_from_config(
            config=config.model,
            device=device
        )
        
        # Load model weights - only need the state_dict here
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Ensure model is in evaluation mode
        model.eval()
        
        return model, config
        
    except FileNotFoundError as e:
        print(f"Experiment directory or checkpoint not found: {e}")
        raise
    except Exception as e:
        print(f"Error loading model from experiment: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_experiment(exp_dir, dataset_name="high_snr_fixed", batch_size=16, 
                   total_samples=512, device="cuda", load_best=False):
    """
    Load a model and data loader from an experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
        dataset_name: Name of dataset to load
        batch_size: Batch size for data loader
        total_samples: Total number of samples in the dataset
        device: Device to load model on ("cpu" or "cuda")
        load_best: Whether to load the best model checkpoint (vs final)
        
    Returns:
        tuple: (model, config, data_loader)
    """
    # Load model using our local implementation
    model, config = load_model_from_experiment(exp_dir, load_best=load_best, device=device)
    
    # Create data loader
    data_loader = create_data_loader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        total_samples=total_samples,
        device=device
    )
    
    return model, config, data_loader


def save_results(results, filepath):
    """
    Save evaluation results to file.
    
    Args:
        results: Evaluation results (single dictionary or list of dictionaries)
        filepath: Path to save to
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata
    metadata = {
        'save_timestamp': datetime.now().isoformat(),
        'is_batched': isinstance(results, list)
    }
    
    # Prepare the save dictionary
    save_dict = {'metadata': metadata}
    
    # Process results based on type
    if isinstance(results, list):
        # Multiple batch results
        metadata['num_batches'] = len(results)
        
        # Add each batch with a unique key
        for i, batch_result in enumerate(results):
            for key, value in batch_result.items():
                # Process based on type
                if isinstance(value, (np.ndarray, np.number, float, int, bool, str)):
                    save_dict[f'batch{i}_{key}'] = value
                elif isinstance(value, torch.Tensor):
                    save_dict[f'batch{i}_{key}'] = value.detach().cpu().numpy()
                elif isinstance(value, dict):
                    # Handle nested dictionaries (like KMeans results)
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, torch.Tensor):
                            save_dict[f'batch{i}_{key}_{sub_key}'] = sub_value.detach().cpu().numpy()
                        elif isinstance(sub_value, (np.ndarray, np.number, float, int, bool, str)):
                            save_dict[f'batch{i}_{key}_{sub_key}'] = sub_value
    else:
        # Single batch result
        metadata['num_batches'] = 1
        
        # Add each field with its key
        for key, value in results.items():
            if isinstance(value, (np.ndarray, np.number, float, int, bool, str)):
                save_dict[f'result_{key}'] = value
            elif isinstance(value, torch.Tensor):
                save_dict[f'result_{key}'] = value.detach().cpu().numpy()
            elif isinstance(value, dict):
                # Handle nested dictionaries
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        save_dict[f'result_{key}_{sub_key}'] = sub_value.detach().cpu().numpy()
                    elif isinstance(sub_value, (np.ndarray, np.number, float, int, bool, str)):
                        save_dict[f'result_{key}_{sub_key}'] = sub_value
    
    # Save using numpy's savez
    np.savez(filepath, **save_dict)


def load_results(filepath):
    """
    Load evaluation results from file.
    
    Args:
        filepath: Path to load from
        
    Returns:
        Evaluation results (single dictionary or list of dictionaries)
    """
    # Load the saved data
    cached_data = np.load(filepath, allow_pickle=True)
    
    # Extract metadata as a python object
    metadata = cached_data['metadata'].item()
    is_batched = metadata.get('is_batched', False)
    num_batches = metadata.get('num_batches', 0)
    
    if is_batched:
        # Multiple batch results
        results = []
        
        # Load each batch result
        for i in range(num_batches):
            batch_result = {}
            
            # Extract all keys for this batch
            batch_prefix = f'batch{i}_'
            for key in cached_data.keys():
                if key.startswith(batch_prefix):
                    # Check for nested dictionaries
                    parts = key[len(batch_prefix):].split('_')
                    
                    if len(parts) > 1:
                        # This is a nested dictionary entry
                        main_key = parts[0]
                        sub_key = '_'.join(parts[1:])
                        
                        if main_key not in batch_result:
                            batch_result[main_key] = {}
                            
                        batch_result[main_key][sub_key] = cached_data[key]
                    else:
                        # Regular entry
                        field_name = key[len(batch_prefix):]
                        batch_result[field_name] = cached_data[key]
            
            if batch_result:  # Only add if we found data
                results.append(batch_result)
                
        return results
    else:
        # Single batch result
        result = {}
        
        # Extract all fields
        result_prefix = 'result_'
        for key in cached_data.keys():
            if key.startswith(result_prefix):
                # Check for nested dictionaries
                parts = key[len(result_prefix):].split('_')
                
                if len(parts) > 1:
                    # This is a nested dictionary entry
                    main_key = parts[0]
                    sub_key = '_'.join(parts[1:])
                    
                    if main_key not in result:
                        result[main_key] = {}
                        
                    result[main_key][sub_key] = cached_data[key]
                else:
                    # Regular entry
                    field_name = key[len(result_prefix):]
                    result[field_name] = cached_data[key]
                    
        return result


def get_cached_results(model, data_id, snr=None, flow_speed=None, 
                      cache_dir=None, force_recompute=False,
                      kmeans_on_inputs=True, kmeans_on_predictions=True,
                      metrics=None, batch_size=32, device='cuda'):
    """
    Get cached results or compute and cache new results.
    
    Args:
        model: Model to evaluate
        data_id: String identifier for data or dataset name
        snr: Optional SNR values to use (will be converted to flow_speed using model's flow_predictor)
        flow_speed: Optional flow speed values (preferred over SNR if both are provided)
        cache_dir: Directory to store cache, or None for default
        force_recompute: Whether to force recomputation even if cache exists
        kmeans_on_inputs: Whether to run KMeans on input data
        kmeans_on_predictions: Whether to run KMeans on model predictions
        metrics: List of metrics to compute (default: all supported metrics)
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        
    Returns:
        Evaluation results (single dict or list of dicts based on input type)
    """
    # Import locally to avoid circular dependencies
    from .eval_utils import evaluate, evaluate_dataset, evaluate_with_snr, evaluate_with_flow
    
    # Set default metrics if not provided
    if metrics is None:
        metrics = [
            'entropy',
            'log_wasserstein',
            'log_kmeans_wasserstein',
            'log_pred_kmeans_wasserstein'
        ]
    
    # Determine if data_id is a dataset name or a tensor ID
    is_dataset = isinstance(data_id, str) and not data_id.startswith('tensor_')
    
    # Create cache directory if not provided
    if cache_dir is None:
        if hasattr(model, 'experiment_dir'):
            # Use model's experiment directory if available
            cache_dir = os.path.join(model.experiment_dir, 'evaluation_cache')
        else:
            # Use current directory
            cache_dir = os.path.join(os.getcwd(), 'evaluation_cache')
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Determine cache file name
    cache_components = ['eval']
    if is_dataset:
        cache_components.append(data_id)
    else:
        cache_components.append('tensor')
    
    if flow_speed is not None:
        cache_components.append('custom_flow')
    elif snr is not None:
        cache_components.append('custom_snr')
    
    # Add KMeans components to cache name
    if kmeans_on_inputs:
        cache_components.append('kmeans_inputs')
    if kmeans_on_predictions:
        cache_components.append('kmeans_pred')
        
    # Add metrics to cache name
    cache_components.append('_'.join(metrics))
    
    # Create cache filename
    cache_filename = f"{'_'.join(cache_components)}.npz"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Check if cache exists
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached evaluation from {cache_path}")
        return load_results(cache_path)
    
    # If we get here, we need to compute and cache the results
    print(f"Computing evaluation results...")
    
    # Prepare data
    if is_dataset:
        # Load dataset
        data_loader = create_data_loader(
            dataset_name=data_id,
            batch_size=batch_size,
            total_samples=512,
            device=device
        )
        results = evaluate_dataset(
            model=model,
            data_loader=data_loader,
            kmeans_on_inputs=kmeans_on_inputs,
            kmeans_on_predictions=kmeans_on_predictions,
            metrics=metrics,
            device=device
        )
    else:
        # Get data from tensor id
        try:
            # In a real implementation, this would properly load the tensor
            # For now, we'll simulate with a placeholder
            data = torch.randn(10, 100, 2, device=device)  # [batch, samples, dim]
            
            # Prepare targets (for log_wasserstein and similar metrics)
            targets = {
                'centers': torch.randn(5, 2, device=device),  # [n_clusters, dim]
                'labels': torch.randint(0, 5, (100,), device=device)  # [samples]
            }
            
            # Evaluate with flow_speed or snr based on availability
            if flow_speed is not None:
                results = evaluate_with_flow(
                    model=model,
                    data=data,
                    flow_speeds=flow_speed,
                    kmeans_on_inputs=kmeans_on_inputs,
                    kmeans_on_predictions=kmeans_on_predictions,
                    metrics=metrics,
                    batch_size=batch_size,
                    device=device,
                    targets=targets
                )
            elif snr is not None:
                results = evaluate_with_snr(
                    model=model,
                    data=data,
                    snr_values=snr,
                    kmeans_on_inputs=kmeans_on_inputs,
                    kmeans_on_predictions=kmeans_on_predictions,
                    metrics=metrics,
                    batch_size=batch_size,
                    device=device,
                    targets=targets
                )
            else:
                raise ValueError("Either snr or flow_speed must be provided for tensor evaluation")
                
        except Exception as e:
            raise ValueError(f"Cannot evaluate with data_id {data_id}: {str(e)}")
    
    # Save results
    save_results(results, cache_path)
    print(f"Evaluation complete and cached to {cache_path}")
    
    return results 