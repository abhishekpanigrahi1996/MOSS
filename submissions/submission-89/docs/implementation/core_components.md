# Core Components Implementation

This document provides an overview of the core components of the GMM data generation framework.

## RandomState (`utils/random_utils.py`)

Handles reproducible random state management through a thin wrapper around NumPy's random Generator API:

```python
class RandomState:
    def __init__(self, seed=None):
        self.generator = np.random.Generator(np.random.PCG64(seed))
        
    def get_state(self):
        return {"bit_generator": self.generator.bit_generator.state}
        
    def set_state(self, state):
        self.generator = np.random.Generator(np.random.PCG64())
        self.generator.bit_generator.state = state["bit_generator"]
```

## GMMParams (`core/gmm_params.py`)

Defines the parameters of a GMM distribution and provides methods for generating samples:

```python
class GMMParams:
    def __init__(self, dim, n_clusters, weights, means, 
                 control_mode='snr', snr_db=None, noise_std=None, 
                 target_mi=None, mi_factor=None, seed=None):
        self.dim = dim
        self.n_clusters = n_clusters
        self.weights = weights
        self.means = means
        self.control_mode = control_mode
        # ... initialize parameters based on control mode
        
    def generate_samples(self, n_samples, seed=None):
        # Generate random cluster assignments based on weights
        # Then generate samples by adding noise to the cluster means
        pass
```

## NoiseController (`core/noise_controller.py`)

Calculates noise levels based on SNR or mutual information targets:

```python
class NoiseController:
    @staticmethod
    def snr_to_noise_std(means, weights, snr_db):
        # Calculate signal power from means and weights
        # Convert SNR in dB to linear scale
        # Return noise standard deviation
        pass
        
    @staticmethod
    def mi_to_noise_std(means, weights, target_mi):
        # Use the inverse problem solver to find noise level
        # that produces the target mutual information
        pass
        
    @staticmethod
    def mi_factor_to_mi(means, weights, mi_factor):
        # Convert MI factor (0-1) to absolute MI
        # based on the maximum possible MI (entropy of X)
        pass
```

## ParameterGenerator (`core/parameter_generator.py`)

Generates GMM parameters according to specified distributions:

```python
class ParameterGenerator:
    def __init__(self, dim, cluster_config, sample_count_config=None,
                 snr_config=None, mi_config=None, mi_factor_config=None, 
                 seed=None):
        self.dim = dim
        self.cluster_config = cluster_config
        # ... store configurations
        self.random_state = RandomState(seed)
        
    def generate(self, batch_size=1):
        # Generate parameters for batch_size datasets
        params_list = []
        for _ in range(batch_size):
            # Sample parameters from specified distributions
            # Create GMMParams instances
            params_list.append(gmm_params)
        return params_list, sample_count
```

## DataGenerator (`core/data_generator.py`)

Generates data batches from GMM parameters:

```python
class DataGenerator:
    def __init__(self, seed=None):
        self.random_state = RandomState(seed)
        
    def generate_batch(self, params_list, sample_count):
        # Generate samples from each set of parameters
        # Return data and labels
        pass
        
    def generate_training_batch(self, params_list, sample_count):
        # Generate samples formatted for training
        # Return data and labels as tensors
        pass
```