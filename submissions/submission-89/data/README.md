# GMM v2 Data Generation

This is a separate component of the GMM v2 project for data generation using Gaussian Mixture Models.

## Components

- `core/` - Core data generation algorithms
  - `data_generator.py` - Generates GMM data batches
  - `parameter_generator.py` - Creates GMM parameters
  - `gmm_params.py` - Data class for GMM parameters
  - `noise_controller.py` - Handles noise level calculations

- `loaders/` - Data loading interfaces
  - `data_loader.py` - Iterator interface for data generation
  - `data_manager.py` - Factory for data loaders
  - `factory.py` - Creates generators from configuration

- `utils/` - Utility functions
  - `distribution_utils.py` - Parameter distribution sampling
  - `random_utils.py` - Random state management

## Installation

```bash
pip install -e .
```

## Usage

See the documentation and tutorials for examples.
