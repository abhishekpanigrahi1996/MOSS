# GMM Model Evaluation Tutorial

*Submission ID: 89*

## Overview

This submission contains a comprehensive tutorial and evaluation framework for Gaussian Mixture Models (GMMs). The project provides tools for training, evaluating, and visualizing GMM models across different dataset configurations and noise levels.

## Key Features

- **Comprehensive GMM Evaluation**: Complete framework for evaluating GMM models with various configurations
- **Dataset Comparison**: Tools for comparing performance across different dataset types (Simple, Standard, Complex)
- **SNR Analysis**: Evaluation of model performance under different Signal-to-Noise Ratios
- **Visualization Pipeline**: Visualization tools for data exploration and model comparison
- **Training Framework**: Configurable training system with multiple model architectures
- **Baseline Comparisons**: Integration with classical methods like K-means clustering

## Main Components

### Notebooks
- `gmm_evaluation_tutorial.ipynb`: Main tutorial notebook with step-by-step evaluation process

### Configuration
- `config/`: Model, training, and dataset configurations
- `pyproject.toml`: Project dependencies and settings
- `requirements.txt`: Python package requirements

## Environment Setup

### Option 1: Using uv (Recommended)

1. Install uv if you haven't already:
   ```bash
   pip install uv
   ```

2. Create and activate the environment:
   ```bash
   uv sync
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

### Option 2: Using venv and pip

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Getting Started

1. After setting up the environment (using either option above), run the main tutorial:
   ```bash
   jupyter notebook gmm_evaluation_tutorial.ipynb
   ```

## Experiment Types

The tutorial covers several evaluation scenarios:

- **Dataset Complexity**: Comparing model performance on Simple, Standard, and Complex datasets
- **SNR Variations**: Evaluating models under different noise conditions (High, Medium, Low SNR)
- **Model Architecture**: Testing different model configurations (small, medium, large)
- **Training Configurations**: Various training setups and optimization strategies

## Dependencies

The project uses modern Python libraries including:
- PyTorch for model implementation
- NumPy and SciPy for numerical computations
- Matplotlib for visualization
- Jupyter for interactive tutorials
