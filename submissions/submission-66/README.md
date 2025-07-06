# Why Loss Re-weighting Works If You Stop Early: Training Dynamics of Unconstrained Features

This repository contains code and analysis for research on class imbalance and loss reweighting in neural networks. The work investigates why loss reweighting improves early-stage performance on imbalanced datasets.


## Notebook

### Part 1: MNIST Experiment
- Tests standard vs. weighted cross-entropy loss on imbalanced MNIST data
- Shows that weighted loss achieves balanced accuracy much earlier in training
- Uses 3-layer CNN with 10:1 class imbalance ratio
- Includes a video file showing how confusion matrix evolves during training

### Part 2: Theoretical Analysis  
- Uses Unconstrained Features Model (UFM) with SVD analysis
- Shows how reweighting equalizes singular values in the label matrix
- Explains why this leads to better early-stage performance

## Files

- `Loss_reweighting_notebook (1).ipynb`: Complete implementation
- `Weighted_loss_on_imbalanced_data (16).pdf`: Research paper

## Requirements

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn imageio
```

## Usage

1. Run Part 1 cells to reproduce MNIST experiments
2. Run Part 2 cells to reproduce theoretical analysis
3. Modify parameters to experiment with different settings

## Key Results

- **Standard loss**: Majority class reaches 97% accuracy while minority stays at 50%
- **Weighted loss**: Both classes achieve >90% accuracy simultaneously
- **Early stopping**: Weighted loss shows significant advantage in first 20-30 steps

## Citation

If you use this work, please cite the accompanying paper.

---

**Note**: The notebook is self-contained and reproduces all experiments from the paper. 