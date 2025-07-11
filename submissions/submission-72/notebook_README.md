# Polynomial Decomposition Tutorial Notebook

This notebook provides a complete walkthrough of our paper ["Discovering Hidden Algebraic Structures via Transformers with Rank-Aware Beam GRPO"](https://openreview.net/forum?id=lO9q5itiqK). It demonstrates the full pipeline from data generation to our novel Beam GRPO training method.

## Overview

The notebook showcases polynomial decomposition: given an expanded polynomial like `-2816*a^4 -7040*a^3 -7168*a^2 -3460*a -692`, find the inner polynomial `Q(a)` and outer polynomial `P(b)` such that `P(Q(a))` equals the expanded form.

This example corresponds to the first part of the $\mathcal{D}_1$ evaluation axis in our paper, examining the effect of polynomial degrees.

## What's Included

### 1. Setup & Repository Clone
```bash
git clone https://github.com/Jaeha0526/PolynomialDecomposition.git
cd PolynomialDecomposition
```

### 2. Data Generation
Generate 1M training samples with parallel processing:
```python
from using_sympy import generate_all_datasets_parallel

generate_all_datasets_parallel(
    file_directory="data_storage/dataset/single_variable",
    num_train=1000000,    # 1M training samples
    num_test=3000,        # 3K test samples per degree combination
    num_valid=128,        # 128 validation samples
    inner_only=True       # Single variable format
)
```

### 3. Supervised Learning
Train a 6-layer transformer model:
- Architecture: 6 layers, 8 attention heads, 512 embedding dimension
- Training: 10 epochs with teacher forcing
- ~19M parameters optimized for mathematical reasoning

### 4. Evaluation Methods

#### Greedy Search
Fast, deterministic inference selecting the highest probability token at each step:
```python
python Training/mingpt/run.py inequality_evaluate4 \
    --reading_params_path data_storage/model/single_variable_model_best.pt \
    --evaluate_corpus_path data_storage/dataset/single_variable/test_dataset_2_4.txt
```

#### Beam Search
Explores multiple hypotheses simultaneously for better solutions:
```python
python Training/mingpt/run.py debug_beam \
    --beam_width 10 \
    --reading_params_path data_storage/model/single_variable_model_best.pt \
    --evaluate_corpus_path data_storage/dataset/single_variable/test_dataset_2_4.txt
```

### 5. Beam GRPO (Our Novel Contribution)

Our Beam Group Relative Policy Optimization (BGRPO) method improves beam search efficiency by training the model to rank correct solutions higher in the beam:

```bash
cd Training/GRPO && bash run_single_variable_model.sh
```

**Key Innovation**: BGRPO uses rank-aware rewards that incentivize correct solutions to appear earlier in beam search results, reducing the beam width needed for good performance.


## Quick Start

Open `Polynomial_decomposition.ipynb` and run all cells sequentially. The notebook includes detailed explanations and visualizations of each step.

This tutorial demonstrates how transformers can learn to reverse complex algebraic operations and how our BGRPO method makes beam search more efficient for mathematical reasoning tasks.