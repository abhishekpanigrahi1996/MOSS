# Pruning Increases Orderedness in Weight-Tied Recurrent Computation

This repository contains the implementation for the paper "Pruning Increases Orderedness in Weight-Tied Recurrent Computation" (Song, 2025), accepted at the [Methods and Opportunities at Small Scale (MOSS) Workshop](https://sites.google.com/view/moss2025) @ ICML 2025.

## Quick Start

1. **Install dependencies.** In a virtual environment of your choice, run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Explore with notebooks.** To reproduce our results, explore the following Jupyter notebooks:
    - [Main results with complete perceptron layers](notebooks/demo.ipynb)
    - [Visualisation and plotting utilities](notebooks/plotting.ipynb)
    - [Notebooks to generate quantitative analysis for Table 1 in the paper](notebooks/table.ipynb)

## Code Structure
The codebase is organized as follows:
```text
./
├── data.py              # Definitions for datasets, hyperparameters, baseline MLP models
├── layers.py            # CompleteLayer - the main complete perceptron layer implementation
├── training.py          # Training loop with pruning and orderedness tracking
├── experiments.py       # High-level experiment orchestration and reproducibility
├── pruning.py           # Various pruning strategies (random, top-k, tril-damp, etc.)
├── inits.py             # Weight initialization functions (normal, uniform, zeros, etc.)
├── losses.py            # Loss functions
├── evals.py             # Evaluation and visualisation classes
├── funcs.py             # [UNUSED] Utility functions for straight-through estimators
├── utils.py             # Core utilities and orderedness computation
└── notebooks/
    ├── demo.ipynb       # Main experiments and results
    ├── plotting.ipynb   # Visualisation utilities
    └── table.ipynb      # Data generation for Table 1
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{song2025orderednesspruning,
  title={Pruning Increases Orderedness in Weight-Tied Recurrent Computation},
  author={Song, Yiding},
  booktitle={Methods and Opportunities at Small Scale (MOSS) Workshop @ ICML 2025},
  year={2025}
}
```

## License

Copyright (c) 2025 Yiding Song.

All code in this repository is licensed under the the GNU General Public License v3.0. See [`LICENSE`](./LICENSE) file for details.
