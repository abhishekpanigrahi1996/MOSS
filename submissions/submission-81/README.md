## Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry

Code base for the paper "Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry" accepted at ICML 2025 Methods and Opportunities at Small Scale (MOSS) Workshop.

Authors:

Sai Sumedh R. Hindupur, Ekdeep Singh Lubana, Thomas Fel and Demba Ba

Note: Main jupyter notebook(s) which create the figures are `mainfigs.ipynb` in folders `expt_separability/` or `expt_heterogeneity/` (one per experiment).


## Usage

1. *Generate synthetic data*: Run `createdata_separability.ipynb`, `createdata_heterogeneity.ipynb` 
2. *Run Synthetic data experiments*: 

    a. Execute bash script `run_allmodels.sh` in `expt_separability/` or `expt_heterogeneity/`

    b. Run analysis using `mainfigs.ipynb` 


## Overview

1. *SAE Definitions*: `models.py` defines SAEs- ReLU, JumpReLU, TopK and SpaDE
2. *Generating Synthetic Data*: `createdata_separability.ipynb`, `createdata_heterogeneity.ipynb` create datasets of gaussian clusters for the separability and heterogeneity experiments respectively. Choose location to save data by modifying `dataset_dir` in this notebook. 
3. *Synthetic data experiments*: `expt_heterogeneity/` and `expt_separability/` have the training file `train_saes.py` and the jupyter notebook `mainfigs.ipynb` which performs analysis. The relevant experiment settings (`settings.txt`), hyperparameter files (`hyperparameters2.csv`) are also present. The bash script `run_allmodels.sh` runs `train_saes.py` for all choices of hyperparameters in `hyperparameters2.csv`.
4. *Functions for data, training and utilities*: `functions/` includes files to preprocess/load data (`get_data.py`), training pipeline for models (`train_test.py`) and miscellaneous functions (`utils.py`)


Note: _gamma_reg_ is the scaling constant for sparsity regularizer in the loss, _kval_topk_ is the _K_ in TopK