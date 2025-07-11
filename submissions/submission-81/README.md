## Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry

Code base for the paper "Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry" accepted at ICML 2025 Methods and Opportunities at Small Scale (MOSS) Workshop.

Authors:

Sai Sumedh R. Hindupur, Ekdeep Singh Lubana, Thomas Fel and Demba Ba

Note: Main jupyter notebook(s) which create the figures are `mainfigs.ipynb` in folders `expt_separability/` or `expt_heterogeneity/` (one per experiment).


## Usage

1. *Generate synthetic data*: Run `createdata_separability.ipynb`, `createdata_heterogeneity.ipynb` 
2. *Run Synthetic data experiments*: Go to `expt_separability/` or `expt_heterogeneity/`,

    a. Execute bash script `run_allmodels.sh` 

    b. Run analysis using `mainfigs.ipynb` 


## Overview

1. *SAE Definitions*: in `models.py`
2. *Generating Synthetic Data*: `createdata_separability.ipynb`, `createdata_heterogeneity.ipynb` create datasets of gaussian clusters for the separability and heterogeneity experiments respectively. 
3. *Synthetic data experiments*: training file `train_saes.py` experiment settings (`settings.txt`), hyperparameter files (`hyperparameters2.csv`). The bash script `run_allmodels.sh` trains SAEs for all hyperparameters.


Note: _gamma_reg_ is the scaling constant for sparsity regularizer in the loss, _kval_topk_ is the _K_ in TopK