# Dynamic Low-Rank Training with Spectral Regularization: Achieving Robustness in Compressed Representations

## Authors

- Steffen Schotthöfer, Oak Ridge National Laboratory\*
- Lexie Wang, Oak Ridge National Laboratory
- Stefan Schnake, Oak Ridge National Laboratory

\*Corresponding author

## Citation info 

Coming soon

## Overview

This repository provides code for training neural networks using **Dynamic Low-Rank Training (DLRT)** with **Spectral Regularization** to achieve both **compression** and **robustness**. The method dynamically adapts the rank of weight matrices during training and introduces a spectral penalty that controls the tail of the singular spectrum.


## Run as jupyter notebook
1) Start the ipynb "low_rank.ipynb" 
2) Run all cells
3) Output is logged with wandb and in cell output.

## Alternative: Run as python script (recommended way!)

1) Create a local python environment and install the python requirements in a local virtual environment:

    ```
    python3 -m venv ./venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```


2) Run the bash scripts (It's recommended that you use the wandb logging option to visualize the results)
    ```
    sh download_data.sh
    sh run_baseline.sh
    sh run_low_rank.sh
    ```