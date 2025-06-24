# An Empirical Investigation of Initialization Strategies for Kolmogorov–Arnold Networks

This directory contains the supplementary material for the paper *An Empirical Investigation of Initialization Strategies for Kolmogorov–Arnold Networks*, submitted by **Spyros Rigas**, **Dhruv Verma**, **Georgios Alexandridis**, **Yixuan Wang** and accepted at the *ICML 2025 Methods and Opportunities at Small Scale (MOSS)* workshop.


## Contents

- `grid_search.csv`: A `.csv` file with the results for an extensive grid search over different Kolmogorov-Arnold Network architectures and different initialization methods.
- `kan_init_investigation.ipynb`: The main notebook that provides an analysis of the aforementioned grid search results, as well as some experiments using a lightweight and a more heavyweight Kolmogorov-Arnold Network.

## Runtime

The notebook is heavily dependent on [jaxKAN](https://jaxkan.readthedocs.io/en/latest/) (GPU version), however all dependencies - including jaxKAN - are installed by running the first code cell.

When run on Google Colab's free-tier runtime on a T4 GPU, the total runtime for the notebook is approximately 45 minutes.