# Transformers May Learn to Classify In-Context by Context-Adaptive Kernel Gradient Descent
### Authors 
**Sara Dragutinovic**, University of Oxford

**Aaditya K. Singh**, Gatsby Computational Neuroscience Unit, UCL

**Andrew M. Saxe**, Gatsby Computational Neuroscience Unit, UCL
### Overview
In this repository, we provide code accompanying the short paper accepted at the **Methods and Opportunities at Small Scale (MOSS)** workshop at ICML 2025. 
The main empirical result reproduced in the notebook is that both **linear** and **softmax** self-attention mechanisms learn to implement the corresponding gradient descent step derived in the paper after training.

### Running the notebook
To run the experiments in Google Colab, just follow the instructions in `Colab_Main_Notebook.ipynb`. This is the intended way, as recommended by the workshop committee.
Alternativelly, you can run the `Alternative_Notebook.ipynb` notebook on a local GPU machine, in which case you'll also need the python environment specified in `environment.yml`.
