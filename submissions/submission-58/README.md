# Permutations as a testbed for studying the effect of input representations on learning
Sarah M.Scullen, Davis Brown, Robert Jasper, Henry Kvinge, Helen Jenne


## Overview
This code generates examples of different representations of permutations and computes various statistics on this permutation data for use in downstream classification tasks. We provide the code to generate two example permutation datasets and show how to train a simple MLP classification model on the data. 

This code uses SageMath, which is not pip installable. To install within Google colab, follow the instructions in the first code cell of the Installation section. The time to install Sage on colab free-tier gpu varies, although you should expect it to take anywhere between 5-30 minutes depending on system usage. All required packages are imported in the first few cells of the notebook.

Note, that the code to train models assumes the device is set to 'cuda'. As such, the runtime option in colab should be changed from CPU to the free-tier GPU (e.g. T4 GPU) before running the code.


## Contents
Our code is in permutation_representations_notebook.ipynb

