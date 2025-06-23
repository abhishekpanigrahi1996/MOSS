# SynDaCaTE: A Synthetic Dataset For Evaluating Part-Whole Hierarchical Inference

*Jake Levi, Mark van der Wilk*

*Abstract: Learning to infer object representations, and in particular part-whole hierarchies, has been the focus of extensive research in computer vision, in pursuit of improving data efficiency, systematic generalisation, and robustness. Models which are designed to infer part-whole hierarchies, often referred to as capsule networks, are typically trained end-to-end on supervised tasks such as object classification, in which case it is difficult to evaluate whether such a model actually learns to infer part-whole hierarchies, as claimed. To address this difficulty, we present a SYNthetic DAtaset for CApsule Testing and Evaluation, abbreviated as SynDaCaTE, and establish its utility by (1) demonstrating the precise bottleneck in a prominent existing capsule model, and (2) demonstrating that permutation-equivariant self-attention is highly effective for parts-to-wholes inference, which motivates future directions for designing effective inductive biases for computer vision.*

This is the README/notebook for our paper "SynDaCaTE: A Synthetic Dataset For Evaluating Part-Whole Hierarchical Inference", accepted at MOSS at ICML 2025.

Our notebook is available at [`./syndacate.ipynb`](./syndacate.ipynb). The notebook clones and installs our public code implementation, which is available at [`github.com/jakelevi1996/syndacate-public`](https://github.com/jakelevi1996/syndacate-public). All installation instructions and documentation are available in the notebook!

The total running time for training models in this notebook is 2 hours and 3 minutes, using Colab free tier and a T4 GPU. Training durations do not include time taken for dataset generation, which is approximately 5-10 minutes per dataset when running in Colab.
