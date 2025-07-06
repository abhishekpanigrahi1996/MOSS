# Stats or Facts: Decomposing Generalization in Language Models with Small-Scale Models  
**Tina Behnia, Puneesh Deora, Christos Thrampoulidis**

*Accepted at MOSS Workshop, Submission #68, ICML 2025*


## Overview

The notebook **`MCPos.ipynb`** reproduces the main figures in the paper using a smaller-scale setup that runs faster.

While the full-scale experiments require long runtime, the notebook version completes in under 2 hours on a free-tier Colab GPU and replicates the same key behaviors.
(More details in the notebook)

**Code Structure**:

- **`MCPos.ipynb`** – Main notebook for running experiments and generating plots  
- **`src/`** – Supporting code for training, metric computation, and visualization

**How to Run**:

1. At the beginning of the notebook, set the `path` variable to the directory containing the `src/` folder.

2. *(Optional)* Modify the hyperparameters (such as number of templates, training steps, etc, defined in the notebook). 

3. Run the notebook cells step by step to reproduce the figures.
