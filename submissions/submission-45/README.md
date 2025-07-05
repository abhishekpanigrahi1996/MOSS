# Neural Stochastic Differential Equations on Compact State-Spaces

*Accepted at the Methods and Opportunities at Small Scale (MOSS) Workshop, ICML 2025.*

**Authors:**
* Yue-Jane Liu, [MOGU Lab](https://mogu-lab.github.io), Wellesley College
* Malinda Lu, [MOGU Lab](https://mogu-lab.github.io), Wellesley College
* Matthew K. Nock, [Nock Lab](https://nocklab.fas.harvard.edu), Harvard University
* [Yaniv Yacoby](https://yanivyacoby.github.io), [MOGU Lab](https://mogu-lab.github.io), Wellesley College

**Abstract:** 
Many modern probabilistic models rely on SDEs, but their adoption is hampered by instability, poor inductive bias outside bounded domains, and reliance on restrictive dynamics or training tricks. 
While recent work constrains SDEs to compact spaces using reflected dynamics, these approaches lack continuous dynamics and efficient high-order solvers, limiting interpretability and applicability. 
We propose a novel class of neural SDEs on compact polyhedral spaces with continuous dynamics, amenable to higher-order solvers, and with favorable inductive bias.

**Instructions:**
1. Install the dependencies from `requirements.txt` into your own environment. If you run into trouble when installing `pypoman`, it may be due to one of its dependencies, `pycddlib`. In this case, we recommend [this guide](https://pycddlib.readthedocs.io/en/latest/quickstart.html).
2. After preparing the environment, open `demo.ipynb` with Jupyter lab.

