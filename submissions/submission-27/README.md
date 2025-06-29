# What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers

Authors: Pulkit Gopalani, Wei Hu (University of Michigan)

The notebook `loss_plateau.ipynb` contains code to reproduce results on the moving-window-sum (MWS) task in the main paper. 

- The `# Model Definition` cell defines `GPTLinear` used as the Transformer model in our experiments. Similarly `# Data` cell defines the `MovingWindowSum` class for generating data for the moving-window-sum task.

- Please modify any (model / training / data) hyperparameters in the cell marked `# Config`. 

- We use [Wandb](https://wandb.ai/) for logging metrics during training; please use your Wandb credentials in the notebook (location marked with `TODO` comments in the `# Train` cell).

