# Dataset Distillation for Memorized Data: Soft Labels can Leak Held-Out Teacher Knowledge

> Freya Behrens, Lenka Zdeborová

**Abstract** Dataset and knowledge distillation transfer capabilities between models. 
Their efficiency is often linked to structure in the data. 
However, next to general skills, modern neural networks encode specific facts, but if and how such memorized information is transferred remains less understood.
To analyze the transfer of memorized information in isolation, we consider finite random i.i.d. datasets where generalization is a priori impossible and a successful teacher fit implies pure memorization.
Yet, we show that students can learn non-trivial accuracy on held out memorized teacher data they never directly observed - in some cases up to perfect accuracy. 
This notebook showcases this phenomenon in three different contexts, and sets up the framework required for a deeper empirical and theoretical analysis. 


**Running the code**
Notebook requires the `plotting_helper.py` to be in the same folder. It takes roughly ~2.5hrs to run the experiments.

Requires the following libraries
- torch
- matplotlib
- transformers
- tqdm
- scikit-learn

Feel free to contact us if there are any questions.
