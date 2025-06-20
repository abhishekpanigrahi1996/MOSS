# MP-SAE: Matching Pursuit Sparse Autoencoders

This repo contains code for the paper:  
**Evaluating Sparse Autoencoders: From Shallow Design to Matching Pursuit**  
Accepted at **ICML 2025 Methods and Opportunities at Small Scale (MOSS)** workshop.

**Authors**:  
Valérie Costa, Thomas Fel, Ekdeep Singh Lubana, Bahareh Tolooshams, Demba Ba

 

## Code Structure

The code is implemented in a single notebook: `main.ipynb`. It is organized as follows:

- **Utils**  
  Helper functions for JumpReLU.

- **SAE Architectures**  
  Definitions of MP-SAE and shallow sparse autoencoders using ReLU, JumpReLU, TopK, and BatchTopK.

- **Train**
  - **Config**: Model selection and hyperparameter settings.
  - **SAE**: Model initialization.
  - **Data**: MNIST loading and preprocessing.
  - **Training**: Training loop for the selected SAE.

- **Evaluation**
  - **Load Model**: Load a trained model and set it to evaluation mode.  
  - **Input**: Load inputs and select an index for evaluation.  
  - **Reconstruction**: Visualize the reconstruction of the selected input, showing progressive reconstruction.


## Requirements

The code depends on the following Python packages:

- `torch`
- `torchvision`
- `matplotlib`
- `tqdm`