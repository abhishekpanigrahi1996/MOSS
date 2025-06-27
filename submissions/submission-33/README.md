# Exploring Diverse Solutions for Underdetermined Problems

**Authors**: Eric Volkmann<sup>1</sup>, Andreas Radler<sup>1</sup>, Johannes Brandstetter<sup>1,2</sup>, Arturs Berzins<sup>1</sup>

<sup>1</sup> LIT AI Lab, Institute for Machine Learning, JKU Linz, Austria

<sup>2</sup> Emmi AI GmbH, Linz, Austria

This directory contains the notebook `MOSS_final.ipynb`, which accompanies our workshop paper "Exploring Diverse Solutions for Underdetermined Problems", accepted at the MOSS workshop at ICML 2025. The notebook demonstrates the nearest-neighbor diversity loss on finite vector and function spaces, with illustrative experiments and visualizations.


## File Overview

- `MOSS_final.ipynb`: Main notebook to run experiments and create plots. 
- `horse_shoe.py`, `model_defs.py`, `sampling_primitives.py`, `util.py`: Supporting modules for geometry, models, and utilities.

## Notebook Structure

- **Horseshoe Example:** Demonstrates diversity losses on a finite vector space.
- **Flat Parametric Curve:** Shows diversity loss on simple parametric curves.
- **Parametric Curve on Manifold:** Explores diversity on a parametric manifold.

## How to Run

1. **Install Required Libraries**

   Before running the notebook, ensure you have the following Python libraries installed:

   ```sh
   pip install torch numpy matplotlib tqdm k3d
   ```

2. **Run the Notebook**

   Open the `MOSS_final.ipynb` notebook and execute the cells in order. All results and visualizations will be generated. The notebook contains comments and markdown sections to walk you through the experiments step by step.
   The experiments are small enough to run on a laptop CPU within a few minutes. Utility code for setting up the models, the geometry and sampling etc. is provided in `util/`. 


If you have questions about the code or want to get in touch, feel free to reach out via email 

volkmann@ml.jku.at 