# Exploring Diverse Solutions for Underdetermined Problems

This submission contains the notebook `MOSS_final.ipynb`, which accompanies the MOSS workshop submission "Exploring Diverse Solutions for Underdetermined Problems". The notebook demonstrates the nearest-neighbor diversity loss on finite vector and function spaces, with illustrative experiments and visualizations.

## Structure

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

   Open the `MOSS_final.ipynb` Jupyter Notebook execute the cells in order. The notebook will generate all results and visualizations. The notebook contains comments and markdown sections to walk you through the experiments step by step.
   The experiments are small enough to even run on a laptop CPU within a few minutes. Utility code for setting up the models,
   the geometry and sampling etc. is provided in `util/`. 

## File Overview

- `MOSS_final.ipynb`: Main notebook to run experiments and create plots. 
- `horse_shoe.py`, `model_defs.py`, `sampling_primitives.py`, `util.py`: Supporting modules for geometry, models, and utilities.
