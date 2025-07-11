# Understanding How Chess-Playing Language Models Compute Linear Board Representations

This notebook reproduces the analyses described in *"Understanding How Chess-Playing Language Models
Compute Linear Board Representations."* The notebook is self contained - its accompanying explanations are sufficiently detailed such that readers can understand the methodology and results without referring to the paper. 

It has been tested end-to-end on Google Colab (GPU or CPU runtimes, though GPU is recommended).

## Quick start

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Wc_JAj2cbZXSzdZL3FAzEaiSrnm8dgTo#scrollTo=L2EhYz-txHkM)

1. Click the badge above or upload the notebook to Colab.  
2. Work through the cells at the desired pace

## Data & model (Hugging Face)

| Resource | Link |
|-----------|------|
| Processed dataset | <https://huggingface.co/datasets/spherical-chisel/ChessGPT-Interp> |
| Model | <https://huggingface.co/spherical-chisel/ChessGPT-Interp> |

Both are downloaded automatically in the notebook via huggingface. Models and datasets are adapted from https://github.com/adamkarvonen/chess_llm_interpretability.

### Regenerating the dataset

The `precomputed_game_cache.pt` cache found on hugging face was produced using `lichess_train.csv` by running

```bash
python generate_data.py
```
