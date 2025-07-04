# 🧠 Foundation Models on a Budget: Approximating Blocks in Large Vision Models

**[Irene Cannistraci](https://irene.cannistraci.dev/)**, Simone Antonelli, Emanuele Palumbo, Thomas M. Sutter, Emanuele Rodolà, Bastian Rieck†, Julia E. Vogt† 
*Accepted at Methods and Opportunities at Small Scale (MOSS), ICML 2025*, Vancouver, Canada

📄 [OpenReview](https://openreview.net/forum?id=XI9tNjMZhd) | 📚 [arXiv](https://arxiv.org/abs/2410.04941) | 
📖 [BibTeX](#bibtex)

> For questions contact: **irene.cannistraci@inf.ehtz.ch**

## 🚀 Getting Started

### 1. Required Files

Ensure the following files are present in your project directory:

- `dictionaries.py`  
- `module.py`  
- `train_NN.py`  
- `utils.py`  

### 2. Running the Notebook

Open the notebook and follow the step-by-step cells. To test different configurations, you can modify the `run_encoding()` function. The one in the notebook runs on Google Colab free tier, but you can test different configurations:

```python
run_encoding(
    dataset_name = "cifar10",              # ✏️ Replace with any Hugging Face dataset
    encoder_name = "facebook/dinov2-small",# ✏️ Choose a model from MODEL2CONFIGS in dictionaries.py
    translator_name = "linear",
    seed = 0,
    samples_to_extract = 500,
    batch_size = 256,
    skips = [[], [(10, 11)]]               # ✏️ Skip blocks with [(start_layer, end_layer)] format
)
```

## ☁️ Alternative: link to Google Drive folder
📂 Upload [this folder](https://drive.google.com/drive/folders/1pr83bzGn3inj_7q6BDAD8qq7BACpKxXX?usp=sharing) to your Drive and run the code on Colab!

## BibTeX

```bibtex
@misc{cannistraci2025foundationmodelsbudgetapproximating,
      title={Foundation Models on a Budget: Approximating Blocks in Large Vision Models}, 
      author={Irene Cannistraci and Simone Antonelli and Emanuele Palumbo and Thomas M. Sutter and Emanuele Rodolà and Bastian Rieck and Julia E. Vogt},
      year={2025},
      eprint={2410.04941},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.04941}, 
}
```