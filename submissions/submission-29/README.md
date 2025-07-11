# Encoding Domain Insights into Multi-modal Fusion

## Abstract

Using small-scale experiments with real and synthetic tasks, we compare multi-modal fusion methods, including a proposed ‘Product Fusion’, to demonstrate how encoding task-specific static fusion logic affects performance. Our results highlight a crucial trade-off: aligning fusion design with task features boosts clean-data accuracy with limited data but significantly diminishes robustness to noisy inputs.

## Setup

### 1. Environment Setup

Create the conda environment from the provided `environment.yml` file to install all necessary dependencies.

`conda env create -f environment.yml`


### 2. Download Dataset

The following code will automatically download and extract the dataset.
```
import requests, zipfile, io, os

def download_and_unzip_dropbox(dropbox_url, extract_to="."):
download_url = dropbox_url.replace("?dl=0", "?dl=1")
response = requests.get(download_url)
response.raise_for_status()
with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
zip_ref.extractall(extract_to)
print(f"Extracted to {os.path.abspath(extract_to)}")

dropbox_url = "https://dl.dropboxusercontent.com/scl/fi/bk0gcvfy2ytni5eeu4gah/dataset.zip?rlkey=da701q1fc3xas3s1et76h5f7a&st=ft5099v2&dl=0"
download_and_unzip_dropbox(dropbox_url)

```
## Running the Experiments

The `main_train` function in the notebook trains the models. To reproduce the paper's results, run the training for each configuration by setting the `config` file (`concat_tweet.yaml`, `product_tweet.yaml`, `tensorfusion_tweet.yaml`) and the `synth` flag (`True` or `False`).

An example for training Concatenation Fusion on real data:

`main_train(config="concat_tweet.yaml", synth=False)`


After training, use the corresponding cells in the notebook to generate the accuracy, ROC, and robustness plots.

## Citation

If you use this code in your research, please cite our paper:
```
@inproceedings{michaels2025encoding,
title={Encoding Domain Insights into Multi-modal Fusion: Improved Performance at the Cost of Robustness},
author={Jackson Michaels and Sidong Zhang and Ina Fiterau},
booktitle={Methods and Opportunities at Small Scale (MOSS), ICML},
year={2025}
}
```