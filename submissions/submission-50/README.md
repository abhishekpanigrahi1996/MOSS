# **AdaptMI : Adaptive Skill-based In-context Math Instructions for Small Language Models**

To test a small-scale version of AdaptMI, please follow the instructions in `adaptmi.ipynb`.

Before running the notebook, we recommend reading the following preparation steps:

### **Environmental setup:**

you'll need 2 different conda environments. Please follow these steps to install them. (We also included these steps in the notebook)

```bash
conda create -n matheval python=3.10
conda activate matheval

cd evaluation/latex2sympy
pip install -e .
cd ..
pip install torch
pip install -r requirements.txt
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3
conda install ipykernel
```

```bash
conda create -n classify python=3.10.9
conda activate classify

git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
git checkout 55cc214c767741e83ee7b346e5e13e6c03b7b9fa
pip install -e .

pip3 install torch==2.1.2 torchvision torchaudio
pip install flash-attn

git clone https://github.com/lm-sys/FastChat.git
cd FastChat
pip install -e .

git clone https://github.com/WeiXiongUST/RLHF-Reward-Modeling.git
pip install deepspeed

pip install -r math-rm/requirements.txt
conda install ipykernel
```

`adaptmi.ipynb` includes three stages: **👉 Stage 1-1**, **👉 Stage 1-2**, and **👉 Stage 2**. You should activate different environments before running them:

- Stage 1-1: `matheval`
- Stage 1-2: `classify `
- Stage 2: `matheval`

### Now, please open `adaptmi.ipynb` and run the code step by step!
