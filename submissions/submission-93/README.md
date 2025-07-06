## LLaDA R3 (Submission #93)


### 1. High-level Idea
The code builds on the paper "**Large Language Model Diffusion for Math Reasoning**" and implements three increasingly sophisticated sampling strategies:

1. **Vanilla-PRM** (`generate_vanilla_prm.py`)   Baseline block-by-block decoding scored with a Preference Reward Model (PRM).
2. **Back-Masking** (`generate_backmasking.py`)   Adds a *look-back* window; low-scoring blocks are re-masked and re-decoded in parallel.
3. **Window-Score + Batch-Refine** (`generate_optimized.py`)   Vectorised, memory-friendly rewrite with batched PRM calls and aggressive refinement.

All three methods decode the answer *block-wise* (default 32 tokens per block) and therefore require only a few hundred forward passes instead of thousands of autoregressive steps.

---

### 2. Directory Layout
```
submissions/submission-93/
├── generate_backmasking.py        # Back-masking sampler (k-parallel candidates)
├── generate_optimized.py          # Optimised sampler (batched, window K)
├── generate_vanilla_prm.py        # Simple baseline sampler
├── llada_main_bon.py              # CLI for dataset evaluation / ablation
├── modal_app.py                   # Minimal Modal.com wrapper for A100 deployment
├── math_test_data.csv             # Tiny test set (question / boxed-answer)
├── llada.ipynb                    # Colab notebook – interactive demo
├── main.py                        # Empty placeholder, safe to ignore
└── README.md                      # ← you are here
```

---

### 3. Quick Start (Local GPU)
1. **Create env**
```bash
conda create -n llada93 python=3.10 -y
conda activate llada93
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124  # or cpu
pip install transformers accelerate datasets pandas gptqmodel
```
2. **(Optional) HuggingFace token** – set once so models auto-download:
```bash
export HF_TOKEN="<your-private-token>"
```
3. **Single prompt demo** (uses default 4-bit GPTQ weights – ≈ 13-16 GB VRAM):
```python
from transformers import AutoTokenizer
from gptqmodel import GPTQModel
from generate_backmasking import generate

MODEL_ID  = "FunAGI/LLaDA-8B-Instruct-gptqmodel-4bit"
PRM_ID    = "HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2"  # lightweight 1.5B PRM

model     = GPTQModel.load(MODEL_ID, device="cuda", trust_remote_code=True)
prm_model = GPTQModel.load(PRM_ID,  device="cuda", trust_remote_code=True)  # or AutoModel

tokenizer      = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")
prm_tokenizer  = AutoTokenizer.from_pretrained(PRM_ID, trust_remote_code=True)

question = "What is the value of x if 2x + 3 = 7?"
chat     = tokenizer.apply_chat_template([{"role":"user","content":question}], add_generation_prompt=True, tokenize=False)
prompt   = tokenizer(chat, return_tensors="pt").input_ids.to(model.device)

out = generate(
    model=model,
    prompt=prompt,
    prm_model=prm_model,
    tokenizer=tokenizer,
    prm_tokenizer=prm_tokenizer,
    # sampling-hyper-params (defaults shown)
    steps=128,
    gen_length=512,
    block_length=32,
    temperature=0.6,
)
print(tokenizer.decode(out[0, prompt.shape[1]:], skip_special_tokens=True))
```

---

### 4. Re-producing the Paper-style Evaluation
`llada_main_bon.py` bundles dataset loading, answer extraction and auto-grading:
```bash
python submissions/submission-93/llada_main_bon.py \
       --model FunAGI/LLaDA-8B-Instruct-gptqmodel-4bit \
       --prm   HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2
```
The script iterates through `math_test_data.csv`, generates an answer **with back-masking**, and appends per-question metrics to `math_evaluation_results_detailed_n=5-optimcsv` inside `/my_vol` (configurable).

Key flags are defined at the top of the file – edit in-place or wrap the call in your own script.

---

### 5. Running on Modal
If you prefer to build a docker-free cloud endpoint, **Modal** is fully supported:
```bash
modal run submissions/submission-93/modal_app.py
```
The app mounts *all code files* and requests an **A100-80 GB**. Inside the function body you can call `run_llada()` which in turn executes `llada_main_bon.run_evaluation()`.

---

### 6. Customising the Samplers
All three samplers expose a **single `generate()`** API with the following shared arguments:
```
model, prompt, prm_model, tokenizer, prm_tokenizer,
steps, gen_length, block_length, temperature,
cfg_scale, remasking, mask_id
```
Additional algorithm-specific parameters are documented in the function signature:
* `num_candidates_per_block` / `num_demasking_candidates`
* `backmasking_lookback`, `backmasking_threshold`, `backmasking_alpha`, `backmasking_intensity`
* `backmasking_frequency`, `num_refinement_samples`, …

Feel free to tweak these for speed / quality trade-offs. The code is *torch-script free* and thus hack-friendly.

---

### 7. Tips & Troubleshooting
* **GPU OOM?** Load the 1.5 B PRM or the 4-bit 8 B LLaDA weights first. CPU inference is possible but extremely slow.
* **exllama backend buffer error** – add
  ```python
  from gptqmodel import exllama_set_max_input_length
  model = exllama_set_max_input_length(model, max_input_length=3000)
  ```
  before the first generation call if your sequences exceed the default 2765 tokens.
* **PRM scatter error on older GPUs** – the code will automatically fall back to *random scores* so that generation continues; for full accuracy run on CUDA 11+.

---

### 8. Citation
If you use this implementation in academic work please cite the original LLaDA and Qwen-PRM papers.

### 0. Interactive Notebook Demo
If you prefer an **out-of-the-box playground**, open `llada.ipynb`:

* **Colab** – click the notebook in the repo UI or visit `https://colab.research.google.com/github/<your-fork>/blob/main/submissions/submission-93/llada.ipynb`. Colab will prompt for a HuggingFace token and spin up a T4 GPU.
* **Local Jupyter** – after the *Quick-Start* environment setup:
  ```bash
  jupyter notebook submissions/submission-93/llada.ipynb
  ```
  The first code cell installs missing Python packages if needed. Make sure you export `HF_TOKEN` before executing the cells so model downloads succeed.

The notebook walks through:
1. Loading the quantised 8 B LLaDA model & lightweight 1.5 B PRM.
2. Running a **single-question demo** with full back-masking pipeline.
3. (Optional) Evaluating a CSV of math problems.

Feel free to tweak generation parameters directly in the cells and re-run — no extra scripting required.

---


