# Continuous Chain of Thought (CoT2)
This repository provides part of the implementation for the paper **"Continuous Chain of Thought Enables Parallel Exploration and Reasoning"** ([arXiv link](https://arxiv.org/abs/2505.23648)).

## Abstract

Current language models generate chain-of-thought traces by autoregressively sampling tokens from a finite vocabulary. While this discrete sampling has achieved remarkable success, conducting \underline{c}hain-\underline{o}f-\underline{t}hought with \underline{co}ntinuously-valued \underline{t}okens (CoT2) offers a richer and more expressive alternative. 

Our work examines the benefits of CoT2 through logical reasoning tasks that inherently require search capabilities and provide optimization and exploration methods for CoT2. Theoretically, we show that CoT2 allows the model to track multiple traces in parallel and quantify its benefits for inference efficiency. Notably, one layer transformer equipped with CoT2 can provably solve the combinatorial "subset sum problem" given sufficient embedding dimension. These insights lead to a novel and effective supervision strategy where we match the softmax outputs to the empirical token distributions of a set of target traces. 

Complementing this, we introduce sampling strategies that unlock policy optimization and self-improvement for CoT2. Our first strategy samples and composes $K$ discrete tokens at each decoding step to control the level of parallelism, and reduces to standard CoT when $K=1$. Our second strategy relies on continuous exploration over the probability simplex. Experiments confirm that policy optimization with CoT2 indeed improves the performance of the model beyond its initial discrete or continuous supervision.

## Notebook

This notebook demonstrates the training process for discrete CoT and CoT2 models on the MNNS task, highlighting the advantages of the CoT2 model trained via our proposed continuous supervised fine-tuning (CSFT). The demonstration emphasizes CoT2's improved performance in tasks requiring search-based reasoning. Both models can be trained within approximately 3 hours for 200 epochs; however, training for 1000 epochs is recommended for complete experimental results.

## Citation
If you find our paper helpful for your research, please consider citing our paper:

```bibtex
@misc{gozeten2025continuouschainthoughtenables,
      title={Continuous Chain of Thought Enables Parallel Exploration and Reasoning}, 
      author={Halil Alperen Gozeten and M. Emrullah Ildiz and Xuechen Zhang and Hrayr Harutyunyan and Ankit Singh Rawat and Samet Oymak},
      year={2025},
      eprint={2505.23648},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.23648}, 
}
```
