# Reasoning by Superposition
Implementation for **"Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought"** ([paper](https://arxiv.org/abs/2505.12514)).


## Background

Large Language Models (LLMs) have demonstrated remarkable performance in many applications, including challenging reasoning problems via chain-of-thoughts (CoTs) techniques that generate *"thinking tokens"* before answering the questions. While existing theoretical works demonstrate that CoTs with discrete tokens boost the capability of LLMs, recent work on **continuous CoTs** lacks a theoretical understanding of why it outperforms discrete counterparts in various reasoning tasks such as **directed graph reachability**, a fundamental graph reasoning problem that includes many practical domain applications as special cases.

In this paper, we prove that a **two-layer transformer with D steps of continuous CoTs** can solve the directed graph reachability problem, where *D* is the diameter of the graph, while the best known result of constant-depth transformers with **discrete CoTs** requires *O(n²)* decoding steps where *n* is the number of vertices (*D < n*). 

In our construction, each continuous thought vector is a **superposition state** that encodes multiple search frontiers simultaneously (i.e., parallel **breadth-first search (BFS)**), while discrete CoTs must choose a single path sampled from the superposition state, which leads to **sequential search** that requires many more steps and may be trapped into local solutions.

We also performed extensive experiments to verify that our theoretical construction aligns well with the empirical solution obtained via training dynamics. Notably, **encoding of multiple search frontiers as a superposition state automatically emerges** in training continuous CoTs, without explicit supervision to guide the model to explore multiple paths simultaneously.


## Notebook

This interactive notebook visualizes the superpositional reasoning of Coconut model, which aligns with the key intuitions of the theoretical construction in this work: **Layer 1** establishes the query context, **Layer 2** expands the search frontier, and the latent vectors encode reachable state sets in a continuous, distributed form—realizing the theoretical construction in the paper.

## Getting Started

1. Clone the repo and install [dependencies](requirements.txt).
2. Run [`notebook.ipynb`](notebook.ipynb)

## Citation
If you find this work useful, please cite:

```bibtex
@misc{zhu2025reasoning,
  title={Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought},
  author={Zhu, Hanlin and Hao, Shibo and Hu, Zhiting and Jiao, Jiantao and Russell, Stuart and Tian, Yuandong},
  year={2025},
  eprint={2505.12514},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
