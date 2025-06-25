# Evaluating Generalization and Representation Stability in Small LMs via Prompting, Fine-Tuning and Out-of-Distribution Prompts

**Rahul Raja, Arpita Vats**  
📌 Accepted at *Methods and Opportunities at Small Scale (MOSS), ICML 2025, Vancouver, Canada*

---
##  Paper 
You can access the paper on [arXiv here](https://arxiv.org/abs/2506.17289).

## Abstract

We investigate the generalization capabilities of small language models under two popular adaptation paradigms:  
few-shot prompting and supervised fine-tuning. While prompting is often favored for its parameter efficiency and  
flexibility, it remains unclear how robust this approach is in low-resource settings and under distributional shifts.  
This paper presents a comparative study of prompting and fine-tuning across task formats, prompt styles, and  
model scales, with a focus on their behavior in both in-distribution and out-of-distribution (OOD) settings.  
Beyond accuracy, we analyze the internal representations learned by each approach to assess the stability and  
abstraction of task-specific features. Our findings highlight critical differences in how small models internalize  
and generalize knowledge under different adaptation strategies. This work offers practical guidance for model  
selection in low-data regimes and contributes empirical insight into the ongoing debate over prompting versus  
fine-tuning.

---

## 1. Introduction

Few-shot prompting and supervised fine-tuning are two widely adopted strategies for adapting pretrained language models  
(LMs) to downstream tasks. Prompting adapts models by conditioning on in-context examples at inference time without  
updating model parameters (Brown et al., 2020), whereas fine-tuning involves directly optimizing the model on labeled  
data. While prompting is attractive for its flexibility and efficiency, its reliability in low-resource settings—particularly for  
small-scale language models like GPT-2 (Radford et al., 2019) and DistilGPT2 (Sanh et al., 2019)—remains uncertain.

In this work, we present a systematic comparison of prompting and fine-tuning using three GPT-2 variants: distilgpt2,  
gpt2, and gpt2-medium, evaluated across a suite of language understanding tasks. We investigate three central questions:  
(1) How does prompting performance scale with the number of in-context examples compared to fine-tuning under an  
equivalent data budget? (2) How well does each method generalize to out-of-distribution (OOD) prompt templates? (3) How  
stable are their internal representations across prompt variations?

We begin with prompting on synthetic multi-task tasks (sentiment, grammar correction, arithmetic, plural forms), followed  
by fine-tuning on IMDb sentiment classification. In addition to standard accuracy comparisons, we use t-SNE to analyze the  
structure of prompt and hidden-layer representations across models and prompt styles.

This study reveals important distinctions in how small LMs internalize supervision under different adaptation strategies,  
with implications for generalization and representational robustness in low-data regimes.

##  Citation

If you use this work, please cite:

```bibtex
@misc{raja2025evaluatinggeneralizationrepresentationstability,
  title        = {Evaluating Generalization and Representation Stability in Small LMs via Prompting, Fine-Tuning and Out-of-Distribution Prompts}, 
  author       = {Rahul Raja and Arpita Vats},
  year         = {2025},
  eprint       = {2506.17289},
  archivePrefix= {arXiv},
  primaryClass = {cs.AI},
  url          = {https://arxiv.org/abs/2506.17289}, 
}

