# ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training

**Feijiang Han, Xiaodong Yu, Jianheng Tang, Qingyun Zeng, Licheng Guo, Lyle Ungar**  
📌 Accepted at *Methods and Opportunities at Small Scale (MOSS), ICML 2025, Vancouver, Canada*



You can try the latest version directly in Google Colab:

🔗 [Run on Google Colab](https://colab.research.google.com/drive/1JUBOZqMxfbMR1-rRowJkh8xw60JWb12V?usp=sharing)

To run this demo:

1. Open the notebook using the link above.
2. Follow the step-by-step cells to test the model.
3. (Optional) Modify the code to experiment with your own inputs or datasets.



---

##  Paper 

You can access the paper and blog on [AlpharXiv here](https://www.alphaxiv.org/abs/2505.11739).

## Abstract

Training-free methods for enhancing large language models (LLMs) have attracted growing interest recently, with token-level attention tuning emerging as an interpretable and promising direction. However, existing methods typically rely on auxiliary mechanisms to identify important or irrelevant task-specific tokens, introducing potential bias and limiting applicability.
In this work, we uncover a surprising and elegant alternative: the semantically empty initial token (e.g., <BOS> in Llama) serves as a powerful and underexplored control point for optimizing model behavior. Through theoretical analysis, we show that tuning the initial token’s attention sharpens or flattens the attention distribution over subsequent tokens, and its role as an attention sink amplifies this effect. Empirically, we find that: (1) tuning its attention improves LLM performance across tasks more effectively than tuning other task-specific tokens; (2) the effect follows a consistent trend across layers, with earlier layers having greater impact, but varies across attention heads, with different heads showing distinct preferences in how they attend to this token.
Based on these findings, we propose \textbf{ZeroTuning}, a training-free approach that improves LLM performance by applying head-specific attention adjustments to this special token. Despite tuning only one token, ZeroTuning achieves higher average performance on text classification, multiple-choice QA, and multi-turn conversation tasks across models such as LLama, Qwen, and DeepSeek. For example, ZeroTuning improves Llama-3.1-8B by 11.71\% on classification tasks, 2.64\% on QA tasks, and raises its multi-turn score from 7.804 to 7.966. The method is also robust to limited resources, few-shot settings, long contexts, quantization, decoding strategies, and prompt variations.
Our work sheds light on a previously overlooked control point in LLMs, offering new insights into both inference-time tuning and model interpretability.

##  Citation

If you use this work, please cite:

```bibtex
@article{han2025zerotuning,
  title={ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training},
  author={Han, Feijiang and Yu, Xiaodong and Tang, Jianheng and Ungar, Lyle},
  journal={arXiv preprint arXiv:2505.11739},
  year={2025}
}
```
