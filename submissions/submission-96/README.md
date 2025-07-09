Jupyter notebook implementing the Threshold Relative Attention mechanism for improved length generalisation as presented [in this paper.](https://arxiv.org/abs/2503.23174)

### Abstract:
Transformers struggle with length generalisation, displaying poor performance even on basic tasks. We test whether these limitations can be explained through two key failures of the self-attention mechanism. The first is the inability to fully remove irrelevant information. The second is tied to position, even if the dot product between a key and query is highly negative (i.e. an irrelevant key) learned positional biases may unintentionally up-weight such information - dangerous when distances become out of distribution. Put together, these two failure cases lead to compounding generalisation difficulties. We test whether they can be mitigated through the combination of a) selective sparsity - completely removing irrelevant keys from the attention softmax and b) contextualised relative distance - distance is only considered as between the query and the keys that matter. We show how refactoring the attention mechanism with these two mitigations in place can substantially improve generalisation capabilities of decoder only transformers.

### Usage:
You can upload this notebook to colab and verify for yourself that TRA successfully generalises to the Flip-Flop tasks. 

The code is self-contained so you shouldn't have to do anything other than connect to the GPU and run! 

Training takes two hours on T4 and 20 minutes on A100.





