Jupyter notebook implementing Threshold Relative Attention:

***Abstract***

Transformers struggle with length generalisation, displaying poor performance even on basic tasks. We test whether these limitations can be explained through two key failures of the self-attention mechanism. The first is the inability to fully remove irrelevant information. The second is tied to position, even if the dot product between a key and query is highly negative (i.e. an irrelevant key) learned positional biases may unintentionally up-weight such information - dangerous when distances become out of distribution. Put together, these two failure cases lead to compounding generalisation difficulties. We test whether they can be mitigated through the combination of a) selective sparsity - completely removing irrelevant keys from the attention softmax and b) contextualised relative distance - distance is only considered as between the query and the keys that matter. We show how refactoring the attention mechanism with these two mitigations in place can substantially improve generalisation capabilities of decoder only transformers.

To run just upload the notebook to colab and you should be good to go! Training takes around 2 hours on a T4 and
20 minutes on A100. 



