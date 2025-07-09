import jax
from jax import numpy as jnp
from jax import nn
import numpy as np
from flax import linen as nn
from flax.linen import initializers as nni
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import random as jr
from util import *
import optax

class Task:
    def __init__(self, vocab_size, class0, class1):
        self.vocab_size = vocab_size
        self.class0 = class0
        self.class1 = class1
    
    def sample(self, length, key, L):
        keys = jr.split(key, 2)
        p = jr.dirichlet(keys[0], jnp.ones(self.vocab_size))
        x = jr.choice(keys[1], self.vocab_size, (length,), p=p)
        counts0 = jnp.cumsum(x == self.class0)
        counts1 = jnp.cumsum(x == self.class1)
        total_counts = counts0 + counts1
        # y = jnp.where(counts0 > self.gamma * counts1, 0, 1)
        y = jnp.where(total_counts > 0, (counts1 - counts0)/(counts0 + counts1), 0)[1:] # regression task, only predict starting from 2nd pos
        y = jnp.sin(L*y)
        return x, y

class ModPSeq2SeqTask:
    def __init__(self, p, k):
        self.p = p
        self.k = k
    
    # length = sequence_length/self.p
    def sample(self, seq_length, key):
        # every index k mod p has different prob
        keys = jr.split(key, 2)
        p = jr.beta(keys[0], a = 1, b = 1, shape = (self.p,))
        
        repeats = (seq_length + self.p - 1) // self.p
        X = jr.bernoulli(keys[1], p, (repeats, self.p)).astype(jnp.int32)
        
        # index from first token in sequence
        sums = jnp.cumsum(X[:, self.k])
        counts = jnp.arange(1, repeats+1)
        means = sums/counts
        y = jnp.repeat(means, self.p)[self.p-self.k-1:seq_length-self.k] # predict on positions p through seq_length.

        X = X.reshape(-1)[:seq_length]
        return X, y