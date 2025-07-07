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

# augment with periodic positional embeddings
class PeriodicPositionalEncoding(nn.Module):
    d: int
    delta: int
    max_seq_length: int = 256
    
    @nn.compact
    def __call__(self, x):
        wpe = self.param('wpe', nni.normal(1.), (self.delta, self.d))
        repeat_count = (self.max_seq_length + self.delta - 1) // self.delta
        wpe_big = jnp.tile(wpe, (repeat_count, 1))
        return wpe_big
        

class Transformer(nn.Module):
    vocab_size: int
    max_length: int
    output_size: int
    d: int
    heads: int
    width: int # width of MLP
    delta: int # periodicity
    use_pos: bool # whether to use positional embeddings
    
    
    def attn(self, x, Q, K):
        T = x.shape[-2]
        attn = jnp.einsum("...ij,jm,km,...lk -> ...il", x, Q, K, x)
        # attn = jnp.log(jnp.arange(1, T+1))[None, :, None] * attn / self.d
        attn = attn/self.d
        attn = jnp.where(jnp.tri(T), attn, -jnp.inf)
        attn = nn.softmax(attn)
        attn = jnp.einsum("...ij,...jk->...ik", attn, x)
        return attn
    
    def embed(self, x, wte):
        out = wte[x]
        return out
    
    @nn.compact
    def __call__(self, x):
        
        B, T = x.shape[:2]
    
        
        wte = self.param('wte', nni.normal(1.), (self.vocab_size, self.d))
        unembed = self.param('unembed', nni.normal(1.), (self.d,))
        
        Q = self.param('Q', nni.normal(1./jnp.sqrt(self.d)), (self.heads, self.d, self.d))
        K = self.param('K', nni.normal(1./jnp.sqrt(self.d)), (self.heads, self.d, self.d))
        O = self.param('O', nni.normal(1./jnp.sqrt(self.d)), (self.d, self.d))
        V = self.param('V', nni.normal(1./jnp.sqrt(self.d)), (self.d, self.d))
        
        x = self.embed(x, wte)
        if self.use_pos:
            pos_enc = PeriodicPositionalEncoding(d=self.d, delta=self.delta, max_seq_length=self.max_length)(x)
            x = x + pos_enc[:T]
        
        attn = jax.vmap(self.attn, (None, 0, 0), -2)(x, Q, K)
        print(attn.shape)
        attn = attn.reshape(*attn.shape[:-2], -1)
        print(attn.shape)
        attn = attn@V@O
        x = attn + x
        
        #muP init
        z = nn.Dense(self.width, kernel_init=nni.normal(jnp.sqrt(1/self.d)), use_bias=True, name = 'layer1')(x)
        z = nn.relu(z)
        z = nn.Dense(self.d, kernel_init=nni.normal(jnp.sqrt(1/self.width)), use_bias=False, name = 'layer2')(z)
        x = z + x
        
        # output should be a length [batch, T] sequence
        return x@unembed/self.d