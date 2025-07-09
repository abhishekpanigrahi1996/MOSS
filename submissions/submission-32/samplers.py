'''
This file contains all utils to construct data samplers for
our synthetic datasets.
'''

from jax import Array

import jax
from jax import numpy as jnp


def make_balanced_classification_queries_sampler(
  n_labels: int,
  context_len: int,
  n_samples: int,
  X_dim: int,
  noise_scale: float = 0.0
  ):
  """Build a function which samples data as specified.
  """
  def sample(key: Array):
    X_key, Xq_key, yq_key, W_key = jax.random.split(key, 4)
    X = jax.random.normal(X_key, (n_samples, n_labels*20*context_len, X_dim))
    X = X / jnp.linalg.norm(X, axis=2, keepdims=True)
    W = jax.random.normal(W_key, (n_samples, X_dim, n_labels))
    W = W / jnp.linalg.norm(W, axis=1, keepdims=True)
    Y = jnp.einsum("ijk, ikn -> ijn", X, W)
    X_qs = jax.random.normal(Xq_key, (n_samples, n_labels * 30, X_dim))
    X_qs = X_qs / jnp.linalg.norm(X_qs, axis=2, keepdims=True)
    q_labs = jax.random.randint(yq_key, (n_samples,), 0, 5)
    Y_qs = jnp.argmax(jnp.einsum("ijk, ikn -> ijn", X_qs, W), axis=-1)
    indices = jnp.argsort(jax.vmap(lambda y_qs, lab: y_qs==lab)(Y_qs, q_labs))[:, -1]
    X_q = X_qs[jnp.arange(n_samples), indices][:, None, :]
    Y_q = jnp.eye(n_labels)[Y_qs[jnp.arange(n_samples), indices]][:, None, :]
        
    labels = jnp.argmax(Y, axis=-1) # (n_samples, _ * cont_len)
    per_label = [labels==i for i in range(n_labels)] # (n_labels, n_samples, _ * cont_len)
    per_label = [per_label[i] & (jnp.cumsum(per_label[i], axis=-1)<=context_len/n_labels) for i in range(n_labels)] # (n_labels, n_samples, _ * cont_len)
    
    mask = jnp.any(jnp.array(per_label), axis=0)

    X = X[jnp.arange(n_samples)[:, None], jnp.argsort(mask)[:, - (context_len//n_labels)*n_labels:]]
    Y = Y[jnp.arange(n_samples)[:, None], jnp.argsort(mask)[:, - (context_len//n_labels)*n_labels:]]
    Y = jnp.eye(n_labels)[jnp.argmax(Y, axis=-1)]
    
    return {'examples': jnp.concatenate((X, X_q), axis=1), 'labels': jnp.concatenate((Y, Y_q), axis=1), 'vectors': W}
  return sample