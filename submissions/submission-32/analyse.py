import jax
from jax import numpy as jnp
import jax.nn as jnn
import equinox as eqx

@eqx.filter_jit
def analyse2(data, model1, model2, fwd_fn1, fwd_fn2, key=jax.random.PRNGKey(10)):
    """Analyse alignement between 2 models."""
    x, y = data['examples'], data['labels']
    keys = jax.random.split(key, x.shape[0])

    pred_fn1 = lambda x_, y_, k_: jnn.softmax(fwd_fn1(model=model1, x=x_, y=y_, key=k_)['out'][-1, :]) # for one sample only, of shape (context_size, dim)
    pred_fn2 = lambda x_, y_, k_: jnn.softmax(fwd_fn2(model=model2, x=x_, y=y_, key=k_)['out'][-1, :])


    pred_y1 = jax.vmap(pred_fn1, in_axes=(0, 0, 0))(x, y, keys)
    pred_y2 = jax.vmap(pred_fn2, in_axes=(0, 0, 0))(x, y, keys)
    pred_norm = jnp.mean(jnp.linalg.norm(pred_y1 - pred_y2, axis=-1)) / y.shape[-1] # Divide by number of classes, such that the metric is comparable to linear regression setting

    grads1 = jax.vmap(jax.jacfwd(pred_fn1, argnums=0), in_axes=(0,0,0))(x, y, keys)[:, :, -1, :] 
    grads_norm1 = jnp.linalg.norm(grads1, axis=-1)

    grads2 = jax.vmap(jax.jacfwd(pred_fn2, argnums=0), in_axes=(0,0,0))(x, y, keys)[:, :, -1, :] # (N, C, x_dim)
    grads_norm2 = jnp.linalg.norm(grads2, axis=-1)

    dot_products = jnp.einsum('ikj,ikj->ik', grads1/(grads_norm1[..., None]), grads2/(grads_norm2[..., None]))
    dot = jnp.mean(dot_products)
    norm = jnp.mean(jnp.linalg.norm(grads1-grads2, axis=-1)) / y.shape[-1] # Divide by number of classes, such that the metric is comparable to linear regression setting
    return dot, norm, pred_norm