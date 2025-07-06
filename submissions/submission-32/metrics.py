import jax.numpy as jnp
import jax
from jax import Array

def accuracy(pred_y: Array, y: Array) -> Array:
  predicted_class = jnp.argmax(pred_y, axis=-1)
  true_class = jnp.argmax(y, axis=-1)
  return predicted_class == true_class


def entropy(pred_y):
  pred = jax.nn.softmax(pred_y, axis=-1)
  return -jnp.sum(pred * jnp.log(pred + 1e-16), axis=-1)

def p_correct(pred_y, y):
  prob = jax.nn.softmax(pred_y, axis=-1)
  corr_class = jnp.argmax(y, axis=-1)
  return prob[jnp.arange(pred_y.shape[0]), corr_class]


def ce(pred_y: Array, y: Array) -> Array:
  pred_y = jax.nn.log_softmax(pred_y, axis=-1)
  return -jnp.sum(pred_y * y, axis=-1)

def class_mse(pred_y: Array, y: Array) -> Array:
  prob = jax.nn.softmax(pred_y, axis=-1)
  return 0.5 * jnp.sum((prob - y) ** 2, axis=-1)

def mse(pred_y: Array, y: Array) -> Array:
  return 0.5 * jnp.sum((pred_y - y) ** 2, axis=-1)