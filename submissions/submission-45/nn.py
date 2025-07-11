import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jrandom
import chex

from utils import safe_zip


WEIGHT_INIT_STD = 1.0


def count_weights_given_layers(layers):
    return sum(i * o + o for i, o in safe_zip(layers[:-1], layers[1:]))


class NeuralNetwork:
    def __init__(self, layers, key, activation_fn, output_activation_fn):
        self.layers = layers
        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn

        # initialize random weights
        initializer = jnn.initializers.glorot_normal()

        key_layers = jrandom.split(key, len(layers) - 1)
        weights = []
        for i, o, key_l in safe_zip(self.layers[:-1], self.layers[1:], key_layers):
            weights.append(initializer(key_l, (i + 1, o)).flatten() * WEIGHT_INIT_STD)

        self.weights = jnp.concatenate(weights, axis=-1)
        chex.assert_shape(self.weights, (count_weights_given_layers(layers),))

    def __call__(self, t, x, args):
        idx = 0
        r = x

        for i, o in safe_zip(self.layers[:-1], self.layers[1:])[:-1]:
            W = self.weights[idx : (idx + i * o)].reshape(i, o)
            idx += i * o

            b = self.weights[idx : (idx + o)]
            idx += o

            r = jnp.matmul(r, W) + b
            r = self.activation_fn(r)

        i = self.layers[-2]
        o = self.layers[-1]

        W = self.weights[idx : (idx + i * o)].reshape(i, o)
        idx += i * o

        b = self.weights[idx : (idx + o)]
        idx += o
        chex.assert_size(self.weights, idx)

        output = jnp.matmul(r, W) + b
        if self.output_activation_fn is not None:
            return self.output_activation_fn(output)

        return output

    
