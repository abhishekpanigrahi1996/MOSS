import jax
import jax.numpy as jnp


def safe_zip(*args):
    if len(args) > 0:
        first = args[0]
        for a in args[1:]:
            assert len(a) == len(first)

    return list(zip(*args))


def inv_sigmoid(x, tol=1e-8):
    return -jnp.log((1 / (x + tol)) - 1.0)


