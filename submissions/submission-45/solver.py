import jax
import jax.numpy as jnp
import jax.random as jrandom
import chex
import diffrax


def wong_zakai_expansion(t, t1, wzs):
    '''
    Pathwise expansion of Brownian Motion
    '''
    chex.assert_rank(wzs, 2)

    def body_fun(accum, i): # Karhunen–Loeve series
        phi = (
            ((2.0 / t1) ** 0.5)
            * jnp.cos((2.0 * i - 1.0) * jnp.pi * t / (2.0 * t1))
        )

        return None, phi

    _, expansion = jax.lax.scan(body_fun, None, jnp.arange(wzs.shape[0]) + 1)
    expansion = expansion[..., None]

    prod = wzs * expansion
    chex.assert_equal_shape((wzs, prod))

    dBdt = jnp.sum(prod, axis=0)
    chex.assert_shape(dBdt, (wzs.shape[-1],))

    return dBdt


def solve_sde(dynamics, x0, ts, key, solver, dt0=0.01, truncated=None):
    drift = dynamics.drift
    diffusion = dynamics.diffusion

    if truncated is not None:
        xi = jrandom.normal(key, shape=(truncated, 1))

        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift),
            diffrax.ODETerm(
                lambda t, x, args: (
                    diffusion(t, x, args)
                    * wong_zakai_expansion(t, ts[-1], xi)
                )
            ),
        )

    else:
        brownian_motion = diffrax.VirtualBrownianTree(
            ts[0], ts[-1], tol=1e-3, shape=(), key=key
        )

        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift),
            diffrax.ControlTerm(diffusion, brownian_motion),
        )

    sol = diffrax.diffeqsolve(
        terms,
        solver,
        ts[0],
        ts[-1],
        dt0=dt0,
        y0=x0,
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=1000000,
    )

    return sol.ys

