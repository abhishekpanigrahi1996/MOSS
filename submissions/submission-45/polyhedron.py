import numpy as np
import jax.numpy as jnp
import chex
import pypoman


class Polyhedron:
    def __init__(
        self, 
        U, 
        V, 
        normalize=True, 
        name=None, 
        viz_bounds=(-0.1, 1.1, -0.1, 1.1),
    ):
        chex.assert_rank((U, V), 2)
        chex.assert_equal_shape((U, V))
        
        self.U = U
        self.V = V / jnp.linalg.norm(V, axis=-1)[..., None]

        self.viz_bounds = viz_bounds
        self.name = name

        A = np.array(-self.V).astype(np.double)
        b = np.array(jnp.sum(-self.U * self.V, axis=-1, keepdims=True)).astype(np.double)
        self.center = jnp.array(
            pypoman.polyhedron.compute_chebyshev_center(A, b)
        )

        self.vertices = jnp.array(
            pypoman.duality.compute_polytope_vertices(A, b)
        )

    def get_chebyshev_center(self):        
        return self.center

    def get_vertices(self):
        return self.vertices
        
    def distances_to_boundaries(self, z):
        chex.assert_rank(z, 1)
    
        d = jnp.sum((z - self.U) * self.V, axis=-1) / jnp.linalg.norm(self.V, axis=-1)
        chex.assert_shape(d, (len(self.V),))
    
        return d

    def is_feasible(self, z):
        chex.assert_rank(z, 1)
    
        b = (jnp.sum((z - self.U) * self.V, axis=-1) >= 0).all()
        chex.assert_shape(b, ())
    
        return b

    def get_num_boundaries(self):
        return len(self.V)

    def get_dims(self):
        return self.V.shape[-1]
    
    def get_viz_bounds(self):
        return self.viz_bounds

    def get_name(self):
        return self.name

    
