import matplotlib
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import jax
import jax.numpy as jnp
import chex

from utils import *


def plot_polyhedron(ax, polyhedron, center=False, vertices=False):    
    b = polyhedron.get_viz_bounds()
    for u, v in safe_zip(polyhedron.U, polyhedron.V):
        chex.assert_shape((u, v), (2,))

        if v[1] != 0.0:
            x = jnp.linspace(b[0], b[1], 100)
            y = -(x - u[0]) * v[0] / v[1] + u[1]
        else:
            y = jnp.linspace(b[2], b[3], 100)
            x = -(y - u[1]) * v[1] / v[0] + u[0] 
        
        ax.plot(x, y, color='black', linestyle='dashed', linewidth=3.0)

        if center:
            c = polyhedron.get_chebyshev_center()
            ax.scatter(c[0], c[1], color='black', marker='*', s=170.0)

        if vertices:
            verts = polyhedron.get_vertices()
            ax.scatter(verts[:, 0], verts[:, 1], color='black', marker='o', s=90.0)
            

def eval_fn_on_meshgrid(fn, bounds, num_points=100):
    a1, b1, a2, b2 = bounds
    z1 = jnp.linspace(a1, b1, num_points)
    z2 = jnp.linspace(a2, b2, num_points)
    X, Y = jnp.meshgrid(z1, z2)
    
    Z = jax.vmap(fn)(
        jnp.array((X, Y)).reshape(2, -1).T,
    ).reshape(num_points, num_points, -1)
    
    return X, Y, Z


def style_ax(
        ax,
        lw_major=1.0,
        lw_minor=0.5,
        color_major='black',
        color_minor='gray',
        alpha_minor=0.5,
        tick_length=3.0,
        remove_yticks=False,
):
    ax.xaxis.minorticks_on()
    if not remove_yticks:
        ax.yaxis.minorticks_on()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.axhline(0.0, c=color_major, linewidth=lw_major, zorder=-1)
    ax.set_yticks(jnp.arange(11) / 10.0, minor=(not remove_yticks))
    ax.set_yticks([0.0, 1.0], minor=False)
    ax.axhline(1.0, c=color_major, linewidth=lw_major, zorder=-1)

    ax.grid(
        axis='y',
        which='minor',
        zorder=-1,
        color=color_minor,
        linewidth=lw_minor,
        alpha=alpha_minor,
    )

    if not remove_yticks:
        ax.yaxis.set_minor_formatter('{x:.1f}')
    ax.yaxis.set_major_formatter('{x:.1f}')

    ax.yaxis.set_tick_params(
        which='major',
        width=lw_major,
        length=tick_length,
        color=color_major,
    )
    ax.yaxis.set_tick_params(
        which='minor',
        width=lw_minor,
        length=tick_length,
        color=color_minor,
    )

    ax.set_xlabel(r'Time, $t$')
    #ax.set_xlabel(r'Time:')
    #ax.xaxis.set_label_coords(-0.078, -0.055)

    
def parametric_plot(x, y, c, ax, **lc_kwargs):
    """
    Taken from:
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    x_midpts = jnp.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = jnp.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = jnp.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, jnp.newaxis, :]
    coord_mid = jnp.column_stack((x, y))[:, jnp.newaxis, :]
    coord_end = jnp.column_stack((x_midpts[1:], y_midpts[1:]))[:, jnp.newaxis, :]
    segments = jnp.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)
