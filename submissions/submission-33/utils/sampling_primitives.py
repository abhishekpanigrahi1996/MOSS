import torch
import numpy as np

def sample_bbox(bnds, N=1000):
    """
    Sample N points in nx dimenions from a rectangular domain given by bnds.
    bnds[i] gives min, max along dimension i
    """
    return bnds[:,0] + (bnds[:,-1] - bnds[:,0])*torch.rand(N, bnds.shape[0]) ## Sample bx number of points within the specified domain

def get_meshgrid_in_domain2d(bounds, nx=400, ny=400):
    """Meshgrid the domain"""
    x0s = np.linspace(*bounds[0], nx)
    x1s = np.linspace(*bounds[1], ny)
    X0, X1 = np.meshgrid(x0s, x1s)
    xs = np.vstack([X0.ravel(), X1.ravel()]).T
    xs = torch.tensor(xs).float()
    return X0, X1, xs

def get_meshgrid_in_domain3d(bounds, nx=400, ny=400, nz=400):
    """Meshgrid the domain"""
    x0s = np.linspace(*bounds[0], nx)
    x1s = np.linspace(*bounds[1], ny)
    x2s = np.linspace(*bounds[2], nz)
    X0, X1, X2 = np.meshgrid(x0s, x1s, x2s)
    xs = np.vstack([X0.ravel(), X1.ravel(), X2.ravel()]).T
    xs = torch.tensor(xs).float()
    return X0, X1, X2, xs