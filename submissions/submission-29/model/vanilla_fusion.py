import torch.nn as nn
import torch
from torch.autograd import Variable
import math

class ProductConcat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        prod = X[0] * X[1]
        inv_prod = X[0] * X[1].flip(0)
        return torch.cat([prod, inv_prod], dim=1)

class Concatenate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.cat(X, dim=1)

class TensorFusion(nn.Module):
    """
    Implementation of TensorFusion Networks.

    See https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py for more and the original code.
    """

    def __init__(self):
        """Instantiates TensorFusion Network Module."""
        super().__init__()

    def forward(self, modalities):
        """
        Forward Pass of TensorFusion.

        :param modalities: An iterable of modalities to combine.
        """
        if len(modalities) == 1:
            return modalities[0]

        mod0 = modalities[0]
        nonfeature_size = mod0.shape[:-1]

        m = torch.cat((Variable(torch.ones(
            *nonfeature_size, 1).type(mod0.dtype).to(mod0.device), requires_grad=False), mod0), dim=-1)
        for mod in modalities[1:]:
            mod = torch.cat((Variable(torch.ones(
                *nonfeature_size, 1).type(mod.dtype).to(mod.device), requires_grad=False), mod), dim=-1)
            fused = torch.einsum('...i,...j->...ij', m, mod)
            m = fused.reshape([*nonfeature_size, -1])

        return m

def find_concat_idx(mod_sizes):
    mod_indices = []
    interval = 1
    full_size = 1

    for m in mod_sizes:
        full_size *= (m+1)

    for mod_size in reversed(mod_sizes):
        cur_mod_indices = []
        for i in range(1, mod_size+1):
            cur_mod_indices.append(i * interval)

        mod_indices.append(cur_mod_indices)
        interval *= (mod_size + 1)

    mod_indices = reversed(mod_indices)

    return sum(mod_indices, [])


if __name__ == "__main__":
    t1 = torch.tensor([[2, 3], [3, 4], [5, 6]])
    t2 = torch.tensor([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
    t3 = torch.tensor([[16, 17], [18, 19], [18, 20]])
    t4 = torch.tensor([[20, 21], [22, 23], [24, 20]])
    tf = TensorFusionAlt()
    indices = find_concat_idx([t1.shape[1], t2.shape[1], t3.shape[1], t4.shape[1]])
    print(indices)
    print(tf(t1, t2, t3, t4)[:, indices])
