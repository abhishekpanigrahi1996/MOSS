'''Various miscellaneous modules
'''

import torch
from math import prod


class GlobalPooling(torch.nn.Module):
    def __init__(self):
        super(GlobalPooling, self).__init__()

    def forward(self, x):
        # apply global average pooling
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, 2)
        x = x.view(x.size(0), -1)

        return x


class View(torch.nn.Module):
    """Basic reshape module.
    """
    def __init__(self, *shape):
        """
        Args:
            *shape: Input shape.
        """
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        """Reshapes tensor.
        Args:
            input: Input tensor.
        Returns:
            torch.Tensor: Flattened tensor.
        """
        return input.view(*self.shape)


class Unfold(torch.nn.Module):
    """Module for unfolding tensor.
    Performs strided crops on 2d (image) tensors. Stride is assumed to be half the crop size.
    """
    def __init__(self, img_size, fold_size, stride=None, final_size=None):
        """
        Args:
            img_size: Input size.
            fold_size: Crop size.
        """
        super(Unfold, self).__init__()

        fold_stride = stride or fold_size // 2
        self.fold_size = fold_size
        self.fold_stride = fold_stride
        self.n_locs = final_size or (2 * int(img_size / fold_size) - 1)
        self.unfold = torch.nn.Unfold((self.fold_size, self.fold_size),
                                      stride=(self.fold_stride, self.fold_stride))

    def forward(self, x):
        """Unfolds tensor.
        Args:
            x: Input tensor.
        Returns:
            torch.Tensor: Unfolded tensor.
        """
        N = x.size(0)
        x = self.unfold(x).reshape(N, -1, self.fold_size, self.fold_size, self.n_locs * self.n_locs)\
            .permute(0, 4, 1, 2, 3)\
            .reshape(N * self.n_locs * self.n_locs, -1, self.fold_size, self.fold_size)
        return x.contiguous()


class Fold(torch.nn.Module):
    """Module (re)folding tensor.
    Undoes the strided crops above. Works only on 1x1.
    """
    def __init__(self, n_locs):
        """
        Args:
            n_locs: Number of locations in output.
        """
        super(Fold, self).__init__()
        self.n_locs = n_locs

    def forward(self, x):
        """(Re)folds tensor.
        Args:
            x: Input tensor.
        Returns:
            torch.Tensor: Refolded tensor.
        """
        x_shape = x.size()[1:]

        if len(x_shape) == 3:
            dim_c, dim_x, dim_y = x_shape
        else:
            dim_c = x_shape[0]
            dim_x = 1
            dim_y = 1
        x = x.reshape(-1, self.n_locs * self.n_locs, dim_c, dim_x * dim_y)
        x = x.reshape(-1, self.n_locs * self.n_locs, dim_c, dim_x * dim_y)\
            .permute(0, 2, 3, 1)\
            .reshape(-1, dim_c * dim_x * dim_y, self.n_locs, self.n_locs).contiguous()
        return x


class Permute(torch.nn.Module):
    """Module for permuting axes.
    """
    def __init__(self, *perm):
        """
        Args:
            *perm: Permute axes.
        """
        super(Permute, self).__init__()
        self.perm = perm

    def forward(self, input):
        """Permutes axes of tensor.
        Args:
            input: Input tensor.
        Returns:
            torch.Tensor: permuted tensor.
        """
        return input.permute(*self.perm)


class Expand2d(torch.nn.Module):
    """Expands 2d tensor to 4d
    """

    def forward(self, input):
        """Expands tensor to 4d.
        Args:
            input: Input tensor.
        Returns:
            torch.Tensor: expanded tensor.
        """
        size = input.size()

        if len(size) == 4:
            return input
        elif len(size) == 2:
            return input[:, None, None, :]
        else:
            raise ValueError()


class Flatten(torch.nn.Module):
    def __init__(self, args=None):
        super(Flatten, self).__init__()
        self.args = args
        return

    def forward(self, input):
        return input.view(input.size(0), -1)


class Downsample(torch.nn.Module):
    def __init__(self, shape, mode):
        super(Downsample, self).__init__()
        self.shape = shape
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.shape, mode=self.mode)


# class GlobalPooling2D(torch.nn.Module):
#     def __init__(self):
#         super(GlobalPooling2D, self).__init__()
#
#     def forward(self, x):
#         # apply global average pooling
#         x = x.view(x.size(0), x.size(1), -1)
#         x = torch.mean(x, 2)
#         x = x.view(x.size(0), -1)
#
#         return x


class SameShapeLinear(torch.nn.Module):
    def __init__(self, img_dim, txt_dim):
        super().__init__()
        self.flatten_img_dim = prod(img_dim)
        self.linear = torch.nn.Linear(self.flatten_img_dim + txt_dim, self.flatten_img_dim)

    def forward(self, img, text):
        N, C, H, W = img.shape
        assert self.flatten_img_dim == C * H * W
        flatten_img = img.view(-1, self.flatten_img_dim)
        flatten_output = self.linear(torch.cat((flatten_img, text), dim=1)).view(-1, C, H, W)
        return flatten_output
