import tensorly as tly

tly.set_backend("pytorch")


import torch
import torch.nn as nn


# Define low-rank layer
class CustomConv2d(nn.Conv2d):

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride,
        padding,
        groups,
        dilation,
        bias=False,
        original_layer=None,
    ):
        """
        Initializes a low-rank layer with factorized weight matrices.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): If True, includes a bias term. Default is True.
            original_layer (nn.Module, optional): An existing layer to copy weights and bias from. If provided, initializes
                the low-rank layer using singular value decomposition of the original layer's weight matrix.

        Raises:
            ValueError: If in_features or out_features are not specified when original_layer is None.
        """
        super().__init__(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )
        del self.weight
        self.weight = original_layer.weight
        if self.bias is not None:
            del self.bias
            self.bias = original_layer.bias
        self.r = [
            self.weight.shape[0],
            self.weight.shape[1],
            self.weight.shape[2],
            self.weight.shape[3],
        ]
        self.unfold_mode = 0

    def prepare_save(self):
        pass

    def print_parameters(self):
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape}")

    @torch.no_grad()
    def compute_lr_params(self):
        return self.weight.shape[0] * self.weight.shape[1]

    @torch.no_grad()
    def compute_dense_params(self):
        return self.weight.shape[0] * self.weight.shape[1]

    def reset_low_rank_parameters(self) -> None:
        pass

    @torch.no_grad()
    def set_grad_zero(self) -> None:
        pass

    @torch.no_grad()
    def set_basis_grad_zero(self) -> None:
        pass

    @torch.no_grad()
    def deactivate_basis_grads(self) -> None:
        pass

    @torch.no_grad()
    def augment(self, optimizer) -> None:
        pass

    @torch.no_grad()
    def truncate(self, optimizer) -> None:
        pass

    def robustness_regularization(self, beta):
        """
        Computes the robustness regularization term for the low-rank layer.

        The robustness regularization term is given by the squared Frobenius norm of the difference
        between the core tensor S times its transpose and the identity matrix times the squared
        Frobenius norm of S divided by the rank of S.

        Args:
            beta (float): A parameter that controls the strength of the regularization term.

        Returns:
            float: The value of the robustness regularization term.
        """

        s_unfold = tly.base.unfold(
            self.weight,
            mode=self.unfold_mode,
        )
        return beta * (
            torch.linalg.norm(
                s_unfold @ s_unfold.T
                - torch.eye(self.r[0], device=s_unfold.device)
                * torch.linalg.norm(s_unfold) ** 2
                / self.r[self.unfold_mode]
            )
        )

    @torch.no_grad()
    def get_condition_nr(self):

        s_unfold = tly.base.unfold(
            self.weight,
            mode=self.unfold_mode,
        )
        return torch.linalg.cond(s_unfold)

    @torch.no_grad()
    def get_singular_spectrum(self):
        """
        Computes the singular spectrum of the unfolded core tensor.

        This function performs a singular value decomposition (SVD) on the unfolded core tensor
        of the low-rank layer and returns the singular values. These singular values represent
        the singular spectrum of the tensor and can provide insights into the properties and
        rank of the tensor.

        Returns:
            numpy.ndarray: A NumPy array containing the singular values of the unfolded core tensor.
        """

        P, d, Q = torch.linalg.svd(
            tly.base.unfold(
                self.weight,
                mode=self.unfold_mode,
            )
        )

        return d.detach().cpu().numpy()

    @torch.no_grad()
    def print_parameters(self):
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape}")

    @torch.no_grad()
    def compute_lr_params(self):
        product_r = 1
        for r in self.r:
            product_r *= r

        return product_r

    @torch.no_grad()
    def compute_dense_params(self):
        # Same as low-rank parameters here
        product_r = 1
        for r in self.r:
            product_r *= r

        return product_r
