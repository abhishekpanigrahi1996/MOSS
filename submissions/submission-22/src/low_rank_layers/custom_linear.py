import torch
import torch.nn as nn


# Define low-rank layer
class CustomLinearLayer(nn.Linear):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
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
        super().__init__(in_features, out_features, bias)
        del self.weight
        self.weight = original_layer.weight
        if self.bias is not None:
            del self.bias
            self.bias = original_layer.bias
        self.r = min(self.weight.shape[0], self.weight.shape[1])

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

    @torch.no_grad()
    def get_singular_spectrum(self):
        """
        Computes the singular spectrum of the core tensor.

        This function performs a singular value decomposition (SVD) on S
        of the low-rank layer and returns the singular values. These singular values represent
        the singular spectrum of S and can provide insights into the properties and
        rank of the tensor.

        Returns:
            numpy.ndarray: A NumPy array containing the singular values of S.
        """
        P, d, Q = torch.linalg.svd(self.weight)

        # d_out = np.zeros(min(d.shape))
        # d_out[: self.r] =
        return d.detach().cpu().numpy()

    @torch.no_grad()
    def get_condition_nr(self):
        return torch.linalg.cond(self.weight)

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

        return beta * (
            torch.linalg.norm(
                self.weight @ self.weight.T
                - torch.eye(self.out_features, device=self.weight.device)
                * torch.linalg.norm(self.weight) ** 2
                / self.out_features
            )
        )
