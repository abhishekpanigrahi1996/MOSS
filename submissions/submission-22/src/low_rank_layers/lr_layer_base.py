import torch
import torch.nn as nn
import math


# Define low-rank layer
class LowRankLayerBase(nn.Linear):

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        rmax=100,
        rmin=2,
        init_rank=50,
        original_layer=None,
    ):
        """
        Initializes a low-rank layer with factorized weight matrices.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            bias (bool, optional): If True, includes a bias term. Default is True.
            rmax (int, optional): Maximum rank for augmentation. Default is 100.
            rmin (int, optional): Minimum rank for truncation. Default is 2.
            init_rank (int, optional): Initial rank for the factorized weight. If negative, defaults to rmax. Default is 50.
            original_layer (nn.Module, optional): An existing layer to copy weights and bias from. If provided, initializes
                the low-rank layer using singular value decomposition of the original layer's weight matrix.

        Raises:
            ValueError: If in_features or out_features are not specified when original_layer is None.
        """
        super().__init__(in_features, out_features, bias)
        del self.weight  # Not needed
        # set rank and truncation tolerance

        if original_layer is None:
            if in_features == -1 or out_features == -1:
                raise ValueError("Input and output sizes must be specified")

            self.rmax = int(
                min(int(min(in_features, out_features) / 2), rmax)
            )  # maximal rank for augmentation
            self.rmin = rmin  # minimal rank for truncation
            if init_rank < 0:
                self.r = self.rmax
            else:
                self.r = int(init_rank)
            # declare factorized bases
            self.U = nn.Parameter(torch.empty(in_features, self.rmax))
            self.V = nn.Parameter(torch.empty(out_features, self.rmax))

            self.reset_low_rank_parameters()
            # ensure that U and V are orthonormal
            self.U.data, _ = torch.linalg.qr(self.U, "reduced")
            self.V.data, _ = torch.linalg.qr(self.V, "reduced")

            # initilize coefficient matrix
            self.singular_values, _ = torch.sort(
                torch.randn(self.rmax) ** 2, descending=True
            )
            self.S = nn.Parameter(torch.diag(self.singular_values))
        else:

            self.rmax = int(min(min(in_features, out_features), rmax))
            self.rmin = rmin
            self.r = self.rmax  # max(min(init_rank, self.rmax), self.rmin)
            P, d, QT = torch.linalg.svd(original_layer.weight.T)
            self.U = nn.Parameter(P[:, : self.rmax])
            self.V = nn.Parameter(QT.T[:, : self.rmax])
            self.S = nn.Parameter(torch.diag(d[: self.rmax]))

            if original_layer.bias is not None:
                self.bias = nn.Parameter(original_layer.bias.clone())

    def prepare_save(self):
        """
        Prepares the layer for saving by adjusting the ranks and parameters.

        This method updates the `rmax`, `U`, `V`, and `S` attributes to reflect the
        current ranks `r`. It ensures that the parameters are correctly sized
        for saving the model state.
        """
        # self.rmax = self.r
        # self.U = nn.Parameter(self.U[:, : self.r])
        # self.V = nn.Parameter(self.V[:, : self.r])
        # self.S = nn.Parameter(self.S[: self.r, : self.r])

    # ---- Save additional parameters ----
    def extra_repr(self):
        return f"rmax={self.rmax}, rank={self.r}"

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # Add custom attributes
        destination[prefix + "r"] = self.r

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Load the saved attributes
        self.r = state_dict.pop(prefix + "r", self.r)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def print_parameters(self):
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape}")

    @torch.no_grad()
    def compute_lr_params(self):
        return self.U.shape[0] * self.r + self.r**2 + self.V.shape[0] * self.r

    @torch.no_grad()
    def compute_dense_params(self):
        return self.U.shape[0] * self.V.shape[0]

    def reset_low_rank_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.S, a=math.sqrt(5)) # Dont initialize S
        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.V.T)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def set_grad_zero(self) -> None:
        self.U.grad.zero_()
        if self.S.grad is not None:
            self.S.grad.zero_()
        self.V.grad.zero_()
        if self.bias.grad is not None:
            self.bias.grad.zero_()

    @torch.no_grad()
    def set_basis_grad_zero(self) -> None:
        self.U.grad.zero_()
        self.V.grad.zero_()
        if self.bias is not None:
            self.bias.grad.zero_()

    @torch.no_grad()
    def deactivate_basis_grads(self) -> None:
        self.U.requires_grad_(False)
        self.V.requires_grad_(False)

    def forward(self, x) -> torch.Tensor:
        """Returns the output of the layer. The formula implemented is output = U*S*V'*x + bias.
        Args:
            x: input to layer
        Returns:
            output of layer
        """
        out = ((x @ self.U[:, : self.r]) @ self.S[: self.r, : self.r]) @ self.V[
            :, : self.r
        ].T
        if self.bias is not None:
            return out + self.bias
        return out

    @torch.no_grad()
    def augment(self) -> None:
        pass

    @torch.no_grad()
    def truncate(self) -> None:
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
        P, d, Q = torch.linalg.svd(self.S[: self.r, : self.r])

        # d_out = np.zeros(min(d.shape))
        # d_out[: self.r] =
        return d.detach().cpu().numpy()

    @torch.no_grad()
    def get_condition_nr(self):
        return torch.linalg.cond(self.S[: self.r, : self.r])
