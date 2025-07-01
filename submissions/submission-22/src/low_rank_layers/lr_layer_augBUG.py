import torch
import torch.nn as nn

from src.low_rank_layers.lr_layer_base import LowRankLayerBase


# Define low-rank layer
class LowRankLayerAugBUG(LowRankLayerBase):

    def __init__(
        self,
        in_features=-1,
        out_features=-1,
        bias=True,
        rmax=100,
        rmin=2,
        init_rank=50,
        tol=0.1,
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
            tol (float, optional): Error tolerance for the low-rank approximation. Default is 0.1.
            original_layer (nn.Module, optional): An existing layer to copy weights and bias from. If provided, initializes
                the low-rank layer using singular value decomposition of the original layer's weight matrix.

        Raises:
            ValueError: If in_features or out_features are not specified when original_layer is None.
        """
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            rmax=rmax,
            rmin=rmin,
            init_rank=init_rank,
            original_layer=original_layer,
        )

        self.tol = tol

    @torch.no_grad()
    def augment(
        self,
        optimizer,
    ) -> None:
        """Performs a steepest descend training update on specified low-rank factors
        Args:
            learning_rate: learning rate for training
            dlrt_step: sepcifies step that is taken. Can be 'K', 'L' or 'S'
            adaptive: specifies if fixed-rank or rank-adaptivity is used
        """

        r1 = min(self.rmax, 2 * self.r)
        # K step
        U1, _ = torch.linalg.qr(
            torch.cat((self.U[:, : self.r], -self.U.grad[:, : self.r]), 1),
            "reduced",
        )

        # L step
        V1, _ = torch.linalg.qr(
            torch.cat((self.V[:, : self.r], -self.V.grad[:, : self.r]), 1),
            "reduced",
        )

        # Basis projection

        M = U1[:, :r1].T @ self.U[:, : self.r]  # r1 x r
        N = self.V[:, : self.r].T @ V1[:, :r1]  # r x r1
        # Project coefficients

        self.S.data[:r1, :r1] = M @ self.S[: self.r, : self.r] @ N
        if optimizer.state[self.S]:
            state = optimizer.state[self.S]
            state["exp_avg"][:r1, :r1] = M @ state["exp_avg"][: self.r, : self.r] @ N
            state["exp_avg_sq"][:r1, :r1] = (
                M @ state["exp_avg_sq"].sqrt()[: self.r, : self.r] @ N
            ).pow(2)
        # update basis
        self.U.data[:, :r1] = U1[:, :r1]
        self.V.data[:, :r1] = V1[:, :r1]

        self.r = r1

    @torch.no_grad()
    def truncate(
        self,
        optimizer,
    ) -> None:
        """Truncates the weight matrix to a new rank"""

        P, d, Q = torch.linalg.svd(self.S[: self.r, : self.r])

        tol = self.tol * torch.linalg.norm(d)
        r1 = self.r
        for j in range(0, self.r):
            tmp = torch.linalg.norm(d[j : self.r])
            if tmp < tol:
                r1 = j
                break

        # Check if new ranks is withing legal bounds
        r1 = min(r1, self.rmax)
        r1 = max(r1, self.rmin)

        # update s
        self.S.data[:r1, :r1] = torch.diag(d[:r1])

        # update u and v
        self.U.data[:, :r1] = self.U[:, : self.r] @ P[:, :r1]
        self.V.data[:, :r1] = self.V[:, : self.r] @ Q.T[:, :r1]

        # Manage optimizer state and gradients
        state = optimizer.state[self.S]
        m_1 = P.T[:r1, :] @ state["exp_avg"][: self.r, : self.r] @ Q.T[:, :r1]
        state["exp_avg"][:r1, :r1] = m_1
        v_1 = P.T[:r1, :] @ state["exp_avg_sq"].sqrt()[: self.r, : self.r] @ Q.T[:, :r1]
        state["exp_avg_sq"][:r1, :r1] = v_1.pow(2)

        # Update Rank
        self.r = int(r1)

    def robustness_projection(self, beta):
        """
        Performs a robustness projection on the singular values of the core tensor.

        This function stabilizes the singular values by clamping them within a specified range
        around a reference value. The reference singular value `s_ref` is calculated as the root
        mean square of the singular values. The clamping range is determined by a parameter `beta`
        and ensures that the singular values remain close to the reference value.

        Args:
            beta (float): A parameter that controls the clamping range around the reference singular
                        value. Higher values allow for a wider range of singular values.

        Modifies:
            self.S: The core tensor of the low-rank layer, where its singular values are adjusted
                    according to the clamping operation.
        """

        P, d, Q = torch.linalg.svd(self.S[: self.r, : self.r])

        s_ref = torch.sqrt(torch.sum(d**2) / self.r)
        epsilon = beta * s_ref / (2 + beta)
        d_clamped = torch.clamp(d, min=s_ref - epsilon, max=s_ref + epsilon)
        print(d[0] / d[-1], d_clamped[0] / d_clamped[-1])
        # print(d_clamped[0], d_clamped[-1])
        # print("===")
        self.S.data[: self.r, : self.r] = P @ torch.diag(d_clamped) @ Q

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
        # W = self.U[:, : self.r] @ self.S[: self.r, : self.r] @ self.V[:, : self.r].T
        # print(W.shape)
        # return beta * (
        #    torch.linalg.norm(
        #        W @ W.T
        #        - torch.eye(W.shape[0], device=self.S.device)
        #        * torch.linalg.norm(W) ** 2
        #        / self.r
        #    )
        # )
        return beta * (
            torch.linalg.norm(
                self.S[: self.r, : self.r] @ self.S[: self.r, : self.r].T
                - torch.eye(self.r, device=self.S.device)
                * torch.linalg.norm(self.S[: self.r, : self.r]) ** 2
                / self.r
            )
        )


class LowRankLayerAugBUGL1Truncation(LowRankLayerAugBUG):

    def __init__(
        self, input_size, output_size, rmax, rmin=2, init_cr=0.5, tol=0.1, bias=True
    ):
        """Constructs a low-rank layer of the form U*S*V'*x + b, where
           U, S, V represent the facorized weight W and b is the bias vector
        Args:
            input_size: input dimension of weight W
            output_size: output dimension of weight W, dimension of bias b
            rank: initial rank of factorized weight
            adaptive: set True if layer isrank-adaptive
            init_compression: initial compression of neural network
        """
        super(LowRankLayerAugBUGL1Truncation, self).__init__(
            input_size, output_size, rmax, rmin, init_cr, tol, bias
        )

    @torch.no_grad()
    def truncate(self, learning_rate) -> None:
        """Truncates the weight matrix to a new rank"""
        P, d, QT = torch.linalg.svd(self.S[: self.r, : self.r])

        tol = self.tol * torch.linalg.norm(d)
        r1 = self.r
        d = d * (1 - learning_rate * tol / d)  # l1 truncation
        # torch.relu((1 - learning_rate * tol / d))
        for j in range(0, self.r):  # find new rank
            if d[j] <= 0:
                r1 = j
                break

        r1 = min(r1, self.rmax)
        r1 = max(r1, 10)

        # update s
        self.S.data[:r1, :r1] = torch.diag(d[:r1])
        self.Sinv[:r1, :r1] = torch.diag(1.0 / d[:r1])

        # update u and v
        self.U.data[:, :r1] = torch.matmul(self.U[:, : self.r], P[:, :r1])
        self.V.data[:, :r1] = torch.matmul(self.V[:, : self.r], QT.T[:, :r1])
        self.r = int(r1)
