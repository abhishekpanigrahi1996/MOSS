import torch
import collections

collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping

import tltorch
import math
from torch import nn
import tensorly as tly

tly.set_backend("pytorch")


class Conv2dLowRankLayerAugBUG(torch.nn.Conv2d):

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
        dynamic_modes=[0, 1],
        rmax=100,
        rmin=7,
        init_rank=50,
        tol=0.1,
        original_layer=None,
    ) -> None:
        """
        Initializer for the convolutional low rank layer (filterwise), extention of the classical Pytorch's convolutional layer.
        INPUTS:
        in_channels: number of input channels (Pytorch's standard)
        out_channels: number of output channels (Pytorch's standard)
        kernel_size : kernel_size for the convolutional filter (Pytorch's standard)
        dilation : dilation of the convolution (Pytorch's standard)
        padding : padding of the convolution (Pytorch's standard)
        stride : stride of the filter (Pytorch's standard)
        bias  : flag variable for the bias to be included (Pytorch's standard)
        step : string variable ('K','L' or 'S') for which forward phase to use
        rank : rank variable, None if the layer has to be treated as a classical Pytorch Linear layer (with weight and bias). If
                it is an int then it's either the starting rank for adaptive or the fixed rank for the layer.
        fixed : flag variable, True if the rank has to be fixed (KLS training on this layer)
        load_weights : variables to load (Pytorch standard, to finish)
        dtype : Type of the tensors (Pytorch standard, to finish)
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
        del self.weight  # Not needed

        self.tau = tol
        self.dynamic_modes = dynamic_modes
        if original_layer is None:
            self.dims = [self.out_channels, self.in_channels] + list(self.kernel_size)

            # make sure that there are at least 3 channels, for rgb images
            self.r = [max(init_rank, 3) for d in self.dims[:2]] + self.dims[2::]
            self.rmax = [min(int(d / 2), rmax) for d in self.dims]
            self.rmin = [max(3, rmin) for d in self.dims]

            self.S = torch.nn.Parameter(torch.empty(size=self.rmax), requires_grad=True)
            self.Us = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(torch.empty(size=(d, r)), requires_grad=True)
                    for d, r in zip(self.dims, self.rmax)
                ]
            )

            self.reset_tucker_parameters()  # parameter intitialization
        else:
            stride = original_layer.stride
            padding = original_layer.padding
            dilation = original_layer.dilation

            self.dims = [
                original_layer.out_channels,
                original_layer.in_channels,
            ] + list(original_layer.kernel_size)

            self.rmin = [3 for d in self.dims]

            self.rmax = [
                (
                    int(min(self.dims[i] / 2, rmax))
                    if i < 2
                    else int(min(self.dims[i], rmax))
                )
                for i in range(4)
            ]
            self.r = [self.rmax[i] for i in range(len(self.rmax))]

            core, factors = tly.decomposition.tucker(
                original_layer.weight, rank=self.dims
            )
            self.Us = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(factor[:, : self.rmax[i]])
                    for i, factor in enumerate(factors)
                ]
            )
            self.S = torch.nn.Parameter(
                core[: self.rmax[0], : self.rmax[1], : self.rmax[2], : self.rmax[3]]
            )

            if original_layer.bias is not None:
                self.bias = nn.Parameter(original_layer.bias.clone())

            else:
                self.bias = None

    def prepare_save(self):
        """
        Prepares the layer for saving by adjusting the ranks and parameters.

        This method updates the `rmax`, `Us`, and `S` attributes to reflect the
        current ranks `r`. It ensures that the parameters are correctly sized
        for saving the model state.
        """
        # self.rmax = [r for r in self.r]

        # self.Us = torch.nn.ParameterList(
        #    [torch.nn.Parameter(U[:, : self.r[i]]) for i, U in enumerate(self.Us)]
        # )
        # self.S = torch.nn.Parameter(
        #    self.S[: self.r[0], : self.r[1], : self.r[2], : self.r[3]]
        # )

    # ---- Save additional parameters ----
    def extra_repr(self):
        return f"rank={self.r}"

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        # Add custom attributes
        destination[prefix + "r"] = self.r

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Load the saved attributes
        self.r = state_dict.pop(prefix + "r", self.r)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, input):
        """
        forward phase for the convolutional layer. It has to contain the three different
        phases for the steps 'K','L' and 'S' in order to be optimizable using dlrt.
        Every step is rewritten in terms of the tucker decomposition of the kernel tensor
        """
        C = self.S[: self.r[0], : self.r[1], : self.r[2], : self.r[3]]
        Us = [U[:, : self.r[i]] for i, U in enumerate(self.Us)]

        result = tltorch.functional.tucker_conv(
            input,
            tucker_tensor=tltorch.TuckerTensor(C, Us, rank=self.r),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        if self.bias is not None:

            result += self.bias.view(1, -1, 1, 1)
        return result

    @torch.no_grad()
    def augment(
        self,
        optimizer,
    ):
        """
        Augmentation step for a Tucker layer. The function performs
        gradient-based update of the factor matrices and the core tensor.
        The new rank is taken as 2*r or max rank, whichever is smaller.
        The function projects the core tensor onto the new basis and
        updates the factor matrices.
        """
        Ms = []

        r1 = [r for r in self.r]
        for i in range(4):  # Only augment the dyamic modes
            if i in self.dynamic_modes:
                r1[i] = min(
                    2 * self.r[i], self.rmax[i]
                )  # take new rank as 2*r or max rank

                U1 = torch.linalg.qr(
                    torch.cat(
                        (
                            self.Us[i].data[:, : self.r[i]],
                            -self.Us[i].grad[:, : self.r[i]],
                        ),
                        axis=1,
                    ),
                    "reduced",
                )[0]

                Ms.append(
                    U1[:, : r1[i]].T @ self.Us[i][:, : self.r[i]]
                )  # dims:r x r_old
                self.Us[i][:, : r1[i]] = U1[:, : r1[i]]
            else:
                Ms.append(torch.eye(self.r[i]))
                # = self.Us[i][:, : r1[i]].T @ self.Us[i][:, : self.r[i]]

        # project S onto new basis
        if self.dynamic_modes == [0, 1]:  # default, accelerated code
            self.S.data[: r1[0], : r1[1], : r1[2], : r1[3]] = torch.einsum(
                "abcd,ia,jb->ijcd",
                self.S.data[
                    : self.r[0],
                    : self.r[1],
                    : self.r[2],
                    : self.r[3],
                ],
                Ms[0],
                Ms[1],
            )
            if optimizer.state[self.S]:
                state = optimizer.state[self.S]
                state["exp_avg"][: r1[0], : r1[1], : r1[2], : r1[3]] = torch.einsum(
                    "abcd,ia,jb->ijcd",
                    state["exp_avg"][
                        : self.r[0],
                        : self.r[1],
                        : self.r[2],
                        : self.r[3],
                    ],
                    Ms[0],
                    Ms[1],
                )
                state["exp_avg_sq"][: r1[0], : r1[1], : r1[2], : r1[3]] = torch.einsum(
                    "abcd,ia,jb->ijcd",
                    state["exp_avg_sq"][
                        : self.r[0],
                        : self.r[1],
                        : self.r[2],
                        : self.r[3],
                    ].sqrt(),
                    Ms[0],
                    Ms[1],
                ).pow(2)
        else:
            #    exit("not dynamic mode not tested yet")
            self.S.data[: r1[0], : r1[1], : r1[2], : r1[3]] = torch.einsum(
                "abcd,ia,jb,kc,ld->ijkl",
                self.S.data[
                    : self.r[0],
                    : self.r[1],
                    : self.r[2],
                    : self.r[3],
                ],
                Ms[0],
                Ms[1],
                Ms[2],
                Ms[3],
            )
            if optimizer.state[self.S]:
                state = optimizer.state[self.S]
                state["exp_avg"][: r1[0], : r1[1], : r1[2], : r1[3]] = torch.einsum(
                    "abcd,ia,jb,kc,ld->ijkl",
                    state["exp_avg"][
                        : self.r[0],
                        : self.r[1],
                        : self.r[2],
                        : self.r[3],
                    ],
                    Ms[0],
                    Ms[1],
                )
                state["exp_avg_sq"][: r1[0], : r1[1], : r1[2], : r1[3]] = torch.einsum(
                    "abcd,ia,jb,kc,ld->ijkl",
                    state["exp_avg_sq"][
                        : self.r[0],
                        : self.r[1],
                        : self.r[2],
                        : self.r[3],
                    ].sqrt(),
                    Ms[0],
                    Ms[1],
                ).pow(2)
        del Ms
        self.r = [r for r in r1]

    @torch.no_grad()
    def truncate(
        self,
        optimizer,
    ):
        r_hat = [r for r in self.r]
        Ps = []
        for i in range(4):
            if i in self.dynamic_modes:  # only truncate dynamic modes
                MAT_i_C = tly.base.unfold(
                    self.S[: r_hat[0], : r_hat[1], : r_hat[2], : r_hat[3]], mode=i
                )

                P, d, _ = torch.linalg.svd(MAT_i_C, full_matrices=False)

                Ps.append(P)

                tol = self.tau * torch.linalg.norm(d)
                r_new = r_hat[i]
                for j in range(0, r_hat[i]):
                    tmp = torch.linalg.norm(d[j : r_hat[i]])
                    if tmp < tol:
                        r_new = j
                        break

                self.r[i] = max(min(r_new, self.rmax[i]), self.rmin[i])  # rank update

                # update U
                self.Us[i].data[:, : self.r[i]] = (
                    self.Us[i].data[:, : r_hat[i]] @ P[: r_hat[i], : self.r[i]]
                )
            else:
                Ps.append(torch.eye(self.r[i]))

        # update Core
        if self.dynamic_modes == [0, 1]:  # default, accelerated code
            self.S.data[: self.r[0], : self.r[1], : self.r[2], : self.r[3]] = (
                torch.einsum(
                    "abcd,ia,jb->ijcd",
                    self.S.data[: r_hat[0], : r_hat[1], : r_hat[2], : r_hat[3]],
                    Ps[0][: r_hat[0], : self.r[0]].T,
                    Ps[1][: r_hat[1], : self.r[1]].T,
                )
            )
            # Manage optimizer state and gradients
            state = optimizer.state[self.S]
            state["exp_avg"][: self.r[0], : self.r[1], : self.r[2], : self.r[3]] = (
                torch.einsum(
                    "abcd,ia,jb->ijcd",
                    state["exp_avg"][: r_hat[0], : r_hat[1], : r_hat[2], : r_hat[3]],
                    Ps[0][: r_hat[0], : self.r[0]].T,
                    Ps[1][: r_hat[1], : self.r[1]].T,
                )
            )
            state["exp_avg_sq"][: self.r[0], : self.r[1], : self.r[2], : self.r[3]] = (
                torch.einsum(
                    "abcd,ia,jb->ijcd",
                    state["exp_avg_sq"][
                        : r_hat[0], : r_hat[1], : r_hat[2], : r_hat[3]
                    ].sqrt(),
                    Ps[0][: r_hat[0], : self.r[0]].T,
                    Ps[1][: r_hat[1], : self.r[1]].T,
                )
            ).pow(2)

        else:
            exit("not dynamic mode not tested yet")
            self.S.data[: self.r[0], : self.r[1], : self.r[2], : self.r[3]] = (
                torch.einsum(
                    "abcd,ia,jb,kc,ld->ijkl",
                    self.S.data[: r_hat[0], : r_hat[1], : r_hat[2], : r_hat[3]],
                    Ps[0][: r_hat[0], : self.r[0]].T,
                    Ps[1][: r_hat[1], : self.r[1]].T,
                    Ps[2][: r_hat[2], : self.r[2]].T,
                    Ps[3][: r_hat[3], : self.r[3]].T,
                )
            )
            # Manage optimizer state and gradients
            state = optimizer.state[self.S]
            state["exp_avg"][: self.r[0], : self.r[1], : self.r[2], : self.r[3]] = (
                torch.einsum(
                    "abcd,ia,jb,kc,ld->ijkl",
                    state["exp_avg"][: r_hat[0], : r_hat[1], : r_hat[2], : r_hat[3]],
                    Ps[0][: r_hat[0], : self.r[0]].T,
                    Ps[1][: r_hat[1], : self.r[1]].T,
                    Ps[2][: r_hat[2], : self.r[2]].T,
                    Ps[3][: r_hat[3], : self.r[3]].T,
                )
            )
            state["exp_avg_sq"][: self.r[0], : self.r[1], : self.r[2], : self.r[3]] = (
                torch.einsum(
                    "abcd,ia,jb,kc,ld->ijkl",
                    state["exp_avg_sq"][
                        : r_hat[0], : r_hat[1], : r_hat[2], : r_hat[3]
                    ].sqrt(),
                    Ps[0][: r_hat[0], : self.r[0]].T,
                    Ps[1][: r_hat[1], : self.r[1]].T,
                    Ps[2][: r_hat[2], : self.r[2]].T,
                    Ps[3][: r_hat[3], : self.r[3]].T,
                )
            ).pow(2)

        del Ps

    @torch.no_grad()
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
        S_unfold = tly.base.unfold(
            self.S[: self.r[0], : self.r[1], : self.r[2], : self.r[3]], mode=0
        )

        P, d, Q = torch.linalg.svd(S_unfold, full_matrices=False)

        s_ref = torch.sqrt(torch.sum(d**2) / self.r[0])
        epsilon = beta * s_ref / (2 + beta)
        d_clamped = torch.clamp(d, min=s_ref - epsilon, max=s_ref + epsilon)
        tmp = tly.base.fold(P @ torch.diag(d_clamped) @ Q, mode=0, shape=self.r)

        self.S.data[: self.r[0], : self.r[1], : self.r[2], : self.r[3]] = tmp

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
        # S_full = torch.einsum(
        #    "abcd,ia,jb,kc,ld->ijkl",
        #    self.S.data[
        #        : self.r[0],
        #        : self.r[1],
        #        : self.r[2],
        #        : self.r[3],
        #    ],
        #    self.Us[0][: self.r[0], :],
        #    self.Us[1][: self.r[1], :],
        #    self.Us[2][: self.r[2], :],
        #    self.Us[3][: self.r[3], :],
        # )
        s_unfold = tly.base.unfold(
            self.S[: self.r[0], : self.r[1], : self.r[2], : self.r[3]],
            # S_full,
            mode=0,
        )

        return beta * (
            torch.linalg.norm(
                s_unfold @ s_unfold.T
                - torch.eye(self.r[0], device=s_unfold.device)
                * torch.linalg.norm(s_unfold) ** 2
                / self.r[0]
            )
        )

    @torch.no_grad()
    def get_condition_nr(self):

        s_unfold = tly.base.unfold(
            self.S[: self.r[0], : self.r[1], : self.r[2], : self.r[3]], mode=0
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
                self.S[: self.r[0], : self.r[1], : self.r[2], : self.r[3]], mode=0
            )
        )

        return d.detach().cpu().numpy()

    @torch.no_grad()
    def set_basis_grad_zero(self):
        for i in range(len(self.Us)):
            self.Us[i].grad.zero_()

    @torch.no_grad()
    def set_grad_zero(self):
        for i in range(len(self.Us)):
            self.Us[i].grad.zero_()
        if self.S.grad is not None:
            self.S.grad.zero_()

    @torch.no_grad()
    def deactivate_basis_grads(self) -> None:
        for i in range(len(self.Us)):
            self.Us[i].requires_grad_(False)

    @torch.no_grad()
    def print_parameters(self):
        for name, param in self.named_parameters():
            print(f"{name}: {param.shape}")

    @torch.no_grad()
    def compute_lr_params(self):
        sum_terms = 0

        for i in range(len(self.Us)):
            sum_terms += self.Us[i].shape[0] * self.r[i]

        product_r = 1
        for r in self.r:
            product_r *= r

        return sum_terms + product_r

    @torch.no_grad()
    def compute_dense_params(self):
        product_terms = 1

        for i in range(len(self.Us)):
            product_terms *= self.Us[i].shape[0]

        return product_terms

    @torch.no_grad()
    def reset_tucker_parameters(self):
        torch.nn.init.kaiming_uniform_(self.S, a=math.sqrt(5))
        for i in range(len(self.dims)):
            torch.nn.init.kaiming_uniform_(self.Us[i], a=math.sqrt(5))

            # Orthonormalize bases
            self.Us[i].data, _ = torch.linalg.qr(self.Us[i].data, "reduced")
