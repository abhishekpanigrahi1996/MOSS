from torch import vmap
from torch._functorch.eager_transforms import jacfwd, jacrev
from torch._functorch.functional_call import functional_call

from turtle import st
from typing import Any

from cycler import V
import einops
import torch

from util.model_utils import tensor_product_xz


class NetWithPartials:
    '''
    The only stateful part of the model is the parameters.
    '''
    
    @staticmethod
    def create_from_model(model):
        ## Parameters for stateless model
        params = dict(model.named_parameters())
        
        def f(params, x, z):
            return functional_call(model, params, (x, z))
        vf = vmap(f, in_dims=(None, 0, 0), out_dims=(0))  ## params, [bxz, nx], [bxz, nz] -> [bxz, ny]
        ## Note the difference: in the in_dims and out_dims we want to vectorize in the 0-th dimension
        ## Jacobian
        f_x = jacrev(f, argnums=1)  ## params, [nx], [nz] -> [nx, ny]
        vf_x = vmap(f_x, in_dims=(None, 0, 0), out_dims=(0))  ## params, [bxz, nx], [bxz, nz] -> [bxz, ny, nx]
        ## Hessian
        f_xx = jacfwd(f_x, argnums=1)  ## params, [nx], [nz] -> [nx, ny, nx]
        vf_xx = vmap(f_xx, in_dims=(None, 0, 0), out_dims=(0))  ## params, [bxz, nx], [bxz, nz] -> [bxz, ny, nx, nx]
        
        return NetWithPartials(params=params, f=f, vf=vf, vf_x=vf_x, vf_xx=vf_xx)
            
    
    def __init__(self, params, f, vf, vf_x, vf_xx) -> None:
        self.params = params
        self.f_ = f
        self.vf_ = vf
        self.vf_x_ = vf_x
        self.vf_xx_ = vf_xx
    
    def __call__(self, *args: Any) -> Any:
        return self.vf(*args) # params and p are already included in the vf call
    
    def f(self, *args: Any) -> Any:
        return self.f_(self.params, *args)
    
    def vf(self, *args: Any) -> Any:
        return self.vf_(self.params, *args)
    
    def vf_x(self, *args: Any) -> Any:
        return self.vf_x_(self.params, *args)
    
    def vf_xx(self, *args: Any) -> Any:
        return self.vf_xx_(self.params, *args)