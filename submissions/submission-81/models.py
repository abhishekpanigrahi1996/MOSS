import torch
from sparsemax import Sparsemax
import torch.nn as nn
import torch.nn.functional as F
import torch
from functions.utils import softplus_inverse, jumprelu

class SAE(torch.nn.Module):
    def __init__(self, dimin=2, width=5, sae_type='relu', kval_topk=None, \
        normalize_decoder=False, lambda_init = None, auxkval=None):
        """
        dimin: (int)
            input dimension
        width: (int)
            width of the encoder
        sae_type: (str)
            one of 'relu', 'topk', 'topk_relu', 'jumprelu', 'sparsemax_lintx', 'sparsemax_dist'
            topk_relu: uses relu followed by topk, follows the implementation of TopK SAE in Gao et al "Scaling and Evaluating Sparse Autoencoders"
            topk: uses only topk without relu
            sparsemax_dist: used in SpaDE
        kval_topk: (int)
            k in topk or topk_relu sae_type
            ignored if sae_type is not topk or topk_relu
        normalize_decoder: (bool)
            whether to normalize the decoder weights to unit norm in each forward pass
        lambda_init: (float)
            initial value of lambda (inverse temperature). If None, set to 1/(2*dimin) (default)
            A scaling constant used to scale pre-activations in encoder for all SAEs.
            It is trainable and important in sparsemax_dist, but is a redundant, fixed parameter in other SAEs.
        auxkval: (int)
            auxiliary k value, used in constructing the auxiliary loss for topk/ topk_relu sae_type
        """
        super(SAE, self).__init__()
        #store parameters in SAE
        self.sae_type = sae_type
        self.width = width
        self.dimin = dimin
        self.normalize_decoder = normalize_decoder

        #inv. temperature parameter (mainly useful for sparsemax_dist but also used in others as a redundant scaling constant)
        lambda_init = 1/(2*dimin) if lambda_init is None else lambda_init         
        lambda_pre = softplus_inverse(lambda_init) #allow using softplus to ensure lambda is positive
        
        
        #weights (A) and biases (b) of encoder (e) and decoder (d)
        self.be = nn.Parameter(torch.zeros((1, width)))
        self.Ae = nn.Parameter(torch.randn((width, dimin))) #N(0,1) init
        self.bd = nn.Parameter(torch.zeros((1, dimin)))
        self.Ad = nn.Parameter(torch.randn((dimin, width))) #N(0,1) init
        with torch.no_grad():
            self.Ad.copy_(self.Ae.T) #at init, decoder is the transpose of encoder

        #topk-specific parameters
        if sae_type=='topk' or sae_type=='topk_relu':
            self.deadcounts = nn.Parameter(torch.zeros((width)), requires_grad=False)
            self.auxkval = kval_topk if auxkval is None else auxkval
        if kval_topk is not None:
            self.kval_topk = kval_topk

        #jumprelu specific parameters
        if sae_type=='jumprelu':
            self.logthreshold = nn.Parameter(torch.log(1e-3*torch.ones((1, width))))
            self.bandwidth = 1e-3 #width of rectangle used in approx grad of jumprelu wrt threshold
        
        #sparsemax-specific parameters (for SpaDE)- make lambda trainable
        if 'sparsemax' in sae_type:
            self.lambda_pre = nn.Parameter(lambda_pre) #trainable parameter (~inv temp) for sparsemax
        else:
            self.lambda_pre = nn.Parameter(lambda_pre, requires_grad=False) #not trainable

    @property
    def lambda_val(self): #lambda_val is lambda, forced to be positive with softplus=log(1+exp(lambda_pre))
        return F.softplus(self.lambda_pre)

    def forward(self, x, return_hidden=False):
        lam = self.lambda_val #compute positive parameter lambda with softplus

        #ReLU SAE
        if self.sae_type=='relu':
            x = x-self.bd #pre-encoder bias
            x = torch.matmul(x, self.Ae.T) + self.be
            xint = F.relu(lam*x)
            if self.normalize_decoder: #unit norm
                eps = 1e-6
                Ad_unit = self.Ad / (eps+torch.linalg.norm(self.Ad, dim=0, keepdim=True))
                x = torch.matmul(xint, Ad_unit.T) + self.bd
            else:
                x = torch.matmul(xint, self.Ad.T) + self.bd

        #TopK without ReLU- not used
        elif self.sae_type=='topk':
            x = x-self.bd #pre-encoder bias
            x = torch.matmul(x, self.Ae.T)
            _, topk_indices = torch.topk(x, self.kval_topk, dim=-1)
            mask = torch.zeros_like(x)
            mask.scatter_(-1, topk_indices, 1)
            xint = x * mask* lam
            if self.normalize_decoder: #unit norm
                eps = 1e-6
                Ad_unit = self.Ad / (eps + torch.linalg.norm(self.Ad, dim=0, keepdim=True))
                x = torch.matmul(xint, Ad_unit.T) + self.bd
            else:
                x = torch.matmul(xint, self.Ad.T) + self.bd

        #TopK SAE (with ReLU)
        elif self.sae_type=='topk_relu':
            x = x-self.bd #pre-encoder bias
            x = torch.matmul(x, self.Ae.T)
            _, topk_indices = torch.topk(F.relu(x), self.kval_topk, dim=-1)
            mask = torch.zeros_like(x)
            mask.scatter_(-1, topk_indices, 1)
            xint = F.relu(x) * mask* lam
            if self.normalize_decoder: #unit norm
                eps = 1e-6
                Ad_unit = self.Ad / (eps + torch.linalg.norm(self.Ad, dim=0, keepdim=True))
                x = torch.matmul(xint, Ad_unit.T) + self.bd
            else:
                x = torch.matmul(xint, self.Ad.T) + self.bd

        #JumpReLU SAE
        elif self.sae_type=='jumprelu':
            x = x-self.bd #pre-encoder bias
            x = torch.matmul(x, self.Ae.T) + self.be
            x = F.relu(lam*x)
            threshold = torch.exp(self.logthreshold)
            xint = jumprelu(x, threshold, self.bandwidth)
            x = torch.matmul(xint, self.Ad.T) + self.bd
            if self.normalize_decoder: #unit norm
                eps = 1e-6
                Ad_unit = self.Ad / (eps+torch.linalg.norm(self.Ad, dim=0, keepdim=True))
                x = torch.matmul(xint, Ad_unit.T) + self.bd
            else:
                x = torch.matmul(xint, self.Ad.T) + self.bd

        #Sparsemax with linear transform (not used)
        elif self.sae_type=='sparsemax_lintx':
            x = x-self.bd #pre-encoder bias
            x = torch.matmul(x, self.Ae.T) + self.be
            sm = Sparsemax(dim=-1)
            xint = sm(lam*x)
            x = torch.matmul(xint, self.Ad.T) + self.bd

        #Sparsemax with distance transform (SpaDE)
        elif self.sae_type=='sparsemax_dist':
            x = -lam*torch.square(torch.norm(x.unsqueeze(1)-self.Ae.unsqueeze(0), dim=-1))
            sm = Sparsemax(dim=-1)
            xint = sm(x)
            x = torch.matmul(xint, self.Ad.T)

        else:
            raise ValueError('Invalid sae_type')
        
        if not return_hidden:
            return x
        else:
            return x, xint

