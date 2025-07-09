from functools import partial
from typing import Optional
from jax import Array
import jax
import jax.numpy as jnp
import jax.nn as jnn
import equinox as eqx

from metrics import ce, mse

class cqk_cpv_fit(eqx.Module):
    """Fitting the optimal cqk and cpv
    Note: Currently works only for classification"""
    
    cqk: float
    cpv: float
    loss: Optional[float]
    losses_matrix: Optional[Array]
    
    @staticmethod
    def get_loss(X, y, cqk, cpv, dots):
        attn = jnn.softmax(cqk / jnp.sqrt(len(X[-1])+len(y[-1])) * dots) # (num_samples,) CHECK IF THIS IS STABLE
        vals = cpv * y[:-1]
        pred = jnp.sum(attn[:, None] * vals, axis=0)
        return ce(pred[None, :], y[-1:])
    
    @staticmethod
    def fwd_fn(model, x, y, **kwargs):
        return {'out': model(examples=x, labels=y)}
    
    @staticmethod
    def find_cqk_cpv(cqk, cpv, eval_data, cqk_low, cqk_high, cpv_low, cpv_high, num):
        if cqk is None: cqk_scan_range = jnp.logspace(cqk_low, cqk_high, num=num)
        else: cqk_scan_range = [cqk]
        
        if cpv is None: cpv_scan_range = jnp.logspace(cpv_low, cpv_high, num=num)
        else: cpv_scan_range = [cpv]
        
        if (cqk is None) or (cpv is None):
            print("Doing search for optimal cqk and cpv")
            assert eval_data is not None, "Need to provide data for the grid search"
            
            X = eval_data['examples']
            xq = X[:, -1]
            dots = jnp.einsum('Nnd, Nd->Nn', X[:,:-1], xq)
            def eval_cqk_cpv(cqk, cpv):
                return  jnp.mean(jax.vmap(partial(cqk_cpv_fit.get_loss, cqk=cqk, cpv=cpv))(X=X, y=eval_data['labels'], dots=dots))
        
            losses_matrix = []
            
            for cqk in cqk_scan_range:
                losses_matrix.append([])
                for cpv in cpv_scan_range:
                    losses_matrix[-1].append(eval_cqk_cpv(cqk, cpv))
            losses_matrix = jnp.array(losses_matrix)
            i, j = jnp.unravel_index(jnp.argmin(losses_matrix), losses_matrix.shape)
            opt_cqk, opt_cpv = cqk_scan_range[i], cpv_scan_range[j]
            print(f"Found optimal cqk: {opt_cqk}, and optimal cpv: {opt_cpv}")
            return opt_cqk, opt_cpv, losses_matrix[i, j], losses_matrix
            
        else: return cqk, cpv, None, None
    
    def __init__(self, cqk=None, cpv=None, eval_data=None, cqk_low=-1, cqk_high=2, cpv_low=-1, cpv_high=2, num=100):
        super().__init__()
        
        self.cqk, self.cpv, self.loss, self.losses_matrix = cqk_cpv_fit.find_cqk_cpv(cqk, cpv, eval_data, cqk_low, cqk_high, cpv_low, cpv_high, num)
    
    def __call__(self, examples: Array, labels: Array, **kwargs):
        xq = examples[-1]
        attn = jnn.softmax(self.cqk / jnp.sqrt(len(xq)+len(labels[-1])) * jnp.einsum('nd, d->n', examples[:-1], xq)) # (num_samples,) CHECK IF THIS IS STABLE
        vals = self.cpv * labels[:-1]
        pred = jnp.sum(attn[:, None] * vals, axis=0)
        return pred[None, :]
        
        
class gd(eqx.Module):
    """One step of gradient descent
    Works only for both regression and classification"""
    
    lr: float
    loss: Optional[float]
    losses_matrix: Optional[Array]
    classification: bool
    
    @staticmethod
    def get_loss(X, y, lr, classification):
        loss_fn = ce if classification else mse
        xq = X[-1:, :]
        def loss(w_, x_, y_):
            y_pred = jnp.einsum('sd,dc->sc', x_[:-1, :], w_)
            return jnp.mean(loss_fn(y_pred, y_[:-1, :]))
        w0 = jnp.zeros((X.shape[-1], y.shape[-1]))
        w1 = w0 - lr * jax.grad(loss, argnums=0)(w0, X, y)
        pred_y = jnp.einsum('bd,dc->bc', xq, w1)
        y = y[-1:, :]
        return loss_fn(pred_y, y)
    
    @staticmethod
    def fwd_fn(model, x, y, **kwargs):
        return {'out': model(examples=x, labels=y)}
    
    @staticmethod
    def find_lr(lr, eval_data, classification):
        if lr is None: 
            lr_scan_range = jnp.logspace(0, 2.5, num=100)
            print("Doing search for optimal learning rate for gradient descent")
            assert eval_data is not None, "Need to provide data for the grid search"
            
            def eval_lr(lr):
                return  jnp.mean(jax.vmap(partial(gd.get_loss, lr=lr, classification=classification))(X=eval_data['examples'], y=eval_data['labels']))
        
            losses_matrix = []
            for lr in lr_scan_range:
                losses_matrix.append(eval_lr(lr))
            losses_matrix = jnp.array(losses_matrix)
            i = jnp.argmin(losses_matrix)
            opt_lr = lr_scan_range[i]
            print(f"Found optimal learning rate: {opt_lr}")
            return opt_lr, losses_matrix[i], losses_matrix
            
        else: return lr, None, None
    
    def __init__(self, lr=None, eval_data=None, classification=True):
        super().__init__()
        
        self.classification = classification
        self.lr, self.loss, self.losses_matrix = gd.find_lr(lr, eval_data, classification)
    
    def __call__(self, examples: Array, labels: Array, **kwargs):
        loss_fn = ce if self.classification else mse
        xq = examples[-1:, :]
        def loss(w_, x_, y_):
            y_pred = jnp.einsum('sd,dc->sc', x_[:-1, :], w_)
            return jnp.mean(loss_fn(y_pred, y_[:-1, :]))
        w0 = jnp.zeros((examples.shape[-1], labels.shape[-1]))
        w1 = w0 - self.lr * jax.grad(loss, argnums=0)(w0, examples, labels)
        pred = jnp.einsum('d,dc->c', xq[0], w1)
        return pred[None, :]