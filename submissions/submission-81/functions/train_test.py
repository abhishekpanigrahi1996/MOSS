import torch
import math
from inspect import signature
from functions.utils import step_fn
import torch.nn.functional as F

def train(dataloader, model, optimizer, update_status_fn=None, \
            regularizer=None, encoder_reg=False, gamma_reg=0.1, \
                return_concept_loss=False, num_concepts=4, clip_grad=False, deadcount_threshold=20):
    """ 
    Trains model on training inputs using mini-batches
    
    dataloader: torch.utils.data.DataLoader object
    model: torch.nn.Module object
    optimizer: torch.optim object
    update_status_fn: function to update status (save progress to status file)
    regularizer: one of (None, 'l1', 'l0', 'dist_weighted_l1')
        if None, no regularization
        if 'l1', L1 regularization on latent features
        if 'l0', L0 regularization (with differentiable step (straight-through est))
        if 'dist_weighted_l1', distance-weighted L1 regularization on latent features (from the KDS paper)
    encoder_reg: if True, use encoder weights for distance-weighted L1 regularization
        if False, use decoder weights
        default False
    gamma_reg: regularization weight
        default 1.0
    return_concept_loss: if True, return MSE grouped by concept instead of mean over mini-batches
        default False
    num_concepts: number of concepts in the dataset
        default 4
    clip_grad: if True, clip gradients to max norm of 1.0
        default False
    deadcount_threshold: number of batches for which a neuron must be dead to be considered dead (useful for auxloss in TopK SAE)
        default 20
    """
    size = len(dataloader.dataset)
    model.train()
    status_update_freq=100
    train_loss_trend = []
    loss_concept = torch.zeros((num_concepts,))
    numex_concept = torch.zeros((num_concepts,))
    device = next(model.parameters()).device

    for batch, W in enumerate(dataloader):
        if isinstance(W, list): #X,y (including labels- for concepts)
            X = W[0]
            y = W[1] #label for concepts within the data

        X = X.to(device)
        pred, xlatent = model(X, return_hidden=True)

        # L1 regularizer- used for ReLU SAE
        if regularizer=='l1':
            loss_reg = torch.mean(torch.sum(torch.abs(xlatent), dim=-1)) #sum feature, mean batch 
        
        # Distance-weighted L1 regularizer- used for SpaDE (sparsemax_dist)
        elif regularizer=='dist_weighted_l1':
            if encoder_reg: #distance based regularizer uses encoder weights
                dist_penalty_encoder = (X.unsqueeze(1)-model.Ae.unsqueeze(0)).pow(2).sum(dim=-1)
                loss_reg = (dist_penalty_encoder*xlatent).sum(dim=-1).mean()
            else: #use decoder weights in dist-based regularizer
                dist_penalty = (X.unsqueeze(1)-model.Ad.T.unsqueeze(0)).pow(2).sum(dim=-1)
                loss_reg = (dist_penalty*xlatent).sum(dim=-1).mean()
        
        #L0 regularizer- used for JumpReLU SAE
        elif regularizer=='l0':
            bandwidth = 1e-3
            loss_reg = torch.mean(torch.sum(step_fn(xlatent, torch.exp(model.logthreshold), bandwidth), dim=-1))
        
        #auxiliary loss used for TopK SAE- to reduce dead latents
        elif regularizer=='auxloss':
            err = X-pred #unexplained variance in data
            deadlatents = torch.all(torch.abs(xlatent)<1e-12, dim=0).to(device) #dead latent features in this batch
            if not hasattr(model, 'deadcounts'):
                raise ValueError("Model does not have deadcounts attribute")
            model.deadcounts[deadlatents] = model.deadcounts[deadlatents] + 1
            model.deadcounts[~deadlatents] = 0.0 #reset if a latent is not dead now
            deadlatents = (model.deadcounts>=deadcount_threshold).squeeze() #dead for atleast deadcount_threshold (#batches)
            xt = X-model.bd
            xt = torch.matmul(xt, model.Ae.T)
            xt = F.relu(model.lambda_val*xt)
            xt = xt*deadlatents.reshape(1,-1)
            if deadlatents.sum()<model.auxkval:
                auxkval = deadlatents.sum()
            else:
                auxkval = model.auxkval
            _, topk_indices = torch.topk(xt, auxkval, dim=-1)
            mask = torch.zeros_like(xt)
            mask.scatter_(-1, topk_indices, 1)
            xt = xt*mask
            if model.normalize_decoder:
                eps = 1e-6
                Ad_unit = model.Ad / (eps + torch.linalg.norm(model.Ad, dim=0, keepdim=True))
                xt = torch.matmul(xt, Ad_unit.T) + model.bd
            else:
                xt = torch.matmul(xt, model.Ad.T) + model.bd
            loss_reg = torch.mean(torch.sum(torch.pow(err-xt, 2), dim=-1))
        
        #no regularizer
        elif regularizer is None:
            loss_reg = torch.tensor([0.0], device=device)
        
        # Initialize storage for per-class loss
        loss_per_sample = torch.sum(torch.pow(pred-X, 2), dim=-1)
        loss_mse = torch.mean(loss_per_sample)
        loss = loss_mse + gamma_reg*loss_reg
        y_unique = torch.unique(y)
        for idx in range(len(y_unique)):
            yid = y_unique[idx]
            loss_concept[yid] = loss_concept[yid] + torch.sum(loss_per_sample[y==yid])
            numex_concept[yid] = numex_concept[yid] + torch.sum(y==yid) # #samples of yid in mini-batch
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            max_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step() #weights updated for a mini-batch
        train_loss_trend.append([loss_mse.item(), gamma_reg*loss_reg.item()])
        if batch % status_update_freq == 0:
            loss_mse_val, loss_reg_val = loss_mse.item(), gamma_reg*loss_reg.item()
            current = (batch + 1) * len(X)
            update_status_fn(f"loss: {loss_mse_val:>7f}, {loss_reg_val:>7f},  [{current:>5d}/{size:>5d}]")
    
    loss_concept_persample = torch.zeros((num_concepts,))
    for i in range(num_concepts):
        if numex_concept[i]>0:
            loss_concept_persample[i] = loss_concept[i]/numex_concept[i]
        else: #0 examples of this concept in this batch
            loss_concept_persample[i] = float('nan')
    
    if not return_concept_loss:
        return torch.mean(torch.tensor(train_loss_trend), dim=0) #returns [mse, reg] loss
    else:
        return *torch.mean(torch.tensor(train_loss_trend), dim=0), loss_concept_persample


def test(dataloader, model, update_status_fn=None, \
         regularizer=None, encoder_reg=False, gamma_reg=1.0, \
            return_concept_loss=False, num_concepts=4):
    """
    Evaluates model on test inputs using mini-batches
    Returns average test loss over mini-batches
    
    data loader: torch.utils.data.DataLoader object
    model: torch.nn.Module object
    loss_fn: loss function
    update_status_fn: function to update status (save progress to status file)
    return_accuracy: if True, return accuracy along with loss
    regularizer: one of (None, 'l1', 'dist_weighted_l1')
        if None, no regularization
        if 'l1', L1 regularization on latent features
        if 'dist_weighted_l1', distance-weighted L1 regularization on latent features (from the KDS paper)
    encoder_reg: if True, use encoder weights for distance-weighted L1 regularization
        if False, use decoder weights
        default False
    gamma_reg: regularization weight
        default 1.0
    return_concept_loss: if True, return MSE grouped by concept instead of mean over mini-batches
        default False
    num_concepts: number of concepts in the dataset
        default 4
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    test_loss_mse, test_loss_reg = 0,0
    loss_concept = torch.zeros((num_concepts,))
    numex_concept = torch.zeros((num_concepts,))
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for W in dataloader:
            if isinstance(W, list): #X,y (including labels)
                X = W[0] #use only X (not labels y)
                y = W[1]
            X = X.to(device)
            pred, xlatent = model(X, return_hidden=True)
            
            # L1 regularizer- used for ReLU SAE
            if regularizer=='l1':
                loss_reg = torch.mean(torch.sum(torch.abs(xlatent), dim=-1)) #sum along feature, mean along batch dim
            
            # Distance-weighted L1 regularizer- used for SpaDE (sparsemax_dist)
            elif regularizer=='dist_weighted_l1':
                if encoder_reg: #distance based regularizer uses encoder weights
                    dist_penalty_encoder = (X.unsqueeze(1)-model.Ae.unsqueeze(0)).pow(2).sum(dim=-1)
                    loss_reg = (dist_penalty_encoder*xlatent).sum(dim=-1).mean()
                else: #use decoder weights in dist-based regularizer
                    dist_penalty = (X.unsqueeze(1)-model.Ad.T.unsqueeze(0)).pow(2).sum(dim=-1)
                    loss_reg = (dist_penalty*xlatent).sum(dim=-1).mean() 
            
            # L0 regularizer- used for JumpReLU SAE
            elif regularizer=='l0':
                bandwidth = 1e-3
                loss_reg = torch.mean(torch.sum(step_fn(xlatent, torch.exp(model.logthreshold), bandwidth), dim=-1))     
            
            # no regularizer
            elif regularizer is None:
                loss_reg = torch.tensor([0.0], device=device)                  
            
            # Initialize storage for per-class loss
            loss_per_sample = torch.sum(torch.pow(pred-X, 2), dim=-1)
            loss_mse = torch.mean(loss_per_sample)
            y_unique = torch.unique(y)
            for id in range(len(y_unique)):
                yid = y_unique[id]
                loss_concept[yid] = loss_concept[yid] + torch.sum(loss_per_sample[y==yid])
                numex_concept[yid] = numex_concept[yid] + torch.sum(y==yid) # number of examples of yid in mini-batch
            loss = loss_mse + gamma_reg*loss_reg
            test_loss += loss
            test_loss_mse += loss_mse
            test_loss_reg += gamma_reg*loss_reg
    
    test_loss/= num_batches
    test_loss = test_loss.item()
    test_loss_mse/= num_batches
    test_loss_reg/= num_batches
    
    if update_status_fn is not None:
        update_status_fn(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    loss_concept_persample = torch.zeros((num_concepts,))
    
    for i in range(num_concepts):
        if numex_concept[i]>0:
            loss_concept_persample[i] = loss_concept[i]/numex_concept[i]
        else: #0 examples of this concept in this batch
            loss_concept_persample[i] = float('nan')
    
    if not return_concept_loss:
        return test_loss_mse, test_loss_reg
    else:
        return test_loss_mse, test_loss_reg, loss_concept_persample