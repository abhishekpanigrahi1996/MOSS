import torch
import numpy as np
import random
import argparse
from datetime import date
import torch.nn as nn
from matplotlib.colors import Normalize, hsv_to_rgb
import math
import csv
import os

def seed_worker(worker_id):
        """
        Utility function for reproducibility in Dataloader behavior
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        

def resample_deadlatents(model, dataloader, num_batches=15, threshold = 0.998, noise_label=3):
    """
    model: (torch.nn.Module) model whose dead neurons are to be resampled
    dataloader: (torch.utils.data.DataLoader) dataloader for the dataset
    num_batches: (int) number of mini-batches to use for resampling
    threshold: (float) fraction of datapoints for which a neuron must be silent to become 'dead'
    noise_label: (int) label of noise points in the dataset: to 
        avoid picking noise points while resampling
    """
    big_batch_data = []
    big_batch_labels = []
    device = next(model.parameters()).device
    for i, (data, labels) in enumerate(dataloader):
        if i >= num_batches:  # 15 batches
            break
        big_batch_data.append(data)
        big_batch_labels.append(labels)
    big_batch_data = torch.cat(big_batch_data, dim=0)
    big_batch_labels = torch.cat(big_batch_labels, dim=0)

    big_batch_data = big_batch_data.to(device)
    big_batch_labels = big_batch_labels.to(device)
    pred, latents = model(big_batch_data,return_hidden=True)
    #identify dead neurons from latents: those that have small latents for 99.8% of data
    dead_neurons = (latents.abs() < 1e-5).float().mean(dim=0) > threshold
    numdead = torch.sum(dead_neurons)
    # print(f"Resampling {numdead} dead neurons at t={t+1} epochs")
    #compute loss of model on different examples
    mseloss_per_ex = torch.sum(torch.pow(pred-big_batch_data, 2), dim=-1)
    #get top numdead data points with highest mse; ensure that you pick data points and not noise (outlier) points
    _, top_indices = torch.topk(mseloss_per_ex*((big_batch_labels!=noise_label).squeeze()), numdead) 
    #avoid noise (label=4) points by making their mse 0 for getting top_indices
    #reinit dead neurons' encoder weights with poorly explained examples
    if numdead>0:
        dead_neuron_indices = torch.nonzero(dead_neurons).squeeze()
        if dead_neuron_indices.dim()==0:
            dead_neuron_indices = dead_neuron_indices.unsqueeze(0)
        for i in range(len(dead_neuron_indices)):
            dead_i = dead_neuron_indices[i]
            model.Ae[dead_i].data.copy_(big_batch_data[top_indices[i],:].view(-1)) #encoder weight
            model.Ad[:,dead_i].data.copy_(big_batch_data[top_indices[i],:].view(-1)) #decoder weight



def linear_pieces(model, data, receptive_fields=False, activation_heatmaps = False):
    """
    get linear pieces of a piecewise linear model with different colors per piece
    also returns receptive fields and activation heatmaps if receptive_fields and activation_heatmaps are True resp.
    model: (torch.nn.Module) model to get linear pieces from
    data: (torch.Tensor) data- a meshgrid- to use for getting linear pieces
    """
    numneuro = model.width
    np.random.seed(1)
    hues = np.linspace(0, 1, numneuro, endpoint=False)  # Evenly spaced hues
    hues = np.random.permutation(hues)  # Randomize order
    saturation = 0.9  # High saturation for vibrant colors
    value = 0.9  # High brightness
    neuron_colors = hsv_to_rgb(np.array([[h, saturation, value] for h in hues]))  # Convert to RGB
    _, hidden = model(data, return_hidden=True)
    Mact = (hidden>0).float()
    dims = math.ceil(math.sqrt(data.shape[0]))
    linpieces = np.dot(Mact.numpy(), neuron_colors)
    norm = Normalize()
    linpieces = norm(linpieces)
    linpieces = linpieces.reshape(dims, dims, 3).transpose(1, 0, 2)

    if receptive_fields:
        Rfall = []
        norm = Normalize()
        for k in range(model.width):
            Rf = torch.zeros_like(Mact)
            Rf[:, k] = Mact[:, k]
            Rf = np.dot(Rf.numpy(), neuron_colors)            
            Rf = norm(Rf)
            Rf = Rf.reshape(dims, dims, 3).transpose(1, 0, 2)
            Rfall.append(Rf)
        
    if activation_heatmaps:
        Heatall = []
        for k in range(model.width):
            heat = torch.zeros_like(hidden)
            heat[:, k] = hidden[:,k] #use activation, not just active/inactive
            heat = np.dot(heat.detach().numpy(), neuron_colors)
            heat = norm(heat)
            heat = heat.reshape(dims, dims, 3).transpose(1, 0, 2)
            Heatall.append(heat)
    results = [linpieces]
    if receptive_fields:
        results.append(Rfall)
    if activation_heatmaps:
        results.append(Heatall)
    return tuple(results)
    # if not receptive_fields:
    #     return linpieces
    # else:
    #     return linpieces, Rfall


def softplus_inverse(input, beta=1.0, threshold=20.0):
        """"
        inverse of the softplus function in torch
        """
        if isinstance(input, float):
                input = torch.tensor([input])
        if input*beta<threshold:
                return (1/beta)*torch.log(torch.exp(beta*input)-1.0)
        else:
              return input[0]


def read_hyperparameters(task_id, file_path):
    """
    Reads the hyperparameters from a CSV file for the given task ID.
    """
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if i == task_id:
                return row
    return None

def custom_type(input_string): #for lambda input
    # Assume the input format is "str, float"
    parts = input_string.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Input must be in the format 'str,float'")
    mode = parts[0]
    value = float(parts[1])
    
    return (mode, value)

# Define a simple class to hold arguments as attributes
import ast

# Define a simple class to hold arguments as attributes
class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def load_args_from_file(file_path):
    args_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Strip whitespace and ignore empty lines or lines without '='
            line = line.strip()
            if not line or '=' not in line:
                continue  # Skip this line

            # Split the line into key and value
            key, value = line.split('=', 1)  # Limit the split to 1 split in case value has '=' in it

            # Safely evaluate the value to determine its correct data type
            try:
                args_dict[key.strip()] = ast.literal_eval(value.strip())
            except (ValueError, SyntaxError):
                # If evaluation fails, keep the value as a string
                args_dict[key.strip()] = value.strip()
    
    return Args(**args_dict)

def stable_rank(A):
    """
    Computes the stable rank of a matrix A.
    """
    return  torch.linalg.matrix_norm(A, ord='fro')**2 / torch.linalg.matrix_norm(A, ord=2)**2

#get exptnames from folder names within saved_models
def get_exptnames():
    exptnames = []
    for root, dirs, files in os.walk('saved_models'):
        for dir in dirs:
            if 'checkpoints_' in dir:
                exptnames.append(dir.split('checkpoints_')[1])
    #sort exptnames to have all with same sae_type (name after the first _ within exptnames) together; after saename, the float following must be used for sorting
    order = {'relu': 0, 'jumprelu': 1, 'spade': 3, 'topk': 2}
    exptnames = sorted(exptnames, key=lambda x: (order[x.split('_')[1]]))
    return exptnames

#get sae_types from expt_names: it will be the string after first _, except for spade, which should be repaced by sparsemax_dist
def get_sae_types(exptnames):
    sae_types = []
    for exptname in exptnames:
        if 'spade' in exptname:
            sae_types.append('sparsemax_dist')
        elif 'topk_relu' in exptname:
            sae_types.append('topk_relu')
        else:
            sae_types.append(exptname.split('_')[1])
    return sae_types


# functions for the jumprelu nonlinearity (adapted from Rajamanoharan et al, "Jumping Ahead ..." (2024))

def rectangle(x):
    # rectangle function
    return ((x >= -0.5) & (x <= 0.5)).float()

class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold, bandwidth):
        if not isinstance(threshold, torch.Tensor):
            threshold = torch.tensor(threshold, dtype=input.dtype, device=input.device)
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(bandwidth, dtype=input.dtype, device=input.device)
        ctx.save_for_backward(input, threshold, bandwidth)
        return (input > threshold).type(input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        grad_input = 0.0*grad_output #no ste to input
        grad_threshold = (
            -(1.0 / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)
        return grad_input, grad_threshold, None  # None for bandwidth since const

def step_fn(input, threshold, bandwidth):
    return StepFunction.apply(input, threshold, bandwidth)

class JumpReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        if not isinstance(bandwidth, torch.Tensor):
            bandwidth = torch.tensor(bandwidth, dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, threshold, bandwidth)
        return x*(x>threshold)

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth = ctx.saved_tensors
        # Compute gradients
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * rectangle((x - threshold) / bandwidth)
            * grad_output
        ).sum(dim=0, keepdim=True)  # Aggregating across batch dimension
        return x_grad, threshold_grad, None  # None for bandwidth since const

def jumprelu(x, threshold, bandwidth):
    return JumpReLU.apply(x, threshold, bandwidth)
