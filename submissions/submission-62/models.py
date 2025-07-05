import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
# from utils import *
import numpy as np
import math


def decay_mask(seq_length, gamma):
    # Create a tensor with the powers of gamma
    powers = torch.arange(seq_length).unsqueeze(1) - torch.arange(seq_length).unsqueeze(0)
    # Create the mask using the condition
    mask = torch.where(powers >= 0, gamma ** powers.float(), torch.zeros_like(powers, dtype=torch.float))
    
    return mask



class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, mask_type='causal', gamma=1, bias=False):
        super().__init__()
        self.gamma = gamma
        self.out_channels = out_channels
        
       

       
        self.query_projection = nn.Linear(in_channels, out_channels, bias=bias)
        self.key_projection = nn.Linear(in_channels, out_channels, bias=bias)
        
        self.value_projection = nn.Linear(in_channels, out_channels, bias=bias)
        self.mask_type = mask_type

    def forward(self, x):
        batch_size, seq_length, hidden_size = x.size()

        # Project inputs to query, key, and value
        
        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)


        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.out_channels)
        

        if self.mask_type == 'causal':
            mask = torch.triu(torch.ones((1, seq_length, seq_length)).to(x.device)).transpose(-2, -1).reshape(1, seq_length, seq_length)
            mask = mask.repeat(batch_size, 1, 1)
            mask = torch.log(mask)
            scores = scores + mask


        elif self.mask_type == 'decay':  
            mask = decay_mask(seq_length=seq_length, gamma=self.gamma).to(x.device)
            mask = mask.repeat(batch_size, 1, 1)
            mask = torch.log(mask)
            scores = scores + mask


        

        # Apply softmax to get attention probabilities
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention to the values
        attention_output = torch.matmul(attention_weights, value)

        return attention_output

class SAN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mask_type='causal', gamma=1,
                       hidden_channels=128, num_attn_layers=2, num_mlp_layers=3, bias=False):
        super().__init__()
        self.attn_layers = nn.ModuleList()
        self.mlp_layers = nn.ModuleList()
        self.num_attn_layers = num_attn_layers
        self.num_mlp_layers = num_mlp_layers


        self.attn_layers.append(SelfAttention(in_channels=in_channels,
                                              out_channels=hidden_channels, 
                                              bias=bias, 
                                              mask_type=mask_type,
                                              gamma=gamma
                                              ))

        for _ in range(self.num_attn_layers-1):
            self.attn_layers.append(SelfAttention(in_channels=hidden_channels,
                                                  out_channels=hidden_channels,
                                                  bias=bias, 
                                                  mask_type=mask_type,
                                                  gamma=gamma
                                                  ))
        for _ in range(self.num_mlp_layers):
            self.mlp_layers.append(nn.Linear(hidden_channels, hidden_channels))
        self.lin = torch.nn.Linear(hidden_channels, out_channels)


    def forward(self, x):
        for layer in self.attn_layers:
            x = layer(x)

        # take the query vector as the input to the mlp:
        x = x[:,-1,:]

        for layer in self.mlp_layers:
            x = layer(x)
            x = F.relu(x)

        x = self.lin(x)

        return x