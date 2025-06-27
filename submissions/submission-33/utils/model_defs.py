import torch
from torch import nn
from torch.nn.functional import relu
from torch import sin, tanh


class Net(nn.Module):
    
    def __init__(self, layer_widths):
        super(Net, self).__init__()
        ## Feature-sizes
        self.layer_widths = layer_widths
        ## Fully-connected layers
        self.fcs = nn.ModuleList([nn.Linear(in_features, out_features)
            for in_features, out_features in zip(self.layer_widths[:-1],self.layer_widths[1:])])
        ## Depth
        self.D = len(self.fcs)
        
    def forward(self, x):
        x_ = self.fcs[0](x)
        for i in range(2,self.D+1):
            x_ = self.fcs[i-1](sin(x_))
        return x_
    
    
class ConditionalNet(nn.Module):
    
    def __init__(self, layer_widths):
        super(ConditionalNet, self).__init__()
        ## Feature-sizes
        self.layer_widths = layer_widths
        ## Fully-connected layers
        self.fcs = nn.ModuleList([nn.Linear(in_features, out_features)
            for in_features, out_features in zip(self.layer_widths[:-1],self.layer_widths[1:])])
        ## Depth
        self.D = len(self.fcs)
        
    def forward(self, x, z):
        x_ = self.fcs[0](torch.hstack([x, z]))
        for i in range(2,self.D+1):
            x_ = self.fcs[i-1](sin(x_))
        return x_