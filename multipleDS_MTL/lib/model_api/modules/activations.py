import torch
import torch.nn as nn

def set_activation_function(activation_type):
    assert activation_type is not None
    if activation_type == 'relu':
        return nn.ReLU(inplace=True)
    elif activation_type == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    elif activation_type == 'prelu':
        return nn.PReLU(device='cuda')
    elif activation_type == 'relu6':
        return nn.ReLU6(inplace=True)
    elif activation_type == 'tanh':
        return nn.Tanh()
    elif activation_type == 'gelu':
        return nn.GELU()
    
    