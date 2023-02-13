from collections import OrderedDict

import os
import torch
import torch.nn as nn
import glob

import torchvision
from torchvision.models.resnet import resnet50

model = resnet50(pretrained=True)
params = torch.load('/root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth')

new_layer = OrderedDict()
new_stem = OrderedDict() 
for k, v in params.items():
    if 'layer' in k:
        new_layer[k] = v
        
    else:
        if not 'fc' in k:
            k_ = k.replace('1', '')
            new_stem[k_] = v        

torch.save(new_layer, '/root/volume/pre_weights/resnet50_IM1K_only_layer.pth')
torch.save(new_stem, '/root/volume/pre_weights/resnet50_IM1K_only_stem.pth')
            