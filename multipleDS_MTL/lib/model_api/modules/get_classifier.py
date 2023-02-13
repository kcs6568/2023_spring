from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClfHead(nn.Module):
    def __init__(self,
                 backbone_type,
                 in_channel,
                 num_classes,
                 middle_channle=None,
                 use_avgpool=True) -> None:
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1,1)) if use_avgpool else None
        
        if 'resnet' in backbone_type or 'resnext' in backbone_type:
            self.classifier = nn.Linear(in_channel, num_classes)
            
        elif 'mobilenet' in backbone_type:
            self.classifier = nn.Sequential(
                nn.Linear(in_channel, middle_channle),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(middle_channle, num_classes),
            )
        
        else:
            raise ValueError
        
        self.criterion = nn.CrossEntropyLoss()

        
    def forward(self, feats, targets=None):
        # assert isinstance(feats, OrderedDict) or isinstance(feats, dict)
        
        if isinstance(feats, OrderedDict) or isinstance(feats, dict):
            _, x = feats.popitem()
        else:
            x = feats

        if self.avg:
            x = self.avg(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        
        if self.training:
            assert targets is not None
            losses = self.criterion(out, targets)
            losses = dict(clf_loss=losses)
            
            return losses
        
        else:
            return out
    
    
class ClfStem(nn.Module):
    def __init__(self,
                 sample_dimension=3,
                 out_channels=64,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 maxpool=None,
                 relu=None) -> None:
        super().__init__()
        self.conv = nn.Conv2d(sample_dimension, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if relu == 'hardswish':
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
        
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = maxpool
        
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        
        if self.maxpool:
            x = self.maxpool(x)
            
        return x
    
    
def build_classifier(backbone_type, num_classes, head_cfg):
    return ClfHead(backbone_type, num_classes=num_classes, **head_cfg)
    
    