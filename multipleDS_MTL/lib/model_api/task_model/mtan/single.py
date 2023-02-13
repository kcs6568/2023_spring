from dataclasses import replace
import numpy as np
from copy import deepcopy
from scipy.special import softmax
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules.get_detector import build_detector, DetStem
from ...modules.get_backbone import build_backbone
from ...modules.get_segmentor import build_segmentor, SegStem
from ...modules.get_classifier import build_classifier, ClfStem
from ...backbones.resnet import IdentityBottleneck, conv1x1, Bottleneck
from ....apis.warmup import set_decay_fucntion


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)


def make_stem(task, cfg, backbone=None, stem_weight=None):
    if task == 'clf':
        stem = ClfStem(**cfg['stem'])
        stem.apply(init_weights)
    
    elif task == 'det':
        stem = DetStem(**cfg['stem'])
        if stem_weight is not None:
            ckpt = torch.load(stem_weight)
            stem.load_state_dict(ckpt, strict=False)
        
    elif task == 'seg':
        stem = SegStem(**cfg['stem'])
        if stem_weight is not None:
            ckpt = torch.load(stem_weight)
            stem.load_state_dict(ckpt, strict=False)
    
    return stem
    
    
def make_head(task, backbone, dense_task, num_classes, fpn_channel=256, head_cfg=None):
    if task == 'clf':
        head = build_classifier(
            backbone, num_classes, head_cfg)
    
    elif task == 'det':
        head = build_detector(
            backbone,
            dense_task, 
            fpn_channel, 
            num_classes)
    
    elif task == 'seg':
        head = build_segmentor(
            dense_task, num_classes, cfg_dict=head_cfg)
    
    head.apply(init_weights)
    return head


class MTANSingle(nn.Module):
    def __init__(self,
                 backbone,
                 detector,
                 segmentor,
                 task_cfg,
                 **kwargs
                 ) -> None:
        super().__init__()
        backbone_network = build_backbone(
            backbone, detector, segmentor, kwargs)
        
        self.blocks = []
        self.ds = []
        self.num_per_block = []
        self.channel_per_block = []
        for _, p in backbone_network.body.named_children():
            block = []
            self.num_per_block.append(len(p))
            for m, q in p.named_children():
                if m == '0':
                    self.ds.append(q.downsample)
                    q.downsample = None
                    
                block.append(q)
                
            self.blocks.append(nn.ModuleList(block))
        
        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fpn = backbone_network.fpn
        
        self.dset = list(task_cfg.keys())[0]
        self.task = task_cfg[self.dset]['task']
        self.return_layers = task_cfg[self.dset]['return_layers']
        stem_weight = kwargs['state_dict']['stem']
        self.stem = make_stem(self.task, task_cfg[self.dset], stem_weight)
        
        dense_task = detector if detector else segmentor
        self.head = make_head(self.task, backbone, dense_task,
                              task_cfg[self.dset]['num_classes'],
                              head_cfg=task_cfg[self.dset]['head'])
        
        attention_ch = kwargs['attention_channels']
        
        self.dilation_count = 2
        self.is_downsample_for_next_feat = kwargs['is_downsample_for_next_feat']
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.att_encoder1 = self.att_layer(    attention_ch[0], attention_ch[0]//4, attention_ch[0])
        self.att_encoder2 = self.att_layer(2 * attention_ch[1], attention_ch[1]//4, attention_ch[1])
        self.att_encoder3 = self.att_layer(2 * attention_ch[2], attention_ch[2]//4, attention_ch[2])
        self.att_encoder4 = self.att_layer(2 * attention_ch[3], attention_ch[3]//4, attention_ch[3])
        
        self.is_global_attetion = kwargs['is_global_attetion']
        self.global_attention = nn.ModuleList([
            self.global_att_layer(attention_ch[idx], attention_ch[idx+1] // 4) for idx in range(len(attention_ch)-1)])
        
    
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel))
            # nn.Sigmoid())
           
            
    def global_att_layer(self, in_channel, out_channel):
        channel_downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                nn.BatchNorm2d(4 * out_channel))
        return IdentityBottleneck(in_channel, out_channel, downsample=channel_downsample)
            
    
    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        # for dset, (images, _) in data_dict.items():
        for dset, images in data_dict.items():    
            stem_feats.update({dset: self.stem_dict[dset](images)})
        return stem_feats
    
    
    def get_features(self, images):
        backbone_feats = OrderedDict()
        
        att_feat = None
        att_mask = None
        
        last_feat = None
        last_sec_feat = None
        
        feat = self.stem(images)
        # print("2")
        
        block_count = 0
        for layer_idx, num_blocks in enumerate(self.num_per_block):
            for block_idx in range(num_blocks):
                # if leaky relu function is not in-place, process will be stucked in specific one gpu randomly
                identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                feat = F.leaky_relu_(self.blocks[layer_idx][block_idx](feat) + identity) 
                
                block_count += 1
                
                if block_idx == num_blocks - 2:
                    last_sec_feat = feat
                
                elif block_idx == num_blocks - 1:
                    last_feat = feat
            
            print(layer_idx, block_idx, block_count, last_sec_feat.size(), last_feat.size())
            att_feat = last_sec_feat if layer_idx == 0 else torch.cat((last_sec_feat, att_mask), dim=1)
            att_feat = getattr(self, f"att_encoder{layer_idx+1}")(att_feat).sigmoid_()
            
            att_mask = att_feat * last_feat
            
            print(att_mask.size())
            
            if block_idx == (num_blocks - 1):
                if str(layer_idx) in self.return_layers:
                    backbone_feats.update({str(layer_idx): att_mask})
                    
            if self.is_global_attetion[layer_idx]:
                att_mask = self.global_attention[layer_idx](att_mask)
            
            if self.is_downsample_for_next_feat[layer_idx]:
                att_mask = self.down_sampling(att_mask) 
            
        return backbone_feats
    
    
    def forward_train(self, images, targets, backbone_features, other_hyp):
        total_losses = OrderedDict()
        
        if self.task == 'clf':
            losses = self.head(backbone_features, targets)
            
        elif self.task == 'det':
            fpn_feat = self.fpn(backbone_features)
            losses = self.head(images, fpn_feat,
                                    self.stem.transform, 
                                origin_targets=targets)
            
        elif self.task == 'seg':
            losses = self.head(
                backbone_features, targets, input_shape=targets.shape[-2:])
        
        print(losses)
        
        losses = {f"feat_{self.dset}_{k}": l for k, l in losses.items()}
        total_losses.update(losses)
        
        return total_losses
    
    
    def forward_val(self, images, backbone_features, other_hyp):
        dset = list(other_hyp["task_list"].keys())[0]
        task = list(other_hyp["task_list"].values())[0]
        head = self.head_dict[dset]
        
        back_feats = backbone_features[dset]
        
        if task == 'det':
            fpn_feat = self.fpn(back_feats)
            predictions = head(images, fpn_feat, self.stem_dict[dset].transform)
            
        else:
            if task == 'seg':
                predictions = head(
                    back_feats, input_shape=images.shape[-2:])
        
            else:
                predictions = head(back_feats)
            
            predictions = dict(outputs=predictions)
        
        return predictions
    
    
    def forward(self, data_dict, hyp):
        images = data_dict[self.dset][0]
        targets = deepcopy(data_dict[self.dset][1])
        shared_features = self.get_features(images)
        
        if self.training:
            return self.forward_train(images, targets, shared_features, hyp)
        else:
            return self.forward_val(images, targets, shared_features, hyp)
        
        # if self.training:
        #     return self.forward_train(data_dict, shared_features, hyp)
        # else:
        #     return self.forward_val(data_dict, shared_features, hyp)