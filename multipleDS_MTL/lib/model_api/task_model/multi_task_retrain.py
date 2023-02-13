from dataclasses import replace
from turtle import forward
import numpy as np
from scipy.special import softmax
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..modules.get_detector import build_detector, DetStem
from ..modules.get_backbone import build_backbone
from ..modules.get_segmentor import build_segmentor, SegStem
from ..modules.get_classifier import build_classifier, ClfStem
from ...apis.loss_lib import AutomaticWeightedLoss
from ...apis.loss_lib import shared_gate_loss, disjointed_gate_loss, gate_similarity_loss, non_shared_gate_loss
from ...apis.warmup import PolynomialDecay, ExponentialDecay, set_decay_fucntion


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


class AIGGate(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel//4, 1)
        self.bn = nn.BatchNorm2d(in_channel//4)
        self.conv2 = nn.Conv2d(in_channel//4, 2, 1)
        
    def forward(self, x):
        x = F.avg_pool2d(x, x.shape[2:])
        x = F.leaky_relu(self.bn(self.conv1(x)))
        x = self.conv2(x)
        
        return x


class RetrainMTL(nn.Module):
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
                # self.channel_per_block.append(q.out_channels)
                
            self.blocks.append(nn.ModuleList(block))
        self.current_iter = 0
        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fpn = backbone_network.fpn
        
        self.stem_dict = nn.ModuleDict()
        self.head_dict = nn.ModuleDict()
        
        self.return_layers = {}
        data_list = []
        
        stem_weight = kwargs['state_dict']['stem']
        for data, cfg in task_cfg.items():
            data_list.append(data)
            self.return_layers.update({data: cfg['return_layers']})
            
            task = cfg['task']
            num_classes = cfg['num_classes']
            if task == 'clf':
                stem = ClfStem(**cfg['stem'])
                head = build_classifier(
                    backbone, num_classes, cfg['head'])
                stem.apply(init_weights)
                
            elif task == 'det':
                stem = DetStem(**cfg['stem'])
                
                head_kwargs = {'num_anchors': len(backbone_network.body.return_layers)+1}
                head = build_detector(
                    backbone, detector, 
                    backbone_network.fpn_out_channels, num_classes, **head_kwargs)
                if stem_weight is not None:
                    ckpt = torch.load(stem_weight)
                    stem.load_state_dict(ckpt, strict=False)
                    print("!!!Load weights for detection stem layer!!!")
            
            elif task == 'seg':
                stem = SegStem(**cfg['stem'])
                head = build_segmentor(segmentor, num_classes=num_classes, cfg_dict=cfg['head'])
                if stem_weight is not None:
                    ckpt = torch.load(stem_weight)
                    stem.load_state_dict(ckpt, strict=False)
                    print("!!!Load weights for segmentation stem layer!!!")
            
            head.apply(init_weights)
            self.stem_dict.update({data: stem})
            self.head_dict.update({data: head})
        
        self.make_gate_logits(data_list, len(task_cfg))    
        # self.policys = {dset: torch.zeros(gate.size()).float() for dset, gate in self.task_gating_params.items()}
        
    
    def fix_gate(self, policy):
        fixed = torch.ones(self.num_fixed_gate, 2).cuda()
        fixed[:, 1] = 0
        policy = torch.cat((fixed.float(), policy[self.num_fixed_gate:].float()), dim=0)
        
        return policy
        
    
    def make_gate_logits(self, data_list, num_task):
        logit_dict = {}
        for t_id in range(num_task):
            task_logits = Variable((0.5 * torch.ones(sum(self.num_per_block), 2)), requires_grad=True)
            # task_logits = 0.5 * torch.ones(sum(self.num_per_block), 2)
            # task_logits = nn.init.kaiming_uniform_(task_logits)
            
            logit_dict.update(
                {data_list[t_id]: nn.Parameter(task_logits, requires_grad=False)})
            
        self.task_gating_params = nn.ParameterDict(logit_dict)
        
    
    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        for dset, (images, _) in data_dict.items():
            stem_feats.update({dset: self.stem_dict[dset](images)})
            # exit()
            
        return stem_feats
    
    
    def forward_train(self, data_dict, other_hyp):
        total_losses = OrderedDict()
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        dset_list = list(other_hyp["task_list"].keys())
        
        data = self._extract_stem_feats(data_dict)
        # self.policys = self.task_gating_params

        for dset, feat in data.items():
            block_count = 0
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    if self.task_gating_params[dset][block_count, 0] == 1:
                        identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                        feat = F.leaky_relu(self.blocks[layer_idx][block_idx](feat) + identity)
                    
                    else:
                        feat = self.ds[layer_idx](feat) if block_idx == 0 else feat
                    
                    
                    # identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                    # feat = F.leaky_relu(self.blocks[layer_idx][block_idx](feat) + identity)
                    # feat = feat * self.task_gating_params[dset][block_count, 0] + identity * self.task_gating_params[dset][block_count, 1]
                    
                    block_count += 1
                    
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): feat})
                
        features = {}
        for dset, back_feats in backbone_feats.items():
            task = other_hyp["task_list"][dset]
            head = self.head_dict[dset]
            
            if self.training:
                targets = data_dict[dset][1]
                
                if task == 'clf':
                    losses = head(back_feats, targets)
                    
                elif task == 'det':
                    fpn_feat = self.fpn(back_feats)
                    losses = head(data_dict[dset][0], fpn_feat,
                                            self.stem_dict[dset].transform, 
                                        origin_targets=targets)
                    
                elif task == 'seg':
                    losses = head(
                        back_feats, targets, input_shape=targets.shape[-2:])
                
                losses = {f"feat_{dset}_{k}": l for k, l in losses.items()}
                features.update(losses)    
                total_losses.update(losses)
        
        self.current_iter += 1
        return total_losses
            
    
    def forward_eval(self, data_dict, other_hyp):
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        data = self._extract_stem_feats(data_dict)
        # self.policys = self.task_gating_params

        # print(self.policys[dset_list[0]])
        
        for dset, feat in data.items():
            block_count = 0
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    if self.task_gating_params[dset][block_count, 0] == 1:
                        identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                        feat = F.leaky_relu(self.blocks[layer_idx][block_idx](feat) + identity)
                        # feat = feat + identity
                    
                    else:
                        feat = self.ds[layer_idx](feat) if block_idx == 0 else feat
                    
                    block_count += 1
                    
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): feat})
        dset = list(other_hyp["task_list"].keys())[0]
        task = list(other_hyp["task_list"].values())[0]
        head = self.head_dict[dset]
        
        back_feats = backbone_feats[dset]
        
        if task == 'det':
            fpn_feat = self.fpn(back_feats)
            predictions = head(data_dict[dset][0], fpn_feat, self.stem_dict[dset].transform)
            
        else:
            if task == 'seg':
                predictions = head(
                    back_feats, input_shape=data_dict[dset][0].shape[-2:])
        
            else:
                predictions = head(back_feats)
            
            predictions = dict(outputs=predictions)

        return predictions
    

    def forward(self, data_dict, kwargs):
        if self.training:
            return self.forward_train(data_dict, kwargs)
        else:
            return self.forward_eval(data_dict, kwargs)


    # def _forward_val(self, images, kwargs):
    #     dset = list(kwargs.keys())[0]
    #     task = list(kwargs.values())[0]
        
    #     stem, head = self.stem_dict[dset], self.head_dict[dset]
    #     stem_feats = stem(images)
        
    #     if task == 'det':
    #         back_feats = self.backbone(stem_feats)
    #         predictions = head(images, back_feats, stem.transform)
    #         return predictions
        
    #     else:
    #         back_feats = self.backbone.body(stem_feats)
    #         if task == 'seg':
    #             predictions = head(
    #                 back_feats, input_shape=images.shape[-2:])
        
    #         else:
    #             predictions = head(back_feats)
            
    #         return dict(outputs=predictions)