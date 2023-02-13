from dataclasses import replace
import numpy as np
from scipy.special import softmax
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.get_detector import build_detector, DetStem
from ..modules.get_backbone import build_backbone
from ..modules.get_segmentor import build_segmentor, SegStem
from ..modules.get_classifier import build_classifier, ClfStem


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


class MTLNoGate(nn.Module):
    def __init__(self,
                 backbone,
                 detector,
                 segmentor,
                 task_cfg,
                 **kwargs
                 ) -> None:
        super().__init__()
        
        self.backbone = build_backbone(
            backbone, detector, segmentor, kwargs)
        
        self.stem_dict = nn.ModuleDict()
        self.head_dict = nn.ModuleDict()
        
        self.return_dict = {}
        
        stem_weight = kwargs['state_dict']['stem']
        for data, cfg in task_cfg.items():
            task = cfg['task']
            num_classes = cfg['num_classes']
            self.return_dict.update({data: cfg["return_layers"]})
            
            if task == 'clf':
                stem = ClfStem(**cfg['stem'])
                head = build_classifier(
                    backbone, num_classes, cfg['head'])
                stem.apply(init_weights)
                
            elif task == 'det':
                stem = DetStem(**cfg['stem'])
                
                head_kwargs = {'num_anchors': len(self.backbone.body.return_layers)+1}
                head = build_detector(
                    backbone, detector, 
                    self.backbone.fpn_out_channels, num_classes, **head_kwargs)
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
        
        
    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        for dset, (images, _) in data_dict.items():
            stem_feats.update({dset: self.stem_dict[dset](images)})
            
        return stem_feats
    
    def _extract_backbone_feats(self, stem_feats, tasks):
        backbone_feats = OrderedDict()
        
        for dset, feats in stem_feats.items():
            task = tasks[dset]
            return_layers = self.return_dict[task]
            if task == 'clf':
                features = self.backbone.body(feats, return_layers)
            
            elif task == 'det':
                features = self.backbone(feats, return_layers)
            
            elif task == 'seg':
                features = self.backbone.body(feats, return_layers)
            
            backbone_feats.update({dset: features})
            
        return backbone_feats
    
    
    def _foward_train(self, data_dict, tasks):
        total_losses = OrderedDict()
        task_list = tasks['task_list']
        stem_feats = self._extract_stem_feats(data_dict, task_list)
        backbone_feats = self._extract_backbone_feats(stem_feats, task_list)
        
        for dset, back_feats in backbone_feats.items():
            task = tasks[dset]
            head = self.head_dict[dset]
            targets = data_dict[dset][1]
            
            if task == 'clf':
                losses = head(back_feats, targets)
                
            elif task == 'det':
                losses = head(data_dict[dset][0], back_feats,
                                        self.stem_dict[dset].transform, 
                                       origin_targets=targets)
                
            elif task == 'seg':
                losses = head(
                    back_feats, targets, input_shape=targets.shape[-2:])
                
            losses = {f"{dset}_{k}": l for k, l in losses.items()}
            total_losses.update(losses)
        
        
        
        return total_losses
    

    # def _foward_train(self, data_dict, tasks):
    #     total_losses = OrderedDict()
        
    #     for dset, (images, targets) in data_dict.items():
    #         task = tasks[dset]
    #         dset_task = f"{dset}_{task}"
    #         stem, head = self.stem_dict[dset], self.head_dict[dset]
            
    #         if task == 'clf':
    #             stem_feats = stem(images)
    #             back_feats = self.backbone.body(stem_feats)
    #             losses = head(back_feats, targets)
                
    #         elif task == 'det':
    #             stem_feats = stem(images)
    #             back_feats = self.backbone(stem_feats)
    #             losses = head(images, back_feats, stem.transform, origin_targets=targets)
                
    #         elif task == 'seg':
    #             stem_feats = stem(images)
    #             back_feats = self.backbone.body(stem_feats)
    #             losses = head(back_feats, targets, input_shape=targets.shape[-2:])
                
    #         losses = {f"{dset}_{k}": l for k, l in losses.items()}
    #         total_losses.update(losses)
            
    #     return total_losses
    
    
    def _forward_val(self, images, kwargs):
        dset = list(kwargs['task_list'].keys())[0]
        task = list(kwargs['task_list'].values())[0]
        
        data = images[dset][0]
        
        stem, head = self.stem_dict[dset], self.head_dict[dset]
        stem_feats = stem(data)
        
        if task == 'det':
            back_feats = self.backbone(stem_feats)
            predictions = head(data, back_feats, stem.transform)
            return predictions
        
        else:
            back_feats = self.backbone.body(stem_feats)
            if task == 'seg':
                predictions = head(
                    back_feats, input_shape=data.shape[-2:])
        
            else:
                predictions = head(back_feats)
            
            return dict(outputs=predictions)
        
    
    def forward(self, data_dict, kwargs):
        if self.training:
            return self._foward_train(data_dict, kwargs)

        else:
            return self._forward_val(data_dict, kwargs)




# class MTLNoGate(nn.Module):
#     def __init__(self,
#                  backbone,
#                  detector,
#                  segmentor,
#                  task_cfg,
#                  **kwargs
#                  ) -> None:
#         super().__init__()
#         backbone_network = build_backbone(
#             backbone, detector, segmentor, kwargs)
        
#         self.blocks = []
#         self.ds = []
#         self.num_per_block = []

#         is_freeze_blocks = True if kwargs['nonfreezing_blocks'] is not None else False
        
#         current_block_num = 0
#         for _, p in backbone_network.body.named_children():
#             block = []
#             self.num_per_block.append(len(p))
#             for m, q in p.named_children():
#                 if m == '0':
#                     self.ds.append(q.downsample)
#                     q.downsample = None
                
#                 block.append(q)
#                 current_block_num += 1
                
#             self.blocks.append(nn.ModuleList(block))
        
        
#         if is_freeze_blocks and isinstance(kwargs['nonfreezing_blocks'], int):
#             for block in self.blocks[:kwargs['nonfreezing_blocks']]:
#                 for p in block.parameters():
#                     p.requires_grad = False
                
#         policy = torch.tensor([1 for _ in range(sum(self.num_per_block))]).float()
#         policy[kwargs['nonfreezing_blocks']:] = 0
#         self.policy = policy
        
#         self.blocks = nn.ModuleList(self.blocks)
#         self.ds = nn.ModuleList(self.ds)
#         self.fpn = backbone_network.fpn
        
#         self.stem_dict = nn.ModuleDict()
#         self.head_dict = nn.ModuleDict()
        
#         self.return_layers = {}
#         data_list = []
        
#         stem_weight = kwargs['state_dict']['stem']
#         for data, cfg in task_cfg.items():
#             data_list.append(data)
#             self.return_layers.update({data: cfg['return_layers']})
            
#             task = cfg['task']
#             num_classes = cfg['num_classes']
#             if task == 'clf':
#                 stem = ClfStem(**cfg['stem'])
#                 head = build_classifier(
#                     backbone, num_classes, cfg['head'])
#                 stem.apply(init_weights)
                
#             elif task == 'det':
#                 stem = DetStem(**cfg['stem'])
                
#                 head_kwargs = {'num_anchors': len(backbone_network.body.return_layers)+1}
#                 head = build_detector(
#                     backbone, detector, 
#                     backbone_network.fpn_out_channels, num_classes, **head_kwargs)
#                 if stem_weight is not None:
#                     ckpt = torch.load(stem_weight)
#                     stem.load_state_dict(ckpt, strict=False)
#                     print("!!!Load weights for detection stem layer!!!")
            
#             elif task == 'seg':
#                 stem = SegStem(**cfg['stem'])
#                 head = build_segmentor(segmentor, num_classes=num_classes, cfg_dict=cfg['head'])
#                 if stem_weight is not None:
#                     ckpt = torch.load(stem_weight)
#                     stem.load_state_dict(ckpt, strict=False)
#                     print("!!!Load weights for segmentation stem layer!!!")
            
#             head.apply(init_weights)
#             self.stem_dict.update({data: stem})
#             self.head_dict.update({data: head})
        
        
#     def _extract_stem_feats(self, data_dict):
#         stem_feats = OrderedDict()
        
#         for dset, (images, _) in data_dict.items():
#             stem_feats.update({dset: self.stem_dict[dset](images)})
#         return stem_feats
    
    
#     def get_features(self, data_dict, other_hyp):
#         total_losses = OrderedDict()
#         backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        
#         data = self._extract_stem_feats(data_dict)
#         block_count = 0
#         for dset, feat in data.items():
#             for layer_idx, num_blocks in enumerate(self.num_per_block):
#                 for block_idx in range(num_blocks):
#                     identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                    
#                     if self.policy[block_count] == 1:
#                         feat = F.leaky_relu(self.blocks[layer_idx][block_idx](feat) + identity)
#                     else:
#                         feat = identity
                    
#                 if block_idx == (num_blocks - 1):
#                     if str(layer_idx) in self.return_layers[dset]:
#                         backbone_feats[dset].update({str(layer_idx): feat})
                    
#                 block_count += 1
        
#         if self.training:
#             features = {}
#             for dset, back_feats in backbone_feats.items():
#                 task = other_hyp["task_list"][dset]
#                 head = self.head_dict[dset]
                
#                 if self.training:
#                     targets = data_dict[dset][1]
                    
#                     if task == 'clf':
#                         losses = head(back_feats, targets)
                        
#                     elif task == 'det':
#                         fpn_feat = self.fpn(back_feats)
#                         losses = head(data_dict[dset][0], fpn_feat,
#                                                 self.stem_dict[dset].transform, 
#                                             origin_targets=targets)
                        
#                     elif task == 'seg':
#                         losses = head(
#                             back_feats, targets, input_shape=targets.shape[-2:])
                    
#                     losses = {f"feat_{dset}_{k}": l for k, l in losses.items()}
#                     features.update(losses)    
#                     total_losses.update(losses)

#             return total_losses
            
#         else:
#             dset = list(other_hyp["task_list"].keys())[0]
#             task = list(other_hyp["task_list"].values())[0]
#             head = self.head_dict[dset]
            
#             back_feats = backbone_feats[dset]
            
#             if task == 'det':
#                 fpn_feat = self.fpn(back_feats)
#                 predictions = head(data_dict[dset][0], fpn_feat, self.stem_dict[dset].transform)
                
#             else:
#                 if task == 'seg':
#                     predictions = head(
#                         back_feats, input_shape=data_dict[dset][0].shape[-2:])
            
#                 else:
#                     predictions = head(back_feats)
                
#                 predictions = dict(outputs=predictions)

#             return predictions


#     def forward(self, data_dict, kwargs):
#         return self.get_features(data_dict, kwargs)
