from dataclasses import replace
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.get_detector import build_detector, DetStem
from ..modules.get_backbone import build_backbone
from ..modules.get_segmentor import build_segmentor, SegStem
from ..modules.get_classifier import build_classifier, ClfStem
# from ...apis.weighting import GradNorm as GN_module


def init_weights(m, type="kaiming"):
    if isinstance(m, nn.Conv2d):
        if type == 'kaiming':
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif type == 'xavier':
            nn.init.xavier_normal_(m.weight)
        
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.Linear):
        if type == 'kaiming':
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif type == 'xavier':
            nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class GradNorm(nn.Module):
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

        # self.layers = []
        
        # self.shared_grad = {}
        # def save_grad(name):
        #     def hook(grad):
        #          self.shared_grad[name] = grad
        #     return hook
        
        for _, p in backbone_network.body.named_children():
            # self.layers.append(p)
            
            block = []
            self.num_per_block.append(len(p))
            for m, q in p.named_children():
                if m == '0':
                    self.ds.append(q.downsample)
                    q.downsample = None
                
                block.append(q)
                
            self.blocks.append(nn.ModuleList(block))
        
        # self.blocks[-1][-1].conv3.register_hook(save_grad('shared_grad'))
            
        # self.layers = nn.ModuleList(self.layers)
        
        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fpn = backbone_network.fpn
        
        
        
        
        self.stem_dict = nn.ModuleDict()
        self.head_dict = nn.ModuleDict()
        
        self.return_layers = {}
        data_list = []
        self.activation = nn.LeakyReLU(inplace=True)
        # stem_weight = kwargs['state_dict']['stem']
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
                # if stem_weight is not None:
                #     ckpt = torch.load(stem_weight)
                #     stem.load_state_dict(ckpt, strict=False)
                #     print("!!!Load weights for detection stem layer!!!")
            
            elif task == 'seg':
                stem = SegStem(**cfg['stem'])
                head = build_segmentor(segmentor, num_classes=num_classes, cfg_dict=cfg['head'])
                # if stem_weight is not None:
                #     ckpt = torch.load(stem_weight)
                #     stem.load_state_dict(ckpt, strict=False)
                #     print("!!!Load weights for segmentation stem layer!!!")
            
            head.apply(init_weights)
            self.stem_dict.update({data: stem})
            self.head_dict.update({data: head})
            
        self.final_layer = 16
        self.n_tasks = len(data_list)
        
        w_dict = {}
        for d in data_list:
            weight = torch.tensor(1, dtype=torch.float, requires_grad=True)
            w_dict.update({d: nn.Parameter(weight, requires_grad=True)})
        
        # self.loss_weights = nn.ParameterDict(w_dict)
        
    
    def get_last_shared_module(self):
        last_module = self.blocks[-1][-1].conv3
        # last_module = self.layers[-1][-1].conv3
        return last_module
    
    
    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        for dset, (images, _) in data_dict.items():
            stem_feats.update({dset: self.stem_dict[dset](images)})
        return stem_feats
    
    
    def get_features(self, data_dict, other_hyp):
        total_losses = OrderedDict()
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        
        data = self._extract_stem_feats(data_dict)
        
        for dset, feat in data.items():
            print_cnt = 0
            # for layer_idx, layer in enumerate(self.layers):
            #     feat = self.activation(layer(feat))
            
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                    feat = self.activation(self.blocks[layer_idx][block_idx](feat) + identity)
                    
                if str(layer_idx) in self.return_layers[dset]:
                    backbone_feats[dset].update({str(layer_idx): feat})
                
                print_cnt += 1
        
        # for dset, feat in data.items():
        #     block_count=0
        #     # print(f"{dset} geration")
        #     for layer_idx, num_blocks in enumerate(self.num_per_block):
        #         for block_idx in range(num_blocks):
        #             identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
        #             feat = F.leaky_relu(self.blocks[layer_idx][block_idx](feat) + identity)
        
        
                   
        if self.training:
            for dset, back_feats in backbone_feats.items():
                # print(f"start to get the {dset} loss")
                task = other_hyp["task_list"][dset]
                head = self.head_dict[dset]
                
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
                total_losses.update(losses)
                
            return total_losses
            
        else:
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
        print("aaa")
        return self.get_features(data_dict, kwargs)
