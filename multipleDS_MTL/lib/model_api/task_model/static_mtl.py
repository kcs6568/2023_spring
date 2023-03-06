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


class StaticMTL(nn.Module):
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
        
        if kwargs['backbone_weight'] is not None:
            backbone_weight = torch.load(kwargs['backbone_weight'])
            backbone_network.body.load_state_dict(backbone_weight, strict=True)
            print("!!!Loaded pretrained body weights!!!")
        
        self.num_per_block = []
        blocks = []
        ds = []

        for _, p in backbone_network.body.named_children():
            block = []
            self.num_per_block.append(len(p))
            for m, q in p.named_children():
                if m == '0':
                    ds.append(q.downsample)
                    q.downsample = None
                
                block.append(q)
                
            blocks.append(nn.ModuleList(block))
        
        self.encoder = nn.ModuleDict({
            'block': nn.ModuleList(blocks),
            'ds': nn.ModuleList(ds)
        })
        
        # self.blocks = []
        # self.ds = []
        # self.num_per_block = []

        # for _, p in backbone_network.body.named_children():
        #     block = []
        #     self.num_per_block.append(len(p))
        #     for m, q in p.named_children():
        #         if m == '0':
        #             self.ds.append(q.downsample)
        #             q.downsample = None
                
        #         block.append(q)
                
        #     self.blocks.append(nn.ModuleList(block))
                
        # self.blocks = nn.ModuleList(self.blocks)
        # self.ds = nn.ModuleList(self.ds)
        
        self.fpn = backbone_network.fpn
        
        
        self.stem_dict = nn.ModuleDict()
        self.head_dict = nn.ModuleDict()
        
        self.return_layers = {}
        data_list = []
        
        if kwargs['backbone_weight'] is not None:
            stem_weight = kwargs['stem_weight']
        else:
            stem_weight = None
        
        shared_stem_configs = {
            'activation_function': kwargs['activation_function'],
            'stem_weight': stem_weight
        }
        
        shared_head_configs = {
            'activation_function': kwargs['activation_function']
        }
        
        for data, cfg in task_cfg.items():
            data_list.append(data)
            self.return_layers.update({data: cfg['return_layers']})
            
            if 'stem' in cfg:
                stem_cfg = cfg['stem']
            else:
                head_cfg = {}
            
            if 'head' in cfg:
                head_cfg = cfg['head']
            else:
                head_cfg = {}
            
            stem_cfg.update(shared_stem_configs)
            head_cfg.update(shared_head_configs)
            
            task = cfg['task']
            num_classes = cfg['num_classes']
            if task == 'clf':
                stem = ClfStem(**stem_cfg)
                head = build_classifier(
                    backbone, num_classes, head_cfg)
                stem.apply(init_weights)
                
            elif task == 'det':
                stem = DetStem(**stem_cfg)
                
                head_cfg.update({'num_anchors': len(backbone_network.body.return_layers)+1})
                head = build_detector(
                    backbone, detector, 
                    backbone_network.fpn_out_channels, num_classes, **head_cfg)
                
                # detection = build_detector(
                #     backbone, detector, 
                #     backbone_network.fpn_out_channels, num_classes, **head_cfg)
                
                # if backbone_network.fpn is not None:
                #     head = nn.ModuleDict({
                #         'fpn': backbone_network.fpn,
                #         'detector': detection
                #     })
                
            
            elif task == 'seg':
                stem = SegStem(**stem_cfg)
                head = build_segmentor(segmentor, num_classes=num_classes, cfg_dict=head_cfg)
            
            head.apply(init_weights)
            self.stem_dict.update({data: stem})
            self.head_dict.update({data: head})
        
        
    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        for dset, (images, _) in data_dict.items():
            stem_feats.update({dset: self.stem_dict[dset](images)})
        return stem_feats
    
    
    def get_features(self, data_dict, other_hyp):
        # for dd, smp in data_dict.items():
        #     print(smp)
        
        # for ddd, smp in data_dict.items():
        #     print(ddd, len(smp[0]))
            
        # exit()
        
        total_losses = OrderedDict()
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        
        data = self._extract_stem_feats(data_dict)
        
        block_module = self.encoder['block']
        ds_module = self.encoder['ds']
        
        for dset, feat in data.items():
            block_count=0
            # print(f"{dset} geration")
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    identity = ds_module[layer_idx](feat) if block_idx == 0 else feat
                    feat = F.leaky_relu(block_module[layer_idx][block_idx](feat) + identity)
                    
                    block_count += 1
                    
                    # print(f"block {block_count} finish")
                    
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): feat})
                        # print(f"{dset} return feature saved")
            # print()
                    
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
                    
                    # fpn_feat = head['fpn'](back_feats)
                    # losses = head['detector'](data_dict[dset][0], fpn_feat,
                    #                         self.stem_dict[dset].transform, 
                    #                     origin_targets=targets)
                    
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
                
                # fpn_feat = head['fpn'](back_feats)
                # predictions = head['detector'](data_dict[dset][0], fpn_feat, self.stem_dict[dset].transform)
                
            else:
                if task == 'seg':
                    predictions = head(
                        back_feats, input_shape=data_dict[dset][0].shape[-2:])
            
                else:
                    predictions = head(back_feats)
                
                predictions = dict(outputs=predictions)
            
            return predictions

    
    # def replace_batch_setting_for_DP(self, data_dict):
    

    def forward(self, data_dict, kwargs):
        for ddd, smp in data_dict.items():
            if 'coco' in ddd:
                count = smp[0].shape[0]
                start = torch.cuda.current_device() * count
                end = start + count
                
                reshaped_samples = self.stem_dict[ddd].transform.recover_all_batches(smp[0], start, end)
                new_target = smp[1][start:end]
                
                data_dict[ddd] = (reshaped_samples, new_target)
                # print(data_dict[ddd])
                # exit()
                
        
        return self.get_features(data_dict, kwargs)
