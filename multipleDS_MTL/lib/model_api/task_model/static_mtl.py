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
from ...apis.gradient_based import define_gradient_method
from ...apis.weighting_based import define_weighting_method
from ...utils.dist_utils import get_rank


def init_kaiming_uniform(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=1e-2, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=1e-2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    

def init_kaiming_normal(m):
    # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #     nn.init.kaiming_normal_(m.weight, a=1e-2, mode='fan_in', nonlinearity='leaky_relu')
    #     if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=1e-2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=1e-2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
    
def init_xavier_uniform(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    

def init_xavier_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



def init_weights(m, type="kaiming", distribution='uniform'):
    def init(m):
        if type == 'kaiming':
            if distribution == 'uniform': initializer = nn.init.kaiming_uniform_
            elif distribution == 'normal': initializer = nn.init.normal_
            
            initializer(m.weight, nonlinearity='leaky_relu')
            
        elif type == 'xavier':
            if distribution == 'uniform': initializer = nn.init.xavier_uniform_
            elif distribution == 'normal': initializer = nn.init.xavier_normal_
            initializer(m.weight)
    
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init(m)
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
    
    # if isinstance(m, nn.Conv2d):
    #     if type == 'kaiming':
    #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    #     elif type == 'xavier':
    #         nn.init.xavier_normal_(m.weight)
        
    #     if m.bias is not None:
    #         nn.init.constant_(m.bias, 0)
            
    # elif isinstance(m, nn.BatchNorm2d):
    #     nn.init.constant_(m.weight, 1)
    #     nn.init.constant_(m.bias, 0)
        
    # elif isinstance(m, nn.Linear):
    #     if type == 'kaiming':
    #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    #     elif type == 'xavier':
    #         nn.init.xavier_normal_(m.weight)
    #     nn.init.constant_(m.bias, 0)


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
        
        self.fpn = backbone_network.fpn
        
        self.stem = nn.ModuleDict()
        self.head = nn.ModuleDict()
        
        self.return_layers = {}
        datasets = []
        
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
        
        init_type = kwargs['init_type'] if 'init_type' in kwargs else None
        init_dist = kwargs['init_dist'] if 'init_dist' in kwargs else None
        
        init_function = None
        if init_type is not None and init_dist is not None:
            if kwargs['init_type'] == 'kaiming':
                if kwargs['init_dist'] == 'uniform':
                    init_function = init_kaiming_uniform
                elif kwargs['init_dist'] == 'normal':
                    init_function = init_kaiming_normal
                    
            elif kwargs['init_type'] == 'xavier':
                if kwargs['init_dist'] == 'uniform':
                    init_function = init_xavier_uniform
                elif kwargs['init_dist'] == 'normal':
                    init_function = init_xavier_normal
        
        
        use_fpn = True if kwargs['use_fpn'] else False
        
        for data, cfg in task_cfg.items():
            datasets.append(data)
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

                if init_function is not None: stem.apply(init_function) 
                
            elif task == 'det':
                stem_cfg['stem_weight'] = kwargs['stem_weight']
                stem = DetStem(**stem_cfg)
                
                head_cfg.update({'num_anchors': len(backbone_network.body.return_layers)+1})
                head = build_detector(
                    backbone, detector, 
                    backbone_network.fpn_out_channels, num_classes, **head_cfg)
                
                if use_fpn:
                    if init_function is not None: backbone_network.fpn.apply(init_function)
                    head = nn.ModuleDict({
                        'fpn': backbone_network.fpn,
                        'detector': head
                    })
                
            elif task == 'seg':
                stem_cfg['stem_weight'] = kwargs['stem_weight']
                stem = SegStem(**stem_cfg)
                head = build_segmentor(segmentor, num_classes=num_classes, cfg_dict=head_cfg)
            
            if init_function is not None: head.apply(init_function)
            self.stem.update({data: stem})
            self.head.update({data: head})
        self.datasets = datasets
        
        if 'weight_method' in kwargs:
            if kwargs['weight_method'] is not None:
                wm_param = kwargs['weight_method']
                w_type = wm_param.pop('type')
                init_param = wm_param.pop('init_param')
                self.weighting_method = define_weighting_method(w_type)
                if init_param: self.weighting_method.init_params(self.datasets, **wm_param)
            else: self.weighting_method = None
        else: self.weighting_method = None
        
        if 'grad_method' in kwargs:
            if kwargs['grad_method'] is not None:
                gm_param = kwargs['grad_method']
                g_type = gm_param.pop('type')
                self.grad_method = define_gradient_method(g_type)
                self.grad_method.init_params(self.datasets, **gm_param)
                self.all_shared_params_numel, self.each_param_numel = self.compute_shared_encoder_numel()
                
                if 'weight_method_for_grad' in gm_param:
                    if gm_param['weight_method_for_grad'] is not None:
                        gw_params = gm_param['weight_method_for_grad']
                        gw_type = gw_params.pop('type')
                        init_param = gw_params.pop('init_param')
                        grad_weighting_method = define_weighting_method(gw_type)
                        if init_param: grad_weighting_method.init_params(self.datasets, **gw_params)
                        setattr(self.grad_method, 'weighting_method', grad_weighting_method)
                        
                        
            else: self.grad_method = None
        else: self.grad_method = None
        
        # self.all_shared_params_numel, self.each_param_numel = self.compute_shared_encoder_numel()
        self.activation_function = kwargs['activation_function']
    
        
    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        for dset, (images, _) in data_dict.items():
            stem_feats.update({dset: self.stem[dset](images)})
        return stem_feats
    
    
    def get_features(self, data_dict, other_hyp):
        total_losses = OrderedDict()
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        
        data = self._extract_stem_feats(data_dict)
        
        block_module = self.encoder['block']
        ds_module = self.encoder['ds']
        
        for dset, feat in data.items():
            block_count=0
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    identity = ds_module[layer_idx](feat) if block_idx == 0 else feat
                    feat = self.activation_function(block_module[layer_idx][block_idx](feat) + identity)
                    
                    block_count += 1
                    
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): feat})
                    
        if self.training:
            for dset, back_feats in backbone_feats.items():
                # print(f"start to get the {dset} loss")
                task = other_hyp["task_list"][dset]
                head = self.head[dset]
                
                targets = data_dict[dset][1]
                
                if task == 'clf':
                    losses = head(back_feats, targets)
                    
                elif task == 'det':
                    fpn_feat = head['fpn'](back_feats)
                    losses = head['detector'](data_dict[dset][0], fpn_feat,
                                            self.stem[dset].transform, 
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
            head = self.head[dset]
            
            back_feats = backbone_feats[dset]
            
            if task == 'det':
                fpn_feat = head['fpn'](back_feats)
                predictions = head['detector'](data_dict[dset][0], fpn_feat, self.stem[dset].transform)
                
            else:
                if task == 'seg':
                    predictions = head(
                        back_feats, input_shape=data_dict[dset][0].shape[-2:])
            
                else:
                    predictions = head(back_feats)
                
                predictions = dict(outputs=predictions)
            
            return predictions

    

    def forward(self, data_dict, kwargs):
        return self.get_features(data_dict, kwargs)
