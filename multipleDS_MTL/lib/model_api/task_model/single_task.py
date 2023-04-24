from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.get_detector import build_detector, DetStem
from ..modules.get_backbone import build_backbone
from ..modules.get_segmentor import build_segmentor, SegStem
from ..modules.get_classifier import build_classifier, ClfStem
from ...apis.loss_lib import AutomaticWeightedLoss


# def init_weights(m, type="kaiming"):
#     if isinstance(m, nn.Conv2d):
#         if type == 'kaiming':
#             nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#         elif type == 'xavier':
#             nn.init.xavier_normal_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
            
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)
        
#     elif isinstance(m, nn.Linear):
#         if type == 'kaiming':
#             nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#         elif type == 'xavier':
#             nn.init.xavier_normal_(m.weight)
#         nn.init.constant_(m.bias, 0)


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



def make_stem(task, stem_cfg, backbone=None, stem_weight=None, init_func=init_kaiming_uniform):
    if task == 'clf':
        stem = ClfStem(**stem_cfg)
        stem.apply(init_func)
    
    elif task == 'det':
        stem = DetStem(**stem_cfg)
        if stem_weight is not None:
            stem.load_state_dict(stem_weight, strict=True)
        
        if 'mobilenetv3' in backbone:
            for p in stem.parameters():
                p.requires_grad = False
    
    elif task == 'seg':
        stem = SegStem(**stem_cfg)
        if stem_weight is not None:
            stem.load_state_dict(stem_weight, strict=True)
    
    return stem
    
    
def make_head(task, backbone, num_classes, dense_task=None, fpn_channel=256, 
              head_cfg=None, init_func=init_kaiming_uniform):
    if task == 'clf':
        head = build_classifier(
            backbone, num_classes, head_cfg)
    
    elif task == 'det':
        assert dense_task is not None
        head_cfg.update({})
        head = build_detector(
            backbone,
            dense_task, 
            fpn_channel, 
            num_classes,
            **head_cfg)
    
    elif task == 'seg':
        assert dense_task is not None
        head = build_segmentor(
            dense_task, num_classes, cfg_dict=head_cfg)
    
    head.apply(init_func)
    return head


class SingleTaskNetwork(nn.Module):
    def __init__(self,
                 backbone,
                 detector,
                 segmentor,
                 dataset,
                 task_cfg,
                 **kwargs
                 ) -> None:
        super().__init__()
        backbone_network = build_backbone(
            backbone, detector, segmentor, kwargs)
        
        backbone_weight = None
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
        
        stem_weight = None
        if kwargs['stem_weight'] is not None:
            stem_weight = torch.load(kwargs['stem_weight'])
        self.dset = dataset
        task = task_cfg['task']
        
        stem_cfg = task_cfg['stem']
        head_cfg = task_cfg['head'] if 'head' in task_cfg else {}
        
        stem_cfg.update({'activation_function': kwargs['activation_function']})
        head_cfg.update({'activation_function': kwargs['activation_function']})
        
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
        
        dense_task_name = None
        use_fpn = False
        if task == 'det':
            dense_task_name = detector
            if kwargs['use_fpn']:
                use_fpn = True
                head_cfg.update({'num_anchors': len(backbone_network.body.return_layers)+1})
        
        elif task == 'seg':
            dense_task_name = segmentor
        
        self.stem = make_stem(task, stem_cfg, 
                              backbone, 
                              stem_weight, 
                              init_func=init_function)
        
        head = make_head(task, backbone, 
                              task_cfg['num_classes'],
                              dense_task_name,
                              head_cfg=head_cfg,
                              init_func=init_function)
        
        if use_fpn:
            backbone_network.fpn.apply(init_function)
            self.head = nn.ModuleDict({
                'fpn': backbone_network.fpn,
                'detector': head
            })
        else: self.head = head
        
        if task == 'clf':
            self.task_forward = self._forward_clf
        elif task == 'det':
            self.task_forward = self._forward_coco
        elif task == 'seg':
            self.task_forward = self._forward_voc
        
        self.task = task    
        self.return_layers = task_cfg['return_layers']
    
    
    def _forward_bakcbone(self, images):
        feat = self.stem(images)
        backbone_feats = {}
        
        block_module = self.encoder['block']
        ds_module = self.encoder['ds']
        for layer_idx, num_blocks in enumerate(self.num_per_block):
            for block_idx in range(num_blocks):
                identity = ds_module[layer_idx](feat) if block_idx == 0 else feat
                feat = F.leaky_relu(block_module[layer_idx][block_idx](feat) + identity)
                
            if block_idx == (num_blocks - 1):
                if str(layer_idx) in self.return_layers:
                    backbone_feats.update({str(layer_idx): feat})
                    
    
        return backbone_feats
    

    def _forward_clf(self, images, targets=None):
        backbone_features = self._forward_bakcbone(images)
        
        if self.training:
            losses = self.head(backbone_features, targets)
            return losses
        
        else:
            predictions = self.head(backbone_features)
            
            return dict(outputs=predictions)
    
    
    def _forward_coco(self, images, targets=None):
        backbone_features = self._forward_bakcbone(images)
        data = images
        
        fpn_features = self.head['fpn'](backbone_features)
        
        if self.training:
            losses = self.head['detector'](data, fpn_features, 
                                origin_targets=targets,
                                trs_fn=self.stem.transform)
            return losses
        
        else:
            predictions = self.head['detector'](data, fpn_features,                                       
                            trs_fn=self.stem.transform)
            
            return predictions
        
    
    def _forward_voc(self, images, targets=None):
        backbone_features = self._forward_bakcbone(images)
        
        if self.training:
            losses = self.head(backbone_features, targets,
                               input_shape=targets.shape[-2:])
            return losses
        
        else:
            predictions = self.head(
                    backbone_features, input_shape=images.shape[-2:])
            
            # print(predictions.size())
            return dict(outputs=predictions)
            

    def _foward_train(self, data_dict):
        images, targets = data_dict
        loss_dict = self.task_forward(images, targets)
        
        return loss_dict
        
    
    def _forward_val(self, images):
        prediction_results = self.task_forward(images)
        
        return prediction_results
    
    
    def forward(self, data_dict):
        if self.training:
            return self._foward_train(data_dict)

        else:
            return self._forward_val(data_dict)
        
