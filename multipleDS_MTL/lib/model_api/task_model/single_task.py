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



def make_stem(task, stem_cfg, backbone=None, stem_weight=None, init_func=None):
    if task == 'clf':
        stem = ClfStem(**stem_cfg)
        if init_func is not None: stem.apply(init_func)
    
    elif task == 'det':
        stem = DetStem(**stem_cfg)
        # if stem_weight is not None:
        #     stem.load_state_dict(stem_weight, strict=False)
        
        if 'mobilenetv3' in backbone:
            for p in stem.parameters():
                p.requires_grad = False
    
    elif task == 'seg':
        stem = SegStem(**stem_cfg)
        # if stem_weight is not None:
        #     stem.load_state_dict(stem_weight, strict=False)
    
    return stem
    
    
def make_head(task, backbone, num_classes, dense_task=None, fpn_channel=256, 
              head_cfg=None, init_func=None):
    if task == 'clf':
        head = build_classifier(
            backbone, num_classes, head_cfg)
    
    elif task == 'det':
        assert dense_task is not None
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
    
    if init_func is not None: head.apply(init_func)
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
        
        # self.is_preactivation = True if kwargs['bottleneck_type'] == 'preact' else False
                
        backbone_weight = None
        if kwargs['backbone_weight'] is not None:
            backbone_weight = torch.load(kwargs['backbone_weight'], map_location="cpu")
            if kwargs['use_bias']:
                strict=False
                for n in list(backbone_weight.keys()):
                    if "bn" in n:
                       backbone_weight.pop(n) 
            
            else:
                strict=True
            
            backbone_network.body.load_state_dict(backbone_weight, strict=strict)
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
            stem_weight = torch.load(kwargs['stem_weight'], map_location='cpu')
        self.dset = dataset
        task = task_cfg['task']
        
        stem_cfg = task_cfg['stem']
        head_cfg = task_cfg['head'] if 'head' in task_cfg else {}
        
        stem_cfg.update({'activation_function': kwargs['activation_function']})
        stem_cfg.update({'stem_weight': stem_weight})
        
        head_cfg.update({'activation_function': kwargs['activation_function']})
        
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
        
        dense_task_name = None
        use_fpn = False
        if task == 'det':
            dense_task_name = detector
            if kwargs['use_fpn']:
                use_fpn = True
                head_cfg.update({'num_anchors': len(backbone_network.body.return_layers)+1})
        
        elif task == 'seg':
            dense_task_name = segmentor
            head_cfg["dataset"] = self.dset
        
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
            if init_function is not None: backbone_network.fpn.apply(init_function)
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
        
        in_ddp_static = False
        if "in_ddp_static" in kwargs:
            in_ddp_static = kwargs['in_ddp_static']
        
        if not in_ddp_static:
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
        
        self.activation_function = kwargs['activation_function']
    
    
    def _forward_bakcbone(self, images):
        feat = self.stem(images)
        backbone_feats = {}
        
        block_module = self.encoder['block']
        ds_module = self.encoder['ds']
        for layer_idx, num_blocks in enumerate(self.num_per_block):
            for block_idx in range(num_blocks):
                identity = ds_module[layer_idx](feat) if block_idx == 0 and ds_module[layer_idx] is not None else feat
                block_F = block_module[layer_idx][block_idx](feat)
                feat = self.activation_function(block_F + identity)
                
                # if self.is_preactivation:
                #     feat = block_F + identity
                # else:
                #     feat = self.activation_function(block_F + identity)
                
                
                
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
                               input_shape=images.shape[-2:])
            return losses
        
        else:
            predictions = self.head(
                    backbone_features, input_shape=images.shape[-2:])
            
            return predictions
            # return dict(outputs=predictions)
            

    def _foward_train(self, data_dict):
        if isinstance(data_dict, tuple):
            images, targets = data_dict
        elif isinstance(data_dict, dict):
            images, targets = data_dict[self.dset]
        
        loss_dict = self.task_forward(images, targets)
        
        final_losses = {f"{self.dset}_{k}": v for k, v in loss_dict.items()}
        return final_losses
        
    
    def _forward_val(self, images):
        if isinstance(images, (dict, OrderedDict)):
            images, _ = images[self.dset]
        prediction_results = self.task_forward(images)
        
        return prediction_results
    
    
    def forward(self, data_dict, kwargs=None):
        if self.training:
            return self._foward_train(data_dict)

        else:
            return self._forward_val(data_dict)
        
