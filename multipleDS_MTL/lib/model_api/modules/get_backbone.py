import torch.nn as nn
from torchvision.ops import misc as misc_nn_ops
# from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.ops.feature_pyramid_network import (FeaturePyramidNetwork, 
                                                     LastLevelP6P7, LastLevelMaxPool)
# from torchvision.models._utils import IntermediateLayerGetter
# from torchvision.models.detection.backbone_utils import BackboneWithFPN

from collections import OrderedDict
from typing import Dict, Optional

from ..backbones.resnet import get_resnet


class IntermediateLayerGetter(nn.ModuleDict):
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        
        layers = OrderedDict()
        
        for name, module in model.named_children():
            if isinstance(module, nn.ModuleDict):
                for n, m in module.items():
                    if n in return_layers:
                        new_k = n
                        layers[new_k] = m
                        del return_layers[n]
            
            else:
                layers[name] = module                        
            
            if not return_layers:
                break
        
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers


    def forward(self, x, return_list=None):
        out = OrderedDict()
        
        if return_list is None:
            return_layers = self.return_layers
        else:
            assert isinstance(return_list, dict) or isinstance(return_list, OrderedDict)
            return_layers = return_list
        
        for name, module in self.items():
            x = module(x)
            if name in return_layers:
                out_name = return_layers[name]
                out[out_name] = x
                
            # if name in self.return_layers:
            #     out_name = self.return_layers[name]
            #     out[out_name] = x
            
        
        return out
    

class BackboneWithFPN(nn.Module):
    def __init__(self, backbone, return_layers, in_channels_list=None, out_channels=None, 
                 extra_blocks=None, use_fpn=True,
                 backbone_type='origin'):
        super(BackboneWithFPN, self).__init__()
        if backbone_type == 'origin':
            self.body = backbone
        elif backbone_type == 'intermediate':
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        
        if use_fpn:
            if extra_blocks is None:
                extra_blocks = LastLevelMaxPool()
                
            assert in_channels_list is not None
            assert out_channels is not None
            assert use_fpn
            
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=extra_blocks,
            )
            self.fpn_out_channels = out_channels
        else:
            self.fpn = nn.Identity()

        self.use_fpn = use_fpn
        
        
    def forward(self, x, return_list = None):
        x = self.body(x, return_list)
        if self.use_fpn:
            if self.fpn:
                x = self.fpn(x)
        
        return x
    

def resnet_fpn_backbone(
    backbone_name,
    backbone_args,
):  
    backbone = get_resnet(backbone_name, **backbone_args)
    assert backbone is not None
    # select layers that wont be frozen
    assert 0 <= backbone_args['trainable_layers'] <= 4
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1'][:backbone_args['trainable_layers']]
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    
    # for name, parameter in backbone.named_parameters():
    #     print(name, parameter.requires_grad)
    
    if backbone_args['extra_blocks'] is None:
        extra_blocks = LastLevelMaxPool()
    else:
        extra_blocks = backbone_args['extra_blocks']

    returned_layers = backbone_args['returned_layers']
    assert isinstance(returned_layers, list) or isinstance(returned_layers, str)
    if returned_layers == 'all':
        returned_layers = [1, 2, 3, 4]
    
    elif returned_layers == 'last':
        returned_layers = [4]
        
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, 
                           extra_blocks=extra_blocks, 
                           use_fpn=backbone_args['use_fpn'], 
                           backbone_type=backbone_args['backbone_type'])


def resnet_without_fpn(
    backbone_name,
    backbone_args,
):  
    backbone = get_resnet(backbone_name, **backbone_args)
    assert backbone is not None
    
    # select layers that wont be frozen
    assert 0 <= backbone_args['trainable_layers'] <= 4
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1'][:backbone_args['trainable_layers']]
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)
    
    returned_layers = backbone_args['returned_layers']
    assert isinstance(returned_layers, list) or isinstance(returned_layers, str)
    if returned_layers == 'all':
        returned_layers = [1, 2, 3, 4]
    
    elif returned_layers == 'last':
        returned_layers = [4]
        
    assert min(returned_layers) > 0 and max(returned_layers) < 5
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    
    
    return BackboneWithFPN(backbone, return_layers, 
                           use_fpn=False, 
                           backbone_type=backbone_args['backbone_type'])


def build_backbone(arch, detector=None,
                   segmentor=None,
                   model_args=None):
    freeze_backbone = model_args.pop('freeze_backbone')
    train_allbackbone = model_args.pop('train_allbackbone')
    freeze_bn = model_args.pop('freeze_bn')
    
    if 'without_fpn' in model_args:
        without_fpn = model_args['without_fpn']
    else:
        without_fpn = False
    
    model_args.update({
        'norm_layer': misc_nn_ops.FrozenBatchNorm2d if freeze_bn else None,
        # 'norm_layer': "fbn" if freeze_bn else model_args['norm_type'],
        'deform_layers': model_args['deform'] if 'deform' in model_args else False,
        'backbone_type': 'intermediate' if not 'backbone_type' in model_args is None else model_args['backbone_type'],
        'extra_blocks': None})
    
    if detector is not None: 
        if 'faster' in detector:
            model_args.update({'use_fpn': model_args['use_fpn']})
        
        elif 'retina' in detector:
            model_args.update({'extra_blocks': LastLevelP6P7(256, 256)})
            model_args.update({'returned_layers': [2, 3, 4]})
        
        else:
            ValueError("The detector name {} is not supported detector.".format(detector))
    
    elif not detector and segmentor:
        assert not model_args['use_fpn']
        model_args.update({'use_fpn': model_args['use_fpn']})
    
    elif not detector and not segmentor:
        assert not model_args['use_fpn']
        model_args.update({'use_fpn': model_args['use_fpn']})
    
    if 'resnet' in arch or 'resnext' in arch:
        def check_return_layers(detector, segmentor):
            if not detector and not segmentor: # single-clf task
                returned_layers = 'last'
                    
            elif (detector and not segmentor) \
                or (not detector and segmentor): # train detection task or segemtation task 
                if segmentor:
                    returned_layers = [3, 4]
                elif detector:
                    returned_layers = 'all'
                    
            elif detector and segmentor:
                returned_layers = 'all'
                    
            return returned_layers
        
        
        dilation_type = model_args.pop('dilation_type')
        if dilation_type == 'fft':
                replace_stride_with_dilation = [False, False, True]
        elif dilation_type == 'fff':
            replace_stride_with_dilation = None
        elif dilation_type == 'ftt':
            replace_stride_with_dilation = [False, True, True]
            
        if freeze_backbone:
            if 'train_specific_layers' in model_args:
                train_specific_layers = model_args.pop('train_specific_layers')
            else:
                train_specific_layers = None
            
            if train_specific_layers is not None:
                trainable_backbone_layers = train_specific_layers
            else:
                trainable_backbone_layers = 0
        elif train_allbackbone:
            trainable_backbone_layers = 4
        else:
            if detector and not segmentor:
                trainable_backbone_layers = 3
            elif not detector and not segmentor:
                trainable_backbone_layers = 0
            else:
                trainable_backbone_layers = 4
                
        model_args.update({'replace_stride_with_dilation': replace_stride_with_dilation})
        model_args.update({'activation_fucntion': model_args['activation_function'] if 'activation_function' in model_args else None})
        model_args.update({'trainable_layers': trainable_backbone_layers})
        
        if without_fpn:
            backbone = resnet_without_fpn(
                arch,
                model_args
            )
            
        else:
            model_args.update({'returned_layers': check_return_layers(detector, segmentor)})
            backbone = resnet_fpn_backbone(
                arch,
                model_args
            )
        

        return backbone



    