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
from ..backbones.mobilenet_v3 import get_mobilenet_v3


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


def mobilenetv3_fpn_backbone(
    backbone_name,
    backbone_args,
    detector,
    segmentor
):
    backbone = get_mobilenet_v3(backbone_name, weight_path=backbone_args.pop('weight_path'), **backbone_args)   # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    feature_layers = backbone.features
    
    return_indices = [i for i, b in enumerate(feature_layers) if getattr(b, "_is_cn", False)] + [len(feature_layers) - 1]
    num_stages = len(return_indices)
    
    if not detector and not segmentor:
        return_indices = [len(feature_layers) - 1]
        num_stages = len(return_indices)
        returned_layers = [num_stages - 1]
    
    elif detector and not segmentor:
        returned_layers = [num_stages - 2, num_stages - 1] # if returned_layer is 'all' or list type
    
    elif not detector and segmentor:
        returned_layers = [num_stages - 4, num_stages - 1]
    
    else: # all model
        returned_layers = [num_stages - 3, num_stages - 2, num_stages - 1]
        # returned_layers = [num_stages - 2, num_stages - 1]

    
    # print(rm,beturned_layers)
    assert min(returned_layers) >= 0 and max(returned_layers) < num_stages

    trainable_type = backbone_args['trainable_layers']
    # Freeze layers before the layers of [freeze_before] value
    if isinstance(trainable_type, str) and trainable_type == 'all':
        freeze_before = 0
        trainable_layers = len(feature_layers)
        
    else:
        trainable_layers = trainable_type
        assert isinstance(trainable_layers, int)
        freeze_before = len(feature_layers) if trainable_layers == 0 else return_indices[num_stages - trainable_layers]
        assert 0 <= trainable_layers <= num_stages
    
    for b in feature_layers[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)
            
    return_layers = {f'{return_indices[k]}': str(v) for v, k in enumerate(returned_layers)}
    
    extra_blocks = backbone_args['extra_blocks']
    in_channels_list = None
    out_channels = 256
    if backbone_args['use_fpn']:
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()
        else:
            extra_blocks = backbone_args['extra_blocks']
        in_channels_list = [feature_layers[return_indices[i]].out_channels for i in returned_layers]
    
    
    print(detector, segmentor)
    print(num_stages)
    print(trainable_layers)
    print(return_indices)
    print(returned_layers)
    print(freeze_before)
    # exit()
            
    return BackboneWithFPN(
        feature_layers, return_layers, in_channels_list, out_channels, 
        extra_blocks=extra_blocks, 
        backbone_type=backbone_args['backbone_type'], 
        use_fpn=backbone_args['use_fpn'])
    


        
    # else:
    #     m = nn.Sequential(
    #         backbone,
    #         # depthwise linear combination of channels to reduce their size
    #         nn.Conv2d(backbone[-1].out_channels, out_channels, 1),
    #     )
    #     m.out_channels = out_channels
    #     return m




def build_backbone(arch, detector=None,
                   segmentor=None,
                   model_args=None):
    backbone_args = {}
    
    freeze_backbone = model_args.pop('freeze_backbone')
    train_allbackbone = model_args.pop('train_allbackbone')
    freeze_bn = model_args.pop('freeze_bn')
    
    # if freeze_backbone:
    #     if not train_allbackbone:
    #         train_allbackbone = False
    # elif train_allbackbone:
    #     if not freeze_backbone:
    #         freeze_backbone = False
    # else:
    #     raise RuntimeError("the value of freeze_backbone must be the opposite value of train_allbackbone and vise versa")           
        
    

    if 'without_fpn' in model_args:
        without_fpn = model_args['without_fpn']
    else:
        without_fpn = False
    
    backbone_args = {
        'norm_layer': misc_nn_ops.FrozenBatchNorm2d if freeze_bn else None,
        'deform_layers': model_args['deform'] if 'deform' in model_args else False,
        'weight_path': model_args['state_dict']['backbone'],
        'backbone_type': 'intermediate' if not 'backbone_type' in model_args is None else model_args['backbone_type'],
        'extra_blocks': None,
    }
    
    if detector is not None: 
        if 'faster' in detector:
            backbone_args.update({'use_fpn': model_args['use_fpn']})
        
        elif 'retina' in detector:
            backbone_args.update({'extra_blocks': LastLevelP6P7(256, 256)})
            backbone_args.update({'returned_layers': [2, 3, 4]})
        
        else:
            ValueError("The detector name {} is not supported detector.".format(detector))
    
    elif not detector and segmentor:
        assert not model_args['use_fpn']
        backbone_args.update({'use_fpn': model_args['use_fpn']})
    
    elif not detector and not segmentor:
        assert not model_args['use_fpn']
        backbone_args.update({'use_fpn': model_args['use_fpn']})
    
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
                
        backbone_args.update({'replace_stride_with_dilation': replace_stride_with_dilation})
        backbone_args.update({'relu_type': model_args['relu_type'] if 'relu_type' in model_args else None})
        backbone_args.update({'trainable_layers': trainable_backbone_layers})
        
        
        if without_fpn:
            backbone = resnet_without_fpn(
                arch,
                backbone_args
            )
        else:
            backbone_args.update({'returned_layers': check_return_layers(detector, segmentor)})
            backbone = resnet_fpn_backbone(
                arch,
                backbone_args
            )
            

    elif 'mobile' in arch:
        if train_allbackbone:
            trainable_layers = 'all'
        else:
            trainable_layers = 3
            
        backbone_args.update({'dilated': model_args['dilated']})
        backbone_args.update({'use_fpn': model_args['use_fpn']})
        backbone_args.update({'trainable_layers': trainable_layers})
        backbone_args.update({'no_st_early': model_args['no_st_early']})
        
        backbone = mobilenetv3_fpn_backbone(
            arch,
            backbone_args,
            detector,
            segmentor
        )
        
    
    else:
        raise ValueError("The backbone name should be required.")
    
    if 'use_kd' in model_args:
        if model_args['use_kd']:
            feature_extractor = resnet_without_fpn(
                arch,
                backbone_args)
            return backbone, feature_extractor
        else:
            return backbone

    else:
        return backbone



    


# norm_layer = misc_nn_ops.FrozenBatchNorm2d if freeze_bn else None
    
    # weight_path = model_args['state_dict']['backbone']
    # deform = model_args['deform']
    # use_fpn = False if not 'use_fpn' in model_args else model_args['use_fpn']
    # backbone_type = 'intermediate' if not 'backbone_type' in model_args is None else model_args['backbone_type']
    # erase_relu = model_args['erase_relu'] if 'erase_relu' in model_args else False




# class BackboneWithNoFPN(nn.Module):
#     def __init__(self, backbone, return_layers):
#         super(BackboneWithNoFPN, self).__init__()
#         self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
#         self.body_out_channel = backbone.last_out_channel


#     def forward(self, x):
#         x = self.body(x)
        
#         return x


# def resnet_backbone(
#     backbone_name,
#     use_segmentor=None,
#     weight_path=None,
#     pretrained=True,
#     norm_layer=torchvision.ops.misc.FrozenBatchNorm2d,
#     trainable_layers=4,
#     returned_layers=4,
# ):
#     if 'res' in backbone_name:
#         if use_segmentor:
#             replace_stride_with_dilation = [False, True, True]
#         else:
#             replace_stride_with_dilation = None
            
#         kwargs = {
#                 'replace_stride_with_dilation': replace_stride_with_dilation
#             }
        
#         backbone = get_resnet(backbone_name, weight_path, **kwargs)
    
#     assert 0 <= trainable_layers <= 4
#     layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1'][:trainable_layers]
    
#     for name, parameter in backbone.named_parameters():
#         if all([not name.startswith(layer) for layer in layers_to_train]):
#             parameter.requires_grad_(False)
    
#     # for n, p in backbone.named_parameters():
#     #     print(n, p.requires_grad)
        
#     if returned_layers is None:
#         returned_layers = [1, 2, 3, 4]
#     assert min(returned_layers) > 0 and max(returned_layers) < 5
#     return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    
    
#     return BackboneWithNoFPN(backbone, return_layers)