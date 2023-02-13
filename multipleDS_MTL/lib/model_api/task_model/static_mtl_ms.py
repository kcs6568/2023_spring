from dataclasses import replace
import numpy as np
from scipy.special import softmax
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ..modules.get_detector import build_detector, DetStem
from ..modules.get_backbone import build_backbone
from ..modules.get_segmentor import build_segmentor, SegStem
from ..modules.get_classifier import build_classifier, ClfStem


class ExtraFPNBlock(nn.Module):
    """
    Base class for the extra block in the FPN.

    Args:
        results (List[Tensor]): the result of the FPN
        x (List[Tensor]): the original feature maps
        names (List[str]): the names for each one of the
            original feature maps

    Returns:
        results (List[Tensor]): the extended set of results
            of the FPN
        names (List[str]): the extended set of names for the results
    """
    def forward(
        self,
        results: List[Tensor],
        x: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        pass


class FeaturePyramidNetwork(nn.Module):
    """
    Module that adds a FPN from on top of a set of feature maps. This is based on
    `"Feature Pyramid Network for Object Detection" <https://arxiv.org/abs/1612.03144>`_.

    The feature maps are currently supposed to be in increasing depth
    order.

    The input to the model is expected to be an OrderedDict[Tensor], containing
    the feature maps on top of which the FPN will be added.

    Args:
        in_channels_list (list[int]): number of channels for each feature map that
            is passed to the module
        out_channels (int): number of channels of the FPN representation
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names

    Examples::

        >>> m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
        >>> # get some dummy data
        >>> x = OrderedDict()
        >>> x['feat0'] = torch.rand(1, 10, 64, 64)
        >>> x['feat2'] = torch.rand(1, 20, 16, 16)
        >>> x['feat3'] = torch.rand(1, 30, 8, 8)
        >>> # compute the FPN on top of x
        >>> output = m(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('feat0', torch.Size([1, 5, 64, 64])),
        >>>    ('feat2', torch.Size([1, 5, 16, 16])),
        >>>    ('feat3', torch.Size([1, 5, 8, 8]))]

    """
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int,
        extra_blocks: Optional[ExtraFPNBlock] = None,
    ):
        super(FeaturePyramidNetwork, self).__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        if extra_blocks is not None:
            assert isinstance(extra_blocks, ExtraFPNBlock)
        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.

        Args:
            x (OrderedDict[Tensor]): feature maps for each feature level.

        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(ExtraFPNBlock):
    """
    Applies a max_pool2d on top of the last feature map
    """
    def forward(
        self,
        x: List[Tensor],
        y: List[Tensor],
        names: List[str],
    ) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))
        return x, names


def get_adaptive_scale_factor(main_size, target_size):
    height_scale_factor = target_size[0] / main_size[0]
    width_scale_factor = target_size[1] / main_size[1]
    
    return height_scale_factor, width_scale_factor

 
class MultiScaleModule(nn.Module):
    def __init__(self, main_task) -> None:
        super(MultiScaleModule, self).__init__()
        self.main_task = main_task
        
                
    def forward(self, all_feats, fusion_type='sum', upsample_mode='nearest', extended_return=True):
        main_feat = all_feats[self.main_task]
        B, C, H, W = main_feat.shape
        
        total_minibatches = B
        
        feat_list = []
        for dset, feat in all_feats.items():
            if dset == self.main_task:
                feat_list.extend(feat)
                continue

            feat_shape = feat.shape
            total_minibatches += feat_shape[0]
            
            is_high_height = True if feat_shape[2] > H else False
            is_high_width = True if feat_shape[3] > W else False
            size_check = torch.tensor([is_high_height, is_high_width])
                         
            # width and height are higher than main task feature size
            if torch.all(size_check):
                feat_ = nn.AdaptiveAvgPool2d((H, W))(feat)
            # width and height are smaller than main task feature size
            elif not torch.all(size_check):
                feat_ = nn.Upsample((H, W), mode=upsample_mode)(feat)
            
            # one the width or height is smaller (or higher) than main task feature map
            elif torch.any(size_check):
                # h_scale, w_scale = get_adaptive_scale_factor([H, W], feat_shape[-2:])
                target_size = min(feat_shape[-2:])
                
                # stride_size_h = int(( - 1) // feat_shape[2] + 1)
                # stride_size_w = int((input_size[1] - 1) // output_size[1] + 1)
                
                # 둘 중 하나 선택
                # feat_ = nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1)(feat)
                feat_ = nn.AdaptiveAvgPool2d((target_size, target_size))(feat)
                
                feat_ = nn.Upsample((H, W), mode=upsample_mode)(feat_)
            
            if extended_return:        
                feat_list.extend(feat_)
            else:
                feat_list.append(feat_)

        if total_minibatches != len(all_feats):
            return torch.stack(feat_list, dim=0)
        
        else:
            tmp = []
            _ = [tmp.extend(f) for f in feat_list]
            
            return torch.stack(tmp, dim=0)
        
        
                
class FusionLayer(nn.Module):
    def __init__(
        self,
        main_task,
        main_batches,
        in_channel,
        out_channel=None,
        reduce_ratio=4,
        conv_depth=1,
        conv_type='separable',
        style='sym',
        # fusion_type='sum',
        **kwargs
        ) -> None:
        super(FusionLayer, self).__init__()
        self.main_task = main_task
        self.main_batches = main_batches
        self.multi_scaler = MultiScaleModule(main_task)
        
        if out_channel is None:
            out_channel = in_channel
        
        reduced_channel = in_channel // reduce_ratio
        if conv_type == 'base':
            self.layers = nn.Sequential(
                nn.Conv2d(in_channel, reduced_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(reduced_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduced_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )
            
        elif conv_type == 'separable':
            assert style is not None
            
            if style == 'inline':
                self.layers = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
                    nn.Conv2d(in_channel, out_channel, kernel_size=1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True)
                )
            
            elif style == 'sym': # symmetric channel convert
                self.layers = nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel),
                    nn.Conv2d(in_channel, reduced_channel, kernel_size=1),
                    nn.BatchNorm2d(reduced_channel),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(reduced_channel, reduced_channel, kernel_size=3, padding=1, groups=reduced_channel),
                    nn.Conv2d(reduced_channel, out_channel, kernel_size=1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(inplace=True)
                )
                
                
    def forward(self, feat_dict, selection_mode='random'):
        scaled_features = self.multi_scaler(feat_dict)

        out = self.layers(scaled_features)
        all_B = out.shape[0]
        
        if selection_mode == 'random':
            ranges = torch.ones(all_B).float()
            ranges = torch.div(ranges, all_B)
            selected_idx = torch.multinomial(
                ranges, num_samples=self.main_batches, replacement=False).to("cuda")

        selected_feats = torch.index_select(out, dim=0, index=selected_idx)
        del out
        
        return selected_feats
        

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


class MSStaticMTL(nn.Module):
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
        # self.fpn = backbone_network.fpn
        
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
                
                # head_kwargs = {'num_anchors': len(backbone_network.body.return_layers)+1}
                
                head = build_detector(
                    backbone, detector, 
                    kwargs['det_outchannel'], num_classes)
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

        fusion_layers = {}
        det_channel_converter = {}
        for l in range(len(self.num_per_block)):
            name=f'layer{l+1}'
            
            tmp = {}
            for dset in data_list:
                tmp.update({dset: FusionLayer(
                    dset, 
                    kwargs['minibatches'][dset],
                    kwargs['channels'][l])})
            det_channel_converter.update(
                {name: nn.Conv2d(kwargs['channels'][l], kwargs['det_outchannel'], kernel_size=1)})
            
            fusion_layers.update({name: nn.ModuleDict(tmp)})
        
        self.fusion_layers = nn.ModuleDict(fusion_layers)
        self.det_channel_converter = nn.ModuleDict(det_channel_converter)
        self.data_list = data_list
        self.det_last_maxpool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)
        
    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        for dset, (images, _) in data_dict.items():
            stem_feats.update({dset: self.stem_dict[dset](images)})
        return stem_feats
    
    
    def get_train_features(self, data_dict):
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        
        data = self._extract_stem_feats(data_dict)
        
        for layer_idx, num_blocks in enumerate(self.num_per_block):
            layer_out = {}
            for dset, feat in data.items():
                for block_idx in range(num_blocks):
                    identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                    feat = F.leaky_relu_(self.blocks[layer_idx][block_idx](feat) + identity) 
                
                layer_out.update({dset: feat})
            
            fusion_layer = getattr(self.fusion_layers, f'layer{layer_idx+1}')
            
            for d in self.data_list:
                selected_out = F.leaky_relu_(fusion_layer[d](layer_out) + layer_out[d])
                data.update({d: selected_out})
                
                if d == 'minicoco':
                    selected_out = self.det_channel_converter[f'layer{layer_idx+1}'](selected_out)
                
                if str(layer_idx) in self.return_layers[d]:
                        backbone_feats[d].update({str(layer_idx): selected_out})

        if 'minicoco' in self.data_list:
            det_last_pool = self.det_last_maxpool(backbone_feats['minicoco']['3'])
            backbone_feats['minicoco'].update({'pool': det_last_pool})
        
        return backbone_feats
    
    
    def forward_train(self, origin_data, backbone_features, other_hyp):
        total_losses = OrderedDict()
        
        for dset, back_feats in backbone_features.items():
            task = other_hyp["task_list"][dset]
            head = self.head_dict[dset]
            
            targets = origin_data[dset][1]
            
            if task == 'clf':
                losses = head(back_feats, targets)
                
            elif task == 'det':
                losses = head(origin_data[dset][0], back_feats,
                                        self.stem_dict[dset].transform, 
                                    origin_targets=targets)
                
            elif task == 'seg':
                losses = head(
                    back_feats, targets, input_shape=targets.shape[-2:])
            
            losses = {f"feat_{dset}_{k}": l for k, l in losses.items()}
            total_losses.update(losses)
            
        return total_losses
    
    
    def forward_val(self, origin_data, other_hyp):
        dset = list(other_hyp["task_list"].keys())[0]
        task = list(other_hyp["task_list"].values())[0]
        backbone_feats = OrderedDict()
        
        feat = self.stem_dict[dset](origin_data[dset][0])
        
        for layer_idx, num_blocks in enumerate(self.num_per_block):
            for block_idx in range(num_blocks):
                identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                feat = F.leaky_relu_(self.blocks[layer_idx][block_idx](feat) + identity) 
        
            if str(layer_idx) in self.return_layers[dset]:
                if dset == 'minicoco':
                    back_feat = self.det_channel_converter[f'layer{layer_idx+1}'](feat)
                else:
                    back_feat = feat
                backbone_feats.update({str(layer_idx): back_feat})
        
        if dset == 'minicoco':
            det_last_pool = self.det_last_maxpool(backbone_feats['3'])
            backbone_feats.update({'pool': det_last_pool})
        
        head = self.head_dict[dset]
        
        if task == 'det':
            predictions = head(origin_data[dset][0], backbone_feats, self.stem_dict[dset].transform)
            
        else:
            if task == 'seg':
                predictions = head(
                    backbone_feats, input_shape=origin_data[dset][0].shape[-2:])
        
            else:
                predictions = head(backbone_feats)
            
            predictions = dict(outputs=predictions)
        
        return predictions
    
    
    def forward(self, data_dict, hyp):
        if self.training:
            shared_features = self.get_train_features(data_dict)
            return self.forward_train(data_dict, shared_features, hyp)
        else:
            return self.forward_val(data_dict, hyp)
