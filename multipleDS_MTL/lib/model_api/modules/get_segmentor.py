from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F


class SegStem(nn.Module):
    def __init__(self,
                 out_channels=64,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 use_maxpool=True,
                 relu=None
                ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if relu == 'hardswish':
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True)
            
        if use_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = None
        
        
    def forward(self, x):
        if self.training:
            assert x.size()[2] == x.size()[3]
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        
        if self.maxpool:
            x = self.maxpool(x)
        # exit()
        return x



class FCNHead(nn.Module):
    def __init__(self, in_channels=2048, inter_channels=None, 
                 num_classes=21, use_aux=True, aux_channel=None, num_skip_aux=1) -> None:
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4 if inter_channels is None else inter_channels
        self.fcn_head = self._make_fcn_head(in_channels, inter_channels, num_classes)
        self.interplation = nn.Upsample()
        if use_aux:
            '''
            get aux feature at the specific location (if this value is 1, the popitem() operation will be operated once)
            '''
            self.num_skip_aux = num_skip_aux 
            aux_inchannels = in_channels // 2 if aux_channel is None else aux_channel
            self.aux_head = self._make_fcn_head(aux_inchannels, aux_inchannels//4, num_classes)
            
        # if loss_fn == 'ce':
        #     self.loss_fn = self.ce_loss
        # elif loss_fn == 'dice':
        #     from ...apis.loss_lib import DiceLoss
        #     self.criterion = DiceLoss()
            
    def _make_fcn_head(self, in_channels, inter_channels, num_classes):
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            # nn.ReLU(inplace=True) if relu is None else relu,
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, num_classes, 1)
        ]
        
        return nn.Sequential(*layers)


    def criterion(self, inputs, target):
        losses = {}
        
        for name, x in inputs.items():
            losses[name] = F.cross_entropy(x, target, ignore_index=255)
        
        if "seg_aux_loss" in losses:
            losses["seg_aux_loss"] *= 0.5
         
        return losses
        

    def forward(self, feats, target=None, input_shape=480):
        assert isinstance(feats, dict) or isinstance(feats, OrderedDict)
        # print("1")
        results = OrderedDict()
        _, x = feats.popitem()
        x = self.fcn_head(x)
        
        # print("2")
        x = nn.Upsample(size=input_shape, mode='bilinear', align_corners=False)(x)
        # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        results["seg_out_loss"] = x
        
        # print("3")
        if self.aux_head is not None:
            for _ in range(self.num_skip_aux):
                _, x = feats.popitem()
            # print("4")
            x = self.aux_head(x)
            x = nn.Upsample(size=input_shape, mode='bilinear', align_corners=False)(x)
            # print("5")
            # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            results["seg_aux_loss"] = x
            # print("6")
        
        if self.training:
            # print("7")
            return self.criterion(results, target)

        else:
            return results["seg_out_loss"]
        

def build_segmentor(
    segmentor_name,
    num_classes=21,
    cfg_dict=None,
    detector=None,
    pretrained=False,
    ):
    
    segmentor_name = segmentor_name.lower()
    
    if 'maskrcnn' in segmentor_name:
        from torchvision.ops import MultiScaleRoIAlign
        from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
        
        mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=14,
            sampling_ratio=2)

        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(cfg_dict['out_channels'], mask_layers, mask_dilation)
        
        mask_predictor_in_channels = 256  # == mask_layers[-1]
        mask_dim_reduced = 256
        mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels,
                                            mask_dim_reduced, cfg_dict['num_classes'])
        

        detector.roi_heads.mask_roi_pool = mask_roi_pool
        detector.roi_heads.mask_head = mask_head
        detector.roi_heads.mask_predictor = mask_predictor
        
        return detector

    else:
        if 'fcn' in segmentor_name:
            head = FCNHead(num_classes=num_classes, **cfg_dict)
        
        elif 'deeplab' in segmentor_name:
            head = None    
        
        return head
    
        
