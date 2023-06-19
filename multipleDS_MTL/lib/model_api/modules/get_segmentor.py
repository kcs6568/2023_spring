from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F

from ...apis.loss_lib import get_loss_fn


class SegStem(nn.Module):
    def __init__(self,
                 out_channels=64,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 stem_weight=None,
                 use_maxpool=True,
                 activation_function=nn.ReLU(inplace=True)
                ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if stem_weight is not None:
            # ckpt = torch.load(stem_weight, map_location='cpu')
            self.load_state_dict(stem_weight, strict=False)
            print("!!!Load weights for segmentation stem layer!!!")
        
        self.activation = activation_function
            
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


def make_fcn_head(in_channels, inter_channels, dropout_ratio, num_classes, activation, bias=False):
    layers = [
        nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=bias),
        nn.BatchNorm2d(inter_channels),
        activation,
        nn.Dropout(dropout_ratio),
        nn.Conv2d(inter_channels, num_classes, 1, bias=bias)
    ]
    
    return nn.Sequential(*layers)


class SegAbsHead(object):
    def __init__(self) -> None:
        pass
    
    
    def compute_losses(self):
        pass


class DeepLapHead(nn.Module):
    # def __init__(self, inplanes, num_classes, rate=12):
    def __init__(self, num_classes, dataset, rate, in_channels=2048, dropout_ratio=0.5,
                 activation_function=nn.ReLU(inplace=True)):
        super(DeepLapHead, self).__init__()
        self.activation_function = activation_function
        self.tasks = list(num_classes.keys())
        self.rate_num = len(rate)

        # out_channels = in_channels
        out_channels = in_channels * 2
        
        for task, num_class in num_classes.items():
            is_group = True if in_channels == out_channels else False
            for t_id, r in enumerate(rate):
                setattr(self, f"{task}_{t_id}", self._make_deeplab_head(in_channels, out_channels, 
                                          dropout_ratio, r, num_class, is_group))
        
        # nn.ModuleDict({
        #     task: self._make_deeplab_head(in_channels, out_channels, 
        #                                   dropout_ratio, rate, num_class) for task, num_class in num_classes.items()
        # })
        
        
        # self.head = nn.ModuleDict({
        #     task: self._make_deeplab_head(in_channels, out_channels, 
        #                                   dropout_ratio, rate, num_class) for task, num_class in num_classes.items()
        # })
        
        # self.conv1 = nn.Conv2d(in_channels, outplanes, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=True)
        # self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=1)
        # self.conv3 = nn.Conv2d(outplanes, num_classes, kernel_size=1)
        # 
        # self.dropout = nn.Dropout(dropout_ratio)
        
        self.loss_fn = {task: get_loss_fn(dataset, task) for task in num_classes.keys()}

    
    def _make_deeplab_head(self, in_channels, out_channels, drop_ratio, rate, num_classes, is_group=False):
        if is_group:
            assert in_channels == out_channels
            group_size = in_channels
        else:
            group_size = 1
            
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, groups=group_size, padding=rate, dilation=rate, bias=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, num_classes, kernel_size=1),
            self.activation_function,
            nn.Dropout(drop_ratio),
        )
        
    
    def forward(self, feats, target=None, input_shape=480):
        assert isinstance(feats, dict) or isinstance(feats, OrderedDict)
        all_losses = {}
        
        _, x = feats.popitem()
        
        outputs = {}
        for task in self.tasks:
            t_out = None
            for t_id in range(self.rate_num):
                if t_out is None:
                    t_out = getattr(self, f"{task}_{t_id}")(x)
                else:
                    t_out += getattr(self, f"{task}_{t_id}")(x)
                    
            outputs[task] = t_out
                    
        
        # outputs = {task: head(x) for task, head in self.head.items()}
        
        if self.training:
            for task, seg_out in outputs.items():
                resized = nn.Upsample(size=input_shape, mode='bilinear', align_corners=False)(seg_out)
                all_losses.update({f"{task}_loss": self.loss_fn[task].compute_loss(resized, target[task])})
                
            return all_losses

        else:
            outputs.update({task: nn.Upsample(
                size=input_shape, mode='bilinear', align_corners=False)(out) for task, out in outputs.items()})
            return outputs
         
         
        
        
        
class FCNHead(nn.Module):
    def __init__(self, num_classes, dataset, in_channels=2048, inter_channels=None, 
                 aux_ratio=0.5, use_aux=True, aux_channel=None, dropout_ratio=0.1,
                 bias=False, activation_function=nn.ReLU(inplace=True)) -> None:
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4 if inter_channels is None else inter_channels
        self.task_list = list(num_classes.keys())
        
        task_bias = {t: bias for t in self.task_list}
        task_use_aux = {t: use_aux for t in self.task_list}
        task_dropout_ratio = {t: dropout_ratio for t in self.task_list}
        
        # if bias is None:
        #     bias = {task: False for task in num_classes.keys()}
        # if use_aux is None:
        #     use_aux = {task: False for task in num_classes.keys()}
        # if dropout_ratio is None:
        #     dropout_ratio = {task: 0.1 for task in num_classes.keys()}
        
        self.fcn_head = nn.ModuleDict({
            task: make_fcn_head(in_channels, inter_channels, task_dropout_ratio[task], num_class, 
                                activation_function, task_bias[task]) for task, num_class in num_classes.items()
                })
        
        self.aux_head = {}
        for task, aux in task_use_aux.items():
            self.aux_ratio = float(aux_ratio)
            
            if aux:
                aux_inchannels = in_channels // 2 if aux_channel is None else aux_channel
                self.aux_head[task] = make_fcn_head(aux_inchannels, aux_inchannels//4, task_dropout_ratio[task], num_classes[task], 
                                        activation_function, task_bias[task])
        
        if len(self.aux_head) > 0:
            self.aux_head = nn.ModuleDict(self.aux_head)
        
        self.loss_fn = {task: get_loss_fn(dataset, task) for task in num_classes.keys()}
        
            
    
    def forward(self, feats, target=None, input_shape=480):
        assert isinstance(feats, dict) or isinstance(feats, OrderedDict)
        all_losses = {}
        
        _, x = feats.popitem()
        
        seg_outputs = {task: head(x) for task, head in self.fcn_head.items()}
        
        if self.training:
            for task, seg_out in seg_outputs.items():
                fcn_resized = nn.Upsample(size=input_shape, mode='bilinear', align_corners=False)(seg_out)
                all_losses.update({f"{task}_out_loss": self.loss_fn[task].compute_loss(fcn_resized, target[task])})
                
            # if self.aux_head is not None:
            if len(self.aux_head) > 0:
                _, x = feats.popitem()
                
                aux_outputs = {task: head(x) for task, head in self.aux_head.items()}
                for task, aux_out in aux_outputs.items():
                    aux_resized = nn.Upsample(size=input_shape, mode='bilinear', align_corners=False)(aux_out)
                    all_losses.update({f"{task}_aux_loss": self.loss_fn[task].compute_loss(aux_resized, target[task], is_aux=True)})
            
            return all_losses

        else:
            seg_outputs.update({task: nn.Upsample(
                size=input_shape, mode='bilinear', align_corners=False)(seg_out) for task, seg_out in seg_outputs.items()})
            return seg_outputs
        

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
            # if "num_task" in cfg_dict:
            #     num_head = cfg_dict['num_task']
            # else:
            #     head = FCNHead(num_classes=num_classes, **cfg_dict)
                
            head = FCNHead(num_classes=num_classes, **cfg_dict)
        
        elif 'deeplap' in segmentor_name:
            head = DeepLapHead(num_classes=num_classes, **cfg_dict)
        
        return head
    
        
