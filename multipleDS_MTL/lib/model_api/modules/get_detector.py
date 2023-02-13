# import torch
# import torchvision
# from torchvision import prototype
# from torchvision.models.detection._utils import overwrite_eps
# from torchvision.models.detection.backbone_utils import _validate_trainable_layers
# from torchvision._internally_replaced_utils import load_state_dict_from_url

import warnings
from collections import OrderedDict
from typing import Tuple, List, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign
from .detector_api.anchor_utils import AnchorGenerator
from .detector_api.rpn import RegionProposalNetwork
from .detector_api.roi_heads import RoIHeads
from .detector_api.general_transforms import GeneralizedRCNNTransform

from torchvision.models.detection import RetinaNet


def build_detector(
    backbone_type,
    detector_name, 
    out_channels,
    num_classes,
    pretrained=False,
    progress=True,
    **kwargs):
    detector_name = detector_name.lower()
    
    if 'resnet' in backbone_type or 'resnext' in backbone_type:
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    elif 'mobilenet' in backbone_type:
        anchor_sizes = ((32, 64, 128, 256, 512, ), ) * kwargs['num_anchors']
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
    )
        
    if 'faster' in detector_name:
        if num_classes == 80 or num_classes == 91:
            model = FasterRCNN(out_channels, 
                            num_classes=num_classes, 
                            rpn_anchor_generator=rpn_anchor_generator,
                            **kwargs)
        
        elif num_classes == 20 or num_classes == 21:
            model = FasterRCNN(out_channels, 
                            num_classes=num_classes, 
                            rpn_anchor_generator=rpn_anchor_generator,
                            **kwargs)
            
    elif 'retina' in detector_name:
        raise TypeError("RetinaNet is needed implementation.")
        model = RetinaNet(num_classes, **kwargs)
        
    return model


# def transform_data(min_size=800, max_size=1333, image_mean=None, image_std=None):
#     if image_mean is None:
#         image_mean = [0.485, 0.456, 0.406]
#     if image_std is None:
#         image_std = [0.229, 0.224, 0.225]
        
#     transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    
class DetStem(nn.Module):
    def __init__(self,
                 out_channels=64,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 stem_weight=None,
                 freeze_bn=False,
                 min_size=800, max_size=1333, 
                 image_mean=None, image_std=None,
                 use_maxpool=True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False)
        # self.bn = nn.BatchNorm2d(out_channels)
        bn = misc_nn_ops.FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
        self.bn = bn(out_channels)
        self.activation = nn.ReLU(inplace=True)
            
        if use_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = None
        
        if stem_weight:
            ckpt = torch.load(stem_weight)
            self.load_state_dict(ckpt)
        
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        
        
    def forward(self, images, targets=None):
        '''
        - images (list[Tensor]): images to be processed
        - targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        '''
        
        images, _ = self.transform(images)
        x = self.conv(images.tensors)
        x = self.bn(x)
        x = self.activation(x)
        
        if self.maxpool:
            x = self.maxpool(x)
        
        return x
        

class FasterRCNN(nn.Module):
    def __init__(self, out_channels, num_classes=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 rpn_score_thresh=0.0,
                 # Box parameters
                 box_roi_pool=None, box_head=None, box_predictor=None,
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 **kwargs) -> None:
        super().__init__()
        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))
        
        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError("num_classes should be None when box_predictor is specified")
        else:
            if box_predictor is None:
                raise ValueError("num_classes should not be None when box_predictor "
                                 "is not specified")

        if rpn_anchor_generator is None:
            anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios
            )
        
        relu_type = kwargs['relu_type'] if 'relu_type' in kwargs else None
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels, rpn_anchor_generator.num_anchors_per_location()[0], relu=relu_type
            )

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
            score_thresh=rpn_score_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign( # same as fast-rcnn
                featmap_names=['0', '1', '2', '3'],
                output_size=7,
                sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead( # same as fast-rcnn
                out_channels * resolution ** 2,
                representation_size, relu=relu_type)
            
        if box_predictor is None:
            representation_size = 1024
            box_predictor = FastRCNNPredictor( # same as fast-rcnn
                representation_size,
                num_classes)

        self.roi_heads = RoIHeads(
            # Box
            box_roi_pool, box_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)
        

    def get_original_size(self, images):
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))
            
        return original_image_sizes
            
    
    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections
            
    
    # def forward(self, origins, features, origin_targets=None, trs_targets=None, trs_fn=None):        
    def forward(self, origins, features, trs_fn, origin_targets=None):
        '''
            - origins: original images (not contain target data)
            - features (Tuple(Tensor)): feature data extracted backbone
            - trs_targets: target data transformed in the detection stem layer
        '''
        
        if self.training:
            if origin_targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in origin_targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes = self.get_original_size(origins)
        trs_images, trs_targets = trs_fn(origins, origin_targets)
        
        if trs_targets is not None:
            for target_idx, target in enumerate(trs_targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        
        proposals, proposal_losses = self.rpn(trs_images, features, trs_targets)
        detections, detector_losses = self.roi_heads(features, proposals, trs_images.image_sizes, trs_targets)
        detections = trs_fn.postprocess(detections, trs_images.image_sizes, original_image_sizes)
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses = {'det_'+k: v for k, v in losses.items()}
        
        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
                
            return losses, detections
                
        else:
            return self.eager_outputs(losses, detections)
    

class ConvHeadBlock_A(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ConvHeadBlock_A, self).__init__()
        self.only_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),    
            nn.BatchNorm2d(out_channels)
        )
        
        self.two_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        
    def forward(self, x):
        x1 = self.only_1x1(x)
        x2 = self.two_conv(x)
        
        out = F.relu(x1+x2)
        
        return out
    
    
class ConvHeadBlock_B(nn.Module):
    def __init__(self, in_channels, bottleneck_channels) -> None:
        super(ConvHeadBlock_B, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out = F.relu(out + x)
        
        return out
    
    
class ConvHeadBlock_C(nn.Module):
    def __init__(self, in_channels, bottleneck_channels) -> None:
        super(ConvHeadBlock_C, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        out = F.relu(out + x)
        
        return out   




class MLPConvHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(MLPConvHead, self).__init__()
        self.cls_mlp_head = nn.Sequential(
            nn.Linear(in_channels, representation_size),
            nn.Linear(representation_size, representation_size)
        )
        

    def forward(self, x):
        x = x.flatten(start_dim=1)
        
        cls_features = self.cls_mlp_head(x)
        

        return x
    
    
class MLPConvFastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes, fc_to_conv=False):
        super(MLPConvFastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Conv2d(in_channels, num_classes * 4, kernel_size=1, stride=1)
        # if fc_to_conv:
        #     self.bbox_pred = nn.Conv2d(in_channels, num_classes * 4, kernel_size=1, stride=1)
        # else:
        #     self.bbox_pred = nn.Linear(in_channels, num_classes * 4)       

    def forward(self, x):
        # x: [1024, 1024]
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        # x = x.flatten(start_dim=1)
        # scores = self.cls_score(x)
        # bbox_deltas = self.bbox_pred(x)
        
        # x = x.flatten(start_dim=1)
        scores = self.cls_score(x.flatten(start_dim=1))
        # x = x.unsqueeze
        bbox_deltas = self.bbox_pred(x)
        bbox_deltas = torch.flatten(bbox_deltas, 1)
        
        return scores, bbox_deltas   
    
    
    
class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size, relu=None):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.relu = nn.ReLU(inplace=True)
        # self.relu = F.relu

    def forward(self, x):
        # print("TwoMLP")
        x = x.flatten(start_dim=1)
        # print(x.size())

        # x = self.relu(self.fc6(x))
        # x = self.relu(self.fc7(x))
        x = self.relu(self.fc6(x))
        # print(x.size())
        x = self.relu(self.fc7(x))
        # print(x.size())

        return x
    
    
class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)


    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        # print("FasterRCNNPredictor")
        # print(x.size())
        x = x.flatten(start_dim=1)
        # print(x.size())
        scores = self.cls_score(x)
        # print(scores.size())
        bbox_deltas = self.bbox_pred(x)
        # print(bbox_deltas.size())

        return scores, bbox_deltas
    
    
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
    """

    def __init__(self, in_channels, num_anchors, relu=None):
        super(RPNHead, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=1, stride=1
        )
        for layer in self.children():
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
            
        self.relu = nn.ReLU(inplace=True)
            
        # self.relu = nn.ReLU(inplace=True) if relu is None else relu
        # self.relu = F.relu

    def forward(self, x):
        logits = []
        bbox_reg = []
        for feature in x:
            # t = self.relu(self.conv(feature))
            t = self.relu(self.conv(feature))
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg
    
    