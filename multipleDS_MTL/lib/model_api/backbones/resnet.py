from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
import torchvision
from torch import Tensor

from torchvision.ops import misc as misc_nn_ops
from torchvision.transforms._presets import ImageClassification
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._api import WeightsEnum, Weights
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param

__all__ = [
    "ResNet",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "Wide_ResNet50_2_Weights",
    "Wide_ResNet101_2_Weights",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False,
                 dilation=1,
                 groups=1):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, 
            dilation: int = 1, deform=False, bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    
    if deform:
        return DeformableConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_function = nn.ReLU(inplace=True),
        use_bias=False
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation, bias=use_bias)
        self.bn1 = norm_layer(planes)
        self.activation = activation_function
        self.conv2 = conv3x3(planes, planes, dilation=dilation, bias=use_bias)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        return out


class IdentityBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
         self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_function = nn.ReLU(inplace=True),
        use_bias=False
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, use_bias)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, use_bias)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.activation = activation_function

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out



class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_function = nn.ReLU(inplace=True),
        use_bias=False
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, bias=use_bias)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, bias=use_bias)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, bias=use_bias)
        self.bn3 = norm_layer(planes * self.expansion)
        self.activation = activation_function
        
        self.downsample = downsample
        self.stride = stride
        self.out_channels = planes * self.expansion

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return out
    
    

class PreActBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_function = nn.ReLU(inplace=True),
        use_bias=False
    ) -> None:
        super(PreActBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        
        self.bn1 = norm_layer(inplanes)
        self.conv1 = conv1x1(inplanes, width, bias=use_bias)
        
        self.bn2 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, bias=use_bias)
        
        self.bn3 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, bias=use_bias)
        
        self.activation = activation_function
        
        self.downsample = downsample
        self.stride = stride
        self.out_channels = planes * self.expansion


    def forward(self, x):
        out = self.bn1(x)
        out = self.activation(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.activation(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.activation(out)
        out = self.conv3(out)
        
        return out



class IdentityBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_function = nn.ReLU(inplace=True)
    ) -> None:
        super(IdentityBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation,)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.activation = activation_function    
        self.downsample = downsample
        self.stride = stride
        self.out_channels = planes * self.expansion

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            nn.utils.memory_format
            norm_layer = nn.BatchNorm2d
        
        # else:
        #     if norm_layer == "fbn":
        #         norm_layer = misc_nn_ops.FrozenBatchNorm2d
        #     elif norm_layer == "bn":
        #         norm_layer = nn.BatchNorm2d
        #     elif norm_layer == "layer":
        #         norm_layer = nn.LayerNorm
        #     elif norm_layer == "instance":
        #         norm_layer = nn.InstanceNorm
        #     else:
        #         raise ValueError(f"not enrolled normalization method ({norm_layer})")
            
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        activation_function = kwargs['activation_function']
        assert activation_function is not None
        self.bias = kwargs['use_bias']

        self.in_channels = [64, 128, 256, 512]
        self.layer1 = self._make_layer(block, self.in_channels[0], layers[0],
                                       activation_function=activation_function)
        self.layer2 = self._make_layer(block, self.in_channels[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       activation_function=activation_function)
        self.layer3 = self._make_layer(block, self.in_channels[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       activation_function=activation_function)
        self.layer4 = self._make_layer(block, self.in_channels[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       activation_function=activation_function)
        
        # self.last_out_channel = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
                    
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False,
                    activation_function=nn.ReLU(inplace=True)) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, self.bias),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            # self.base_width, previous_dilation, norm_layer, 
                            self.base_width, self.dilation, norm_layer, 
                            activation_function=activation_function, use_bias=self.bias))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, 
                                activation_function=activation_function, use_bias=self.bias))

        return nn.Sequential(*layers)

    
    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            print('ResNet unknown block error !!!')

        return [bn1, bn2, bn3, bn4]
    
    
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    

_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class ResNet34_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet34-b627a593.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 21797672,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 73.314,
                    "acc@5": 91.420,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


class ResNet50_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet50-0676ba61.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.130,
                    "acc@5": 92.862,
                }
            },
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        url="https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        transforms=partial(ImageClassification, crop_size=224, resize_size=232),
        meta={
            **_COMMON_META,
            "num_params": 25557032,
            "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 80.858,
                    "acc@5": 95.434,
                }
            },
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    weight_path,
    progress: bool,
    **kwargs: Any
) -> ResNet:

# def _resnet(
#     block: Type[Union[BasicBlock, Bottleneck]],
#     layers: List[int],
#     weights: Optional[WeightsEnum],
#     progress: bool,
#     **kwargs: Any,
# ) -> ResNet:

    model = ResNet(block, layers, **kwargs)
    
    # if pretrained:
    #     if weight_path is not None:
    #         state_dict = torch.load('/root/volume/pretrained_weights/resnet50_IM1K_body.pth')
    #         model.load_state_dict(state_dict, strict=True)
    #         print("!!!Loaded pretrained body weights!!!")
    #     else:
    #         print("!!!Not load pretrained body weights!!!")
    
    # if weights is not None:
    #     print("2")
    #     model.load_state_dict(weights.get_state_dict(progress=progress), strict=True)
    #     print(3)
            
    return model


def get_resnet(model, weight_path=None, pretrained=True, **kwargs):
    if model == 'resnet18':
        return resnet18(pretrained=pretrained, weight_path=weight_path, **kwargs)
    elif model == 'resnet34':
        return resnet34(pretrained=pretrained, weight_path=weight_path, **kwargs)
    elif model == 'resnet50':
        return resnet50(pretrained=pretrained, weight_path=weight_path, **kwargs)
        # return resnet50(pretrained, **kwargs)
    elif model == 'resnet101':
        return resnet101(pretrained=pretrained, weight_path=weight_path, **kwargs)
    elif model == 'resnet152':
        return resnet152(pretrained=pretrained, weight_path=weight_path, **kwargs)

    elif model == 'resnext50_32x4d':
        return resnext50_32x4d(pretrained=pretrained, weight_path=weight_path, **kwargs)
    
    elif model == 'wide_resnet50_2':
        return wide_resnet50_2(pretrained=pretrained, weight_path=weight_path, **kwargs)
    

def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, weight_path=None, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    
    bottleneck_type = kwargs['bottleneck_type']
    
    if bottleneck_type == 'default':
        block = BasicBlock
    elif bottleneck_type == 'identity':
        block = IdentityBasicBlock
    elif bottleneck_type == 'preact':
        block = PreActBottleneck
    
    weights = ResNet34_Weights.verify(weights)

    return _resnet(block, [3, 4, 6, 3], weights, progress, **kwargs)
    
    
    
    # return _resnet(block, [3, 4, 6, 3], pretrained, weight_path, progress,
    #                **kwargs)


# @handle_legacy_interface(weights=("pretrained", ResNet50_Weights.IMAGENET1K_V1))
# def resnet50(*, weights: Optional[ResNet50_Weights] = None, progress: bool = True, **kwargs: Any) -> ResNet:
def resnet50(pretrained: bool = False, weight_path=None, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    bottleneck_type = kwargs['bottleneck_type']
    
    if bottleneck_type == 'default':
        block = Bottleneck
    elif bottleneck_type == 'identity':
        block = IdentityBottleneck
    # elif bottleneck_type == 'preact':
    #     block = PreActBottleneck
    
    # weights = ResNet50_Weights.verify(weights)
    # return _resnet(block, [3, 4, 6, 3], weights, progress, **kwargs)
    
    return _resnet(block, [3, 4, 6, 3], pretrained, weight_path, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, weight_path=None, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    
    return _resnet(Bottleneck, [3, 4, 6, 3], pretrained, weight_path, progress,
                   **kwargs)

    
def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)    
    
    
    
    
def setting_resnet_args(args, model_args):
    if 'dilation_type' in args:
        model_args.update({'dilation_type': args.dilation_type})   
    
    if 'relu_type' in args:
        model_args.update({'relu_type': args.relu_type})   
    
    if 'train_specific_layers' in args:
        model_args.update({'train_specific_layers': args.train_specific_layers})
        
    if 'use_awl' in args:
        model_args.update({'use_awl': args.use_awl})
    
    return model_args


