import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.squeeze(x)
        x = x.view(x.size(0), -1) 
        x = self.excitation(x)
        x = x.view(x.size(0), x.size(1), 1, 1)

        return x


class SEConvBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // r, kernel_size=1, stride=1),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // r, in_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.squeeze(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class ChannelSqueeze(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.ch_squeeze = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // r, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // r, in_channels // r, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels // r),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.ch_squeeze(x)
        return x
    
    
class ChannelExcitation(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.ch_excitation = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * r, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels * r),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * r, in_channels * r, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_channels * r),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x = self.ch_excitation(x)
        return x
    
    
class MyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input