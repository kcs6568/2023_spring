import numpy as np

import torch
import PIL
from torchvision.transforms import functional as F

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms


    def __call__(self, image, target):
        for t in self.transforms:
            # if isinstance(image, torch.Tensor):
            #     print(image.size())
            # else:
            #     print(image.size)
            image, target = t(image, target)
                
        return image, target
    
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
    
class ConvertImageDtype:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target
    

# 채널 별 mean 계산
def get_mean(dataset):
    # for i, _ in dataset:
    #     print(type(i))
    #     print(dir(i))
    #     print(type(np.asarray(i)))
    #     print(np.mean(np.asarray(i), axis=(1,2)))
    #     exit()
    meanRGB = [np.mean(np.asarray(image), axis=(1,2)) for image,_ in dataset]
    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])
    return [meanR, meanG, meanB]

# 채널 별 str 계산
def get_std(dataset):
    stdRGB = [np.std(np.asarray(image), axis=(1,2)) for image,_ in dataset]
    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])
    return [stdR, stdG, stdB]