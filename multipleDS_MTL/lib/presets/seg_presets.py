import torch
from lib.transforms.shared_transforms import ConvertImageDtype, Compose
import lib.transforms.seg_transforms as T


class SegmentationPresetTrain:
    def __init__(self, *, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                T.RandomCrop(crop_size),
                T.PILToTensor(),
                ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )
        self.transforms = Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, *, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = Compose(
            [
                T.RandomResize(base_size, base_size),
                T.PILToTensor(),
                ConvertImageDtype(torch.float),
                T.Normalize(mean=mean, std=std),
            ]
        )


    def __call__(self, img, target):
        return self.transform(img, target)

    
    # def __repr__(self):
    #     format_string = self.__class__.__name__ + '('
    #     for t in self.transform:
    #         format_string += '\n'
    #         format_string += '    {0}'.format(t)
    #     format_string += '\n)'
    #     return format_string