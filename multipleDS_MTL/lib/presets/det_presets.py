import torch
from lib.transforms.shared_transforms import ConvertImageDtype, Compose
import lib.transforms.det_transforms as T


class DetectionPresetTrain:
    def __init__(self, data_augmentation, hflip_prob=0.5, mean=(123.0, 117.0, 104.0)):
        if data_augmentation == "hflip":
            self.transforms = Compose(
                [
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    ConvertImageDtype(torch.float),
                ]
            )
        elif data_augmentation == "ssd":
            self.transforms = Compose(
                [
                    T.RandomPhotometricDistort(),
                    T.RandomZoomOut(fill=list(mean)),
                    T.RandomIoUCrop(),
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    ConvertImageDtype(torch.float),
                ]
            )
        elif data_augmentation == "ssdlite":
            self.transforms = Compose(
                [
                    T.RandomIoUCrop(),
                    T.RandomHorizontalFlip(p=hflip_prob),
                    T.PILToTensor(),
                    ConvertImageDtype(torch.float),
                ]
            )
        else:
            raise ValueError(f'Unknown data augmentation policy "{data_augmentation}"')

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        self.transforms = T.ToTensor()

    def __call__(self, img, target):
        return self.transforms(img, target)
