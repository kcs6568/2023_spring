# import json
# import os
# from collections import namedtuple

# import torch
# import torch.utils.data as data
# from PIL import Image
# import numpy as np


# class CityScapes(data.Dataset):
#     """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
#     **Parameters:**
#         - **root** (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
#         - **split** (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
#         - **mode** (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
#         - **transform** (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
#         - **target_transform** (callable, optional): A function/transform that takes in the target and transforms it.
#     """

#     # Based on https://github.com/mcordts/cityscapesScripts
#     CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
#                                                      'has_instances', 'ignore_in_eval', 'color'])
#     classes = [
#         CityscapesClass('unlabeled',            0, 255, 'void', 0, False, True, (0, 0, 0)),
#         CityscapesClass('ego vehicle',          1, 255, 'void', 0, False, True, (0, 0, 0)),
#         CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
#         CityscapesClass('out of roi',           3, 255, 'void', 0, False, True, (0, 0, 0)),
#         CityscapesClass('static',               4, 255, 'void', 0, False, True, (0, 0, 0)),
#         CityscapesClass('dynamic',              5, 255, 'void', 0, False, True, (111, 74, 0)),
#         CityscapesClass('ground',               6, 255, 'void', 0, False, True, (81, 0, 81)),
#         CityscapesClass('road',                 7, 0, 'flat', 1, False, False, (128, 64, 128)),
#         CityscapesClass('sidewalk',             8, 1, 'flat', 1, False, False, (244, 35, 232)),
#         CityscapesClass('parking',              9, 255, 'flat', 1, False, True, (250, 170, 160)),
#         CityscapesClass('rail track',           10, 255, 'flat', 1, False, True, (230, 150, 140)),
#         CityscapesClass('building',             11, 2, 'construction', 2, False, False, (70, 70, 70)),
#         CityscapesClass('wall',                 12, 3, 'construction', 2, False, False, (102, 102, 156)),
#         CityscapesClass('fence',                13, 4, 'construction', 2, False, False, (190, 153, 153)),
#         CityscapesClass('guard rail',           14, 255, 'construction', 2, False, True, (180, 165, 180)),
#         CityscapesClass('bridge',               15, 255, 'construction', 2, False, True, (150, 100, 100)),
#         CityscapesClass('tunnel',               16, 255, 'construction', 2, False, True, (150, 120, 90)),
#         CityscapesClass('pole',                 17, 5, 'object', 3, False, False, (153, 153, 153)),
#         CityscapesClass('polegroup',            18, 255, 'object', 3, False, True, (153, 153, 153)),
#         CityscapesClass('traffic light',        19, 6, 'object', 3, False, False, (250, 170, 30)),
#         CityscapesClass('traffic sign',         20, 7, 'object', 3, False, False, (220, 220, 0)),
#         CityscapesClass('vegetation',           21, 8, 'nature', 4, False, False, (107, 142, 35)),
#         CityscapesClass('terrain',              22, 9, 'nature', 4, False, False, (152, 251, 152)),
#         CityscapesClass('sky',                  23, 10, 'sky', 5, False, False, (70, 130, 180)),
#         CityscapesClass('person',               24, 11, 'human', 6, True, False, (220, 20, 60)),
#         CityscapesClass('rider',                25, 12, 'human', 6, True, False, (255, 0, 0)),
#         CityscapesClass('car',                  26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
#         CityscapesClass('truck',                27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
#         CityscapesClass('bus',                  28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
#         CityscapesClass('caravan',              29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
#         CityscapesClass('trailer',              30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
#         CityscapesClass('train',                31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
#         CityscapesClass('motorcycle',           32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
#         CityscapesClass('bicycle',              33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
#         CityscapesClass('license plate',        -1, 255, 'vehicle', 7, False, True, (0, 0, 142)),
#     ]

#     train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
#     train_id_to_color.append([0, 0, 0])
#     train_id_to_color = np.array(train_id_to_color)
#     # id_to_train_id = np.array([c.train_id for c in classes])
    
#     ignore_index = 255
#     id_to_train_id = {
#         -1: ignore_index, 
#          0: ignore_index,
#          1: ignore_index,
#          2: ignore_index,
#          3: ignore_index,
#          4: ignore_index, 
#          5: ignore_index,
#          6: ignore_index,
#          7: 0,
#          8: 1,
#          9: ignore_index,
#         10: ignore_index,
#         11: 2,
#         12: 3,
#         13: 4,
#         14: ignore_index,
#         15: ignore_index,
#         16: ignore_index,
#         17: 5, 
#         18: ignore_index,
#         19: 6,
#         20: 7,
#         21: 8,
#         22: 9,
#         23: 10,
#         24: 11,
#         25: 12,
#         26: 13,
#         27: 14,
#         28: 15,
#         29: ignore_index,
#         30: ignore_index,
#         31: 16,
#         32: 17,
#         33: 18}
    
#     # train_id_to_color = [(0, 0, 0), (128, 64, 128), (70, 70, 70), (153, 153, 153), (107, 142, 35),
#     #                      (70, 130, 180), (220, 20, 60), (0, 0, 142)]
#     # train_id_to_color = np.array(train_id_to_color)
#     # id_to_train_id = np.array([c.category_id for c in classes], dtype='uint8') - 1

#     def __init__(self, root, split='train', mode='fine', target_type='semantic', transform=None):
#         self.root = os.path.expanduser(root)
#         self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
#         self.target_type = target_type
#         self.images_dir = os.path.join(self.root, 'leftImg8bit', split)

#         self.targets_dir = os.path.join(self.root, self.mode, split)
#         self.transform = transform

#         self.split = split
#         self.images = []
#         self.targets = []
        
#         if not isinstance(target_type, list):
#             self.target_type = [target_type]

#         if split not in ['train', 'test', 'val']:
#             raise ValueError('Invalid split for mode! Please use split="train", split="test"'
#                              ' or split="val"')

#         if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
#             raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
#                                ' specified "split" and "mode" are inside the "root" directory')

#         for city in os.listdir(self.images_dir):
#             img_dir = os.path.join(self.images_dir, city)
#             target_dir = os.path.join(self.targets_dir, city)
            
#             for file_name in os.listdir(img_dir):
#                 target_types = []
#                 for t in self.target_type:
#                     target_name = "{}_{}".format(
#                         file_name.split("_leftImg8bit")[0], self._get_target_suffix(self.mode, t)
#                     )
#                     target_types.append(os.path.join(target_dir, target_name))

#                 self.images.append(os.path.join(img_dir, file_name))
#                 self.targets.append(target_types)

            
#             # for file_name in os.listdir(img_dir):
#             #     self.images.append(os.path.join(img_dir, file_name))
#             #     target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
#             #                                  self._get_target_suffix(self.mode, self.target_type))
#             #     self.targets.append(os.path.join(target_dir, target_name))

#     @classmethod
#     def encode_target(cls, target):
#         return cls.id_to_train_id[np.array(target)]

#     @classmethod
#     def decode_target(cls, target):
#         target[target == 255] = 19
#         #target = target.astype('uint8') + 1
#         return cls.train_id_to_color[target]

#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#         Returns:
#             tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
#             than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
#         """
        
#         image = Image.open(self.images[index]).convert("RGB")

#         targets = []
#         for i, t in enumerate(self.target_type):
#             if t == "polygon":
#                 target = self._load_json(self.targets[index][i])
#             else:
#                 target = Image.open(self.targets[index][i])

#             targets.append(target)

#         target = tuple(targets) if len(targets) > 1 else targets[0]

#         if self.transform is not None:
#             image, target = self.transform(image, target)
        
#         for k, v in self.id_to_train_id.items():
#             target[target == k] = v
            
#         # target = (self.encode_target(target) for target in targets) if len(targets) > 1 else self.encode_target(target)
#         return image, target
        
#         # image = Image.open(self.images[index]).convert('RGB')
#         # target = Image.open(self.targets[index])
#         # if self.transform:
#         #     image, target = self.transform(image, target)
            
#         # target = self.encode_target(target)
#         # return image, target

#     def __len__(self):
#         return len(self.images)

#     def _load_json(self, path):
#         with open(path, 'r') as file:
#             data = json.load(file)
#         return data

#     def _get_target_suffix(self, mode, target_type):
#         if target_type == 'instance':
#             return '{}_instanceIds.png'.format(mode)
#         elif target_type == 'semantic':
#             return '{}_labelIds.png'.format(mode)
#         elif target_type == 'color':
#             return '{}_color.png'.format(mode)
#         elif target_type == 'polygon':
#             return '{}_polygons.json'.format(mode)
#         elif target_type == 'depth':
            # return '{}_disparity.png'.format(mode)


import torch 
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class CityScapes(Dataset):
    """
    num_classes: 19
    """
    
    CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 
                'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

    PALETTE = torch.tensor([[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], 
                [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    ID2TRAINID = {
        0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0, 8: 1, 9: 255, 10: 255,
        11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255, 19: 6, 20: 7,
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
        31: 16, 32: 17, 33: 18, -1: 255}
    
    
    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.label_map = np.arange(256)
        for id, trainid in self.ID2TRAINID.items():
            self.label_map[id] = trainid

        img_path = Path(root) / 'leftImg8bit' / split
        self.files = list(img_path.rglob('*.png'))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('leftImg8bit', 'gtFine').replace('.png', '_labelIds.png')

        # image = io.read_image(img_path)
        # label = io.read_image(lbl_path)
        
        image = Image.open(img_path).convert('RGB')
        label = Image.open(lbl_path)
        
        if self.transform:
            image, label = self.transform(image, label)
        return image, self.encode(label.numpy()).long()

    def encode(self, label: Tensor) -> Tensor:
        label = self.label_map[label]
        return torch.from_numpy(label)
    
    
    
    
    
import os
import json
import numpy as np
import torch
import random
import cv2
from copy import deepcopy
from torchvision import transforms
import pdb


class CityScapes(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, crop_h=None, crop_w=None, num_class=19, small_res=False, opt=None):
        print(self.name())
        json_file = os.path.join(dataroot, 'cityscape.json')
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.dataroot = dataroot
        self.opt = opt
        # FIXIT: debug changes
        # if mode == 'test':
        #     self.groups = info['train']
        # else:
        self.groups = info[mode]
        if crop_h is not None and crop_w is not None:
            self.crop_h = crop_h
            self.crop_w = crop_w
        else:
            if small_res:
                self.crop_h = 128
                self.crop_w = 256
            else:
                self.crop_h = 256
                self.crop_w = 512
        self.mode = mode
        # self.transform = transforms.ToTensor()
        # IMG MEAN is in BGR order
        self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.IMG_MEAN = np.tile(self.IMG_MEAN[np.newaxis, np.newaxis, :], (self.crop_h, self.crop_w, 1))
        self.num_class = num_class

    def __len__(self):
        return len(self.groups)
        # return 16
        # return 6

    @staticmethod
    def __scale__(img, depth, label2, label7, label19):
        """
           Randomly scales the images between 0.5 to 1.5 times the original size.
        """
        # random value between 0.5 and 1.5
        scale = random.random() + 0.5
        h, w, _ = img.shape
        h_new = int(h * scale)
        w_new = int(w * scale)
        img_new = cv2.resize(img, (w_new, h_new))
        depth = np.expand_dims(cv2.resize(depth, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        label2 = np.expand_dims(cv2.resize(label2, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        label7 = np.expand_dims(cv2.resize(label7, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        label19 = np.expand_dims(cv2.resize(label19, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
        return img_new, depth, label2, label7, label19

    @staticmethod
    def __mirror__(img, depth, label2, label7, label19):
        flag = random.random()
        if flag > 0.5:
            img = img[:, ::-1]
            depth = depth[:, ::-1]
            label2 = label2[:, ::-1]
            label7 = label7[:, ::-1]
            label19 = label19[:, ::-1]
        return img, depth, label2, label7, label19

    @staticmethod
    def __random_crop_and_pad_image_and_labels__(img, depth, label2, label7, label19, crop_h, crop_w, ignore_label=-1.0):
        # combining
        label = np.concatenate((label2, label7, label19), axis=2).astype('float32')
        label -= ignore_label
        combined = np.concatenate((img, depth, label), axis=2)
        image_shape = img.shape
        depth_shape = depth.shape
        # padding to the crop size
        pad_shape = [max(image_shape[0], crop_h), max(image_shape[1], crop_w), combined.shape[-1]]
        combined_pad = np.zeros(pad_shape)
        offset_h, offset_w = (pad_shape[0] - image_shape[0])//2, (pad_shape[1] - image_shape[1])//2
        combined_pad[offset_h: offset_h+image_shape[0], offset_w: offset_w+image_shape[1]] = combined
        # cropping
        crop_offset_h, crop_offset_w = pad_shape[0] - crop_h, pad_shape[1] - crop_w
        start_h, start_w = np.random.randint(0, crop_offset_h+1), np.random.randint(0, crop_offset_w+1)
        combined_crop = combined_pad[start_h: start_h+crop_h, start_w: start_w+crop_w]
        # separating
        img_cdim = image_shape[-1]
        img_crop = deepcopy(combined_crop[:, :, :img_cdim])
        depth_cdim = depth_shape[-1]
        depth_crop = deepcopy(combined_crop[:, :, img_cdim: img_cdim + depth_cdim]).astype('float')
        label_crop = combined_crop[:, :, img_cdim + depth_cdim:]
        label_crop = label_crop + ignore_label
        label2_crop = np.expand_dims(label_crop[:, :, 0].astype('int'), axis=-1)
        label7_crop = np.expand_dims(label_crop[:, :, 1].astype('int'), axis=-1)
        label19_crop = np.expand_dims(label_crop[:, :, 2].astype('int'), axis=-1)

        return img_crop, depth_crop, label2_crop, label7_crop, label19_crop

    def __getitem__(self, item):
        # TODO RGB -> BGR
        img_path, depth_path, label2_path, label7_path, label19_path = self.groups[item]
        img = np.load(os.path.join(self.dataroot, img_path))[:, :, ::-1] * 255
        depth = np.load(os.path.join(self.dataroot, depth_path))
        label2 = np.expand_dims(np.load(os.path.join(self.dataroot, label2_path)), axis=-1)
        label7 = np.expand_dims(np.load(os.path.join(self.dataroot, label7_path)), axis=-1)
        label19 = np.expand_dims(np.load(os.path.join(self.dataroot, label19_path)), axis=-1)
        img, depth, label2, label7, label19 = self.__scale__(img, depth, label2, label7, label19)
        img, depth, label2, label7, label19 = self.__mirror__(img, depth, label2, label7, label19)
        img, depth, label2, label7, label19 = self.__random_crop_and_pad_image_and_labels__(img, depth, label2, label7, label19, self.crop_h, self.crop_w)

        img = img.astype('float')
        img -= self.IMG_MEAN
        name = img_path.split('/')[-1]

        if self.num_class == 2:
            seg =  torch.from_numpy(label2).permute(2, 0, 1)
        elif self.num_class == 7:
            seg =  torch.from_numpy(label7).permute(2, 0, 1)
        elif self.num_class == 19:
            seg =  torch.from_numpy(label19).permute(2, 0, 1)
        elif self.num_class == -1:
            seg = torch.from_numpy(label19).permute(2, 0, 1)
        else:
            raise ValueError('%d class is not supported in Cityscapes' % self.num_class)

        batch = {'img': torch.from_numpy(img).permute(2, 0, 1).float(), 'depth': torch.from_numpy(depth).permute(2, 0, 1).float(),
                'label2': torch.from_numpy(label2).permute(2, 0, 1), 'label7': torch.from_numpy(label7).permute(2, 0, 1),
                'label19': torch.from_numpy(label19).permute(2, 0, 1), 'seg': seg, 'name': name}

        img_id = name.split('.')[0]
        if self.opt is not None:
            policy_dir = os.path.join(self.opt['paths']['result_dir'], self.opt['exp_name'], 'policy')
            for t_id, task in enumerate(self.opt['tasks']):
                task_policy_dir = os.path.join(policy_dir, task)
                policy_path = os.path.join(task_policy_dir, img_id + '.npy')
                policy = np.load(policy_path)
                batch['%s_policy' % task] = torch.from_numpy(policy)

        return batch

    def name(self):
        return 'CityScapes'
