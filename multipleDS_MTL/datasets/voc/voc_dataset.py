import os
import collections
from PIL import Image

from torch.utils.data import Dataset

from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse


class VOCDataset(Dataset):
    _DATA_ROOT = '/root/data/mmdataset/VOCdevkit'
    _VALID_YEARS = ['2007', '2012', '0712']
    _VALID_TASK = ['train', 'val', 'trainval', 'test']
        
    _IMAGE_TXT_DIR: str
    _TARGET_DIR: str
    _TARGET_FILE_EXT: str
    
    
    def __init__(self, year, task, transform=None):
        super().__init__()
        self.root = self._DATA_ROOT
        self.task = task
        self.transform = transform
        self.num_classes = 21
        
        assert year in self._VALID_YEARS
        assert self.task in self._VALID_TASK
        voc_root = [os.path.join(self.root, 'VOC'+year)]
        if year == '0712':
            voc_root = [
                os.path.join(self.root, 'VOC2007'),
                os.path.join(self.root, 'VOC2012')]
        
        self.images = []
        self.targets = []
        
        images07 = []
        images12 = []
        
        
        image_size = 0
        target_size = 0
        for v_root in voc_root:
            image_dir = os.path.join(v_root, "ImageSets", self._IMAGE_TXT_DIR)
            image_file = os.path.join(image_dir, self.task.rstrip("\n") + ".txt")
            with open(os.path.join(image_file), "r") as f:
                file_names = [x.strip() for x in f.readlines() if x != '\n']
                
            image_dir = os.path.join(v_root, "JPEGImages")
            images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
            
            target_dir = os.path.join(v_root, self._TARGET_DIR)
            targets = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]
            

            assert len(images) == len(targets)
            
            # if "2007" in v_root:
            #     images07.extend(targets)
            # else:
            #     images12.extend(targets)
            
            self.images.extend(images)
            self.targets.extend(targets)
        
        
        
        assert len(self.images) == len(self.targets)
        
        # print(len(images07), len(images12))
        # exit()
        
        # dup = 0
        # for im in images07:
        #     im_name = im.split("/")[-1]
        #     im_name = im_name.split(".")[0]
            
        #     print(im_name)
            
        #     for im2 in   images12:
        #         im_name2 = im2.split("/")[-1]
        #         im_name2 = im_name2.split(".")[0]
                
        #         if im_name in im_name2:
        #             # print(im_name, im_name2)
        #             dup += 1
        
        # print(dup)
        
        # exit()
        
    def __len__(self) -> int:
        return len(self.images)
    
    
class VOCSegmentation(VOCDataset):
    _IMAGE_TXT_DIR = "Segmentation"
    _TARGET_DIR = "SegmentationClass"
    _TARGET_FILE_EXT = ".png"

    @property
    def masks(self):
        return self.targets

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


class VOCDetection(VOCDataset):
    _IMAGE_TXT_DIR = "Main"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"
    
    DET_CLASSES = {
        'aeroplane:': 1, 
        'bicycle:': 2,
        'bird:': 3,
        'boat:': 4,
        'bottle:': 5,
        'bus:': 6,
        'car:': 7,
        'cat:': 8,
        'chair:': 9,
        'cow:': 10,
        'diningtable:': 11,
        'dog:': 12,
        'horse:': 13,
        'motorbike:': 14,
        'person:': 15,
        'pottedplant:': 16,
        'sheep:': 17,
        'sofa:': 18,
        'train:': 19,
        'tvmonitor:': 20
    }
    
    
    # ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    #            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    #            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
    #            'tvmonitor')

    @property
    def annotations(self):
        return self.targets

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def parse_voc_xml(self, node: ET_Element):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict    
