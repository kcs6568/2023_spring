import os
import json
import numpy as np
import torch
import random
import cv2
from copy import deepcopy
from torchvision import transforms


class NYU_v2(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, cfg):
        json_file = os.path.join(dataroot, 'nyu_v2_3task.json')
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.dataroot = dataroot
        
        available_keys = ["img"]
        if "sseg" in cfg["num_classes"]:
            available_keys.append("seg")
        
        if "depth" in cfg["num_classes"]:
            available_keys.append("depth")
            
        if "sn" in cfg["num_classes"]:
            available_keys.append("normal_mask")
        
        self.mode = mode
        
        group = []
        if "train" in self.mode:
            for k in info.keys():
                if k == "test" or k == "val": continue
                
                for path_list in info[k]:
                    tmp = []
                    for path in path_list:
                        for ava_key in available_keys:
                            if ava_key in path:
                                tmp.append(path)
                                break
                    
                    group.append(tmp)
                    
        else:
            for path_list in info[self.mode]:
                tmp = []
                for path in path_list:
                    for ava_key in available_keys:
                        if ava_key in path:
                            tmp.append(path)
                            break
                
                group.append(tmp)
        
        self.groups = group
        
        if self.mode == "train":
            if cfg["crop_h"] is not None and cfg["crop_w"] is not None:
                self.crop_h = cfg["crop_h"]
                self.crop_w = cfg["crop_w"]
            else:
                if cfg["small_res"]:
                    self.crop_h = 128
                    self.crop_w = 256
                else:
                    self.crop_h = 256
                    self.crop_w = 512
        else:
            self.crop_h = 480
            self.crop_w = 640
            
            # if cfg["small_res"]:
            #     self.crop_h = 128
            #     self.crop_w = 256
            # else:
            #     self.crop_h = 256
            #     self.crop_w = 512
        
        
        # if cfg["crop_h"] is not None and cfg["crop_w"] is not None:
        #     self.crop_h = cfg["crop_h"]
        #     self.crop_w = cfg["crop_w"]
        # else:
        #     if cfg["small_res"]:
        #         self.crop_h = 128
        #         self.crop_w = 256
        #     else:
        #         self.crop_h = 256
        #         self.crop_w = 512
        
        # C.norm_mean = np.array([0.485, 0.456, 0.406])
        # C.norm_std = np.array([0.229, 0.224, 0.225])
        
        # self.norm_mean = np.array([0.485, 0.456, 0.406])
        # self.norm_std = np.array([0.229, 0.224, 0.225])
        
        self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.IMG_MEAN = np.tile(self.IMG_MEAN[np.newaxis, np.newaxis, :], (self.crop_h, self.crop_w, 1))

    def __len__(self):
        return len(self.groups)
        # return 16
        # return 6

    @staticmethod
    def __scale__(batch):
        """
           Randomly scales the images between 0.5 to 1.5 times the original size.
        """
        # random value between 0.5 and 1.5
        scale = random.random() + 0.5
        h, w, _ = batch["img"].shape
        h_new = int(h * scale)
        w_new = int(w * scale)
        batch["img"] = cv2.resize(batch["img"], (w_new, h_new))
        
        if "sseg" in batch:
            batch["sseg"] = np.expand_dims(cv2.resize(batch["sseg"], (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
            
        if "depth" in batch:
            batch["depth"] = np.expand_dims(cv2.resize(batch["depth"], (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)    
            
        if "sn" in batch:
            batch["sn"] = cv2.resize(batch["sn"], (w_new, h_new), interpolation=cv2.INTER_NEAREST)
        
        return batch
        

    @staticmethod
    def __normalize__(img, mean, std):
        # pytorch pretrained model need the input range: 0-1
        img = img.astype(np.float64) / 255.0
        img = img - mean
        img = img / std
        return img
    
    
    @staticmethod
    def __mirror__(batch):
        flag = random.random()
        if flag > 0.5:
            batch["img"] = batch["img"][:, ::-1]
            
            if "sseg" in batch:
                batch["sseg"] = batch["sseg"][:, ::-1]
            if "depth" in batch:
                batch["depth"] = batch["depth"][:, ::-1]
            if "sn" in batch:
                batch["sn"] = batch["sn"][:, ::-1]
            
        return batch


    @staticmethod
    def __random_crop_and_pad_image_and_labels__(batch, crop_h, crop_w, ignore_label=255):
        assert "img" in batch
        assert "sseg" in batch or "depth" in batch or "sn" in batch
        
        # img = batch["img"]
        # label1 = batch["sseg"]
        # label2 = batch["sn"]
        # label3 = batch["depth"]
        
        # label = np.concatenate((label1, label2), axis=2).astype('float32')
        # label -= ignore_label
        # combined = np.concatenate((img, label, label3), axis=2)
        # image_shape = img.shape
        # label3_shape = label3.shape
        # # padding to the crop size
        # pad_shape = [max(image_shape[0], crop_h), max(image_shape[1], crop_w), combined.shape[-1]]
        # combined_pad = np.zeros(pad_shape)
        # offset_h, offset_w = (pad_shape[0] - image_shape[0])//2, (pad_shape[1] - image_shape[1])//2
        # combined_pad[offset_h: offset_h+image_shape[0], offset_w: offset_w+image_shape[1]] = combined
        # # cropping
        # crop_offset_h, crop_offset_w = pad_shape[0] - crop_h, pad_shape[1] - crop_w
        # start_h, start_w = np.random.randint(0, crop_offset_h+1), np.random.randint(0, crop_offset_w+1)
        # combined_crop = combined_pad[start_h: start_h+crop_h, start_w: start_w+crop_w]
        # # separating
        # img_cdim = image_shape[-1]
        # label1_cdim = label1.shape[-1]
        # label3_cdim = label3_shape[-1]
        # batch["img"] = deepcopy(combined_crop[:, :, :img_cdim])
        # batch["depth"] = deepcopy(combined_crop[:, :, -label3_cdim:])
        # label_crop = combined_crop[:, :, img_cdim: -label3_cdim]
        # label_crop = (label_crop + ignore_label).astype('uint8')
        # batch["sseg"] = label_crop[:, :, :label1_cdim]
        # batch["sn"] = label_crop[:, :, label1_cdim:]
        # return batch

        
        # combining
        image_shape = batch["img"].shape
        dim3_labels = []
        shape_dict = {}
        
        if "sseg" in batch: # 1-dimension
            dim3_labels.append(batch["sseg"]-ignore_label)
            shape_dict["sseg"] = batch["sseg"].shape
            
        if "sn" in batch:
            dim3_labels.append(batch["sn"]-ignore_label)
            shape_dict["sn"] = batch["sn"].shape
            
        if "depth" in batch:
            dim3_labels.append(batch["depth"])
            shape_dict["depth"] = batch["depth"].shape
            
        if len(dim3_labels) == 1: dim3_labels = np.asarray(dim3_labels[0]).astype("float32")
        elif len(dim3_labels) > 1: dim3_labels = np.concatenate((dim3_labels), axis=2).astype("float32")
        
        combined = np.concatenate((batch["img"], dim3_labels), axis=2)
        
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
        batch["img"] = deepcopy(combined_crop[:, :, :image_shape[-1]])
        
        cdim_start = image_shape[-1]
        for task, shapes in shape_dict.items():
            cdim_end = cdim_start+shapes[-1]
            batch[task] = combined_crop[:, :, cdim_start:cdim_end]
            cdim_start += shapes[-1]
            
            if task in ["sseg", "sn"]:
                batch[task] = (batch[task] + ignore_label).astype("uint8")
                
        return batch
    

    def __getitem__(self, item):
        all_pathes = self.groups[item]
        
        img_path = all_pathes[0]
        batch = {"img": cv2.imread(os.path.join(self.dataroot, img_path))}
        
        
        
        target_path = all_pathes[1:]
        for tpath in target_path:
            if "seg" in tpath:
                label = np.expand_dims(cv2.imread(os.path.join(self.dataroot, tpath), cv2.IMREAD_GRAYSCALE), axis=-1)
                batch["sseg"] = label
            
            elif "depth" in tpath:
                label = np.expand_dims(np.load(os.path.join(self.dataroot, tpath)), axis=-1)
                batch["depth"] = label
                
            elif "normal_mask" in tpath:
                label = cv2.imread(os.path.join(self.dataroot, tpath))
                batch["sn"] = label
        
        if self.mode == "train":
            batch = self.__scale__(batch)
            batch = self.__mirror__(batch)
            batch = self.__random_crop_and_pad_image_and_labels__(batch, self.crop_h, self.crop_w)
        
        batch["img"] = batch["img"].astype('float')
        batch["img"] -= self.IMG_MEAN
        # batch["img"] = self.__normalize__(batch["img"], self.norm_mean, self.norm_std)
        
        
        batch = {el_name: torch.from_numpy(data).permute(2, 0, 1).float() for el_name, data in batch.items()}
        # if "sseg" in batch:
        #     batch["sseg"] = torch.from_numpy(batch["sseg"]).permute(2, 0, 1)

        img = batch.pop("img")
        
        
        return img, batch

    def name(self):
        return 'NYU_v2'

