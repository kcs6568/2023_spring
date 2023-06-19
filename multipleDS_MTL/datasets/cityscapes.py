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
    def __init__(self, dataroot, mode, cfg):
        json_file = os.path.join(dataroot, 'cityscape.json')
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.dataroot = dataroot
        
        available_keys = ["image"]
        if "sseg" in cfg["num_classes"]:
            self.sseg_classes = cfg["num_classes"]["sseg"]
            self.sseg_classes_str = f"label_{self.sseg_classes}"
            available_keys.append(self.sseg_classes_str)
        
        if "depth" in cfg["num_classes"]:
            available_keys.append("depth")
        
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
            for path_list in info["test"]:
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
            # Adashare offical paper
            self.crop_h = 256
            self.crop_w = 512
            
            # if cfg["small_res"]:
            #     self.crop_h = 128
            #     self.crop_w = 256
            # else:
            #     self.crop_h = 256
            #     self.crop_w = 512
        
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
        
        return batch
        

    @staticmethod
    def __mirror__(batch):
        flag = random.random()
        if flag > 0.5:
            batch["img"] = batch["img"][:, ::-1]
            
            if "sseg" in batch:
                batch["sseg"] = batch["sseg"][:, ::-1]
            if "depth" in batch:
                batch["depth"] = batch["depth"][:, ::-1]
            
        return batch


    @staticmethod
    def __random_crop_and_pad_image_and_labels__(batch, crop_h, crop_w, ignore_label=-1.0):
        assert "img" in batch
        assert "sseg" in batch or "depth" in batch
        
        # combining
        image_shape = batch["img"].shape
        concat_data = [batch["img"]]
        
        img_cdim = 0
        depth_cdim = 0
        
        if "sseg" in batch:
            label = batch["sseg"].astype('float32')
            label -= ignore_label
            concat_data.append(label)
        
        if "depth" in batch:
            concat_data.append(batch["depth"])
            depth_shape =  batch["depth"].shape
            
        combined = np.concatenate((tuple(concat_data)), axis=2)
        
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
        batch["img"] = deepcopy(combined_crop[:, :, :img_cdim])
        
        if "sseg" in batch:
            label_crop = combined_crop[:, :, img_cdim + depth_cdim:]
            label_crop = label_crop + ignore_label
            batch["sseg"] = np.expand_dims(label_crop[:, :, 0].astype('int'), axis=-1)
            
        if "depth" in batch:
            depth_cdim = depth_shape[-1]
            batch["depth"] = deepcopy(combined_crop[:, :, img_cdim: img_cdim + depth_cdim]).astype('float')
        
        return batch
    

    def __getitem__(self, item):
        # TODO RGB -> BGR
        all_pathes = self.groups[item]
        img_path = all_pathes[0]
        target_path = all_pathes[1:]
        
        img = np.load(os.path.join(self.dataroot, img_path))[:, :, ::-1] * 255
        batch = {"img": img}
        
        for tpath in target_path:
            if "depth" in tpath:
                label = np.load(os.path.join(self.dataroot, tpath))
                batch["depth"] = label
            elif "label" in tpath:
                label = np.expand_dims(np.load(os.path.join(self.dataroot, tpath)), axis=-1)    
                batch["sseg"] = label
        
        if self.mode == "train":
            batch = self.__scale__(batch)
            batch = self.__mirror__(batch)
            batch = self.__random_crop_and_pad_image_and_labels__(batch, self.crop_h, self.crop_w)
        
        batch["img"] = batch["img"].astype('float')
        batch["img"] -= self.IMG_MEAN
        
        batch = {el_name: torch.from_numpy(data).permute(2, 0, 1).float() for el_name, data in batch.items()}
        # if "sseg" in batch:
        #     batch["sseg"] = torch.from_numpy(batch["sseg"]).permute(2, 0, 1)

        img = batch.pop("img")
        
        return img, batch
        
    
    # @staticmethod
    # def __scale__(img, depth, label2, label7, label19):
    #     """
    #        Randomly scales the images between 0.5 to 1.5 times the original size.
    #     """
    #     # random value between 0.5 and 1.5
    #     scale = random.random() + 0.5
    #     h, w, _ = img.shape
    #     h_new = int(h * scale)
    #     w_new = int(w * scale)
    #     img_new = cv2.resize(img, (w_new, h_new))
    #     depth = np.expand_dims(cv2.resize(depth, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
    #     label2 = np.expand_dims(cv2.resize(label2, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
    #     label7 = np.expand_dims(cv2.resize(label7, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
    #     label19 = np.expand_dims(cv2.resize(label19, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
    #     return img_new, depth, label2, label7, label19

    # @staticmethod
    # def __mirror__(img, depth, label2, label7, label19):
    #     flag = random.random()
    #     if flag > 0.5:
    #         img = img[:, ::-1]
    #         depth = depth[:, ::-1]
    #         label2 = label2[:, ::-1]
    #         label7 = label7[:, ::-1]
    #         label19 = label19[:, ::-1]
    #     return img, depth, label2, label7, label19

    # @staticmethod
    # def __random_crop_and_pad_image_and_labels__(img, depth, label2, label7, label19, crop_h, crop_w, ignore_label=-1.0):
    #     # combining
    #     label = np.concatenate((label2, label7, label19), axis=2).astype('float32')
    #     label -= ignore_label
    #     combined = np.concatenate((img, depth, label), axis=2)
    #     image_shape = img.shape
    #     depth_shape = depth.shape
    #     # padding to the crop size
    #     pad_shape = [max(image_shape[0], crop_h), max(image_shape[1], crop_w), combined.shape[-1]]
    #     combined_pad = np.zeros(pad_shape)
    #     offset_h, offset_w = (pad_shape[0] - image_shape[0])//2, (pad_shape[1] - image_shape[1])//2
    #     combined_pad[offset_h: offset_h+image_shape[0], offset_w: offset_w+image_shape[1]] = combined
    #     # cropping
    #     crop_offset_h, crop_offset_w = pad_shape[0] - crop_h, pad_shape[1] - crop_w
    #     start_h, start_w = np.random.randint(0, crop_offset_h+1), np.random.randint(0, crop_offset_w+1)
    #     combined_crop = combined_pad[start_h: start_h+crop_h, start_w: start_w+crop_w]
    #     # separating
    #     img_cdim = image_shape[-1]
    #     img_crop = deepcopy(combined_crop[:, :, :img_cdim])
    #     depth_cdim = depth_shape[-1]
    #     depth_crop = deepcopy(combined_crop[:, :, img_cdim: img_cdim + depth_cdim]).astype('float')
    #     label_crop = combined_crop[:, :, img_cdim + depth_cdim:]
    #     label_crop = label_crop + ignore_label
    #     label2_crop = np.expand_dims(label_crop[:, :, 0].astype('int'), axis=-1)
    #     label7_crop = np.expand_dims(label_crop[:, :, 1].astype('int'), axis=-1)
    #     label19_crop = np.expand_dims(label_crop[:, :, 2].astype('int'), axis=-1)

    #     return img_crop, depth_crop, label2_crop, label7_crop, label19_crop

    # def __getitem__(self, item):
    #     # TODO RGB -> BGR
    #     img_path, depth_path, label2_path, label7_path, label19_path = self.groups[item]
    #     img = np.load(os.path.join(self.dataroot, img_path))[:, :, ::-1] * 255
    #     depth = np.load(os.path.join(self.dataroot, depth_path))
    #     label2 = np.expand_dims(np.load(os.path.join(self.dataroot, label2_path)), axis=-1)
    #     label7 = np.expand_dims(np.load(os.path.join(self.dataroot, label7_path)), axis=-1)
    #     label19 = np.expand_dims(np.load(os.path.join(self.dataroot, label19_path)), axis=-1)
    #     if self.mode in ['train', 'train1', 'train2']:
    #         img, depth, label2, label7, label19 = self.__scale__(img, depth, label2, label7, label19)
    #         img, depth, label2, label7, label19 = self.__mirror__(img, depth, label2, label7, label19)
    #         img, depth, label2, label7, label19 = self.__random_crop_and_pad_image_and_labels__(img, depth, label2, label7, label19, self.crop_h, self.crop_w)

    #     img = img.astype('float')
    #     img -= self.IMG_MEAN
    #     name = img_path.split('/')[-1]

    #     if self.num_class == 2:
    #         seg =  torch.from_numpy(label2).permute(2, 0, 1)
    #     elif self.num_class == 7:
    #         seg =  torch.from_numpy(label7).permute(2, 0, 1)
    #     elif self.num_class == 19:
    #         seg =  torch.from_numpy(label19).permute(2, 0, 1)
    #     elif self.num_class == -1:
    #         seg = torch.from_numpy(label19).permute(2, 0, 1)
    #     else:
    #         raise ValueError('%d class is not supported in Cityscapes' % self.num_class)

    #     batch = {'img': torch.from_numpy(img).permute(2, 0, 1).float(), 'depth': torch.from_numpy(depth).permute(2, 0, 1).float(),
    #             'label2': torch.from_numpy(label2).permute(2, 0, 1), 'label7': torch.from_numpy(label7).permute(2, 0, 1),
    #             'label19': torch.from_numpy(label19).permute(2, 0, 1), 'seg': seg, 'name': name}

    #     img_id = name.split('.')[0]
    #     if self.opt is not None:
    #         policy_dir = os.path.join(self.opt['paths']['result_dir'], self.opt['exp_name'], 'policy')
    #         for t_id, task in enumerate(self.opt['tasks']):
    #             task_policy_dir = os.path.join(policy_dir, task)
    #             policy_path = os.path.join(task_policy_dir, img_id + '.npy')
    #             policy = np.load(policy_path)
    #             batch['%s_policy' % task] = torch.from_numpy(policy)

    #     return batch

    def name(self):
        return 'CityScapes'
