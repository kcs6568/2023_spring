import os
import json
import numpy as np
import torch
import random
import cv2
from copy import deepcopy
from PIL import Image
from torchvision import transforms
import pdb


class Taskonomy(torch.utils.data.Dataset):
    def __init__(self, dataroot, mode, cfg):
        json_file = os.path.join(dataroot, 'taskonomy.json')
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.dataroot = dataroot
        
        available_keys = ["domain_rgb"]
        if "sseg" in cfg["num_classes"]:
            available_keys.append("domain_segmentsemantic")
        
        if "depth" in cfg["num_classes"]:
            available_keys.append("domain_depth")
            
        if "sn" in cfg["num_classes"]:
            available_keys.append("domain_normal")
            
        if "keypoint" in cfg["num_classes"]:
            available_keys.append("domain_keypoints2d")
            
        if "edge" in cfg["num_classes"]:
            available_keys.append("edge_texture")
        
        self.mode = mode
        group = []
        if "train" in self.mode:
            for k in info.keys():
                if "train" in k:
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
                self.crop_h = 256
                self.crop_w = 256
        else:
            self.crop_h = 256
            self.crop_w = 256
            
        self.IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        self.IMG_MEAN = np.tile(self.IMG_MEAN[np.newaxis, np.newaxis, :], (self.crop_h, self.crop_w, 1))
        self.prior_factor = np.load(os.path.join(dataroot, 'semseg_prior_factor.npy'))

        self.available_keys = available_keys

    def __len__(self):
        return len(self.groups)


    @staticmethod
    def __scale__(batch):
        """
           Randomly scales the images between 0.5 to 1.5 times the original size.
        """
        # random value between 0.5 and 1.5
        scale = random.random() + 0.5
        h, w, _ = batch["img_p"][0].shape
        h_new = int(h * scale)
        w_new = int(w * scale)
        batch["img_p"][0] = cv2.resize(batch["img_p"][0], (w_new, h_new))
        
        for task in batch.keys():
            if "sseg" in task:
                batch[task] = [np.expand_dims(cv2.resize(t, (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1) for t in batch[task]]
                # batch[task][0] = np.expand_dims(cv2.resize(batch[task][0], (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
                # batch[task][1] = np.expand_dims(cv2.resize(batch[task][1], (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)

            elif "sn" in task:
                batch[task][0] = cv2.resize(batch[task][0], (w_new, h_new), interpolation=cv2.INTER_NEAREST) 
            
            elif "depth" in task:
                batch[task][0] = np.expand_dims(cv2.resize(batch[task][0], (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
                batch[task][1] = np.expand_dims(cv2.resize(batch[task][1], (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
                
            elif "keypoint" in task:
                batch[task][0] = np.expand_dims(cv2.resize(batch[task][0], (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
                
            elif "edge" in task:
                batch[task][0] = np.expand_dims(cv2.resize(batch[task][0], (w_new, h_new), interpolation=cv2.INTER_NEAREST), axis=-1)
                
        return batch


    @staticmethod
    def __mirror__(batch):
        flag = random.random()
        if flag > 0.5:
            for task in batch.keys():
                if "sn" in task:
                    batch[task] = [batch[task][0][:, ::-1]]
                    batch[task][0][:, :, 0] *= -1
                else:
                    batch[task] = [target[:, ::-1] for target in batch[task]]
        return batch


    @staticmethod
    def __random_crop_and_pad_image_and_labels__(batch, crop_h, crop_w, c_dims, ignore_label=255):
        # combining
        # TODO: check the ignoring labels
        img = batch.pop("img_p")[0]
        
        all_targets = sum(list(batch.values()), [])
        label = np.concatenate(all_targets, axis=2).astype('float32')
        label -= ignore_label
        # label = np.concatenate((label2, label7, label19), axis=2).astype('float32')
        # label -= ignore_label
        combined = np.concatenate((img, label), axis=2)
        image_shape = img.shape
        # c_dims = [3, 1, 1, 3, 1, 1, 1, 1]
        assert (sum(c_dims) == combined.shape[2])
        
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
        batch["img_p"] = [deepcopy(combined_crop[:, :, 0: sum(c_dims[:1])])]
        combined_crop[:, :, sum(c_dims[:1]):] += ignore_label
        
        start = 1
        for k, b in batch.items():
            for i in range(len(b)):
                batch[k][i] = deepcopy(combined_crop[:, :, sum(c_dims[:start]): sum(c_dims[:start+1])])
            start += 1
        
        # # seg_crop = deepcopy(combined_crop[:, :, sum(c_dims[:1]): sum(c_dims[:2])])
        # # seg_mask_crop = deepcopy(combined_crop[:, :, sum(c_dims[:2]): sum(c_dims[:3])])
        # # sn_crop = deepcopy(combined_crop[:, :, sum(c_dims[:3]): sum(c_dims[:4])])
        # # depth_crop = deepcopy(combined_crop[:, :, sum(c_dims[:4]): sum(c_dims[:5])])
        # # depth_mask_crop = deepcopy(combined_crop[:, :, sum(c_dims[:5]): sum(c_dims[:6])])
        # # keypoint_crop = deepcopy(combined_crop[:, :, sum(c_dims[:6]): sum(c_dims[:7])])
        # # edge_crop = deepcopy(combined_crop[:, :, sum(c_dims[:7]): sum(c_dims)])
        
        # for task in batch.keys():
        #     if "seg" in task:
        #         batch[task] = [seg_crop, seg_mask_crop]
                
        #     elif "sn" in task:
        #         batch[task] = [sn_crop]
            
        #     elif "depth" in task:
        #         batch[task] = [depth_crop, depth_mask_crop]
                
        #     elif "keypoint" in task:
        #         batch[task] = [keypoint_crop]
                
        #     elif "edge" in task:
        #         batch[task] = [edge_crop]
        
        return batch

    def semantic_segment_rebalanced(self, img, new_dims=(256, 256)):
        '''
        Segmentation
        Returns:
        --------
            pixels: size num_pixels x 3 numpy array
        '''
        img = cv2.resize(img, new_dims, interpolation=cv2.INTER_NEAREST)
        mask = img > 0.1
        mask = mask.astype(float)
        img[img == 0] = 1
        img = img - 1
        rebalance = self.prior_factor[img]
        mask = mask * rebalance
        return img, mask

    @staticmethod
    def rescale_image(img, new_scale=(-1., 1.), current_scale=None, no_clip=False):
        """
        Rescales an image pixel values to target_scale

        Args:
            img: A np.float_32 array, assumed between [0,1]
            new_scale: [min,max]
            current_scale: If not supplied, it is assumed to be in:
                [0, 1]: if dtype=float
                [0, 2^16]: if dtype=uint
                [0, 255]: if dtype=ubyte
        Returns:
            rescaled_image
        """
        img = img.astype('float32')
        # min_val, max_val = img.min(), img.max()
        # img = (img - min_val)/(max_val-min_val)
        if current_scale is not None:
            min_val, max_val = current_scale
            if not no_clip:
                img = np.clip(img, min_val, max_val)
            img = img - min_val
            img /= (max_val - min_val)
        min_val, max_val = new_scale
        img *= (max_val - min_val)
        img += min_val

        return img

    def resize_rescale_image(self, img, new_scale=(-1, 1), new_dims=(256, 256), no_clip=False, current_scale=None):
        """
        Resize an image array with interpolation, and rescale to be
          between
        Parameters
        ----------
        im : (H x W x K) ndarray
        new_dims : (height, width) tuple of new dimensions.
        new_scale : (min, max) tuple of new scale.
        interp_order : interpolation order, default is linear.
        Returns
        -------
        im : resized ndarray with shape (new_dims[0], new_dims[1], K)
        """
        img = img.astype('float32')
        img = cv2.resize(img, new_dims, interpolation=cv2.INTER_NEAREST)
        img = self.rescale_image(img, new_scale, current_scale=current_scale, no_clip=no_clip)
        return img

    def resize_and_rescale_image_log(self, img, new_dims=(256, 256), offset=1., normalizer=np.log(2. ** 16)):
        """
            Resizes and rescales an img to log-linear

            Args:
                img: A np array
                offset: Shifts values by offset before taking log. Prevents
                    taking the log of a negative number
                normalizer: divide by the normalizing factor after taking log
            Returns:
                rescaled_image
        """
        img = np.log(float(offset) + img) / normalizer
        img = cv2.resize(img, new_dims, interpolation=cv2.INTER_NEAREST)
        return img

    @staticmethod
    def mask_if_channel_ge(img, threshold, channel_idx, broadcast_to_shape=None, broadcast_to_dim=None):
        '''
            Returns a mask that masks an entire pixel iff the channel
                specified has values ge a specified value
        '''
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        h, w, c = img.shape
        mask = (img[:, :, channel_idx] < threshold)  # keep if lt
        if len(mask.shape) == 2:
            mask = mask[:, :, np.newaxis].astype(np.float32)
        if broadcast_to_shape is not None:
            return np.broadcast_to(mask, broadcast_to_shape)
        elif broadcast_to_dim is not None:
            return np.broadcast_to(mask, [h, w, broadcast_to_dim])
        else:
            return np.broadcast_to(mask, img.shape)

    def make_depth_mask(self, img, new_dims=(256, 256), broadcast_to_dim=1):
        target_mask = self.mask_if_channel_ge(img, threshold=64500, channel_idx=0, broadcast_to_dim=broadcast_to_dim)
        target_mask = cv2.resize(target_mask, new_dims, interpolation=cv2.INTER_NEAREST)
        target_mask[target_mask < 0.99] = 0.
        return target_mask

    def __getitem__(self, item):
        while True:
            all_pathes = self.groups[item]
            try:
                batch = {}
                img = np.array(Image.open(os.path.join(self.dataroot, all_pathes[0]))).astype('float32')[:, :, ::-1]
                img_p = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)
                batch["img_p"] = [img_p]
                all_dims = [3]
                
                for tpath in all_pathes[1:]:
                    if "domain_segmentsemantic" in tpath:
                        seg = np.array(Image.open(os.path.join(self.dataroot, tpath)))
                        seg_p, seg_mask = self.semantic_segment_rebalanced(seg)
                        seg_p = seg_p.astype('float32')
                        # seg_mask = seg_mask.astype('float32')
                        # batch["seg"] = [seg_p, seg_mask]
                        
                        batch["sseg"] = [seg_p]
                        all_dims.append(1)
                        
                    elif "domain_normal" in tpath:
                        sn = np.array(Image.open(os.path.join(self.dataroot, tpath))).astype('float32') / 255
                        sn_p = self.resize_rescale_image(sn)
                        sn_p = sn_p.astype('float32')
                        batch["sn"] = [sn_p]
                        all_dims.append(3)
                        
                        # when depth estimation is not main task
                        # if "domain_depth" not in self.available_keys:
                        #     depth = np.array(Image.open(os.path.join(self.dataroot, tpath))).astype('float32')
                        #     depth_p = self.resize_and_rescale_image_log(depth)
                        #     depth_mask = self.make_depth_mask(depth)
                        #     depth_mask = depth_mask.astype('float32')
                        #     batch["depth"] = [depth_p, depth_mask]
                        
                    elif "domain_depth" in tpath:
                        depth = np.array(Image.open(os.path.join(self.dataroot, tpath))).astype('float32')
                        depth_p = self.resize_and_rescale_image_log(depth)
                        depth_mask = self.make_depth_mask(depth)
                        depth_mask = depth_mask.astype('float32')
                        batch["depth"] = [depth_p, depth_mask]
                        all_dims.extend([1, 1])
                        
                    elif "domain_keypoints2d" in tpath:
                        keypoint = np.array(Image.open(os.path.join(self.dataroot, tpath))).astype('float32') / (2 ** 16)
                        keypoint_p = self.resize_rescale_image(keypoint, current_scale=(0, 0.005))
                        batch["keypoint"] = [keypoint_p]
                        all_dims.append(1)
                        
                    elif "edge_texture" in tpath:
                        edge = np.array(Image.open(os.path.join(self.dataroot, tpath))).astype('float32') / (2 ** 16)
                        edge_p = self.resize_rescale_image(edge, current_scale=(0, 0.08))
                        batch["edge"] = [edge_p]
                        all_dims.append(1)
                        
            except:
                print('Error in loading %s' % tpath)
                item = 0
            else:
                break
            
        assert len(batch) >= 2

        if self.mode == "train":
            batch = self.__scale__(batch)
            batch = self.__mirror__(batch)
            batch = self.__random_crop_and_pad_image_and_labels__(batch, self.crop_h, self.crop_w, all_dims)

        img_p = batch.pop("img_p")[0]
        img_p = img_p.astype('float32')
        img_p = img_p - self.IMG_MEAN
        img_p = torch.from_numpy(img_p).permute(2, 0, 1).float()
        
        # sn_mask = np.tile(batch["depth"][1], [1, 1, 3])
        # batch["sn"].append(sn_mask)
        
        final_targets = {}
        for task, target in batch.items():
            if len(target) == 1:
                if target[0].ndim == 2:
                    target[0] = target[0][:,:,np.newaxis]
                final_targets[task] = torch.from_numpy(target[0]).permute(2, 0, 1).float()
            
            else:
                name = ["gt", "mask"]
                task_target = {}
                for t_i, t in enumerate(target):
                    if t.ndim == 2:
                        t = t[:,:,np.newaxis]
                    task_target[name[t_i]] = torch.from_numpy(t).permute(2, 0, 1).float()
                    
                final_targets[task] = task_target
        
        return img_p, final_targets
        

    def name(self):
        return 'Taskonomy'
