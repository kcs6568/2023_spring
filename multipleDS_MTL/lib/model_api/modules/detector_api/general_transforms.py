import math
import numpy as np
from copy import deepcopy

import torch
import torchvision

from torch import nn, Tensor
from typing import List, Tuple, Dict, Optional
import torchvision.transforms.functional as tv_F
from .image_list import ImageList
from .roi_heads import paste_masks_in_image


@torch.jit.unused
def _get_shape_onnx(image: Tensor) -> Tensor:
    from torch.onnx import operators
    return operators.shape_as_tensor(image)[-2:]


@torch.jit.unused
def _fake_cast_onnx(v: Tensor) -> float:
    # ONNX requires a tensor but here we fake its type for JIT.
    return v


def _resize_image_and_masks(image: Tensor, self_min_size: float, self_max_size: float,
                            target: Optional[Dict[str, Tensor]] = None,
                            fixed_size: Optional[Tuple[int, int]] = None,
                            ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    if torchvision._is_tracing():
        im_shape = _get_shape_onnx(image)
    else:
        im_shape = torch.tensor(image.shape[-2:])

    size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    recompute_scale_factor: Optional[bool] = None
    if fixed_size is not None:
        size = [fixed_size[1], fixed_size[0]]
    else:
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(self_min_size / min_size, self_max_size / max_size)

        if torchvision._is_tracing():
            scale_factor = _fake_cast_onnx(scale)
        else:
            scale_factor = scale.item()
        recompute_scale_factor = True

    image = torch.nn.functional.interpolate(image[None], size=size, scale_factor=scale_factor, mode='bilinear',
                                            recompute_scale_factor=recompute_scale_factor, align_corners=False)[0]

    if target is None:
        return image, target

    if "masks" in target:
        mask = target["masks"]
        mask = torch.nn.functional.interpolate(mask[:, None].float(), size=size, scale_factor=scale_factor,
                                               recompute_scale_factor=recompute_scale_factor)[:, 0].byte()
        target["masks"] = mask
    return image, target


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size: int, max_size: int, image_mean: List[float], image_std: List[float],
                 size_divisible: int = 32, fixed_size: Optional[Tuple[int, int]] = None):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        # self.image_mean = torch.as_tensor(image_mean)[:, None, None]
        # self.image_std = torch.as_tensor(image_std)[:, None, None]
        self.image_mean = torch.as_tensor(image_mean)[:, None, None]
        self.image_std = torch.as_tensor(image_std)[:, None, None]
        self.size_divisible = size_divisible
        self.fixed_size = fixed_size
        self.save_origin_size = []

    
    def set_cuda_mean_std(self):
        self.image_mean = self.image_mean.to(torch.cuda.current_device())
        self.image_std = self.image_std.to(torch.cuda.current_device())
    
    
    def forward(self,
                images: List[Tensor],
                targets: Optional[List[Dict[str, Tensor]]] = None
                ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy
        
        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
                
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            assert len(image_size) == 2
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image: Tensor) -> Tensor:
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        
        # mean = torch.tensor(self.image_mean).cuda()
        # std = torch.tensor(self.image_std).cuda()
        # print(image.size())
        # print(mean)
        # print(std)
        # print(self.image_mean.size())
        # print(self.image_std.size())
        
        # print(image.device, self.image_mean.device, self.image_std.device)
        # dtype, device = image.dtype, image.device
        # mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        # std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        
        # print(image.device, mean.device, std.device)
        
        # return (image - mean) / std
        
        self.set_cuda_mean_std()
        return (image - self.image_mean) / self.image_std

    def torch_choice(self, k: List[int]) -> int:
        """
        Implements `random.choice` via torch ops so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0., float(len(k))).item())
        return k[index]

    def resize(self,
               image: Tensor,
               target: Optional[Dict[str, Tensor]] = None,
               ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]
        if self.training:
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])
        image, target = _resize_image_and_masks(image, size, float(self.max_size), target, self.fixed_size)

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        return image, target

    # _onnx_batch_images() is an implementation of
    # batch_images() that is supported by ONNX tracing.
    @torch.jit.unused
    def _onnx_batch_images(self, images: List[Tensor], size_divisible: int = 32) -> Tensor:
        max_size = []
        for i in range(images[0].dim()):
            max_size_i = torch.max(torch.stack([img.shape[i] for img in images]).to(torch.float32)).to(torch.int64)
            max_size.append(max_size_i)
        stride = size_divisible
        max_size[1] = (torch.ceil((max_size[1].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size[2] = (torch.ceil((max_size[2].to(torch.float32)) / stride) * stride).to(torch.int64)
        max_size = tuple(max_size)

        # work around for
        # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        # which is not yet supported in onnx
        padded_imgs = []
        for img in images:
            padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
            padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
            padded_imgs.append(padded_img)

        return torch.stack(padded_imgs)

    def max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = deepcopy(the_list[0])
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    
    def match_all_batches_targets(self, images: List[Tensor], targets):
        self.save_origin_size = [list(img.shape) for img in images]
        
        max_size = self.max_by_axis(self.save_origin_size)
        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for idx, img in enumerate(images):
            if img.shape[-2:] == max_size[-2:]: batched_imgs[idx,:,:,:].copy_(img)
            else: batched_imgs[idx] = tv_F.crop(img, 0, 0, max_size[-2], max_size[-1])
            assert torch.equal(img, batched_imgs[idx, :, :img.shape[-2], :img.shape[-1]])
        
        self.keys = list(targets[0].keys())
        self.origin_size = [{k: v.size() for k, v in d.items()} for d in targets]

        gt_sizes = np.array([[len(src[k])for k in self.keys] for src in targets])
        each_max = np.max(gt_sizes, axis=0) # axis=0: row
        
        all_new_samples = []
        for idx, k in enumerate(self.keys):
            all_sample = [anno[k] for anno in targets]
            # if len(all_sample[0].shape) == 2: shape = [len(targets), each_max[idx], all_sample[0].shape[-1]]
            if len(all_sample[0].shape) > 1:
                shape = [len(targets), each_max[idx]]
                shape.extend(list(all_sample[0].shape[1:]))
            else: shape = [len(targets), each_max[idx]]
            
            copied = targets[0][k].new_full(shape, 0)
            for d_idx, d in enumerate(all_sample):
                size_diff = each_max[idx] - d.shape[0]
                
                
                if size_diff == 0: copied[d_idx].copy_(d)
                else:
                    if len(d.shape) == 1: copied[d_idx, :d.shape[0]].copy_(d)
                    elif len(d.shape) == 2: copied[d_idx, :d.shape[0], :d.shape[1]].copy_(d)
            all_new_samples.append(copied)
            
        return batched_imgs, all_new_samples
    
    def recover_all_batches_targets(self, images, targets, start, end):
        batch_lists = []
        for idx, shapes in enumerate(self.save_origin_size[start:end]):
            if shapes[-2:] == images[idx].shape[-2:]:
                batch_lists.append(images[idx].clone())
                assert torch.equal(images[idx], batch_lists[idx])
                
            else:
                new_sample = tv_F.crop(images[idx], 0, 0, shapes[-2], shapes[-1])
                batch_lists.append(new_sample)
                assert torch.equal(images[idx, :, :shapes[-2], :shapes[-1]], batch_lists[idx])
                    
        
        
        anno = [{} for _ in range(end-start)]
        for i, key in enumerate(self.keys):
            for j, (batch_data, shape) in enumerate(zip(targets[i], self.origin_size[start:end])):
                if shape[key][0] == batch_data.shape[0]: anno[j][key] = batch_data
                else:
                    origin_data = batch_data[:shape[key][0]]
                    anno[j][key] = origin_data
        
        return batch_lists, anno
    
    
    def match_all_batches(self, images: List[Tensor]):
        self.save_origin_size = [list(img.shape) for img in images]
        
        max_size = self.max_by_axis(self.save_origin_size)
        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        
        for idx, img in enumerate(images):
            if img.shape[-2:] == max_size[-2:]: batched_imgs[idx,:,:,:].copy_(img)
            else: batched_imgs[idx] = tv_F.crop(img, 0, 0, max_size[-2], max_size[-1])
            assert torch.equal(img, batched_imgs[idx, :, :img.shape[-2], :img.shape[-1]])
            
        return batched_imgs
    
    
    def recover_all_batches(self, images, start, end):
        batch_lists = []
        for idx, shapes in enumerate(self.save_origin_size[start:end]):
            if shapes[-2:] == images[idx].shape[-2:]:
                batch_lists.append(images[idx])
                assert torch.equal(images[idx], batch_lists[idx])
                
            else:
                new_sample = tv_F.crop(images[idx], 0, 0, shapes[-2], shapes[-1])
                batch_lists.append(new_sample)
                assert torch.equal(images[idx, :, :shapes[-2], :shapes[-1]], batch_lists[idx])
                
        return batch_lists
    
    
    def prediction_to_tensor(self, predictions, max_padding_size=50):
        self.output_keys = list(predictions[0].keys())
        
        
        
    
    
    
    
    
    def batch_images(self, images: List[Tensor], size_divisible: int = 32) -> Tensor:
        if torchvision._is_tracing():
            # batch_images() does not export well to ONNX
            # call _onnx_batch_images() instead
            return self._onnx_batch_images(images, size_divisible)

        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)


        return batched_imgs

    def postprocess(self,
                    result: List[Dict[str, Tensor]],
                    image_shapes: List[Tuple[int, int]],
                    original_image_sizes: List[Tuple[int, int]]
                    ) -> List[Dict[str, Tensor]]:
        # if self.training:
        #     return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result
    
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        _indent = '\n    '
        format_string += "{0}Normalize(mean={1}, std={2})".format(_indent, self.image_mean, self.image_std)
        format_string += "{0}Resize(min_size={1}, max_size={2}, mode='bilinear')".format(_indent, self.min_size,
                                                                                         self.max_size)
        format_string += '\n)'
        return format_string


def resize_keypoints(keypoints: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=keypoints.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=keypoints.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    if torch._C._get_tracing_state():
        resized_data_0 = resized_data[:, :, 0] * ratio_w
        resized_data_1 = resized_data[:, :, 1] * ratio_h
        resized_data = torch.stack((resized_data_0, resized_data_1, resized_data[:, :, 2]), dim=2)
    else:
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
    return resized_data


def resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device) /
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
