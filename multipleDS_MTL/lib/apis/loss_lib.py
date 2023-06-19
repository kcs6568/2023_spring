from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import roi_align



class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=4):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    
    def __str__(self) -> str:
        delimeter = " | "
        params = [f"param_{i}: {p}" for i, (p) in enumerate(self.params.data)]
        return delimeter.join(params)
    
    
    def forward(self, total_losses):
        awl_dict = OrderedDict()
        
        for i, (k, v) in enumerate(total_losses.items()):
            losses = sum(list(v.values()))
            awl_dict['awl_'+k] = \
                0.5 / (self.params[i] ** 2) * losses + torch.log(1 + self.params[i] ** 2)
        
        # awl_dict['auto_params'] = str(self)
        return awl_dict


class DiceLoss(nn.Module):
    def __init__(self, num_classes, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, inputs, targets, smooth=1, eps=1e-7):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        losses = {}
        # targets = targets.view(-1)
        
        # for name, x in inputs.items():
        #     print(name, x.size(), targets.size())
        #     continue
        #     inputs = torch.sigmoid(x)       
        #     #flatten label and prediction tensors
        #     inputs = inputs.view(-1)
            
        #     intersection = (inputs * targets).sum()                            
        #     dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
            
        #     losses[name] = 1 - dice
        #     # return 1 - dice
        
        # # exit()
        # # return losses
    

        for name, x in inputs.items():
            true_1_hot = torch.eye(self.num_classes)[targets]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = nn.functional.softmax(x, dim=1)
            true_1_hot = true_1_hot.type(x.type())
            dims = (0,) + tuple(range(2, targets.ndimension()))
            intersection = torch.sum(probas * true_1_hot, dims)
            cardinality = torch.sum(probas + true_1_hot, dims)
            dice_loss = (2. * intersection / (cardinality + eps)).mean()
        
            losses[name] = 1 - dice_loss
        
        return losses
    
    
def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction='sum',
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    """
    Args:
        proposals (list[BoxList])
        mask_logits (Tensor)
        targets (list[BoxList])

    Return:
        mask_loss (Tensor): scalar tensor containing the loss
    """

    def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
        """
        Given segmentation masks and the bounding boxes corresponding
        to the location of the masks in the image, this function
        crops and resizes the masks in the position defined by the
        boxes. This prepares the masks for them to be fed to the
        loss computation as the targets.
        """
        matched_idxs = matched_idxs.to(boxes)
        rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
        gt_masks = gt_masks[:, None].to(rois)
        return roi_align(gt_masks, rois, (M, M), 1.)[:, 0]
    
    discretization_size = mask_logits.shape[-1]
    labels = [gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    mask_loss = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels], mask_targets
    )
    return mask_loss


def cross_entropy_loss(logits, targets):
    return dict(cls_loss=F.cross_entropy(logits, targets))


def cross_entropy_loss_with_aux(logits, targets):
    losses = {}
    
    for name, x in logits.items():
        losses[name] = nn.functional.cross_entropy(x, targets, ignore_index=255)
    
    if "seg_aux_loss" in losses:
        losses["seg_aux_loss"] *= 0.5
        
    return losses


def disjointed_policy_loss(gate_logits, num_blocks, lambda_sparsity,
                           sparsity_weight=1.0, smoothing_alpha=None, return_sum=True):
    loss = 0.
    # if smoothing_alpha is not None:
    #     gt_ = torch.ones(num_blocks, 2).long().cuda()
    #     gt_[:, 0] = 0
    #     gt = torch.tensor([[l*(1-smoothing_alpha) + smoothing_alpha/len(oh) for l in oh] for i, oh in enumerate(gt_)]).float().cuda()
        
    # else:
    #     gt = torch.ones(num_blocks).long().cuda()    
    
    gt = torch.ones(num_blocks).long().cuda()
    
    
        
    # for dset in tasks:
    #     loss += F.cross_entropy(gate_logits[dset], gt)
    
    # if return_sum:    
    #     for logit in gate_logits.values():
    #         loss += F.cross_entropy(logit, gt)
            
    # else:
    #     loss = {}
    #     for data, logit in gate_logits.items(): loss[data] = F.cross_entropy(logit, gt)
    
    all_loss = []
    
    for logit in gate_logits.values():
        # loss = F.cross_entropy(logit, gt, reduction='none') * sparsity_weight
        
        loss = 2 * (
            F.cross_entropy(logit, gt, reduction='none') * sparsity_weight).mean()
        
        all_loss.append(loss * lambda_sparsity)
        
    if return_sum: return sum(all_loss)
    else: return {f"{k}_sparsity_loss": all_loss[i] for i, k in enumerate(gate_logits.keys())}
        

    

class EdgeLoss(nn.Module):
    # def __init__(self, dataset, raw_shape=(321, 321), depth_mask=None):
    def __init__(self, dataset, aux_ratio=0.5):
        super(EdgeLoss, self).__init__()
        self.dataset = dataset
        # self.raw_shape = raw_shape
        self.loss_fn = nn.L1Loss()
        self.aux_ratio = aux_ratio
    
    
    def compute_loss(self, pred, gt, is_aux=False):
        if self.dataset == "taskonomy":
            binary_mask = gt != 255
        
        prediction = pred.masked_select(binary_mask)
        key_gt = gt.masked_select(binary_mask)
        losses = self.loss_fn(prediction, key_gt)
        
        if is_aux:
            return losses * self.aux_ratio
        else: return losses
    
    
class KeypointLoss(nn.Module):
    # def __init__(self, dataset, raw_shape=(321, 321), depth_mask=None):
    def __init__(self, dataset, aux_ratio=0.5):
        super(KeypointLoss, self).__init__()
        self.dataset = dataset
        # self.raw_shape = raw_shape
        self.loss_fn = nn.L1Loss()
        self.aux_ratio = aux_ratio
    
    
    def compute_loss(self, pred, gt, is_aux=False):
        if self.dataset == "taskonomy":
            binary_mask = gt != 255
        
        prediction = pred.masked_select(binary_mask)
        key_gt = gt.masked_select(binary_mask)
        losses = self.loss_fn(prediction, key_gt)
        
        if is_aux:
            return losses * self.aux_ratio
        else: return losses

        
class NormalLoss(nn.Module):
    # def __init__(self, dataset, raw_shape=(321, 321), depth_mask=None):
    def __init__(self, dataset, aux_ratio=0.5):
        super(NormalLoss, self).__init__()
        self.dataset = dataset
        # self.raw_shape = raw_shape
        self.loss_fn = nn.CosineSimilarity()
        self.aux_ratio = aux_ratio
    
    
    def compute_loss(self, pred, gt, is_aux=False):
        prediction = pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        gt = gt.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        labels = (gt.max(dim=1)[0] < 255)
        
        # if hasattr(self, 'normal_mask'):
        #     normal_mask_resize = F.interpolate(self.normal_mask.float(), size=pred.shape[-2:])
        #     gt_mask = normal_mask_resize.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        #     labels = labels and gt_mask.int() == 1

        prediction = prediction[labels]
        gt = gt[labels]

        prediction = F.normalize(prediction)
        gt = F.normalize(gt)

        losses = 1 - self.loss_fn(prediction, gt).mean()     
        
        if is_aux:
            return losses * self.aux_ratio
        else: return losses


class DepthLoss(nn.Module):
    # def __init__(self, dataset, raw_shape=(321, 321), depth_mask=None):
    def __init__(self, dataset, depth_mask=None, aux_ratio=0.5):
        super(DepthLoss, self).__init__()
        self.dataset = dataset
        # self.raw_shape = raw_shape
        self.depth_mask = depth_mask
        self.loss_fn = nn.L1Loss()
        self.aux_ratio = aux_ratio
    
    
    def compute_loss(self, pred, gt, is_aux=False):
        if self.dataset in ['nyuv2', 'cityscapes']:
            binary_mask = (torch.sum(gt, dim=1) > 3 * 1e-5).unsqueeze(1).cuda()
        
        elif self.dataset == 'taskonomy' and "mask" in gt:
            assert gt["mask"] is not None
            binary_mask = (gt["gt"] != 255) * (gt["mask"].int() == 1)
            gt = gt["gt"]
        else:
            raise ValueError('Dataset %s is invalid' % self.dataset)
        
        depth_output = pred.masked_select(binary_mask)
        depth_gt = gt.masked_select(binary_mask)
        
        # torch.sum(torch.abs(depth_pred - depth_resize) * binary_mask) / torch.nonzero(binary_mask).size(0)
        losses = self.loss_fn(depth_output, depth_gt)
        
        if is_aux:
            return losses * self.aux_ratio
        else: return losses
        

class SemSegLoss:
    def __init__(self, dataset, ignore_index=255, aux_ratio=0.5):
        super(SemSegLoss, self).__init__()
        self.dataset = dataset
        
        weight = None
        if dataset == "cityscapes":
            ignore_index = -1
        else:
            ignore_index = ignore_index
            if dataset == "taskonomy":
                weight = np.load("/root/data/tiny-taskonomy/semseg_prior_factor.npy")
                weight = torch.from_numpy(weight).cuda().float()
            
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.aux_ratio = aux_ratio
    
    
    def compute_loss(self, pred, gt, is_aux=False):
        if gt.dim() == 4:
            gt = torch.squeeze(gt, 1)
        gt = gt.long()
        
        if is_aux:
            return self.loss_fn(pred, gt) * self.aux_ratio
        else:
            return self.loss_fn(pred, gt)
            
        
def get_loss_fn(dataset, task, **task_args):
    if task == "sseg":
        return SemSegLoss(dataset, **task_args)
    elif task == "depth":
        return DepthLoss(dataset, **task_args)
    elif task == "sn":
        return NormalLoss(dataset, **task_args)
    elif task == "keypoint":
        return KeypointLoss(dataset, **task_args)
    elif task == "edge":
        return EdgeLoss(dataset, **task_args)
    else:
        raise ValueError("not supported loss type")