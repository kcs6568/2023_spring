from collections import OrderedDict

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


def shared_gate_loss(gate_logits, tasks, num_blocks, same_loss_weight=True):
    # loss_weights = torch.tensor([(num_blocks - (i))/num_blocks for i in range(num_blocks)]).float().cuda()
    # loss_weights = torch.tensor([(num_blocks//2 - (i))/(num_blocks//2) for i in range(num_blocks//2)]).float().cuda()
    # loss_weights = torch.cat((loss_weights, reversed(loss_weights)), dim=0).cuda()
    # loss_weights[:fixed_gate] = 1
    
    if same_loss_weight:
        loss_weights = torch.tensor([1 for _ in range(num_blocks)]).float().cuda()
    else:
        loss_weights = torch.tensor([(num_blocks//2 - (i))/(num_blocks//2) for i in range(num_blocks//2)]).float()
        # loss_weights = torch.tensor([(i+1)/(num_blocks//2) for i in range(num_blocks//2)]).float()
        loss_weights = torch.cat((loss_weights, reversed(loss_weights)), dim=0).cuda()
        
    
    # if len(tasks) == 3:
    #     sharing_loss = gate_logits[tasks[0]][:, 0] - gate_logits[tasks[1]][:, 0] \
    #         - gate_logits[tasks[2]][:, 0]
    # elif len(tasks) == 4:
    #     sharing_loss = (gate_logits[tasks[0]][:, 0] - gate_logits[tasks[1]][:, 0] \
    #         - gate_logits[tasks[2]][:, 0] - gate_logits[tasks[3]][:, 0])
    
    # sharing_loss = torch.sum(loss_weights * torch.abs(sharing_loss))
    
    task_len = len(tasks)
    sharing_loss = \
        sum(
            sum(loss_weights * torch.abs(gate_logits[tasks[i]][:, 0] - gate_logits[tasks[j]][:, 0]) \
                for j in range(i+1, task_len)) \
                    for i in range(task_len))
    
    sharing_loss = torch.sum(sharing_loss)
    
    return sharing_loss


def non_shared_gate_loss(gate_logits, tasks):
    # loss_weights = torch.tensor([(num_blocks - (i+1))/num_blocks for i in range(16)]).float().cuda()
    # loss_weights = torch.arange(num_blocks, 0, -1).float()
    # loss_weights = torch.softmax(loss_weights, 0).cuda()
    
    # loss_weights = torch.tensor([(num_blocks//2 - (i+1))/(num_blocks//2) for i in range(num_blocks//2)]).float().cuda()
    # loss_weights = torch.cat((loss_weights, reversed(loss_weights)), dim=0).cuda()
    
    if len(tasks) == 3:
        non_sharing_loss = gate_logits[tasks[0]][:, 1] - gate_logits[tasks[1]][:, 1] \
            - gate_logits[tasks[2]][:, 1]
    elif len(tasks) == 4:
        non_sharing_loss = (gate_logits[tasks[0]][:, 1] - gate_logits[tasks[1]][:, 1] \
            - gate_logits[tasks[2]][:, 1] - gate_logits[tasks[3]][:, 1])
        
    non_sharing_loss = torch.sum(torch.abs(non_sharing_loss))
    
    return non_sharing_loss


def disjointed_policy_loss(gate_logits, tasks, num_blocks, smoothing_alpha=None):
    loss = 0.
    # alpha = torch.tensor([(num_blocks - (i))/num_blocks for i in range(num_blocks)]).float().cuda()
    # alpha = torch.tensor([0.4 for _ in range(num_blocks)]).float().cuda()
    # alpha[:fixed_gate] = 0
    if smoothing_alpha is not None:
        gt_ = torch.ones(num_blocks, 2).long().cuda()
        gt = torch.tensor([[l*(1-smoothing_alpha) + smoothing_alpha/len(oh) for l in oh] for i, oh in enumerate(gt_)]).float().cuda()
        
    else:
        gt = torch.ones(num_blocks).long().cuda()    
        
    for dset in tasks:
        # gt = torch.ones(len(gate_logits[dset])).long().cuda()    
        loss += F.cross_entropy(gate_logits[dset], gt)
        
    return loss



def disjointed_gate_loss(gate_logits, tasks, num_blocks, smoothing_alpha=None, sparsity_weight=None):
    loss = 0.
    if smoothing_alpha is not None:
        gt_ = torch.ones(num_blocks, 2).long().cuda()
        gt = torch.tensor([[l*(1-smoothing_alpha) + smoothing_alpha/len(oh) for l in oh] for i, oh in enumerate(gt_)]).float().cuda()
        
    else:
        gt = torch.ones(num_blocks).long().cuda()    
    # print(sparsity_weight)
    if sparsity_weight is None:
        for dset in tasks:
            loss += F.cross_entropy(gate_logits[dset], gt)
    else:
        for dset, s_type in sparsity_weight.items():
            # print(dset, s_type)
            if isinstance(s_type, float):
                task_loss = s_type * F.cross_entropy(gate_logits[dset], gt)
                # print(dset, s_type, task_loss)
                loss += task_loss
            
            else:
                ce_loss = F.cross_entropy(gate_logits[dset], gt, reduction='none')
                if s_type == 'b2t':
                    s_weight = torch.tensor([(i+1) / num_blocks for i in range(num_blocks)]).float().cuda()
                    # task_loss = torch.mean((s_weight * F.cross_entropy(gate_logits[dset], gt, reduction='none')))
                    
                elif s_type == 't2b':
                    s_weight = torch.tensor([(num_blocks - i) / num_blocks for i in range(num_blocks)]).float().cuda()
                    # task_loss = torch.mean((s_weight * F.cross_entropy(gate_logits[dset], gt, reduction='none')))
                    
                elif s_type == 'sym':
                    s_weight = torch.tensor([(num_blocks//2 - (i))/(num_blocks//2) for i in range(num_blocks//2)]).float()
                    s_weight = torch.cat((s_weight, reversed(s_weight)), dim=0).cuda()
                    # task_loss = torch.mean((s_weight * F.cross_entropy(gate_logits[dset], gt, reduction='none')))
                
                # print(dset)
                # print(ce_loss)
                # print(s_weight)
                task_loss = torch.mean(s_weight * ce_loss)
                # print(task_loss)
                loss += task_loss
                # print(loss)
                # print()
                
                
                
        
        # for dset in tasks:
        #     task_loss = sparsity_weight[dset] * F.cross_entropy(gate_logits[dset], gt)
        #     loss += task_loss
        
    return loss



def gate_similarity_loss(gate_logits, shift_func='exp'):
    similarity_loss = 0.
    for logits in gate_logits.values():
        similarity_loss +=torch.exp(F.cosine_similarity(logits[:, 0], logits[:, 1], dim=0))
    
    return similarity_loss
        
        
def feature_cosine_similarity(feat, policy_feat):
    feat = F.avg_pool2d(feat, feat.shape[2:]).view(feat.shape[0], -1)
    policy_feat = F.avg_pool2d(policy_feat, policy_feat.shape[2:]).view(policy_feat.shape[0], -1)
    
    return F.cosine_similarity(feat, policy_feat, dim=1)
    # print(feat)
    # print(policy_feat)
    
    # print(feat.size(), policy_feat.size())
    # print(F.cosine_similarity(feat, policy_feat, dim=0))
    # print(F.cosine_similarity(feat, policy_feat, dim=1))
    # exit()
    # for f, p_f in zip(feat, policy_feat):
    #     print(f.size(), p_f.size())
    #     print(F.cosine_similarity(f, p_f, dim=0))
    #     print(F.cosine_similarity(f, p_f, dim=-1))
    #     print()
        
    # exit()
    
    
    # return sum(F.cosine_similarity(f, p_f, dim=0) for f, p_f in zip(feat, policy_feat))

    