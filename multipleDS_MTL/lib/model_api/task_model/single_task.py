from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.get_detector import build_detector, DetStem
from ..modules.get_backbone import build_backbone
from ..modules.get_segmentor import build_segmentor, SegStem
from ..modules.get_classifier import build_classifier, ClfStem
from ...apis.loss_lib import AutomaticWeightedLoss


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)


def make_stem(task, cfg, backbone=None, stem_weight=None):
    if task == 'clf':
        stem = ClfStem(**cfg['stem'])
        stem.apply(init_weights)
    
    elif task == 'det':
        stem = DetStem(**cfg['stem'])
        if stem_weight is not None:
            ckpt = torch.load(stem_weight)
            stem.load_state_dict(ckpt, strict=False)
        
        if 'mobilenetv3' in backbone:
            for p in stem.parameters():
                p.requires_grad = False
    
    elif task == 'seg':
        stem = SegStem(**cfg['stem'])
        if stem_weight is not None:
            ckpt = torch.load(stem_weight)
            stem.load_state_dict(ckpt, strict=False)
    
    return stem
    
    
def make_head(task, backbone, dense_task, num_classes, fpn_channel=256, head_cfg=None):
    if task == 'clf':
        head = build_classifier(
            backbone, num_classes, head_cfg)
    
    elif task == 'det':
        head = build_detector(
            backbone,
            dense_task, 
            fpn_channel, 
            num_classes)
    
    elif task == 'seg':
        head = build_segmentor(
            dense_task, num_classes, cfg_dict=head_cfg)
    
    head.apply(init_weights)
    return head


class SingleTaskNetwork(nn.Module):
    def __init__(self,
                 backbone,
                 detector,
                 segmentor,
                 dataset,
                 task_cfg,
                 **kwargs
                 ) -> None:
        super().__init__()
        # self.backbone = build_backbone(
        #     backbone, detector, segmentor, weight_path=kwargs['state_dict']['backbone'],
        #     train_allbackbone=kwargs['train_allbackbone'],
        #     use_fpn=kwargs['use_fpn'],
        #     freeze_all_backbone_layers=kwargs['freeze_backbone'],
        #     freeze_bn=kwargs['freeze_bn'],
        #     dilation_type=kwargs['dilation_type'],
        #     backbone_type=kwargs['backbone_type'])
        
        self.backbone = build_backbone(
            backbone, detector, segmentor, kwargs)
        
        self.dset = dataset
        task = task_cfg['task']
        # self.task = task_cfg[self.dset]['task']
        
        stem_weight = kwargs['state_dict']['stem']
        self.stem = make_stem(task, task_cfg, backbone, stem_weight)
        
        dense_task = detector if detector else segmentor
        self.head = make_head(task, backbone, dense_task,
                              task_cfg['num_classes'],
                              head_cfg=task_cfg['head'])
        
        if task == 'clf':
            self.task_forward = self._forward_clf
        elif task == 'det':
            self.task_forward = self._forward_coco
        elif task == 'seg':
            self.task_forward = self._forward_voc
        
    
    def _forward_clf(self, images, targets=None):
        stem_feats = self.stem(images)
        backbone_features = self.backbone(stem_feats)
        
        # print(stem_feats.size())
        # for k, v in backbone_features.items():
        #     print(k, v.size())
        # exit()
        
        if self.training:
            losses = self.head(backbone_features, targets)
            return losses
        
        else:
            predictions = self.head(backbone_features)
            
            return dict(outputs=predictions)
    
    
    def _forward_coco(self, images, targets=None, other_hyp=None):
        dset_list = list(other_hyp["task_list"].keys())
        data = images[dset_list[0]][0]
        feats = self.stem(data)
        features = self.backbone(feats)
        
        if self.training:
            losses = self.head(data, features, 
                                origin_targets=targets,
                                trs_fn=self.stem.transform)
            return losses
        
        else:
            predictions = self.head(data, features,                                       
                            trs_fn=self.stem.transform)
            
            return predictions
        
    
    def _forward_voc(self, images, targets=None):
        feats = self.stem(images)
        features = self.backbone(feats)
        
        # for _, v in features.items():
        #     print(v.size())
            
        # exit()
        
        if self.training:
            losses = self.head(features, targets,
                               input_shape=targets.shape[-2:])
            return losses
        
        else:
            predictions = self.head(
                    features, input_shape=images.shape[-2:])
            
            # print(predictions.size())
            return dict(outputs=predictions)
            

    def _foward_train(self, data_dict, kwargs):
        images, targets = data_dict[self.dset]
        loss_dict = self.task_forward(images, targets, other_hyp=kwargs)
        
        return loss_dict
        
    
    def _forward_val(self, images, kwargs):
        # print(images)
        prediction_results = self.task_forward(images, other_hyp=kwargs)
        
        return prediction_results
    
    
    def forward(self, data_dict, kwargs=None):
        if self.training:
            return self._foward_train(data_dict, kwargs)

        else:
            return self._forward_val(data_dict, kwargs)
        
