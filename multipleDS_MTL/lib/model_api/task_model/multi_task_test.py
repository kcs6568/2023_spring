from dataclasses import replace
import numpy as np
from scipy.special import softmax
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.get_detector import build_detector, DetStem
from ..modules.get_backbone import build_backbone
from ..modules.get_segmentor import build_segmentor, SegStem
from ..modules.get_classifier import build_classifier, ClfStem
from ...apis.loss_lib import AutomaticWeightedLoss
from ...apis.loss_lib import shared_gate_loss, disjointed_gate_loss
from ...apis.warmup import PolynomialDecay, ExponentialDecay


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


class MultiTaskNetwork(nn.Module):
    def __init__(self,
                 backbone,
                 detector,
                 segmentor,
                 task_cfg,
                 **kwargs
                 ) -> None:
        super().__init__()
        backbone_network = build_backbone(
            backbone, detector, segmentor, kwargs)
        
        self.temp_decay = ExponentialDecay(
            kwargs['temperature'], kwargs['max_iter'], kwargs['gamma'])
        self.temperature = kwargs['temperature']
        
        self.current_iter = 0
        self.is_hardsampling = kwargs['is_hardsampling']
        
        random_policy = torch.rand(16)
        random_policy = torch.softmax(random_policy, 0) > 0.07
        # self.random_policy = random_policy.int()
        
        self.blocks = []
        self.ds = []
        self.num_per_block = []
        
        block_count = 0
        for _, p in backbone_network.body.named_children():
            block = []
            self.num_per_block.append(len(p))
            for m, q in p.named_children():
                if m == '0':
                    self.ds.append(q.downsample)
                    q.downsample = None
                block.append(q)
                block_count += 1
            self.blocks.append(nn.ModuleList(block))
        
        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fpn = backbone_network.fpn
        
        self.stem_dict = nn.ModuleDict()
        self.head_dict = nn.ModuleDict()
        
        self.return_layers = {}
        data_list = []
        
        stem_weight = kwargs['state_dict']['stem']
        for data, cfg in task_cfg.items():
            data_list.append(data)
            self.return_layers.update({data: cfg['return_layers']})
            
            task = cfg['task']
            num_classes = cfg['num_classes']
            if task == 'clf':
                stem = ClfStem(**cfg['stem'])
                head = build_classifier(
                    backbone, num_classes, cfg['head'])
                stem.apply(init_weights)
                
            elif task == 'det':
                stem = DetStem(**cfg['stem'])
                
                head_kwargs = {'num_anchors': len(backbone_network.body.return_layers)+1}
                head = build_detector(
                    backbone, detector, 
                    backbone_network.fpn_out_channels, num_classes, **head_kwargs)
                if stem_weight is not None:
                    ckpt = torch.load(stem_weight)
                    stem.load_state_dict(ckpt, strict=False)
                    print("!!!Load weights for detection stem layer!!!")
            
            elif task == 'seg':
                stem = SegStem(**cfg['stem'])
                head = build_segmentor(segmentor, num_classes=num_classes, cfg_dict=cfg['head'])
                if stem_weight is not None:
                    ckpt = torch.load(stem_weight)
                    stem.load_state_dict(ckpt, strict=False)
                    print("!!!Load weights for segmentation stem layer!!!")
            
            head.apply(init_weights)
            self.stem_dict.update({data: stem})
            self.head_dict.update({data: head})
        
        print("!!!Loading the pretrained weight....!!!")
        pretrained_ckpt = torch.load(kwargs['pretrained_weight'], map_location=torch.device('cpu'))
        self.load_state_dict(pretrained_ckpt['model'])
        print("!!!Complete to loading the weight!!!")
        
        if kwargs['use_gate']:
            self.make_gate_logits(data_list, len(task_cfg))    
        
        if kwargs['freeze_stem']:
            for p in self.stem_dict.parameters():
                p.requires_grad = False
        
        
        self.current_task = None
        self.policys = None
        
        
    def make_gate_logits(self, data_list, num_task):
        logit_dict = {}
        for t_id in range(num_task):
            task_logits = 0.5 * torch.ones(sum(self.num_per_block), 2)
            # task_logits = nn.init.kaiming_uniform_(task_logits)
            
            logit_dict.update(
                {data_list[t_id]: nn.Parameter(task_logits, requires_grad=True)})
            
        self.task_gating_params = nn.ParameterDict(logit_dict)
        
        
    # def train_sample_policy(self):
    #     policys = {}
    #     for dset, prob in self.task_gating_params.items():
    #         random_prob = torch.randn(prob.size()).cuda()
    #         policy = F.gumbel_softmax(random_prob, self.temperature, hard=self.is_hardsampling)
    #         # policy = F.gumbel_softmax(prob, self.temperature, hard=self.is_hardsampling)
    #         policys.update({dset: policy.float()})
        
    #     return policys
    
    
    # def test_sample_policy(self, dset):
    #     task_policy = []
    #     # task_logits = self.task_gating_params[dset]
    #     task_logits = torch.randn(16, 2).cuda()
    #     logits = softmax(task_logits.detach().cpu().numpy(), axis=-1)
        
    #     for l in logits:
    #         sampled = np.random.choice((1, 0), p=l)
    #         policy = [sampled, 1 - sampled]
    #         task_policy.append(policy)
        
    #     policy = torch.from_numpy(np.array(task_policy)).cuda()
        
    #     return {dset: policy}
    
    
    def train_sample_policy(self, dset_list):
        policys = {}
        for dset in dset_list:
            random_prob = torch.randn(sum(self.num_per_block), 2).cuda()
            policy = F.gumbel_softmax(random_prob, self.temperature, hard=self.is_hardsampling)
            # policy = F.gumbel_softmax(prob, self.temperature, hard=self.is_hardsampling)
            policys.update({dset: policy.float()})
        
        return policys
    
    
    def test_sample_policy(self, dset):
        task_policy = []
        # task_logits = self.task_gating_params[dset]
        task_logits = torch.randn(16, 2).cuda()
        logits = softmax(task_logits.detach().cpu().numpy(), axis=-1)
        
        for l in logits:
            sampled = np.random.choice((1, 0), p=l)
            policy = [sampled, 1 - sampled]
            task_policy.append(policy)
        
        policy = torch.from_numpy(np.array(task_policy)).cuda()
        
        return {dset: policy}
    
    
    def decay_temperature(self):
        self.temperature = self.temp_decay.decay_temp(self.current_iter+1)
    
        
    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        for dset, (images, _) in data_dict.items():
            stem_feats.update({dset: self.stem_dict[dset](images)})
        return stem_feats
    
    
    def _fix_alltozero_policies(self, dset, block_num, direction):
        policies = torch.ones(sum(self.num_per_block), 2)
        policies[:, 1] = 0 
        
        '''
        start policy
        1 0
        1 0
        1 0
        1 0
        '''
        
        if direction == 'topdown':
            policies[block_num:, 0] = 0
            policies[block_num:, 1] = 1
            
        elif direction == 'bottomup':
            policies[:block_num, 0] = 0
            policies[:block_num, 1] = 1    
            
        elif direction == 'one':
            policies[block_num, 0] = 0
            policies[block_num, 1] = 1            
            
        return {dset: policies.cuda()}
    
    def _fix_zerotoall_policies(self, dset, block_num, direction):
        policies = torch.ones(sum(self.num_per_block), 2)
        policies[:, 0] = 0
        
        '''
        0 1
        0 1
        0 1
        0 1
        '''
        
        if direction == 'topdown':
            policies[block_num:, 0] = 1 # make policy for using to the block_num-th blocks
            policies[block_num:, 1] = 0 # make policy for not using from the block_num-th to last blocks
            
        elif direction == 'bottomup':
            policies[:block_num, 0] = 1 # make policy for using to the block_num-th blocks
            policies[:block_num, 1] = 0 # make policy for not using from the block_num-th to last blocks
            
        return {dset: policies.cuda()}
    
    
    def get_features(self, data_dict, other_hyp):
        total_losses = OrderedDict()
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        
        data = self._extract_stem_feats(data_dict)
        dset_list = list(other_hyp["task_list"].keys())
        
        if self.training:
            # policys = self.train_sample_policy()
            policys = self.train_sample_policy(dset_list)
            self.decay_temperature()
            self.current_iter += 1
        else:
            # policys = self.test_sample_policy(dset_list[0])
            if other_hyp["effect_type"] == "zerotoall":
                self.policys = self._fix_zerotoall_policies(dset_list[0], other_hyp["block_num"], other_hyp["effect_direction"])
            elif other_hyp["effect_type"] == "alltozero":
                self.policys = self._fix_alltozero_policies(dset_list[0], other_hyp["block_num"], other_hyp["effect_direction"])

        # print(self.policys)
        
            
        block_count = 0
        for dset, feat in data.items():
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                    feat = F.leaky_relu(self.blocks[layer_idx][block_idx](feat) + identity)
                    feat = feat * self.policys[dset][block_count, 0] + identity * self.policys[dset][block_count, 1]
                    
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): feat})
                    
                block_count += 1
        
        if self.training:
            features = {}
            for dset, back_feats in backbone_feats.items():
                task = other_hyp["task_list"][dset]
                head = self.head_dict[dset]
                
                if self.training:
                    targets = data_dict[dset][1]
                    
                    if task == 'clf':
                        losses = head(back_feats, targets)
                        
                    elif task == 'det':
                        fpn_feat = self.fpn(back_feats)
                        losses = head(data_dict[dset][0], fpn_feat,
                                                self.stem_dict[dset].transform, 
                                            origin_targets=targets)
                        
                    elif task == 'seg':
                        losses = head(
                            back_feats, targets, input_shape=targets.shape[-2:])
                    
                    losses = {f"feat_{dset}_{k}": l for k, l in losses.items()}
                    features.update(losses)    
                    total_losses.update(losses)
            
            # task_list = list(data_dict.keys())
            # sharing_loss = shared_gate_loss(self.task_gating_params, task_list, sum(self.num_per_block))
            # disjointed_loss = disjointed_gate_loss(self.task_gating_params, task_list, sum(self.num_per_block))
            
            # total_losses.update({"sharing": sharing_loss})
            # total_losses.update({"disjointed": disjointed_loss})
            
            # total_losses.update({"sharing": self.sharing_weight * sharing_loss})
            # total_losses.update({"disjoined": self.disjointed_weight * disjoint_loss})

            return total_losses
            
        else:
            dset = list(other_hyp["task_list"].keys())[0]
            task = list(other_hyp["task_list"].values())[0]
            head = self.head_dict[dset]
            
            back_feats = backbone_feats[dset]
            
            if task == 'det':
                fpn_feat = self.fpn(back_feats)
                predictions = head(data_dict[dset][0], fpn_feat, self.stem_dict[dset].transform)
                
            else:
                if task == 'seg':
                    predictions = head(
                        back_feats, input_shape=data_dict[dset][0].shape[-2:])
            
                else:
                    predictions = head(back_feats)
                
                predictions = dict(outputs=predictions)

            return predictions


    def forward(self, data_dict, kwargs):
        return self.get_features(data_dict, kwargs)


    # def _forward_val(self, images, kwargs):
    #     dset = list(kwargs.keys())[0]
    #     task = list(kwargs.values())[0]
        
    #     stem, head = self.stem_dict[dset], self.head_dict[dset]
    #     stem_feats = stem(images)
        
    #     if task == 'det':
    #         back_feats = self.backbone(stem_feats)
    #         predictions = head(images, back_feats, stem.transform)
    #         return predictions
        
    #     else:
    #         back_feats = self.backbone.body(stem_feats)
    #         if task == 'seg':
    #             predictions = head(
    #                 back_feats, input_shape=images.shape[-2:])
        
    #         else:
    #             predictions = head(back_feats)
            
    #         return dict(outputs=predictions)


