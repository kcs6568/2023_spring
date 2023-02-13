from dataclasses import replace
from turtle import forward
import numpy as np
from scipy.special import softmax
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ..modules.get_detector import build_detector, DetStem
from ..modules.get_backbone import build_backbone
from ..modules.get_segmentor import build_segmentor, SegStem
from ..modules.get_classifier import build_classifier, ClfStem
from ...apis.loss_lib import AutomaticWeightedLoss
from ...apis.loss_lib import shared_gate_loss, disjointed_gate_loss, gate_similarity_loss, non_shared_gate_loss
from ...apis.warmup import PolynomialDecay, ExponentialDecay, set_decay_fucntion


# class LearnableGumbel(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         temp = nn.init.xavier_uniform_(torch.rand(1))
        
#         self.learnable_temp = nn.Parameter(temp, requires_grad=True)
        
        
#     def forward(self, logits):
        


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


class AIGGate(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel//4, 1)
        self.bn = nn.BatchNorm2d(in_channel//4)
        self.conv2 = nn.Conv2d(in_channel//4, 2, 1)
        
    def forward(self, x):
        x = F.avg_pool2d(x, x.shape[2:])
        x = F.leaky_relu(self.bn(self.conv1(x)))
        x = self.conv2(x)
        
        return x


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
        # self.is_retrain = kwargs['is_retrain']
        self.label_smoothing_alpha = kwargs['label_smoothing_alpha']
        kwargs['decay_settings'].update({'max_iter': kwargs['max_iter']})
        self.temp_decay = set_decay_fucntion(kwargs['decay_settings'])
        self.temperature = kwargs['decay_settings']['temperature']
    
        self.current_iter = 0
        self.is_hardsampling = kwargs['is_hardsampling']
        self.same_loss_weight = kwargs['same_loss_weight']
        self.sparsity_weight = kwargs['sparsity_weight']
        # print(self.sparsity_weight)
        
        self.blocks = []
        self.ds = []
        self.num_per_block = []
        self.channel_per_block = []
        for _, p in backbone_network.body.named_children():
            block = []
            self.num_per_block.append(len(p))
            for m, q in p.named_children():
                if m == '0':
                    self.ds.append(q.downsample)
                    q.downsample = None
                    
                block.append(q)
                # self.channel_per_block.append(q.out_channels)
                
            self.blocks.append(nn.ModuleList(block))
        
        # print(self.channel_per_block)
        
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
        
        self.make_gate_logits(data_list, len(task_cfg))    
        self.policys = {dset: torch.zeros(gate.size()).float() for dset, gate in self.task_gating_params.items()}
        
        self.use_sharing = kwargs['use_sharing']
        self.use_disjointed = kwargs['use_disjointed']
        
    
    def make_gate_logits(self, data_list, num_task):
        logit_dict = {}
        for t_id in range(num_task):
            task_logits = Variable((0.5 * torch.ones(sum(self.num_per_block), 2)), requires_grad=True)
            
            # requires_grad = False if self.is_retrain else True
            logit_dict.update(
                {data_list[t_id]: nn.Parameter(task_logits, requires_grad=True)})
            
        self.task_gating_params = nn.ParameterDict(logit_dict)
        
    
    def train_sample_policy(self):
        policys = {}
        for dset, prob in self.task_gating_params.items():
            policy = F.gumbel_softmax(prob, self.temperature, hard=self.is_hardsampling)
            # policy = torch.softmax(prob, dim=1)
            
            policys.update({dset: policy.float()})
            
        return policys
    
    
    def test_sample_policy(self, dset):
        task_policy = []
        task_logits = self.task_gating_params[dset]
        
        if self.is_hardsampling:
            hard_gate = torch.argmax(task_logits, dim=1)
            policy = torch.stack((1-hard_gate, hard_gate), dim=1).cuda()
            
        else:
            logits = softmax(task_logits.detach().cpu().numpy(), axis=-1)
            for l in logits:
                sampled = np.random.choice((1, 0), p=l)
                policy = [sampled, 1 - sampled]
                # policy = [1, 0]
                task_policy.append(policy)
            
            policy = torch.from_numpy(np.array(task_policy)).cuda()
        
        return {dset: policy}
        
    
    def decay_temperature(self):
        self.temperature = self.temp_decay.decay_temp(self.current_iter)
    
    
    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        for dset, (images, _) in data_dict.items():
            stem_feats.update({dset: self.stem_dict[dset](images)})
            # exit()
            
        return stem_feats
    
    def forward_train(self, data_dict, other_hyp):
        total_losses = OrderedDict()
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        dset_list = list(other_hyp["task_list"].keys())
        
        data = self._extract_stem_feats(data_dict)
        self.policys = self.train_sample_policy()
        self.current_iter += 1
        self.decay_temperature()
                
        for dset, feat in data.items():
            block_count = 0
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                    feat = F.leaky_relu(self.blocks[layer_idx][block_idx](feat) + identity)
                    feat = feat * self.policys[dset][block_count, 0] + identity * self.policys[dset][block_count, 1]
                    
                    block_count += 1
                    
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): feat})
                
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
            
            
            task_list = list(data_dict.keys())
            if self.use_sharing:
                sharing_loss = shared_gate_loss(self.task_gating_params, task_list, sum(self.num_per_block), same_loss_weight=self.same_loss_weight)
                total_losses.update({"sharing": sharing_loss})
            
            if self.use_disjointed:     
                # print(self.sparsity_weight)           
                disjointed_loss = disjointed_gate_loss(
                    self.task_gating_params, 
                    task_list, 
                    sum(self.num_per_block), 
                    smoothing_alpha=self.label_smoothing_alpha,
                    sparsity_weight=self.sparsity_weight)
                total_losses.update({"disjointed": disjointed_loss})
        
        return total_losses
            
    
    def forward_eval(self, data_dict, other_hyp):
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        dset_list = list(other_hyp["task_list"].keys())
        
        data = self._extract_stem_feats(data_dict)
        self.policys = self.test_sample_policy(dset_list[0])

        for dset, feat in data.items():
            block_count = 0
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    if self.policys[dset][block_count, 0]:
                        identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                        feat = F.leaky_relu(self.blocks[layer_idx][block_idx](feat) + identity)
                        # feat = feat * self.policys[dset][block_count, 0] + identity * self.policys[dset][block_count, 1]
                    else:
                        identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                        feat = identity * self.policys[dset][block_count, 1]
                    
                    block_count += 1
                    
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): feat})
                
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
        if self.training:
            return self.forward_train(data_dict, kwargs)
        else:
            return self.forward_eval(data_dict, kwargs)





    
    # def get_features(self, data_dict, other_hyp):
    #     total_losses = OrderedDict()
    #     backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
    #     dset_list = list(other_hyp["task_list"].keys())
        
    #     data = self._extract_stem_feats(data_dict)
    #     if not self.is_retrain:
    #         if self.training:
    #             self.policys = self.train_sample_policy()
    #             self.current_iter += 1
    #             self.decay_temperature()
                
    #         else:
    #             self.policys = self.test_sample_policy(dset_list[0])
        
    #     else:
    #         self.policys = self.task_gating_params

    #     # print(self.policys[dset_list[0]])
        
    #     for dset, feat in data.items():
    #         block_count = 0
    #         for layer_idx, num_blocks in enumerate(self.num_per_block):
    #             for block_idx in range(num_blocks):
    #                 # if self.policys[dset][block_count, 0] == 1:
    #                 #     # print("Block used")
    #                 #     identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
    #                 #     feat = F.leaky_relu(self.blocks[layer_idx][block_idx](feat) + identity)
    #                 # else:
    #                 #     # print("Block not used. Identity only used")
    #                 #     feat = self.ds[layer_idx](feat) if block_idx == 0 else feat
                    
    #                 identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
    #                 feat = F.leaky_relu(self.blocks[layer_idx][block_idx](feat) + identity)
    #                 feat = feat * self.policys[dset][block_count, 0] + identity * self.policys[dset][block_count, 1]
                    
    #                 block_count += 1
                    
    #             if block_idx == (num_blocks - 1):
    #                 if str(layer_idx) in self.return_layers[dset]:
    #                     backbone_feats[dset].update({str(layer_idx): feat})
                
    #     if self.training:
    #         features = {}
    #         for dset, back_feats in backbone_feats.items():
    #             task = other_hyp["task_list"][dset]
    #             head = self.head_dict[dset]
                
    #             if self.training:
    #                 targets = data_dict[dset][1]
                    
    #                 if task == 'clf':
    #                     losses = head(back_feats, targets)
                        
    #                 elif task == 'det':
    #                     fpn_feat = self.fpn(back_feats)
    #                     losses = head(data_dict[dset][0], fpn_feat,
    #                                             self.stem_dict[dset].transform, 
    #                                         origin_targets=targets)
                        
    #                 elif task == 'seg':
    #                     losses = head(
    #                         back_feats, targets, input_shape=targets.shape[-2:])
                    
    #                 losses = {f"feat_{dset}_{k}": l for k, l in losses.items()}
    #                 features.update(losses)    
    #                 total_losses.update(losses)
            
            
    #         if not self.is_retrain:
    #             task_list = list(data_dict.keys())
    #             # sharing_loss = shared_gate_loss(self.task_gating_params, task_list, sum(self.num_per_block), same_loss_weight=self.same_loss_weight)
    #             # non_sharing_loss = non_shared_gate_loss(self.task_gating_params, task_list)
    #             disjointed_loss = disjointed_gate_loss(
    #                 self.task_gating_params, 
    #                 task_list, 
    #                 sum(self.num_per_block), 
    #                 self.num_fixed_gate,
    #                 smoothing_alpha=self.label_smoothing_alpha)
    #             # similarity_loss = gate_similarity_loss(self.task_gating_params)
                
    #             # total_losses.update({"sharing": sharing_loss})
    #             # total_losses.update({"nonsharing": non_sharing_loss})
    #             total_losses.update({"disjointed": disjointed_loss})
    #             # total_losses.update({"similarity": similarity_loss})
            
    #         return total_losses
            
    #     else:
    #         dset = list(other_hyp["task_list"].keys())[0]
    #         task = list(other_hyp["task_list"].values())[0]
    #         head = self.head_dict[dset]
            
    #         back_feats = backbone_feats[dset]
            
    #         if task == 'det':
    #             fpn_feat = self.fpn(back_feats)
    #             predictions = head(data_dict[dset][0], fpn_feat, self.stem_dict[dset].transform)
                
    #         else:
    #             if task == 'seg':
    #                 predictions = head(
    #                     back_feats, input_shape=data_dict[dset][0].shape[-2:])
            
    #             else:
    #                 predictions = head(back_feats)
                
    #             predictions = dict(outputs=predictions)

    #         return predictions


    # def forward(self, data_dict, kwargs):
    #     return self.get_features(data_dict, kwargs)

