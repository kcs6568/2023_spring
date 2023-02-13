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
from ...apis.loss_lib import shared_gate_loss, disjointed_gate_loss, gate_similarity_loss, non_shared_gate_loss, disjointed_policy_loss
from ...apis.warmup import PolynomialDecay, ExponentialDecay, set_decay_fucntion




class PolicyGenerator(nn.Module):
    def __init__(self, num_blocks, in_channel=64, out_channel=16) -> None:
        super().__init__()
        # self.gen1 = nn.Conv2d(in_channel, out_channel, 1)
        # self.bn = nn.BatchNorm2d(out_channel) 
        self.close_gate = nn.Conv2d(in_channel, num_blocks, 1)
        self.open_gate = nn.Conv2d(in_channel, num_blocks, 1)
        
        # self.act = nn.LeakyReLU(inplace=True)
        self.num_blocks = num_blocks
        
        nn.init.constant_(self.close_gate.weight, 0.5)
        nn.init.constant_(self.open_gate.weight, 0.5)
    
    
    def size(self):
        return torch.Size((self.num_blocks, 2))
    
    
    def sharing_loss(self, raw_policy):
        # a = F.l1_loss
        loss_weights = torch.tensor([(self.num_blocks//2 - (i))/(self.num_blocks//2) for i in range(self.num_blocks//2)]).float().cuda()
        
        for i in range(raw_policy.shape[0]):
            for j in range(i+1, raw_policy.shape[0]):
                loss = loss_weights * F.l1_loss(raw_policy[i][:, 0], raw_policy[j][:, 0])
                loss_norm = sum(p.pow(2.0).sum() for p in self.close_gate.parameters())
                
                print(loss)
                print(loss_norm)
                                     
        
        exit()           
        
        
        sharing_loss = \
            sum(
                sum(loss_weights * F.l1_loss(raw_policy[i][:, 0], raw_policy[j][:, 0]) \
                    for j in range(i+1, raw_policy.shape[0])) \
                        for i in range(raw_policy.shape[0]))
        # sharing_loss = \
        #     sum(
        #         sum(loss_weights * torch.abs(raw_policy[i][:, 0] - raw_policy[j][:, 0]) \
        #             for j in range(i+1, raw_policy.shape[0])) \
        #                 for i in range(raw_policy.shape[0]))
        
        sharing_loss = torch.sum(sharing_loss)
        
        return sharing_loss
    
    
    def disjointed_loss(self, raw_policy):
        gt = torch.ones(raw_policy.shape[0], self.num_blocks, 2).float().cuda()
        disjointed_loss = F.cross_entropy(raw_policy.squeeze(-1), gt)
        
        return disjointed_loss
    
    
    def forward(self, x):
        x = nn.AvgPool2d(x.shape[-2:])(x)
        # x = self.act(self.bn(self.gen1(x)))
        
        close = self.close_gate(x)
        open = self.open_gate(x)
        
        policy = torch.cat([close, open], dim=2)
        # policy = policy.squeeze(-1)
        
        return policy
        


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
        self.is_retrain = kwargs['is_retrain']
        if not self.is_retrain:
            self.label_smoothing_alpha = kwargs['label_smoothing_alpha']
            kwargs['decay_settings'].update({'max_iter': kwargs['max_iter']})
            self.temp_decay = set_decay_fucntion(kwargs['decay_settings'])
            self.temperature = kwargs['decay_settings']['temperature']
        
        
            if kwargs['curriculum_direction'] is None:
                self.curriculum_speed = None
                self.num_fixed_gate = 0
            else:
                self.curriculum_speed = kwargs['curriculum_speed'] 
                self.num_fixed_gate = kwargs['num_fixed_gate']
                if kwargs['num_fixed_gate'] is None:
                    self.num_fixed_gate = 0
            
            self.current_iter = 0
            self.is_hardsampling = kwargs['is_hardsampling']
        
        self.same_loss_weight = kwargs['same_loss_weight']
        
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
        
        stem_weight = kwargs['state_dict']['stem']
        for data, cfg in task_cfg.items():
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
        
        self.make_gate_logits(list(task_cfg.keys()), len(task_cfg))    
        self.raw_policy = None
        self.policys = None
        
          
    def fix_gate(self, policy):
        fixed = torch.ones(self.num_fixed_gate, 2).cuda()
        fixed[:, 1] = 0
        policy = torch.cat((fixed.float(), policy[self.num_fixed_gate:].float()), dim=0)
        
        return policy
        
    
    def make_gate_logits(self, data_list, num_task):
        logit_dict = {}
        for t_id in range(num_task):
            logit_dict.update({data_list[t_id]: PolicyGenerator(sum(self.num_per_block))})
            
        self.task_gating_params = nn.ModuleDict(logit_dict)
            
    
    def train_sample_policy(self):
        policys = {}
        for dset, logit in self.raw_policy.items():
            policy = F.gumbel_softmax(logit, self.temperature, hard=self.is_hardsampling, dim=2)
            
            # if self.num_fixed_gate is not None and self.num_fixed_gate != 0:
            #     policy = self.fix_gate(policy)
                
            # if self.curriculum_speed is not None:
            #     policy = self.policy_curriculum_setting(policy)
            
            policys.update({dset: policy.unsqueeze(-1).unsqueeze(-1).float()})
            
        return policys
    
    
    def test_sample_policy(self, dset, only_max=False):
        task_policy = []
        task_logits = self.raw_policy[dset]
        if only_max:
            prob = torch.softmax(task_logits, dim=2)
            # print(prob)
            # print(prob.size())
            max_prob = torch.argmax(prob, dim=2)
            # print(max_prob)
            # print(max_prob.size())
            hard_gate = torch.stack((1-max_prob, max_prob), dim=2)
            # print(hard_gate)
            # print(hard_gate.size())
            # exit()
            policy = hard_gate.unsqueeze(-1).unsqueeze(-1).cuda()
            
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
    
    
    def decay_num_fixed_gate(self):
        self.num_fixed_gate -= 1
    
        
    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        for dset, (images, _) in data_dict.items():
            stem_feats.update({dset: self.stem_dict[dset](images)})
            # exit()
            
        return stem_feats
    
    
    def _extract_policy(self, stem_feat):
        raw_policy = OrderedDict()
        
        sharing_loss = 0.
        disjointed_loss = 0.
        
        for dset, feat in stem_feat.items():
            p = self.task_gating_params[dset](feat)
            raw_policy.update({dset: p})
            
            disjointed_loss += self.task_gating_params[dset].disjointed_loss(p)
            # sharing_loss += self.task_gating_params[dset].sharing_loss(p)
            
        
        if self.training:
            disjointed_loss /= len(stem_feat)
            # sharing_loss /= len(stem_feat)
            
            # return raw_policy, [sharing_loss, disjointed_loss]
            return raw_policy, [disjointed_loss]
        
        else:
            return raw_policy, None
        
    
    def get_features(self, data_dict, other_hyp):
        total_losses = OrderedDict()
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        dset_list = list(other_hyp["task_list"].keys())
        
        data = self._extract_stem_feats(data_dict)
        self.raw_policy, policy_loss = self._extract_policy(data)
        
        
        if not self.is_retrain:
            if self.training:
                self.policys = self.train_sample_policy()
                self.current_iter += 1
                self.decay_temperature()
                
            else:
                self.policys = self.test_sample_policy(dset_list[0], only_max=True)
        
        else:
            self.policys = self.task_gating_params
            
        
        for dset, feat in data.items():
            block_count = 0
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    # if self.policys[dset][block_count, 0] == 1:
                    #     # print("Block used")
                    #     identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                    #     feat = F.leaky_relu(self.blocks[layer_idx][block_idx](feat) + identity)
                    # else:
                    #     # print("Block not used. Identity only used")
                    #     feat = self.ds[layer_idx](feat) if block_idx == 0 else feat
                    
                    identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                    feat = F.leaky_relu(self.blocks[layer_idx][block_idx](feat) + identity)
                    
                    feat = feat * self.policys[dset][:, block_count, 0] + identity * self.policys[dset][:, block_count, 1]
                    
                    block_count += 1
                    
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): feat})
                
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
            total_losses.update({"disjointed": policy_loss[0]})
            # total_losses.update({"sharing": policy_loss[0]})
            
            # task_list = list(data_dict.keys())
            # disjointed_loss = disjointed_policy_loss(
            #         self.task_gating_params, 
            #         task_list, 
            #         sum(self.num_per_block), 
            #         self.num_fixed_gate,
            #         smoothing_alpha=self.label_smoothing_alpha)
            # total_losses.update({"disjointed": disjointed_loss})
            
            # if not self.is_retrain:
            #     task_list = list(data_dict.keys())
                # sharing_loss = shared_gate_loss(self.task_gating_params, task_list, sum(self.num_per_block), same_loss_weight=self.same_loss_weight)
            #     # non_sharing_loss = non_shared_gate_loss(self.task_gating_params, task_list)
            #     disjointed_loss = disjointed_gate_loss(
            #         self.task_gating_params, 
            #         task_list, 
            #         sum(self.num_per_block), 
            #         self.num_fixed_gate,
            #         smoothing_alpha=self.label_smoothing_alpha)
            #     # similarity_loss = gate_similarity_loss(self.task_gating_params)
                
            #     # total_losses.update({"sharing": sharing_loss})
            #     # total_losses.update({"nonsharing": non_sharing_loss})
            #     total_losses.update({"disjointed": disjointed_loss})
            #     # total_losses.update({"similarity": similarity_loss})
                
                
                
                
            
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
            self.policys[dset] = self.policys[dset].squeeze(-1).squeeze(-1).squeeze(-1)[0]
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