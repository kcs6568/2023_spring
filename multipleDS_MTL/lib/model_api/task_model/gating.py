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

from ...apis.loss_lib import disjointed_policy_loss
from ...apis.warmup import set_decay_fucntion
from ...apis.weighting import define_weighting_method


def init_weights(m, type="kaiming"):
    if isinstance(m, nn.Conv2d):
        if type == 'kaiming':
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif type == 'xavier':
            nn.init.xavier_normal_(m.weight)
        
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.Linear):
        if type == 'kaiming':
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif type == 'xavier':
            nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class GateMTL(nn.Module):
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
        
        if kwargs['backbone_weight'] is not None:
            backbone_weight = torch.load(kwargs['backbone_weight'], map_location="cpu")
            backbone_network.body.load_state_dict(backbone_weight, strict=True)
            print("!!!Loaded pretrained body weights!!!")
        
        self.num_per_block = []
        blocks = []
        ds = []

        for _, p in backbone_network.body.named_children():
            block = []
            self.num_per_block.append(len(p))
            for m, q in p.named_children():
                if m == '0':
                    ds.append(q.downsample)
                    q.downsample = None
                
                block.append(q)
                
            blocks.append(nn.ModuleList(block))
        
        self.encoder = nn.ModuleDict({
            'block': nn.ModuleList(blocks),
            'ds': nn.ModuleList(ds)
        })
        
        # self.blocks = nn.ModuleList(self.blocks)
        # self.ds = nn.ModuleList(self.ds)
        self.fpn = backbone_network.fpn
        
        self.stem_dict = nn.ModuleDict()
        self.head_dict = nn.ModuleDict()
        
        self.return_layers = {}
        data_list = []
        
        if kwargs['backbone_weight'] is not None:
            stem_weight = kwargs['stem_weight']
        else:
            stem_weight = None
            
        shared_stem_configs = {
            'activation_function': kwargs['activation_function'],
            'stem_weight': stem_weight
        }
        
        shared_head_configs = {
            'activation_function': kwargs['activation_function']
        }
        
        for data, cfg in task_cfg.items():
            data_list.append(data)
            self.return_layers.update({data: cfg['return_layers']})
            
            if 'stem' in cfg:
                stem_cfg = cfg['stem']
            else:
                head_cfg = {}
            
            if 'head' in cfg:
                head_cfg = cfg['head']
            else:
                head_cfg = {}
            
            stem_cfg.update(shared_stem_configs)
            head_cfg.update(shared_head_configs)
            
            task = cfg['task']
            num_classes = cfg['num_classes']
            if task == 'clf':
                stem = ClfStem(**stem_cfg)
                head = build_classifier(
                    backbone, num_classes, head_cfg)
                stem.apply(init_weights)
                
            elif task == 'det':
                stem = DetStem(**stem_cfg)
                
                head_cfg.update({'num_anchors': len(backbone_network.body.return_layers)+1})
                
                head = build_detector(
                    backbone, detector, 
                    backbone_network.fpn_out_channels, num_classes, **head_cfg)
                
                
                # detection = build_detector(
                #     backbone, detector, 
                #     backbone_network.fpn_out_channels, num_classes, **head_cfg)
                
                # if backbone_network.fpn is not None:
                #     head = nn.ModuleDict({
                #         'fpn': backbone_network.fpn,
                #         'detector': detection
                #     })
            
            elif task == 'seg':
                stem = SegStem(**stem_cfg)
                head = build_segmentor(segmentor, num_classes=num_classes, cfg_dict=head_cfg)
            
            head.apply(init_weights)
            self.stem_dict.update({data: stem})
            self.head_dict.update({data: head})
        
        if 'static_weight' in kwargs:
            static_weight = torch.load(kwargs['static_weight'], map_location="cpu")['model']
            self.load_state_dict(static_weight, strict=True)
            print("!!!Load weights of Hard Parameter Sharing Baseline!!!")

        self.data_list = data_list
        self._make_gate(kwargs['gate_args'])
        self.current_iters = 0
        
        wm_args = kwargs['weight_method']
        if wm_args is not None:
            wm_type = wm_args.pop('type')
            kw = {'task_list': self.data_list}
            for k, v in wm_args.items(): kw.update({k: v})
            self.wm = define_weighting_method(wm_type, **kw)
        else:
            self.wm = None
            
        self.seperate_features = kwargs['seperate_features']
        self.each_param_numel = self._compute_shared_encoder_numel
        
        self.sim_function = nn.CosineSimilarity(dim=0)
        self.sim_function2 = nn.CosineSimilarity(dim=1)
    
    @property
    def get_gate_policy(self):
        return self.task_gating_params
    
    
    @property
    def _compute_shared_encoder_numel(self):
        all_numel = []
        
        # for p in self.get_shared_encoder.parameters():
        #     each_numel.append(p.data.numel())
        
        block = self.encoder['block']
        for layer_idx, num_blocks in enumerate(self.num_per_block):
            for block_idx in range(num_blocks):
                block_numel = []
                for p in block[layer_idx][block_idx].parameters():
                    block_numel.append(p.data.numel())
                all_numel.append(block_numel)
        
        assert len(all_numel) == sum(self.num_per_block)
        return all_numel
    
    
    def _grad2vec(self, block_idx, block_module):
        grad = torch.zeros(sum(self.each_param_numel[block_idx]))
        count = 0
        for n, param in block_module.named_parameters():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.each_param_numel[block_idx][:count])
                end = sum(self.each_param_numel[block_idx][:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad
    
    
    def collect_shared_grad2vec(self):
        shared_grad2vec = []
        # for dset, feat in data.items():
        block_count=0
        for layer_idx, num_blocks in enumerate(self.num_per_block):
            for block_idx in range(num_blocks):
                block_module = self.encoder['block'][layer_idx][block_idx]
                # block_grad2vec = self._grad2vec(block_count, block_module)
                shared_grad2vec.append(self._grad2vec(block_count, block_module).clone())
                block_count += 1

        return shared_grad2vec
    
    # def compute_sim(self):
    #     shared_grad2vec = self.collect_shared_grad2vec()
        
    #     for block_idx in range(sum(self.num_per_block)):
    #         print("block_idx:", block_idx)
    #         print("***"*60)
    #         # norm_block_grad = torch.div(shared_grad2vec[block_idx], shared_grad2vec[block_idx].norm()).cuda()
    #         norm_block_grad = shared_grad2vec[block_idx].cuda()
    #         for data in self.data_list:
    #             print(data)
    #             print("==="*20)
    #             grad_clone = self.task_gating_params[data].grad[block_idx].clone()
    #             # print(block_grad)
    #             # print(grad_clone)
    #             # print(self.task_gating_params[data].grad)
    #             # print(self.task_gating_params[data][block_idx])
    #             # print(self.task_gating_params[data].grad[block_idx])
    #             # print(self.task_gating_params[data].grad[block_idx][0])
    #             # exit()
    #             print(grad_clone)
    #             print(norm_block_grad)
    #             task_grad = torch.ones_like(norm_block_grad).cuda() * grad_clone[0]
    #             print(task_grad)
    #             task_sim = self.sim_function(norm_block_grad, task_grad)
    #             print(task_sim, torch.neg(task_sim))
    #             grad_clone[0] *= task_sim
                
    #             print(norm_block_grad)
    #             task_grad = torch.ones_like(norm_block_grad).cuda() * grad_clone[1]
    #             print(task_grad)
    #             task_sim = self.sim_function(norm_block_grad, task_grad)
    #             print(task_sim, torch.neg(task_sim))
    #             grad_clone[1] *= task_sim
                
                
    #             print(grad_clone)
    #             self.task_gating_params[data].grad[block_idx].data = grad_clone
    #             print()
    #             print(self.task_gating_params[data].grad)
    #         print()
    #         print()
    #     exit()
    
    # def compute_sim(self):
        # for data in self.data_list: print(self.task_gating_params[data].grad)
        # task_head_norm = []
        # for data, head in self.head_dict.items():
        #     head_norm = 0
        #     for p in head.parameters(): head_norm += p.grad.norm()
        #     task_head_norm.append(head_norm)
            
        # print(task_head_norm)
        
        
        
        # block_norm = torch.zeros(sum(self.num_per_block)).cuda()
        # block_count=0
        # for layer_idx, num_blocks in enumerate(self.num_per_block):
        #     for block_idx in range(num_blocks):
        #         block = self.encoder['block'][layer_idx][block_idx]
        #         norm = 0
                
        #         block_use = []
        #         block_skip = []
        #         for data in self.data_list:
        #             task_sim = []
        #             task_nonsim = []
        #             gate_grad_clone = self.task_gating_params[data].grad[block_count].clone()
                    
        #             for n, p in block.named_parameters():
        #                 use_grad = torch.ones_like(p.grad).cuda() * gate_grad_clone[0]
        #                 skip_grad = torch.ones_like(p.grad).cuda() * gate_grad_clone[1]
                        
        #                 sim = self.sim_function(p.grad, use_grad)
        #                 task_sim.append(sim.mean())
                        
        #                 norm += p.grad.norm()
        #                 sim = self.sim_function(p.grad, skip_grad)
        #                 task_nonsim.append(sim.mean())
                    
        #             if block_norm[block_count] == 0:
        #                 print(block_count, norm)
        #                 block_norm[block_count] = norm
                    
                    

        #             use_grad = torch.stack(task_sim).mean(0)
        #             skip_grad = torch.stack(task_nonsim).mean(0)
        #             # mean_use = use_grad.mean(0)
        #             block_use.append(use_grad)
        #             block_skip.append(skip_grad)
                
        #         for idx, data in enumerate(self.data_list):
        #             self.task_gating_params[data].grad[block_count, 0] *= ((block_norm[block_count]/task_head_norm[idx]) * block_use[idx])
        #             self.task_gating_params[data].grad[block_count, 1] *= ((block_norm[block_count]/task_head_norm[idx]) * block_skip[idx])
        #             # self.task_gating_params[data].grad[block_count, 0] = gate_grad_clone[0] * mean_use
        #         # self.task_gating_params[data].grad[block_count, 1] = gate_grad_clone[1] * mean_skip
                
        #         # sim_prob = torch.softmax(torch.stack(block_use), dim=0)
        #         # # print(block_count, block_use, sim_prob)
        #         # for idx, data in enumerate(self.data_list): 
        #         #     self.task_gating_params[data].grad[block_count, 0] *= sim_prob[idx]
        #         #     self.task_gating_params[data].grad[block_count, 1] *= (1-sim_prob[idx])
                    
        #         block_count += 1
        
        
        # for data in self.data_list: print(self.task_gating_params[data].grad)
        
        
        # exit()
        
            
            
            # for n, p in block.named_parameters():
            #     task_sim = []
            #     task_nonsim = []
            #     grad = p.grad.clone().cuda()
            #     for data in self.data_list:
            #         if block_idx == 0: print(self.task_gating_params[data].grad[block_idx])
                    
            #         gate_grad_clone = self.task_gating_params[data].grad[block_idx].clone()
            #         gate_grad = torch.ones_like(grad).cuda() * gate_grad_clone[0]
            #         sim = self.sim_function(grad, gate_grad)
            #         task_sim.append(sim.mean())
                    
            #         gate_grad = torch.ones_like(grad).cuda() * gate_grad_clone[1]
            #         sim = self.sim_function(grad, gate_grad)
            #         task_nonsim.append(sim.mean())
            
            #     block_sim.append(torch.stack(task_sim))
            #     block_nonsim.append(torch.stack(task_nonsim))
            
            # block_sim = torch.stack(block_sim)
            # block_nonsim = torch.stack(block_nonsim)
            
            # meansim = block_sim.mean(0)
            # meannonsim = block_nonsim.mean(0)
            
            # self.task_gating_params[data].grad[block_idx][0] = 
            
            
            
            
                
                
            
        
        
    
        # for n, p in self.encoder['block'].named_parameters():
        #     for data in self.data_list:
        #         origin_grad = self._grad2vec()
        #         gate_grad = torch.ones_like(origin_grad).cuda() * self.task_gating_params[data][0]
        #         cos_sim = self.sim_function(origin_grad, gate_grad)
            
            
            
            
        # pcgrad = {k: origin_grad.clone().cuda() for k in self.data_list}
    
    
    def compute_sim(self):
        block_count=0
        for layer_idx, num_blocks in enumerate(self.num_per_block):
            for block_idx in range(num_blocks):
                block = self.encoder['block'][layer_idx][block_idx]
                
                for p in block.parameters():
                    # masks = torch.zeros([len(self.data_list)] + list(p.shape)).cuda()
                    
                    # max_gate = float('inf')
                    max_gate = float('-inf')
                    for idx, data in enumerate(self.data_list):
                        # print(data, self.task_gating_params[data].grad[block_count, 0])
                        if max_gate == float('-inf'): max_gate = torch.abs(self.task_gating_params[data].grad[block_count, 0])
                        if torch.abs(self.task_gating_params[data].grad[block_count, 0]) < max_gate:
                            max_gate = torch.abs(self.task_gating_params[data].grad[block_count, 0])
                    # print(max_gate)
                    #     print(torch.abs(self.task_gating_params[data].grad[block_count, 0]))
                    #     masks[idx] = (torch.abs(p.grad) > torch.abs(self.task_gating_params[data].grad[block_count, 0])).int()
                    # filtered_masks = torch.sum(masks, dim=0)
                    # p.grad.data *= filtered_masks
                    filtered_masks = (torch.abs(p.grad) < max_gate).int()
                    p.grad.data *= filtered_masks
                    
    
    def fix_gate(self):
        layer_count = []
        for data in self.data_list:
            distribution = torch.softmax(self.task_gating_params[data], dim=1)
            task_gate = torch.tensor([torch.argmax(gate, dim=0) for gate in distribution])
            final_gate = torch.stack((1-task_gate, task_gate), dim=1).float()

            self.task_gating_params[data].data = final_gate
            self.task_gating_params[data].requires_grad = False
            
            layer_count.append(final_gate[:, 0])
        
        layer_count = torch.sum(torch.stack(layer_count, dim=1), dim=1)
        
        block_count = 0
        for layer_idx, num_blocks in enumerate(self.num_per_block):
            for block_idx in range(num_blocks):
                if layer_count[block_count] == 0:
                    for p in self.encoder['block'][layer_idx][block_idx].parameters():
                        p.requires_grad = False
                block_count += 1        
        
    
    def compute_subnetwork_size(self):
        task_block_params = {}
        total_block_params = []
        for data in self.data_list:
            gate_count = 0
            param = []
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    block_params = 0
                    block_gate = self.task_gating_params[data][gate_count, 0]
                    
                    for p in self.encoder['block'][layer_idx][block_idx].parameters():
                        block_params += p.numel()
                    if len(total_block_params) < sum(self.num_per_block): total_block_params.append(block_params)
                    
                    block_params *= block_gate
                    param.append(block_params)
                    
                    gate_count += 1

            task_block_params[data] = param
        ds_param = []
        
        for ds_idx in range(len(self.num_per_block)):
            ds = 0
            for ds_p in self.encoder['ds'][ds_idx].parameters():
                ds += ds_p.numel()
            ds_param.append(ds)

        return task_block_params, ds_param, total_block_params
    
    
    def _make_gate(self, args):
        decay_args = args.pop('decay_settings')
        self.decay_function = set_decay_fucntion(decay_args)
        
        for k, v in args.items(): setattr(self, k ,v)
        # self.sparsity_weight = gate_args['sparsity_weight']
        logit_dict = {}
        get_grad = True
        for t_id in range(len(self.data_list)):
            task_logits = Variable((0.5 * torch.ones(sum(self.num_per_block), 2, requires_grad=get_grad)), requires_grad=get_grad)
            
            # requires_grad = False if self.is_retrain else True
            logit_dict.update(
                {self.data_list[t_id]: nn.Parameter(task_logits, requires_grad=get_grad)})
            
        self.task_gating_params = nn.ParameterDict(logit_dict)
    

    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        for dset, (images, _) in data_dict.items():
            stem_feats.update({dset: self.stem_dict[dset](images)})
        return stem_feats
    
    
    def train_sample_policy(self):
        policys = {}
        for dset, prob in self.task_gating_params.items():
            policy = F.gumbel_softmax(prob, self.decay_function.temperature, hard=self.is_hardsampling)
            policys.update({dset: policy.float()})
        #     print(self.current_iters, dset, prob, self.is_hardsampling, policy)
        # print()
        return policys
    
    
    def test_sample_policy(self, dset, is_hardsampling=False):
        task_policy = []
        task_logits = self.task_gating_params[dset]
        
        if is_hardsampling:
            hard_gate = torch.argmax(task_logits, dim=1)
            policy = torch.stack((1-hard_gate, hard_gate), dim=1).cuda()
            
        else:
            logits = softmax(task_logits.detach().cpu().numpy(), axis=-1)
            for l in logits:
                sampled = np.random.choice((1, 0), p=l)
                policy = [sampled, 1 - sampled]
                task_policy.append(policy)
            
            policy = torch.from_numpy(np.array(task_policy)).cuda()
        
        return {dset: policy}
    
    def retraining_features(self, data_dict, other_hyp):
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        data = self._extract_stem_feats(data_dict)
        
        block_module = self.encoder['block']
        ds_module = self.encoder['ds']
        
        
        for dset, feat in data.items():
            block_count=0
            layer_gate = self.task_gating_params[dset]
            
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    identity = (ds_module[layer_idx](feat) if block_idx == 0 else feat) * layer_gate[block_count, 1]
                    block_output = block_module[layer_idx][block_idx](feat) * layer_gate[block_count, 0]
                    feat = F.leaky_relu_(block_output + identity)
                    
                    block_count += 1
                    
                    
                    
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): feat})
        if self.training:
            return self._forward_train(data_dict, backbone_feats, other_hyp)
        
        else:
            return self._forward_val(data_dict, backbone_feats, other_hyp)
        

    
    def get_features(self, data_dict, other_hyp):
        self.current_iters += 1
        
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        data = self._extract_stem_feats(data_dict)
        
        if self.training:
            policies = self.train_sample_policy()
            self.decay_function.set_temperature(self.current_iters)
        else:
            dataset = list(other_hyp['task_list'].keys())[0]
            policies = self.test_sample_policy(dataset)
        
        block_module = self.encoder['block']
        ds_module = self.encoder['ds']
        
        for dset, feat in data.items():
            block_count=0
            # print(f"{dset} geration")
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    identity = ds_module[layer_idx](feat) if block_idx == 0 else feat
                    # feat = F.leaky_relu(self.blocks[layer_idx][block_idx](feat) + identity)
                   
                    if self.seperate_features:
                        block_output = F.leaky_relu_(block_module[layer_idx][block_idx](feat))
                    else:
                        block_output = F.leaky_relu_(block_module[layer_idx][block_idx](feat) + identity)
                        
                    feat = (policies[dset][block_count, 0] * block_output) + (policies[dset][block_count, 1] * identity)
                    block_count += 1
                    
                    
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): feat})
                        # print(f"{dset} return feature saved")
                        
        
        if self.training:
            return self._forward_train(data_dict, backbone_feats, other_hyp)
        
        else:
            return self._forward_val(data_dict, backbone_feats, other_hyp)
        
    
    def _forward_train(self, data_dict, backbone_feats, other_hyp):
        total_losses = OrderedDict()
        for dset, back_feats in backbone_feats.items():
            task = other_hyp["task_list"][dset]
            head = self.head_dict[dset]
            targets = data_dict[dset][1]
            
            if task == 'clf':
                losses = head(back_feats, targets)
                
            elif task == 'det':
                fpn_feat = self.fpn(back_feats)
                losses = head(data_dict[dset][0], fpn_feat,
                                        self.stem_dict[dset].transform, 
                                    origin_targets=targets)
                
                # fpn_feat = head['fpn'](back_feats)
                # losses = head['detector'](data_dict[dset][0], fpn_feat,
                #                         self.stem_dict[dset].transform, 
                #                     origin_targets=targets)
                
            elif task == 'seg':
                losses = head(
                    back_feats, targets, input_shape=targets.shape[-2:])
            
            losses = {f"feat_{dset}_{k}": l for k, l in losses.items()}
            total_losses.update(losses)
        
        
        if not self.retrain_phase:
            disjointed_loss = disjointed_policy_loss(
                    self.task_gating_params, 
                    sum(self.num_per_block), 
                    smoothing_alpha=self.label_smoothing_alpha)
            
            total_losses.update({"sparsity": disjointed_loss * self.sparsity_weight})
            
        return total_losses
    
    
    def _forward_val(self, data_dict, backbone_feats, other_hyp):
        dset = list(other_hyp["task_list"].keys())[0]
        task = list(other_hyp["task_list"].values())[0]
        head = self.head_dict[dset]
        
        back_feats = backbone_feats[dset]
                
        if task == 'det':
            fpn_feat = self.fpn(back_feats)
            predictions = head(data_dict[dset][0], fpn_feat, self.stem_dict[dset].transform)
            
            # fpn_feat = head['fpn'](back_feats)
            # predictions = head['detector'](data_dict[dset][0], fpn_feat, self.stem_dict[dset].transform)
            
        else:
            if task == 'seg':
                predictions = head(
                    back_feats, input_shape=data_dict[dset][0].shape[-2:])
        
            else:
                predictions = head(back_feats)
            
            predictions = dict(outputs=predictions)
        
        return predictions
    
             
        # if self.training:
        #     for dset, back_feats in backbone_feats.items():
        #         task = other_hyp["task_list"][dset]
        #         head = self.head_dict[dset]
                
        #         targets = data_dict[dset][1]
                
        #         if task == 'clf':
        #             losses = head(back_feats, targets)
                    
        #         elif task == 'det':
        #             fpn_feat = head['fpn'](back_feats)
        #             losses = head['detector'](data_dict[dset][0], fpn_feat,
        #                                     self.stem_dict[dset].transform, 
        #                                 origin_targets=targets)
                    
        #         elif task == 'seg':
        #             losses = head(
        #                 back_feats, targets, input_shape=targets.shape[-2:])
                
        #         losses = {f"feat_{dset}_{k}": l for k, l in losses.items()}
        #         total_losses.update(losses)
                
        #     disjointed_loss = disjointed_policy_loss(
        #             self.task_gating_params, 
        #             sum(self.num_per_block), 
        #             smoothing_alpha=self.label_smoothing_alpha)
            
        #     # if self.wm is not None:
        #     #     print("=="*60)
        #     #     print(total_losses)
        #     #     print(disjointed_loss)
        #     #     assert isinstance(disjointed_loss, (dict, OrderedDict))
        #     #     wm_gate_loss = self.wm(disjointed_loss)
        #     #     print(wm_gate_loss)
        #     #     print("||"*60)
        #     #     total_losses.update({f"Sparse{self.wm.method_name}_{k}": l for k, l in wm_gate_loss.items()})    
        #     # else:
        #     #     total_losses.update({"disjointed": disjointed_loss * self.sparsity_weight})
        #     total_losses.update({"sparsity": disjointed_loss * self.sparsity_weight})
            
            
                
        #     return total_losses
            
        # else:
        #     dset = list(other_hyp["task_list"].keys())[0]
        #     task = list(other_hyp["task_list"].values())[0]
        #     head = self.head_dict[dset]
            
        #     back_feats = backbone_feats[dset]
                    
        #     if task == 'det':
        #         fpn_feat = head['fpn'](back_feats)
        #         predictions = head['detector'](data_dict[dset][0], fpn_feat, self.stem_dict[dset].transform)
                
        #     else:
        #         if task == 'seg':
        #             predictions = head(
        #                 back_feats, input_shape=data_dict[dset][0].shape[-2:])
            
        #         else:
        #             predictions = head(back_feats)
                
        #         predictions = dict(outputs=predictions)
            
        #     return predictions
        

    def forward(self, data_dict, kwargs):
        assert hasattr(self, 'retrain_phase')
        if self.retrain_phase:
            assert self.retrain_phase
            return self.retraining_features(data_dict, kwargs)
        else:
            assert not self.retrain_phase
            return self.get_features(data_dict, kwargs)

    
    def __str__(self) -> str:
        info = f"[Current Gate Parameter States]\n"
        for data in self.data_list:
            gate = self.task_gating_params[data]
            info += f"{data} (trainable?: {gate.requires_grad}):\n{torch.transpose(gate.data, 0, 1)}\n"
            
        return info