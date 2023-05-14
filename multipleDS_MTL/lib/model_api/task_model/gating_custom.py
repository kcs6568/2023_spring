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

from ...apis.loss_lib import disjointed_policy_loss, disjointed_prob_loss
from ...apis.warmup import set_decay_fucntion
from ...apis.gradient_based import define_gradient_method
from ...apis.weighting_based import define_weighting_method

def init_kaiming_uniform(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=1e-2, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=1e-2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    

def init_kaiming_normal(m):
    # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #     nn.init.kaiming_normal_(m.weight, a=1e-2, mode='fan_in', nonlinearity='leaky_relu')
    #     if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=1e-2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=1e-2, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    
    
def init_xavier_uniform(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    

def init_xavier_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



class CustomGating(nn.Module):
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
        
        init_type = kwargs['init_type'] if 'init_type' in kwargs else None
        init_dist = kwargs['init_dist'] if 'init_dist' in kwargs else None
        init_function = None
        if init_type is not None and init_dist is not None:
            if kwargs['init_type'] == 'kaiming':
                if kwargs['init_dist'] == 'uniform':
                    init_function = init_kaiming_uniform
                elif kwargs['init_dist'] == 'normal':
                    init_function = init_kaiming_normal
                    
            elif kwargs['init_type'] == 'xavier':
                if kwargs['init_dist'] == 'uniform':
                    init_function = init_xavier_uniform
                elif kwargs['init_dist'] == 'normal':
                    init_function = init_xavier_normal
        
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
        
        # self.fpn = backbone_network.fpn
        
        self.stem = nn.ModuleDict()
        self.head = nn.ModuleDict()
        
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
        
        use_fpn = True if kwargs['use_fpn'] else False
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
                
                if init_function is not None: stem.apply(init_function)
                
                
            elif task == 'det':
                stem_cfg['stem_weight'] = kwargs['stem_weight']
                stem = DetStem(**stem_cfg)
                
                head_cfg.update({'num_anchors': len(backbone_network.body.return_layers)+1})
                
                head = build_detector(
                    backbone, detector, 
                    backbone_network.fpn_out_channels, num_classes, **head_cfg)
                
                
                # detection = build_detector(
                #     backbone, detector, 
                #     backbone_network.fpn_out_channels, num_classes, **head_cfg)
                
                if init_function is not None: backbone_network.fpn.apply(init_function)
                head = nn.ModuleDict({
                    'fpn': backbone_network.fpn,
                    'detector': head
                })
            
            elif task == 'seg':
                stem_cfg['stem_weight'] = kwargs['stem_weight']
                stem = SegStem(**stem_cfg)
                head = build_segmentor(segmentor, num_classes=num_classes, cfg_dict=head_cfg)
            
            # head.apply(init_weights)
            self.stem.update({data: stem})
            self.head.update({data: head})
        
        if 'static_weight' in kwargs:
            if kwargs['static_weight'] is not None:
                static_weight = torch.load(kwargs['static_weight'], map_location="cpu")['model']
                self.load_state_dict(static_weight, strict=True)
                print("!!!Load weights of Hard Parameter Sharing Baseline!!!")

        self.decay_function = set_decay_fucntion(kwargs['decay_settings'])
        self.data_list = data_list
        self._make_gate(kwargs['gate_args'])
        self.current_iters = 0
        
        if 'weight_method' in kwargs and kwargs['weight_method'] is not None:
            wm_param = kwargs['weight_method']
            w_type = wm_param.pop('type')
            self.weighting_method = define_weighting_method(w_type)
            if wm_param['init_param']: self.weighting_method.init_params(data_list, **wm_param)
        else: self.weighting_method = None
        
        if 'grad_method' in kwargs and kwargs['grad_method'] is not None:
            gm_param = kwargs['grad_method']
            g_type = gm_param.pop('type')
            self.grad_method = define_gradient_method(g_type)
            if gm_param['init_param']: self.grad_method.init_params(data_list, **gm_param)
            self.compute_grad_dim
        else: self.grad_method = None
        
        self.activation_function = kwargs['activation_function']
        
        self.retrain_phase = kwargs['retrain_phase']
        if self.retrain_phase: self.activation_function.inplace = False
        
    def fix_gate(self):
        layer_count = []
        
        self.exploiting_gate.requires_grad = False
        self.dropping_gate.requires_grad = False
        
        for d_idx, data in enumerate(self.data_list):
            trained_gate = torch.stack(
            [self.exploiting_gate[:, d_idx], self.dropping_gate[:, d_idx]]).t()
            
            distribution = torch.softmax(trained_gate, dim=1)
            task_gate = torch.tensor([torch.argmax(gate, dim=0) for gate in distribution])
            final_gate = torch.stack((1-task_gate, task_gate), dim=1).float()

            self.exploiting_gate[:, d_idx] = final_gate[:, 0]
            self.dropping_gate[:, d_idx] = final_gate[:, 1]
            
            # self.task_gating_params[data].data = final_gate
            # self.task_gating_params[data].requires_grad = False
            
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
        for d_idx, data in enumerate(self.data_list):
            gate_count = 0
            param = []
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    block_params = 0
                    block_gate = self.exploiting_gate[gate_count, d_idx]
                    
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
        for k, v in args.items(): setattr(self, k ,v)
        get_grad = True
        self.exploiting_gate = nn.Parameter(Variable((0.5 * torch.ones(sum(self.num_per_block), 3, requires_grad=get_grad)), requires_grad=get_grad), requires_grad=get_grad)
        self.dropping_gate = nn.Parameter(Variable((0.5 * torch.ones(sum(self.num_per_block), 3, requires_grad=get_grad)), requires_grad=get_grad), requires_grad=get_grad)

    

    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        for dset, (images, _) in data_dict.items():
            stem_feats.update({dset: self.stem[dset](images)})
        return stem_feats
    
    
    def train_sample_policy(self):
        exploiting_prob = F.gumbel_softmax(self.exploiting_gate, self.decay_function.temperature, hard=self.is_hardsampling, dim=1)
        dropping_gate = F.gumbel_softmax(self.dropping_gate, self.decay_function.temperature, hard=self.is_hardsampling, dim=1)
        
        return exploiting_prob, dropping_gate
    
    
    def test_sample_policy(self, dset, is_hardsampling=False):
        task_index = self.data_list.index(dset)
        task_logits = torch.stack(
            [self.exploiting_gate[:, task_index], self.dropping_gate[:, task_index]]).t()
        
        task_policy = []
        
        exploiting = []
        # dropping = []
        if is_hardsampling:
            hard_gate = torch.argmax(task_logits, dim=1)
            policy = torch.stack((1-hard_gate, hard_gate), dim=1).cuda()
            
        else:
            logits = softmax(task_logits.detach().cpu().numpy(), axis=-1)
            for l in logits:
                sampled = np.random.choice((1, 0), p=l)
                policy = [sampled, 1 - sampled]
                exploiting.append(policy[0])
            
            # policy = torch.from_numpy(np.array(task_policy)).cuda()
        
        return torch.from_numpy(np.array(exploiting)).cuda(), None
    
    
    def retraining_features(self, data_dict, other_hyp):
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        data = self._extract_stem_feats(data_dict)
        
        block_module = self.encoder['block']
        ds_module = self.encoder['ds']
            
        for d_idx, dset in enumerate(self.data_list):
            if not dset in data: continue
            
            block_count=0
            feat = data[dset]
            
            layer_gate = self.exploiting_gate[:, d_idx]
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    identity = (ds_module[layer_idx](feat) if block_idx == 0 else feat)
                    
                    if layer_gate[block_count]: feat = self.activation_function(block_module[layer_idx][block_idx](feat) + identity)
                    else: feat = self.activation_function(identity)
                    
                    block_count += 1
                    
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): feat})
        
        if self.training:
            return self._forward_train(data_dict, backbone_feats, other_hyp)
        
        else:
            return self._forward_val(data_dict, backbone_feats, other_hyp)
        
    
    def get_features(self, data_dict, other_hyp):
        # self.current_iters += 1
        
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        data = self._extract_stem_feats(data_dict)
        
        if self.training:
            exploiting, dropping = self.train_sample_policy()
            # self.decay_function.set_temperature(self.current_iters)
        else:
            dataset = list(other_hyp['task_list'].keys())[0]
            exploiting, _ = self.test_sample_policy(dataset)
        
        block_module = self.encoder['block']
        ds_module = self.encoder['ds']
        
        # for dset, feat in data.items():
        for d_idx, dset in enumerate(self.data_list):
            if not self.training and dset != dataset: continue
            
            block_count=0
            feat = data[dset]
            
            # if self.training:
            #     task_exploiting = exploiting[:, d_idx]
            #     task_dropping = dropping[:, d_idx]
            
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    identity = ds_module[layer_idx](feat) if block_idx == 0 else feat
                   
                    if self.training:
                        block_output = block_module[layer_idx][block_idx](feat) + identity
                        feat = self.activation_function(
                                (exploiting[block_count, d_idx] * block_output) + (dropping[block_count, d_idx] * identity)
                            )
                        
                    else:
                        if exploiting[block_count]: feat = self.activation_function(block_module[layer_idx][block_idx](feat) + identity) 
                        else: feat = self.activation_function(identity)
                    
                    block_count += 1
                    
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): feat})
            
        if self.training:
            return self._forward_train(data_dict, backbone_feats, other_hyp, exploiting)
        
        else:
            return self._forward_val(data_dict, backbone_feats, other_hyp)
        
    
    def _forward_train(self, data_dict, backbone_feats, other_hyp, policies=None):
        total_losses = OrderedDict()
        for dset, back_feats in backbone_feats.items():
            task = other_hyp["task_list"][dset]
            head = self.head[dset]
            targets = data_dict[dset][1]
            
            if task == 'clf':
                losses = head(back_feats, targets)
                
            elif task == 'det':
                fpn_feat = head['fpn'](back_feats)
                losses = head['detector'](data_dict[dset][0], fpn_feat,
                                        self.stem[dset].transform, 
                                    origin_targets=targets)
                
            elif task == 'seg':
                losses = head(
                    back_feats, targets, input_shape=targets.shape[-2:])
            
            losses = {f"feat_{dset}_{k}": l for k, l in losses.items()}
            total_losses.update(losses)
        
        
        if not self.retrain_phase:
            disjointed_loss = disjointed_prob_loss(
                    policies, 
                    sum(self.num_per_block),
                    sparsity_weight=self.sparsity_weight, 
                    smoothing_alpha=self.label_smoothing_alpha,
                    return_sum=self.return_sum)
            
            if hasattr(self, 'only_gate_train'):
                if self.only_gate_train:
                    if isinstance(disjointed_loss, dict): total_losses.update(disjointed_loss)
                    else: total_losses.update({"sparsity": disjointed_loss})
            # else: total_losses.update({"sparsity": disjointed_loss * self.sparsity_weight})
            else: total_losses.update({"sparsity": disjointed_loss})
            
        return total_losses
    
    
    def _forward_val(self, data_dict, backbone_feats, other_hyp):
        dset = list(other_hyp["task_list"].keys())[0]
        task = list(other_hyp["task_list"].values())[0]
        self.current_val_dataset = dset
        
        head = self.head[dset]
        back_feats = backbone_feats[dset]
                
        if task == 'det':
            fpn_feat = head['fpn'](back_feats)
            predictions = head['detector'](data_dict[dset][0], fpn_feat, self.stem[dset].transform)
            
            # fpn_feat = head['fpn'](back_feats)
            # predictions = head['detector'](data_dict[dset][0], fpn_feat, self.stem[dset].transform)
            
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
        #         head = self.head[dset]
                
        #         targets = data_dict[dset][1]
                
        #         if task == 'clf':
        #             losses = head(back_feats, targets)
                    
        #         elif task == 'det':
        #             fpn_feat = head['fpn'](back_feats)
        #             losses = head['detector'](data_dict[dset][0], fpn_feat,
        #                                     self.stem[dset].transform, 
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
        #     head = self.head[dset]
            
        #     back_feats = backbone_feats[dset]
                    
        #     if task == 'det':
        #         fpn_feat = head['fpn'](back_feats)
        #         predictions = head['detector'](data_dict[dset][0], fpn_feat, self.stem[dset].transform)
                
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
            
            # return self.retraining_features(data_dict, kwargs)
        else:
            assert not self.retrain_phase
            return self.get_features(data_dict, kwargs)

    
    def __str__(self) -> str:
        if self.training:
            info = f"[Current Gate Parameter States]\n"
            for d_idx, data in enumerate(self.data_list):
                info += f"{data.upper()} (trainable?: {self.exploiting_gate[:, d_idx].requires_grad}):\n"
                info += f"Exploiting: {self.exploiting_gate[:, d_idx].data}\n"
                info += f"Dropping: {self.dropping_gate[:, d_idx].data}\n\n"
            info += f"Temperature: {self.decay_function.temperature}\n"
        
        else:
            current_gate_index = self.data_list.index(self.current_val_dataset)
            info = f"Gate Hard Sampling:\n"
            info += f"Exploiting: {self.exploiting_gate[:, current_gate_index].data}\n"
            info += f"Dropping: {self.dropping_gate[:, current_gate_index].data}\n\n"
        return info
    
    def print_fixed_gate_info(self):
        info = f"[Current Gate Parameter States]\n"
        for d_idx, data in enumerate(self.data_list):
            gate = self.exploiting_gate[:, d_idx]
            info += f"{data.upper()} (trainable?: {gate.requires_grad}):\n{gate.data}\n"
            
        return info