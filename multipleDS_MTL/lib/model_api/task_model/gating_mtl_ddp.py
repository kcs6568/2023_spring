import random
import numpy as np
from copy import deepcopy
from scipy.special import softmax
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.dist_utils import *
from ...utils.dist_utils import get_rank
from ...apis.loss_lib import disjointed_policy_loss
from ...apis.warmup import set_decay_fucntion
from ...apis.gradient_based import define_gradient_method
from ...apis.weighting_based import define_weighting_method
from ...model_api.task_model.single_task import SingleTaskNetwork



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


class DDPGateMTL(nn.Module):
    def __init__(self,
                 backbone,
                 detector,
                 segmentor,
                 task_cfg,
                 **kwargs
                 ) -> None:
        super().__init__()
        self.task_single_network = nn.ModuleDict({
            data: SingleTaskNetwork(
                backbone, detector, segmentor, data, cfg, **kwargs
                ) for data, cfg in task_cfg.items()})
        
        self.datasets = list(task_cfg.keys())
        self.base_dataset = self.datasets[0]
        self.dataset_index = list(range(len(self.datasets)))
        self.num_per_block = self.task_single_network[self.base_dataset].num_per_block
        
        if 'weight_method' in kwargs:
            if kwargs['weight_method'] is not None:
                wm_param = kwargs['weight_method']
                w_type = wm_param.pop('type')
                init_param = wm_param.pop('init_param')
                self.weighting_method = define_weighting_method(w_type)
                if init_param: self.weighting_method.init_params(self.datasets, **wm_param)
                
            else: self.weighting_method = None
        else: self.weighting_method = None
        
        if 'grad_method' in kwargs:
            if kwargs['grad_method'] is not None:
                gm_param = kwargs['grad_method']
                g_type = gm_param.pop('type')
                self.grad_method = define_gradient_method(g_type)
                self.grad_method.init_params(self.datasets, **gm_param)
                self.all_shared_params_numel, self.each_param_numel = self.compute_shared_encoder_numel()
                
                if 'weight_method_for_grad' in gm_param:
                    if gm_param['weight_method_for_grad'] is not None:
                        gw_params = gm_param['weight_method_for_grad']
                        gw_type = gw_params.pop('type')
                        init_param = gw_params.pop('init_param')
                        grad_weighting_method = define_weighting_method(gw_type)
                        if init_param: grad_weighting_method.init_params(self.datasets, **gw_params)
                        setattr(self.grad_method, 'weighting_method', grad_weighting_method)
                        
            else: self.grad_method = None
        else: self.grad_method = None
        
        if 'static_weight' in kwargs:
            static_weight = torch.load(kwargs['static_weight'], map_location="cpu")['model']
            # self.load_state_dict(static_weight, strict=True)
            encoder_weight = {}
            task_stem = {d: {} for d in self.datasets}
            task_head = {d: {} for d in self.datasets}
            
            for k, v in static_weight.items():
                if not 'encoder' in k and (not "stem" in k and not "head" in k and not "fpn" in k):
                    remain = k.find(".")
                    if 'block' in k: new_key = 'block' + k[remain:]
                    else: new_key = 'ds' + k[remain:]
                    encoder_weight[new_key] = v
                
                else:
                    if 'stem' in k:
                        for data in self.datasets:
                            if data in k: 
                                new_key = k.replace(f'stem_dict.{data}.', "")
                                task_stem[data][new_key] = v
                    elif 'head' in k:
                        for data in self.datasets:
                            if data in k:
                                if not 'coco' in k: new_key = k.replace(f'head_dict.{data}.', "")
                                else: new_key = k.replace(f'head_dict.{data}', "detector")
                                task_head[data][new_key] = v
                        
                    elif 'fpn' in k:
                        for data in self.datasets:
                            if not 'coco' in data: continue
                            task_head[data][k] = v
            
            for data, net in self.task_single_network.items():
                net.encoder.load_state_dict(encoder_weight, strict=True)
                net.stem.load_state_dict(task_stem[data], strict=True)
                net.head.load_state_dict(task_head[data], strict=True)
                print(f"!!!Load weights of Hard Parameter Sharing Baseline for {data.upper()}!!!")
                

        self.decay_function = set_decay_fucntion(kwargs['decay_settings'])
        self._make_gate(kwargs['gate_args'])
        self.current_iters = 0
        self.dataset_task_pair = {}
        
        
    def get_shared_encoder(self):
        return self.get_network(self.base_dataset).encoder
    
    
    def compute_shared_encoder_numel(self):
        each_numel = []
        for p in self.get_shared_encoder().parameters():
            each_numel.append(p.data.numel())
        return sum(each_numel), each_numel
    
    
    def compute_task_head_numel(self, task):
        each_numel = []
        for p in self.get_task_head(task).parameters():
            each_numel.append(p.data.numel())
        return [sum(each_numel), each_numel]
    
    
    
    @property
    def grad_zero_shared_encoder(self):
        for net in self.task_single_network.values(): net.encoder.zero_grad()
    
    
    def get_task_stem(self, task):
        return self.task_single_network[task].stem
    
    
    def get_task_head(self, task):
        return self.task_single_network[task].head
    
    
    def get_task_encoder(self, task):
        return self.task_single_network[task].encoder
    
    
    def get_network(self, dataset):
        return self.task_single_network[dataset]
    
    
    def _grad2vec(self, task):
        grad = torch.zeros(self.all_shared_params_numel).to(get_rank())
        count = 0
        
        for n, param in self.task_single_network[task].encoder.named_parameters():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.each_param_numel[:count])
                end = sum(self.each_param_numel[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad
    
    
    def grad2vec(self, task, clone_grad=False):
        grad = torch.zeros(self.all_shared_params_numel).to(get_rank())
        count = 0
        
        for n, param in self.task_single_network[task].encoder.named_parameters():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.each_param_numel[:count])
                end = sum(self.each_param_numel[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
            
        if clone_grad: return grad.clone()
        else: return grad
    
    
    def _grad2vec_head(self, task):
        grad = torch.zeros(self.head_numel[task][0]).to(get_rank())
        count = 0
        
        for n, param in self.get_task_head(task).named_parameters():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.head_numel[task][1][:count])
                end = sum(self.head_numel[task][1][:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad
        
        
    def _reset_grad(self, new_grads):
        count = 0
        for n, param in self.get_shared_encoder().named_parameters():
            beg = 0 if count == 0 else sum(self.each_param_numel[:count])
            end = sum(self.each_param_numel[:(count+1)])
            param.grad = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
        
    
    def _transfer_computed_grad(self):
        for d in self.datasets:
            if d == self.base_dataset: continue
            for base, target in zip(
                self.get_shared_encoder().parameters(),
                self.task_single_network[d].encoder.parameters()):
                target.grad = base.grad
    
    
    def compute_newgrads(self, origin_grad=None, cur_iter=None, total_mean_grad=False):
        if origin_grad is None:
            origin_grad = {k: self._grad2vec(k) for k in self.datasets}
            
        assert origin_grad is not None
                
        if self.grad_method.require_copied_grad: copied_task_grad2vec = {k: origin_grad[k].clone().to(get_rank()) for k in self.datasets}
        else: copied_task_grad2vec = None
        self.grad_zero_shared_encoder 
        
        kwargs = {'iter': cur_iter}
        new_grads = self.grad_method.backward(origin_grad, copied_task_grad2vec, **kwargs)
        
        dist.all_reduce(new_grads)
        new_grads /= dist.get_world_size()
        
        if total_mean_grad: self._reset_grad(new_grads/len(origin_grad))
        else: self._reset_grad(new_grads)
        
        self._transfer_computed_grad()
        
    
    def fix_gate(self):
        layer_count = []
        for data in self.datasets:
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
        for data in self.datasets:
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
        for k, v in args.items(): setattr(self, k ,v)
        # self.sparsity_weight = gate_args['sparsity_weight']
        logit_dict = {}
        get_grad = True
        from torch.autograd import Variable
        
        for t_id in range(len(self.datasets)):
            task_logits = Variable((0.5 * torch.ones(sum(self.num_per_block), 2, requires_grad=get_grad)), requires_grad=get_grad)
            
            # requires_grad = False if self.is_retrain else True
            logit_dict.update(
                {self.datasets[t_id]: nn.Parameter(task_logits, requires_grad=get_grad)})
            
        self.task_gating_params = nn.ParameterDict(logit_dict)
    
    
    def train_sample_policy(self, cur_task):
        policy = F.gumbel_softmax(
            self.task_gating_params[cur_task], self.decay_function.temperature, hard=self.is_hardsampling)
        return policy.float()
    
    
    def test_sample_policy(self, cur_task, is_hardsampling=False):
        task_policy = []
        task_logits = self.task_gating_params[cur_task]
        
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
        
        return policy
    
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
                    # identity = (ds_module[layer_idx](feat) if block_idx == 0 else feat) * layer_gate[block_count, 1]
                    # block_output = block_module[layer_idx][block_idx](feat) * layer_gate[block_count, 0]
                    # feat = F.leaky_relu_(block_output + identity)
                    
                    identity = (ds_module[layer_idx](feat) if block_idx == 0 else feat)
                    
                    if layer_gate[block_count, 0]: feat = F.leaky_relu(block_module[layer_idx][block_idx](feat) + identity)
                    else: feat = F.leaky_relu(identity)
                    block_count += 1
                    
                    
                    
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): feat})
        if self.training:
            return self._forward_train(data_dict, backbone_feats, other_hyp)
        
        else:
            return self._forward_val(data_dict, backbone_feats, other_hyp)
        

    
    def get_features(self, data_dict, cur_task):
        backbone_feats = {}
        feat = self.task_single_network[cur_task].stem(data_dict[0])
        
        if self.training:
            policies = self.train_sample_policy(cur_task)
        else:
            policies = self.test_sample_policy(cur_task)
        
        
        block_module = self.task_single_network[cur_task].encoder['block']
        ds_module = self.task_single_network[cur_task].encoder['ds']
        
        block_count=0
        for layer_idx, num_blocks in enumerate(self.num_per_block):
            for block_idx in range(num_blocks):
                identity = ds_module[layer_idx](feat) if block_idx == 0 else feat
                if self.training:
                    # if self.seperate_features: block_output = F.leaky_relu(block_module[layer_idx][block_idx](feat))
                    # else: block_output = F.leaky_relu(block_module[layer_idx][block_idx](feat) + identity)
                    # feat = (policies[dset][block_count, 0] * block_output) + (policies[dset][block_count, 1] * identity)
                    
                    block_output = block_module[layer_idx][block_idx](feat) + identity
                    feat = F.leaky_relu((policies[block_count, 0] * block_output) + (policies[block_count, 1] * identity))
                    
                else:
                    if policies[block_count, 0]: feat = F.leaky_relu(block_module[layer_idx][block_idx](feat) + identity) 
                    else: feat = F.leaky_relu(identity)
                
                block_count += 1
                
            if block_idx == (num_blocks - 1):
                if str(layer_idx) in self.task_single_network[cur_task].return_layers:
                    backbone_feats.update({str(layer_idx): feat})
                    # print(f"{dset} return feature saved")
                        
        
        if self.training:
            return self._forward_train(data_dict, backbone_feats, cur_task)
        
        else:
            return self._forward_val(data_dict, backbone_feats, cur_task)
        
    
    def _forward_train(self, data_dict, backbone_feats, cur_dataset):
        total_losses = OrderedDict()
        task = self.task_single_network[cur_dataset].task
        head = self.task_single_network[cur_dataset].head
        targets = data_dict[1]
        
        if task == 'clf':
            losses = head(backbone_feats, targets)
            
        elif task == 'det':
            fpn_feat = head.fpn(backbone_feats)
            losses = head.detector(data_dict[0], fpn_feat,
                                    self.task_single_network[cur_dataset].stem.transform, 
                                origin_targets=targets)
            
            # fpn_feat = head['fpn'](back_feats)
            # losses = head['detector'](data_dict[dset][0], fpn_feat,
            #                         self.stem_dict[dset].transform, 
            #                     origin_targets=targets)
            
        elif task == 'seg':
            losses = head(
                backbone_feats, targets, input_shape=targets.shape[-2:])
        
        losses = {f"feat_{cur_dataset}_{k}": l for k, l in losses.items()}
        total_losses.update(losses)
        
        
        if not self.retrain_phase:
            disjointed_loss = disjointed_policy_loss(
                    self.task_gating_params[cur_dataset], 
                    sum(self.num_per_block),
                    sparsity_weight=self.sparsity_weight, 
                    smoothing_alpha=self.label_smoothing_alpha,
                    return_sum=self.return_sum)
            
            if hasattr(self, 'only_gate_train'):
                if self.only_gate_train:
                    if isinstance(disjointed_loss, dict): total_losses.update(disjointed_loss)
                    else: total_losses.update({"sparsity": disjointed_loss})
            # else: total_losses.update({"sparsity": disjointed_loss * self.sparsity_weight})
            else: total_losses.update({f"{cur_dataset}_sparsity": disjointed_loss})
            
        return total_losses
    
    
    def _forward_val(self, data_dict, backbone_feats, cur_dataset):
        task = self.task_single_network[cur_dataset].task
        head = self.task_single_network[cur_dataset].head
        
        if task == 'det':
            fpn_feat = head.fpn(backbone_feats)
            predictions = head.detector(data_dict[0], fpn_feat,
                               self.task_single_network[cur_dataset].stem.transform)
            
            # fpn_feat = head['fpn'](back_feats)
            # predictions = head['detector'](data_dict[dset][0], fpn_feat, self.stem_dict[dset].transform)
            
        else:
            if task == 'seg':
                predictions = head(
                    backbone_feats, input_shape=data_dict[0].shape[-2:])
        
            else:
                predictions = head(backbone_feats)
            
            predictions = dict(outputs=predictions)
        
        return predictions
    
    
    
    def forward(self, data_dict, cur_dataset):
        assert hasattr(self, 'retrain_phase')
        
        if not self.training:
            data_dict = [data_dict]
            cur_dataset = list(cur_dataset['task_list'].keys())[0]
        
        if self.retrain_phase:
            assert self.retrain_phase
            return self.retraining_features(data_dict, cur_dataset)
        else:
            assert not self.retrain_phase
            return self.get_features(data_dict, cur_dataset)
        
        
        # print(cur_dataset)
        # if self.training:
        #     output = self.task_single_network[cur_dataset](data_dict)
        #     losses = {f"{cur_dataset}_{k}": v for k, v in output.items()}
            
        # else:
        #     losses = self.task_single_network[cur_dataset](data_dict)
        
        # return losses
        
        
        
    def __str__(self) -> str:
        info = f"[Current Gate Parameter States]\n"
        for data in self.datasets:
            gate = self.task_gating_params[data]
            info += f"{data} (trainable?: {gate.requires_grad}):\n{torch.transpose(gate.data, 0, 1)}\n"
            
        return info