import os
import random
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...model_api.task_model.single_task import SingleTaskNetwork
from ...utils.dist_utils import *
from ...apis.gradient_based import define_gradient_method
from ...apis.weighting_based import define_weighting_method

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


class DDPStatic(nn.Module):
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


    def get_shared_encoder(self):
        return self.get_network(self.base_dataset).encoder
    
    
    def compute_shared_encoder_numel(self):
        each_numel = []
        for p in self.get_shared_encoder().parameters():
            each_numel.append(p.data.numel())
        return sum(each_numel), each_numel
    
    
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
    

    def compute_newgrads(self, origin_grad=None, cur_iter=None,
                         total_mean_grad=False):
        if origin_grad is None:
            origin_grad = {k: self._grad2vec(k) for k in self.datasets}
            
        assert origin_grad is not None
        
        if self.grad_method.apply_method:
            if self.grad_method.require_copied_grad: copied_task_grad2vec = {k: origin_grad[k].clone().to(get_rank()) for k in self.datasets}
            else: copied_task_grad2vec = None
            self.grad_zero_shared_encoder 
            
            kwargs = {'iter': cur_iter}
            new_grads = self.grad_method.backward(origin_grad, copied_task_grad2vec, **kwargs)
        else:
            new_grads = sum(grad for grad in origin_grad.values())
        
        dist.all_reduce(new_grads)
        new_grads /= dist.get_world_size()
        
        if total_mean_grad: self._reset_grad(new_grads/len(origin_grad))
        else: self._reset_grad(new_grads)
        
        self._transfer_computed_grad()
    
    
    def forward(self, data_dict, kwargs):
        if self.training:
            output = self.task_single_network[kwargs['dataset']](data_dict)
            losses = {f"{kwargs['dataset']}_{k}": v for k, v in output.items()}
            
        else:
            dset = list(kwargs["task_list"].keys())[0]
            losses = self.task_single_network[dset](data_dict)
        
        return losses