import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...model_api.task_model.single_task import SingleTaskNetwork
from ...utils.dist_utils import get_rank
from ...apis.weighting import PCGrad



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


class PCGradMTL(nn.Module):
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
        self.dataset_index = list(range(len(self.datasets)))
        self.weight_method = PCGrad()
        self.base_dataset = list(task_cfg.keys())[0]
        self.all_shared_params_numel, self.each_param_numel = self._compute_shared_encoder_numel

        
    @property
    def get_shared_encoder(self):
        return self.get_network(self.base_dataset).encoder
    
    @property
    def _compute_shared_encoder_numel(self):
        each_numel = []
        for p in self.get_shared_encoder.parameters():
            each_numel.append(p.data.numel())
        return sum(each_numel), each_numel
    
    @property
    def grad_zero_shared_encoder(self):
        for net in self.task_single_network.values(): net.encoder.zero_grad()
    
    
    def get_task_stem(self, task):
        return self.task_single_network[task].stem
    
    
    def get_task_head(self, task):
        return self.task_single_network[task].head
    
    
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
        
        
    def _reset_grad(self, new_grads):
        count = 0
        for n, param in self.get_shared_encoder.named_parameters():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.each_param_numel[:count])
                end = sum(self.each_param_numel[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
        
    
    def _transfer_computed_grad(self):
        for d in self.datasets[1:]:
            for base, target in zip(
                self.get_shared_encoder.parameters(),
                self.task_single_network[d].encoder.parameters()):
                target.grad = base.grad
                
        
    def compute_pcgrad(self):
        origin_grad = {k: self._grad2vec(k) for k in self.datasets}
        pcgrad = {k: origin_grad[k].clone().to(get_rank()) for k in self.datasets}
        
        self.grad_zero_shared_encoder # the grad None and grad zero is different
        
        
        for std_dataset in self.datasets:
            random_task_index = list(range(len(self.datasets))); random.shuffle(random_task_index)
            
            for rand_task_idx in random_task_index:
                rand_dataset = self.datasets[rand_task_idx]
                dot_grad = torch.dot(pcgrad[std_dataset], origin_grad[rand_dataset])
                
                if dot_grad < 0:
                    pcgrad[std_dataset] -= dot_grad * origin_grad[rand_dataset] / (origin_grad[rand_dataset].norm().pow(2))
                    # batch_weight[rand_dataset] -= (dot_grad/(origin_grad[rand_dataset].norm().pow(2))).item()
        
        new_grads = sum(grad for grad in pcgrad.values())
        self._reset_grad(new_grads)
        self._transfer_computed_grad()
    
    
    def forward(self, data_dict, kwargs):
        if self.training:
            task_output = self.task_single_network[kwargs['dataset']](data_dict)
        else:
            dset = list(kwargs["task_list"].keys())[0]
            task_output = self.task_single_network[dset](data_dict)
        
        return task_output