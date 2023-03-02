import numpy as np
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# from ..modules.get_detector import build_detector, DetStem
# from ..modules.get_backbone import build_backbone
# from ..modules.get_segmentor import build_segmentor, SegStem
# from ..modules.get_classifier import build_classifier, ClfStem
# from ...apis.weighting import SimpleWeighting

from ...model_api.task_model.single_task import SingleTaskNetwork
from ...utils.dist_utils import get_rank


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


class MainMethod(nn.Module):
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
        self.base_dataset = list(task_cfg.keys())[0]
        self.all_shared_params_numel, self.each_param_numel = self._compute_shared_encoder_numel
        
        self.make_gate_logits(self.datasets, len(self.datasets))    
        self.policys = {dset: torch.zeros(gate.size()).float() for dset, gate in self.task_gating_params.items()}
        
        
    
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
    
    
    # @property
    # def _normalize_all_grad(self):    
    #     for dset in self.datasets:
    #         for p in self.get_network(dset).encoder.parameters(): p.grad.div_(p.grad.norm())
                
    
    def compute_cosine_sim(self):
        # self._normalize_all_grad
        
        cos = nn.CosineSimilarity(dim=0)
        
        for v1, v2, v3 in zip(
            self.task_single_network['cifar10'].encoder.parameters(),
            self.task_single_network['minicoco'].encoder.parameters(),
            self.task_single_network['voc'].encoder.parameters()):
            unit_v1 = torch.div(v1.grad, v1.grad.norm())
            unit_v2 = torch.div(v2.grad, v2.grad.norm())
            unit_v3 = torch.div(v3.grad, v3.grad.norm())
            
            cos12 = cos(unit_v1, unit_v2)
            cos23 = cos(unit_v2, unit_v3)
            cos13 = cos(unit_v1, unit_v3)
            
            mean_sim = torch.mean(torch.stack([cos12, cos23, cos13]))
            
            v1.grad.mul_(mean_sim)
            v2.grad.mul_(mean_sim)
            v3.grad.mul_(mean_sim)
            

        
    def _transfer_computed_grad(self):
        for d in self.datasets[1:]:
            for base, target in zip(
                self.get_shared_encoder.parameters(),
                self.task_single_network[d].encoder.parameters()):
                target.grad = base.grad
    
    
    def forward(self, data_dict, kwargs):
        if self.training:
            task_output = self.task_single_network[kwargs['dataset']](data_dict)
        else:
            dset = list(kwargs["task_list"].keys())[0]
            task_output = self.task_single_network[dset](data_dict)
        
        return task_output
