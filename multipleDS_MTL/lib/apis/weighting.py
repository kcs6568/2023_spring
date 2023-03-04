import numpy as np
from collections import OrderedDict

import torch, sys, random
import torch.nn as nn
import torch.nn.functional as F

from ..utils.dist_utils import get_rank


class AbsWeighting(nn.Module):
    r"""An abstract class for weighting strategies.
    """
    def __init__(self):
        super(AbsWeighting, self).__init__()
        
    def init_param(self):
        r"""Define and initialize some trainable parameters required by specific weighting methods. 
        """
        pass

    def _compute_grad_dim(self):
        self.grad_index = []
        for param in self.get_share_params():
            self.grad_index.append(param.data.numel())
        self.grad_dim = sum(self.grad_index)

    def _grad2vec(self):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad

    def _compute_grad(self, losses, mode, rep_grad=False):
        '''
        mode: backward, autograd
        '''
        if not rep_grad:
            grads = torch.zeros(self.task_num, self.grad_dim).to(self.device)
            for tn in range(self.task_num):
                if mode == 'backward':
                    losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
                    grads[tn] = self._grad2vec()
                # elif mode == 'autograd':
                #     grad = list(torch.autograd.grad(losses[tn], self.get_share_params(), retain_graph=True))
                #     grads[tn] = torch.cat([g.view(-1) for g in grad])
                # else:
                #     raise ValueError('No support {} mode for gradient computation')
                self.zero_grad_share_params()
        # else:
        #     if not isinstance(self.rep, dict):
        #         grads = torch.zeros(self.task_num, *self.rep.size()).to(self.device)
        #     else:
        #         grads = [torch.zeros(*self.rep[task].size()) for task in self.task_name]
        #     for tn, task in enumerate(self.task_name):
        #         if mode == 'backward':
        #             losses[tn].backward(retain_graph=True) if (tn+1)!=self.task_num else losses[tn].backward()
        #             grads[tn] = self.rep_tasks[task].grad.data.clone()
        return grads

    def _reset_grad(self, new_grads):
        count = 0
        for param in self.get_share_params():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                param.grad.data = new_grads[beg:end].contiguous().view(param.data.size()).data.clone()
            count += 1
            
    def _get_grads(self, losses, mode='backward'):
        r"""This function is used to return the gradients of representations or shared parameters.
        If ``rep_grad`` is ``True``, it returns a list with two elements. The first element is \
        the gradients of the representations with the size of [task_num, batch_size, rep_size]. \
        The second element is the resized gradients with size of [task_num, -1], which means \
        the gradient of each task is resized as a vector.
        If ``rep_grad`` is ``False``, it returns the gradients of the shared parameters with size \
        of [task_num, -1], which means the gradient of each task is resized as a vector.
        """
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode, rep_grad=True)
            if not isinstance(self.rep, dict):
                grads = per_grads.reshape(self.task_num, self.rep.size()[0], -1).sum(1)
            else:
                try:
                    grads = torch.stack(per_grads).sum(1).view(self.task_num, -1)
                except:
                    raise ValueError('The representation dimensions of different tasks must be consistent')
            return [per_grads, grads]
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode)
            return grads
        
    def _backward_new_grads(self, batch_weight, per_grads=None, grads=None):
        r"""This function is used to reset the gradients and make a backward.
        Args:
            batch_weight (torch.Tensor): A tensor with size of [task_num].
            per_grad (torch.Tensor): It is needed if ``rep_grad`` is True. The gradients of the representations.
            grads (torch.Tensor): It is needed if ``rep_grad`` is False. The gradients of the shared parameters. 
        """
        if self.rep_grad:
            if not isinstance(self.rep, dict):
                # transformed_grad = torch.einsum('i, i... -> ...', batch_weight, per_grads)
                transformed_grad = sum([batch_weight[i] * per_grads[i] for i in range(self.task_num)])
                self.rep.backward(transformed_grad)
            else:
                for tn, task in enumerate(self.task_name):
                    rg = True if (tn+1)!=self.task_num else False
                    self.rep[task].backward(batch_weight[tn]*per_grads[tn], retain_graph=rg)
        else:
            # new_grads = torch.einsum('i, i... -> ...', batch_weight, grads)
            new_grads = sum([batch_weight[i] * grads[i] for i in range(self.task_num)])
            self._reset_grad(new_grads)
    
    @property
    def backward(self, losses, **kwargs):
        r"""
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass
    
    
class GradNorm(AbsWeighting):
    r"""Gradient Normalization (GradNorm).
    
    This method is proposed in `GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks (ICML 2018) <http://proceedings.mlr.press/v80/chen18a/chen18a.pdf>`_ \
    and implemented by us.
    Args:
        alpha (float, default=1.5): The strength of the restoring force which pulls tasks back to a common training rate.
    """
    def __init__(self):
        super(GradNorm, self).__init__()
        
    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([1.0]*self.task_num, device=self.device))
        
    def backward(self, losses, **kwargs):
        alpha = kwargs['alpha']
        if self.epoch >= 1:
            loss_scale = self.task_num * F.softmax(self.loss_scale, dim=-1)
            grads = self._get_grads(losses, mode='backward')
            if self.rep_grad:
                per_grads, grads = grads[0], grads[1]
                
            G_per_loss = torch.norm(loss_scale.unsqueeze(1)*grads, p=2, dim=-1)
            G = G_per_loss.mean(0)
            L_i = torch.Tensor([losses[tn].item()/self.train_loss_buffer[tn, 0] for tn in range(self.task_num)]).to(self.device)
            r_i = L_i/L_i.mean()
            constant_term = (G*(r_i**alpha)).detach()
            L_grad = (G_per_loss-constant_term).abs().sum(0)
            L_grad.backward()
            loss_weight = loss_scale.detach().clone()
            
            if self.rep_grad:
                self._backward_new_grads(loss_weight, per_grads=per_grads)
            else:
                self._backward_new_grads(loss_weight, grads=grads)
            return loss_weight.cpu().numpy()
        else:
            loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
            loss.backward()
            return np.ones(self.task_num)
        
        
class UncertaintyWeights(AbsWeighting):
    r"""Uncertainty Weights (UW).
    
    This method is proposed in `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR 2018) <https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf>`_ \
    and implemented by us. 
    """
    def __init__(self, task_list, init_value=-0.5):
        super(UncertaintyWeights, self).__init__()
        if not isinstance(init_value, float):
            init_value = float(init_value)
        self.method_name = "UW"
        logsigma = {}
        for d in task_list:
            logsigma.update({d: nn.Parameter(torch.tensor(init_value, requires_grad=True), requires_grad=True)})

        self.logsigma = nn.ParameterDict(logsigma)
        
    
    def __str__(self):
        param_st = ""
        for dset, k in self.logsigma.items():
            param_st += f"{dset}: {k.data} || "
        return param_st
    
    
    def forward(self, losses, **kwargs):
        logsigma_loss = {k: 1 / (2 * torch.exp(self.logsigma[k])) * v + self.logsigma[k] / 2 for k, v in losses.items()}
        # logsigma_loss = (losses/(2*self.logsigma.exp())+self.logsigma/2).sum()
        return logsigma_loss
        # loss.backward()
        # return (1/(2*torch.exp(self.logsigma))).detach().cpu().numpy()
        
        
class DynamicWeightAverage(AbsWeighting):
    r"""Dynamic Weight Average (DWA).
    
    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_. 
    Args:
        T (float, default=2.0): The softmax temperature.
    """
    def __init__(self, task_list, total_epoch, temperature=2):
        super(DynamicWeightAverage, self).__init__()
        self.train_loss_buffer = {t: torch.zeros([total_epoch]) for t in task_list}
        self.task_num = len(task_list)
        self.temperature = temperature
        self.epoch = 0
        
        
    def set_epoch(self, cur_epoch):
        self.epoch = cur_epoch
        
        
    def save_loss_to_buffer(self, loss_dict):
        for k, loss in loss_dict.items(): self.train_loss_buffer[k][self.epoch] = loss
    
    
    def is_valid_epoch(self):
        return self.epoch > 1
    
    
    def forward(self, losses, **kwargs):
        assert isinstance(losses, (dict, OrderedDict))
        
        w_i = {k: torch.Tensor(
            self.train_loss_buffer[k][self.epoch-1]/self.train_loss_buffer[k][self.epoch-2]
            ).to(get_rank()) for k in self.train_loss_buffer.keys()}
        
        batch_weight = self.task_num*F.softmax(torch.stack(list(w_i.values()))/self.temperature, dim=-1)
        
        total_loss = {f"DWA_{k}_loss": batch_weight[i] * v for i, (k, v) in enumerate(losses.items())}
        
        return total_loss

            
            # w_i = torch.Tensor(self.train_loss_buffer[:,self.epoch-1]/self.train_loss_buffer[:,self.epoch-2]).to(get_rank())
            # batch_weight = self.task_num*F.softmax(w_i/self.temperature, dim=-1)


class SimpleWeighting(AbsWeighting):
    def __init__(self, data_list, initial_value=1.):
        super(SimpleWeighting, self).__init__()
        self.task_indices = {k: i for i, k in enumerate(data_list)}
        self.global_parameter = nn.Parameter(torch.tensor(initial_value, requires_grad=True), requires_grad=True)
        self.task_parameter = nn.ParameterDict({
            d: nn.Parameter(torch.tensor(initial_value, requires_grad=True), requires_grad=True) for d in data_list
        })
    
    def forward(self, losses, **kwargs):
        assert isinstance(losses, (dict, OrderedDict))
        weight_prob = F.softmax(torch.stack(list(self.task_parameter.values())), dim=-1)
        total_loss = {f"simple_{k}_loss": weight_prob[self.task_indices[k]] * v for k, v in (losses.items())}
        
        return total_loss

            
            # w_i = torch.Tensor(self.train_loss_buffer[:,self.epoch-1]/self.train_loss_buffer[:,self.epoch-2]).to(get_rank())
            # batch_weight = self.task_num*F.softmax(w_i/self.temperature, dim=-1)
            
            
class PCGrad(AbsWeighting):
    r"""Project Conflicting Gradients (PCGrad).
    
    This method is proposed in `Gradient Surgery for Multi-Task Learning (NeurIPS 2020) <https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html>`_ \
    and implemented by us.

    .. warning::
            PCGrad is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self):
        super(PCGrad, self).__init__()
        
    def backward(self, losses, **kwargs):
        batch_weight = np.ones(len(losses))
        if self.rep_grad:
            raise ValueError('No support method PCGrad with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward') # [task_num, grad_dim]
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            random.shuffle(task_index)
            for tn_j in task_index:
                g_ij = torch.dot(pc_grads[tn_i], grads[tn_j])
                if g_ij < 0:
                    pc_grads[tn_i] -= g_ij * grads[tn_j] / (grads[tn_j].norm().pow(2))
                    batch_weight[tn_j] -= (g_ij/(grads[tn_j].norm().pow(2))).item()
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        return batch_weight
    
    
def define_weighting_method(weight_method, **kwargs):
    print(weight_method, kwargs)
    if weight_method == 'uw':
        return UncertaintyWeights(**kwargs)
    elif weight_method == 'dwa':
        return DynamicWeightAverage(**kwargs)