import numpy as np
from collections import OrderedDict
from scipy.optimize import minimize

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
    
    
    def compute_shared_encoder_numel(self, shared_encoder):
        each_numel = []
        for p in shared_encoder.parameters():
            each_numel.append(p.data.numel())
        
        self.grad_index = each_numel
        self.grad_dim = sum(each_numel)
        
        # return sum(each_numel), each_numel
    
    
    # def _compute_grad_dim(self, shared_module):
    #     self.grad_index = []
    #     for param in shared_module.parameters():
    #         self.grad_index.append(param.data.numel())
    #     self.grad_dim = sum(self.grad_index)


    def _grad2vec(self, shared_encoder):
        grad = torch.zeros(self.grad_dim)
        count = 0
        for param in shared_encoder.parameters():
            if param.grad is not None:
                beg = 0 if count == 0 else sum(self.grad_index[:count])
                end = sum(self.grad_index[:(count+1)])
                grad[beg:end] = param.grad.data.view(-1)
            count += 1
        return grad


    def _compute_grad(self, losses, datasets, zero_grad_fn, retain_graph=True, rep_grad=False):
        # losses.backward(retain_graph=retain_graph)
        # grads = self._grad2vec()
        # return grads
        
        if not rep_grad:
            return_grads = {}
            for idx, data in enumerate(datasets):
                grads = torch.zeros( self.grad_dim).to(torch.cuda.current_device())
                losses[data].backward(retain_graph=True) if (idx+1)!=len(datasets) else losses[data].backward()
                return_grads[data] = self._grad2vec()
                zero_grad_fn()
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
        
        
class PCGrad(AbsWeighting):
    r"""Project Conflicting Gradients (PCGrad).
    
    This method is proposed in `Gradient Surgery for Multi-Task Learning (NeurIPS 2020) <https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html>`_ \
    and implemented by us.

    .. warning::
            PCGrad is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self):
        super(PCGrad, self).__init__()
        self.t_2 = 0
        self.t_1 = 0
        self.if_start = 100
        self.epoch = 0
        self.count_len = 0
        self.print_str = True
        
        
    def init_params(self, task_list, **params):
        self.task_list = task_list
        for k, v in params.items(): setattr(self, k, v)
        # self.surgery_count = {k: [] for k in self.task_list}
        self.surgery_count = []
        self.epoch_count = {k: 0 for k in self.task_list}
        self.iter_surgery_count = {k: 0 for k in self.task_list}
        
        self.use_alter_step = True if "alter_step" in params else False
        if self.use_alter_step:
            self.apply_method = True if self.epoch in self.alter_step else False
        else: self.apply_method = True
        
    def after_iter(self):
        self.iter_surgery_count = {k: 0 for k in self.task_list}
        self.epoch += 1
        
        if self.use_alter_step:
            self.apply_method = True if self.epoch in self.alter_step else False
    
    
    # def set_epoch(self, epoch):
    #     self.epoch = epoch
        
    # def get_epoch(self):
    #     return self.epoch
    
    
    # def after_iter(self):
    #     self.epoch_count = {k: 0 for k in self.task_list}
    #     self.save_count()

    
    # def save_count(self):
    #     if self.get_epoch() == 0:
    #         self.t_2 = sum(self.surgery_count)
    #         self.count_len = len(self.surgery_count)
    #         # print(self.t_2, self.count_len)
        
    #     elif self.get_epoch() == 1:
    #         self.t_1 = sum(self.surgery_count[self.count_len:])
    #         # print(self.t_1)
    #         self.imf_12 = self.t_1 + self.t_2
    #         # print(self.imf_12)
    #         # self.count_len *= 2
    #     else:
    #         for i in range(2, 0, -1):
    #             start = self.count_len * (self.get_epoch() - (i-1))
    #             end = start + self.count_len
    #             setattr(self, f"t_{str(i)}", sum(self.surgery_count[start:end]))
    #             # print(start, end, getattr(self, f"t_{str(i)}"))
    #             # self.t_2 = sum(self.surgery_count[start:end])
                
    #         self.imf_12 = self.t_1 + self.t_2
    #         # print(self.imf_12)
    #         # print()
    
    
    # def set_information(self, saved_information):
    #     for k, v in saved_information.items(): setattr(self, k, v)    
        
    @property
    def get_surgery_count(self):
        return self.surgery_count
    
    @property
    def get_save_information(self):
        return_vars = {}
        for k, v in vars(self).items():
            if not k.startswith("_"): return_vars.update({f"grad_{k}": v})
        
        if hasattr(self, 'weighting_method'):
            for k, v in vars(self.weighting_method).items():
                if not k.startswith("_"): return_vars.update({f"weight_{k}": v})
                
        return return_vars        
    
    
    def set_saved_information(self, information):
        for k, v in information.items():
            type_position = k.find("_")
            types = k[:type_position]
            key = k[type_position+1:]
            
            if types == "grad":
                setattr(self, key, v)
                
            elif types == "weight":
                assert hasattr(self, "weighting_method")
                setattr(self.weighting_method, key, v)
            
        
    def backward(self, origin_grad, copied_grads=None, 
                 **kwargs):
        assert copied_grads is not None and self.require_copied_grad
        assert len(origin_grad) > 1 and len(copied_grads) > 1
        
        per_iteration_count = 0
        for std_dataset_idx in range(len(self.task_list)):
            random_task_index = list(range(len(self.task_list)))
            random_task_index.remove(std_dataset_idx)
            random.shuffle(random_task_index)
            
            std_task = self.task_list[std_dataset_idx]
            for rand_task_idx in random_task_index:
                rand_task = self.task_list[rand_task_idx]
                dot_grad = torch.dot(copied_grads[std_task], origin_grad[rand_task])
                
                if kwargs['positive_surgery']:
                    if dot_grad > 0:
                        self.epoch_count[std_task] += 1
                        self.iter_surgery_count[std_task] += 1
                        # text += f"{std_task}---{rand_task_idx} (dot: {dot_grad}) | "
                        per_iteration_count += 1
                        copied_grads[std_task] -= dot_grad * origin_grad[rand_task] / (origin_grad[rand_task].norm().pow(2))
                    
                else:
                    if dot_grad < 0:
                        self.epoch_count[std_task] += 1
                        self.iter_surgery_count[std_task] += 1
                        # text += f"{std_task}---{rand_task_idx} (dot: {dot_grad}) | "
                        per_iteration_count += 1
                        copied_grads[std_task] -= dot_grad * origin_grad[rand_task] / (origin_grad[rand_task].norm().pow(2))
        
        self.surgery_count.append(per_iteration_count) 
        
        if not kwargs['positive_surgery']:
            new_grads = sum(grad for grad in copied_grads.values())
            return new_grads
        
        else:
            return copied_grads
    
    
    def __str__(self) -> str:
        text = f"Iteration Surgery Count: {self.iter_surgery_count}"
        return text
        
        
class GradVac(AbsWeighting):
    r"""Gradient Vaccine (GradVac).
    
    This method is proposed in `Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models (ICLR 2021 Spotlight) <https://openreview.net/forum?id=F1vEjWK-lH_>`_ \
    and implemented by us.
    Args:
        beta (float, default=0.5): The exponential moving average (EMA) decay parameter.
    .. warning::
            GradVac is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.
    """
    def __init__(self):
        super(GradVac, self).__init__()
    
    
    def init_params(self, task_list, **params):
        self.task_list = task_list
        self.rho_T = torch.zeros(len(self.task_list), len(self.task_list)).to(torch.cuda.current_device()) # rho value at the previous training step, i.e., iteration
        for k, v in params.items(): setattr(self, k, v)
        
    def backward(self, origin_grad, copied_grads=None, **kwargs):
        assert copied_grads is not None and self.require_copied_grad
        
        for std_dataset_idx in range(len(self.task_list)):
            random_task_index = list(range(len(self.task_list)))
            random_task_index.remove(std_dataset_idx)
            random.shuffle(random_task_index)
            std_task = self.task_list[std_dataset_idx]
            
            for rand_task_idx in random_task_index:
                cur_task = self.task_list[rand_task_idx]
                
                rho_ij = torch.dot(copied_grads[std_task], origin_grad[cur_task]) / (copied_grads[std_task].norm()*origin_grad[cur_task].norm())
                if rho_ij < self.rho_T[std_dataset_idx, rand_task_idx]:
                    w = copied_grads[std_task].norm()*(self.rho_T[std_dataset_idx, rand_task_idx]*(1-rho_ij**2).sqrt()-rho_ij*(1-self.rho_T[std_dataset_idx, rand_task_idx]**2).sqrt())/(origin_grad[cur_task].norm()*(1-self.rho_T[std_dataset_idx, rand_task_idx]**2).sqrt())
                    copied_grads[std_task] += origin_grad[cur_task]*w
                    self.rho_T[std_dataset_idx, rand_task_idx] = (1-self.beta)*self.rho_T[std_dataset_idx, rand_task_idx] + self.beta*rho_ij
        
        new_grads = sum(grad for grad in copied_grads.values())
        return new_grads
        
        # self._reset_grad(new_grads)
        # return batch_weight
    

class CAGrad(AbsWeighting):
    r"""Conflict-Averse Gradient descent (CAGrad).
    
    This method is proposed in `Conflict-Averse Gradient Descent for Multi-task learning (NeurIPS 2021) <https://openreview.net/forum?id=_61Qh8tULj_>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/Cranial-XIX/CAGrad>`_. 
    Args:
        calpha (float, default=0.5): A hyperparameter that controls the convergence rate.
        rescale ({0, 1, 2}, default=1): The type of the gradient rescaling.
    .. warning::
            CAGrad is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.
    """
    def __init__(self):
        super(CAGrad, self).__init__()
        
    
    def init_params(self, task_list=None, **params):
        self.task_list = task_list
        for k, v in params.items(): setattr(self, k, v); print(k, v)
        
        
    def backward(self, origin_grads, copied_grads=None, **kwargs):
        assert copied_grads is None and not self.require_copied_grad
        
        origin_grads = torch.stack(list(origin_grads.values()))
        
        GG = torch.matmul(origin_grads, origin_grads.t()).cpu() # [num_tasks, num_tasks]
        g0_norm = (GG.mean()+1e-8).sqrt() # norm of the average gradient
        x_start = np.ones(len(self.task_list)) / len(self.task_list)
        
        # x_start = np.ones(len(self.task_list)) / len(self.task_num)
        bnds = tuple((0,1) for _ in x_start)
        cons=({'type':'eq','fun':lambda x:1-sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (self.calpha*g0_norm+1e-8).item()
        def objfn(x):
            return (x.reshape(1,-1).dot(A).dot(b.reshape(-1,1))+c*np.sqrt(x.reshape(1,-1).dot(A).dot(x.reshape(-1,1))+1e-8)).sum()
        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(torch.cuda.current_device())
        gw = (origin_grads * ww.view(-1, 1)).sum(0)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm+1e-8)
        g = origin_grads.mean(0) + lmbda * gw
        if self.rescale == 0:
            new_grads = g
        elif self.rescale == 1:
            new_grads = g / (1+self.calpha**2)
        elif self.rescale == 2:
            new_grads = g / (1 + self.calpha)
        else:
            raise ValueError('No support rescale type {}'.format(self.rescale))
        
        return new_grads
        # self._reset_grad(new_grads)
        # return w_cpu


class PosNegPCGrad(AbsWeighting):
    def __init__(self):
        super(PosNegPCGrad, self).__init__()
        self.dir_types = ["pos", "neg"]
        self.sim_func = nn.CosineSimilarity(dim=0)
        self.all_max_task = []
        self.all_pos_count = []
        self.all_neg_count = []
        self.iter_pos_count = 0
        self.iter_neg_count = 0
        
        
        
    def init_params(self, task_list, **params):
        self.task_list = task_list
        for k, v in params.items(): setattr(self, k, v)
        self.iter_max_norm_task_count = {k: 0 for k in self.task_list}
    
    
    def after_iter(self):
        self.iter_pos_count = 0
        self.iter_neg_count = 0
        self.iter_max_norm_task_count.update({k: 0 for k in self.task_list})

    
    @property
    def get_save_information(self):
        return_vars = {}
        for k, v in vars(self).items():
            if not k.startswith("_"): return_vars.update({f"grad_{k}": v})
        
        if hasattr(self, 'weighting_method'):
            for k, v in vars(self.weighting_method).items():
                if not k.startswith("_"): return_vars.update({f"weight_{k}": v})
                
        return return_vars        
    
    
    def pos_neg_grad(self, grads):
        norms = torch.tensor([grad.norm(p=self.norm_type) for grad in grads.values()]).to(get_rank())
        # max_index = torch.topk(norms, k=1)[1]
        # index_list, _ = torch.sort(norms, descending=self.descending)
        # target_index = index_list[0]
        
        target_index = torch.argmin(norms).int()
        self.all_max_task.append(target_index)
        self.iter_max_norm_task_count[self.task_list[target_index]] += 1
        
        others = list(range(len(self.task_list)))
        others.remove(target_index)
        
        pos_grad = [grads[self.task_list[target_index]]]
        neg_grad = []
        
        for idx in others:
            cos_sim = self.sim_func(pos_grad[0], grads[self.task_list[idx]])
            if cos_sim > 0: pos_grad.append(grads[self.task_list[idx]])
            elif cos_sim < 0: neg_grad.append(grads[self.task_list[idx]])

        self.all_pos_count.append(len(pos_grad))
        self.all_neg_count.append(len(neg_grad))
        self.iter_pos_count += len(pos_grad)
        self.iter_neg_count += len(neg_grad)
        
        if len(neg_grad) == 0:
            if self.mean_grad: return sum(pos_grad) / len(pos_grad)
            else: return sum(pos_grad)
        else:
            if self.mean_grad: return {
                "pos": (sum(pos_grad) / len(pos_grad)).to(get_rank()),
                "neg": (sum(neg_grad) / len(neg_grad)).to(get_rank())}
            else: return {"pos": sum(pos_grad).to(get_rank()), "neg": sum(neg_grad).to(get_rank())}
    
    
    
    def backward(self, origin_grad, copied_grads=None, **kwargs):
        assert origin_grad is not None and copied_grads is None
        
        pos_neg_grad = self.pos_neg_grad(origin_grad)
        
        if isinstance(pos_neg_grad, dict):
            direc_type = None
            if self.final_direction == 'min':
                pos_norm, neg_norm = pos_neg_grad["pos"].norm(), pos_neg_grad["neg"].norm()
                direc_type = "neg" if pos_norm > neg_norm else "pos"
            
            elif self.final_direction == 'max':
                pos_norm, neg_norm = pos_neg_grad["pos"].norm(), pos_neg_grad["neg"].norm()
                direc_type = "pos" if pos_norm > neg_norm else "neg"
                
            copied_grads = {k: pos_neg_grad[k].clone().to(get_rank()) for k in self.dir_types}
            
            
            # the direction of negative gradient should head toward the direction of positive gradient
            # copied_grads["neg"] -= torch.dot(pos_neg_grad["pos"], pos_neg_grad["neg"]) * pos_neg_grad["pos"] / (pos_neg_grad["pos"].norm().pow(2))
            
            #############################
            # the direction (i.e., direc_type variable)
            # is the direction of mininum gradient between positive and negative gradient
            #############################
            
            dot_grad = torch.dot(pos_neg_grad["pos"], pos_neg_grad["neg"])
            for type_idx in range(2): # pos and neg
                if type_idx == 0:
                    cri_type = "pos"
                    other_type = "neg"
                else:
                    cri_type = "neg"
                    other_type = "pos"
                
                if direc_type is not None:
                    if direc_type == cri_type: continue
                
                copied_grads[cri_type] -= dot_grad * pos_neg_grad[other_type] / (pos_neg_grad[other_type].norm().pow(2))
            
            new_grads = sum(grad for grad in copied_grads.values())
            
            if self.posneg_mean: return new_grads / 2
            else: return new_grads
        
        else: 
            if self.posneg_mean: return pos_neg_grad / 2
            else: return pos_neg_grad
        
        
    def __str__(self) -> str:
        text = ""
        text += f"Current Iteration Information:\n"
        text += f"Positive Count: {self.iter_pos_count} | Negative Count: {self.iter_neg_count}\n"
        text += f"Target Norm Task Count: {self.iter_max_norm_task_count}"
        return text
        

class BasicGrad:
    def __init__(self):
        super(BasicGrad, self).__init__()
        self.print_str = False
        
        
    def init_params(self, task_list, **params):
        if len(params) != 0:
            for k, v in params.items(): setattr(self, k, v)
    
    
    def backward(self, origin_grad, copied_grads=None, **kwargs):
        assert origin_grad is not None and copied_grads is None
        return torch.stack(list(origin_grad.values())).mean(dim=0)
        
        
def define_gradient_method(type):
    g_type = type
    if g_type == 'pcgrad':
        return PCGrad()
    elif g_type == 'gradvac':
        return GradVac()
    elif g_type == 'cagrad':
        return CAGrad()
    elif g_type == "posnegpcgrad":
        return PosNegPCGrad()
    elif g_type == "basic_mean":
        return BasicGrad()

    else: return None
    
    
    
