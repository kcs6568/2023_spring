import math

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


class Fromage(torch.optim.Optimizer):

    def __init__(self, params, lr=0.01, p_bound=None):
        """The Fromage optimiser.
        Arguments:
            lr (float): The learning rate. 0.01 is a good initial value to try.
            p_bound (float): Restricts the optimisation to a bounded set. A
                value of 2.0 restricts parameter norms to lie within 2x their
                initial norms. This regularises the model class.
        """
        self.p_bound = p_bound
        defaults = dict(lr=lr)
        super(Fromage, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0 and self.p_bound is not None:
                    state['max'] = self.p_bound*p.norm().item()
                
                d_p = p.grad.data
                d_p_norm = p.grad.norm()
                p_norm = p.norm()

                if p_norm > 0.0 and d_p_norm > 0.0:
                    p.data.add_(d_p * (p_norm / d_p_norm), alpha=-group['lr'])
                else:
                    p.data.add_(d_p, alpha=-group['lr'])
                p.data /= math.sqrt(1+group['lr']**2)

                if self.p_bound is not None:
                    p_norm = p.norm().item()
                    if p_norm > state['max']:
                        p.data *= state['max']/p_norm

        return loss
    
    
class AdamP(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, delta=0.1, wd_ratio=0.1, nesterov=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        delta=delta, wd_ratio=wd_ratio, nesterov=nesterov)
        super(AdamP, self).__init__(params, defaults)

    def _channel_view(self, x):
        return x.view(x.size(0), -1)

    def _layer_view(self, x):
        return x.view(1, -1)

    def _cosine_similarity(self, x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        return F.cosine_similarity(x, y, dim=1, eps=eps).abs_()

    def _projection(self, p, grad, perturb, delta, wd_ratio, eps):
        wd = 1
        expand_size = [-1] + [1] * (len(p.shape) - 1)
        for view_func in [self._channel_view, self._layer_view]:

            cosine_sim = self._cosine_similarity(grad, p.data, eps, view_func)

            if cosine_sim.max() < delta / math.sqrt(view_func(p.data).size(1)):
                p_n = p.data / view_func(p.data).norm(dim=1).view(expand_size).add_(eps)
                perturb -= p_n * view_func(p_n * perturb).sum(dim=1).view(expand_size)
                wd = wd_ratio

                return perturb, wd

        return perturb, wd

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                beta1, beta2 = group['betas']
                nesterov = group['nesterov']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                # Adam
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1

                if nesterov:
                    perturb = (beta1 * exp_avg + (1 - beta1) * grad) / denom
                else:
                    perturb = exp_avg / denom

                # Projection
                wd_ratio = 1
                if len(p.shape) > 1:
                    perturb, wd_ratio = self._projection(p, grad, perturb, group['delta'], group['wd_ratio'], group['eps'])

                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'] * wd_ratio)

                # Step
                p.data.add_(perturb, alpha=-step_size)

        return loss




def select_scheduler(optimizer, lr_config, args):
    scheduler_type = lr_config['type']
    
    if scheduler_type  == "step":
        step_size = 1 if not 'step_size' in lr_config else lr_config['step_size']
        gamma = 0.1 if not 'gamma' in lr_config else lr_config['gamma']
        
        main_sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
    elif scheduler_type == "multi":
        milestones = [8, 11] if not 'lr_steps' in lr_config else lr_config['lr_steps']
        gamma = 0.1 if not 'gamma' in lr_config else lr_config['gamma']
        
        main_sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        
    elif scheduler_type == "cosine":
        t_max = lr_config['T_max'] if 'T_max' in lr_config else args.epochs
        eta_min = lr_config['eta_min'] if 'eta_min' in lr_config else 0
        main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        
    elif scheduler_type == "exp":
        main_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_config['gamma'])
        
    elif scheduler_type == "cycle":
        assert args.lr == lr_config['max_lr']
        cycle_momentum = False if 'adam' in args.opt else True
        main_sch = torch.optim.lr_scheduler.CyclicLR(
            optimizer, 
            base_lr=lr_config['base_lr'], max_lr=lr_config['max_lr'], 
            step_size_up=lr_config['step_size_up'], step_size_down=lr_config['step_size_down'], 
            mode=lr_config['mode'], cycle_momentum=cycle_momentum)
        
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{scheduler_type}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    
    return main_sch



def get_optimizer_for_gating(args, model):
    optimizer = {}
    
    main_opt = None
    gate_opt = None
    
    gate_args = None
    if 'gate_opt' in args:
        if args.gate_opt is not None:
            gate_args = args.gate_opt

    if gate_args is not None:
        main_lr = args.lr
        gate_lr = gate_args['gating_lr']
        
        if main_lr != gate_lr:
            main_params = {'params': [p for n, p in model.named_parameters() if p.requires_grad and not 'gating' in n]}
            gate_params = {
                'params': [p for n, p in model.named_parameters() if p.requires_grad and 'gating' in n],
                'lr': gate_lr
                }

            all_params = [main_params, gate_params]
        
        else:
            all_params = [p for p in model.parameters() if p.requires_grad]
    
    else:
        all_params = [p for p in model.parameters() if p.requires_grad]
            
    if args.opt == 'sgd':
        main_opt = torch.optim.SGD(all_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'nesterov':
        main_opt = torch.optim.SGD(all_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.opt =='adam':
        if 'eps' in args:
            main_opt = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay, 
                                        eps=float(args.eps))
        else:
            main_opt = torch.optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt =='adamw':
        main_opt = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    
    
    assert main_opt is not None
    
    if gate_opt is None: # not use the gate-specific optimizer
        return main_opt
    else:
        optimizer.update({"main": main_opt, "gate": gate_opt})
        return optimizer
    
    
def get_scheduler_for_gating(args, optimizer):
    main_sch = None
    gate_sch = None
    
    if 'only_gate_opt' in args:
        if args.only_gate_opt and args.only_gate_step is not None :
            lr_step = args.only_gate_step
        
        else:
            lr_step = args.lr_steps    
    else:
        lr_step = args.lr_steps
        
    if isinstance(optimizer, dict):
        main_opt = optimizer['main']
    else:
        main_opt = optimizer
    
    if args.lr_scheduler == "step":
        main_sch = torch.optim.lr_scheduler.StepLR(main_opt, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == "multi":
        main_sch = torch.optim.lr_scheduler.MultiStepLR(main_opt, milestones=lr_step, gamma=args.gamma)
    elif args.lr_scheduler == "cosine":
        main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(main_opt, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
    
    if args.gating_scheduler is not None:
        assert 'gate' in optimizer
        gate_opt = optimizer['gate']
        
        if args.gating_scheduler == "step":
            gate_sch = torch.optim.lr_scheduler.StepLR(gate_opt, step_size=args.step_size, gamma=args.gamma)
        elif args.gating_scheduler == "multi":
            gate_sch = torch.optim.lr_scheduler.MultiStepLR(gate_opt, milestones=lr_step, gamma=args.gamma)
        elif args.gating_scheduler == "cosine":
            gate_sch = torch.optim.lr_scheduler.CosineAnnealingLR(gate_opt, T_max=args.epochs)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
            )
        
    if gate_sch is not None:
        return {'main': main_sch, 'gate': gate_sch}
    
    else:
        return main_sch


def get_optimizer(args, model):
    if "opt_config" in args:
        assert args.opt_config is not None
        
        
        params = []
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not 'gating' in n]}
        if "sep_keys" in args.opt_config:
            else_dict = {'params': [p for n, p in model.named_parameters() if p.requires_grad and not args.opt_config["sep_keys"] in n]}
            params.append(else_dict)
            
            param_dict = {'params': [p for n, p in model.named_parameters() if p.requires_grad and args.opt_config["sep_keys"] in n], "lr": args.opt_config["lr"]}
            params.append(param_dict)
            
        else:
            params = [p for n, p in model.named_parameters() if p.requires_grad] # in ddp setting
            
    else:
        params = [p for n, p in model.named_parameters() if p.requires_grad] # in ddp setting
    
    if "lr" in args.lr_config:
        args.lr = args.lr_config["lr"]
    
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'nesterov':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.opt =='adam':
        if 'eps' in args: optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, eps=float(args.eps))
        else: optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt =='adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt =='adamp':
        optimizer = AdamP(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "fromage":
        optimizer = Fromage(params, lr=args.lr)    
    

    return optimizer

    
def get_scheduler(args, optimizer):
    assert isinstance(args.step_size, int)
    assert isinstance(args.gamma, float)
    assert isinstance(args.lr_steps, list)
    
    lr_config = args.lr_config
    if isinstance(lr_config, list) and len(lr_config) > 1:
        check_type = torch.zeros(len(lr_config))
        for i, c in enumerate(lr_config.values()):
            if args.lr_scheduler == c['type']: check_type[i] = 1
        assert torch.any(check_type.bool())
        assert sum(check_type) == 1
        config_index = int(check_type.nonzero()[0][0])
        lr_config = args.lr_config[config_index]
    
    else:
        assert args.lr_scheduler == lr_config['type']
    
    
    if lr_config['type'] == 'seq':
        scheduler_seq = []
        for config in lr_config['each_config']:
            scheduler = select_scheduler(optimizer, config, args)
            scheduler_seq.append(scheduler)
        
        main_sch = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=scheduler_seq, milestones=lr_config['milestones'])
    
    else: main_sch = select_scheduler(optimizer, lr_config, args)
    
    return main_sch