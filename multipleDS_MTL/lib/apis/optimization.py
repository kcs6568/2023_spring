from collections import OrderedDict
from pickletools import optimize
from sqlite3 import paramstyle

import torch
from torch.optim.lr_scheduler import _LRScheduler


def select_scheduler(optimizer, lr_config, args):
    scheduler_type = lr_config['type']
    
    if scheduler_type  == "step":
        main_sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_config['step_size'], gamma=lr_config['gamma'])
        
    elif scheduler_type == "multi":
        main_sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_config['lr_steps'], gamma=lr_config['gamma'])
        
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
    if 'gate_opt' in args or args.gate_opt is not None:
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
    params = [p for n, p in model.named_parameters() if p.requires_grad] # in ddp setting
        
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'nesterov':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.opt =='adam':
        if 'eps' in args: optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, eps=float(args.eps))
        else: optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt =='adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

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
    
    # if args.lr_scheduler == "step":
    #     main_sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_config['step_size'], gamma=lr_config['gamma'])
        
    # elif args.lr_scheduler == "multi":
    #     main_sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_config['lr_steps'], gamma=lr_config['gamma'])
        
    # elif args.lr_scheduler == "cosine":
    #     t_max = lr_config['T_max'] if 'T_max' in lr_config else args.epochs
    #     eta_min = lr_config['eta_min'] if 'eta_min' in lr_config else 0
    #     main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        
    # elif args.lr_scheduler == "exp":
    #     main_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_config['gamma'])
        
    # elif args.lr_scheduler == "cycle":
    #     assert args.lr == lr_config['max_lr']
    #     cycle_momentum = False if 'adam' in args.opt else True
    #     main_sch = torch.optim.lr_scheduler.CyclicLR(
    #         optimizer, 
    #         base_lr=lr_config['base_lr'], max_lr=lr_config['max_lr'], 
    #         step_size_up=lr_config['step_size_up'], step_size_down=lr_config['step_size_down'], 
    #         mode=lr_config['mode'], cycle_momentum=cycle_momentum)
    
        
    # else:
    #     raise RuntimeError(
    #         f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
    #     )
    
    return main_sch