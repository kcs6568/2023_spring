from collections import OrderedDict
from pickletools import optimize
from sqlite3 import paramstyle

import torch
from torch.optim.lr_scheduler import _LRScheduler


def get_optimizer_for_gating(args, model):
    optimizer = {}
    
    main_opt = None
    gate_opt = None
    
    print('only_gate_opt' in args)
    
    if 'only_gate_opt' in args:
        if args.only_gate_opt:
            all_params = [{'params': [p for n, p in model.named_parameters() if 'gating' in n and p.requires_grad], "lr": args.gating_lr}]
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

        
        else:
            if args.gating_opt is None:
                if args.use_gate:
                    if args.one_groups:
                        all_params = [p for p in model.parameters()]
                    else:
                        gate_params = {'params': [p for n, p in model.named_parameters() if 'gating' in n and p.requires_grad]}
                        if not args.same_lr:
                            gate_params.update({"lr": float(args.gating_lr)})
                        
                        all_params = [{'params': [p for n, p in model.named_parameters() if not 'gating' in n and p.requires_grad]}, 
                                    gate_params]
                    
                else:
                    if args.is_retrain:
                        all_params = [p for n, p in model.named_parameters() if p.requires_grad]
                    else:    
                        all_params = [p for n, p in model.named_parameters() if not 'gating' in n and p.requires_grad]
                
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
                    
                
                else:
                    gate_params = [p for n, p in model.named_parameters() if 'gating' in n and p.requires_grad]
                    if args.gating_opt == 'sgd':
                        gate_opt = torch.optim.SGD(gate_params, lr=args.gating_lr, momentum=args.momentum, weight_decay=args.weight_decay)
                    elif args.gating_opt =='adam':
                        if 'eps' in args:
                            gate_opt = torch.optim.Adam(backbone_params, lr=args.lr, weight_decay=args.weight_decay, 
                                                        eps=float(args.eps))
                        else:
                            gate_opt = torch.optim.Adam(backbone_params, lr=args.lr, weight_decay=args.weight_decay)    
    
    else:
        main_params = [p for n, p in model.named_parameters() if not 'gating' in n and p.requires_grad]
        gate_params = {'params': [p for n, p in model.named_parameters() if 'gating' in n and p.requires_grad]}
        if args.gating_lr != args.lr:
            gate_params.update({"lr": float(args.gating_lr)})
            
        if args.gating_opt is not None:
            gate_params = [gate_params]
            if args.gating_opt == 'sgd':
                gate_opt = torch.optim.SGD(gate_params, lr=args.gating_lr, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.gating_opt =='adam':
                if 'eps' in args:
                    gate_opt = torch.optim.Adam(gate_params, lr=args.lr, weight_decay=args.weight_decay, 
                                                eps=float(args.eps))
                else:
                    gate_opt = torch.optim.Adam(gate_params, lr=args.lr, weight_decay=args.weight_decay)    
            elif args.gating_opt =='adamw':
                gate_opt = torch.optim.AdamW(gate_params, lr=args.lr, weight_decay=args.weight_decay)
                
            all_params = main_params
            
        else:
            if args.one_groups:
                all_params = [p for p in model.parameters()]
            else:
                print("not one groups")
                all_params = [{'params': [p for n, p in model.named_parameters() if not 'gating' in n and p.requires_grad]}, 
                            gate_params]
                
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
    params = [p for p in model.parameters() if p.requires_grad]
    
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'nesterov':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif args.opt =='adam':
        if 'eps' in args:
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay, 
                                         eps=float(args.eps))
        else:
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
            
    elif args.opt =='adamw':
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    
    
    return optimizer


def get_scheduler(args, optimizer):
    assert isinstance(args.step_size, int)
    assert isinstance(args.gamma, float)
    assert isinstance(args.lr_steps, list)
    
    if args.lr_scheduler == "step":
        main_sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == "multi":
        main_sch = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.gamma)
    elif args.lr_scheduler == "cosine":
        main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )
      
    return main_sch