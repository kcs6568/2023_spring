import os
import yaml
import numpy as np

import torch


def set_args(args):
    if args.cfg:
        with open(args.cfg, 'r') as f:
            configs = yaml.safe_load(f)
        
        for i, j in configs.items():
            setattr(args, i, j)
    
    args.lr = float(args.lr)
    args.gamma = float(args.gamma)
    
    # if len(args.task_bs) == 3:
    #     task_bs = {k: args.task_bs[i]  for i, (k, v) in enumerate(args.task_cfg.items()) if v is not None}
    #     args.task_bs = task_bs
        
    # elif len(args.task_bs) > 3:
    #     task_bs = {data: args.task_bs[i]  for i, (data, v) in enumerate(args.task_cfg.items()) if v is not None}
    #     args.task_bs = task_bs
        
    # else:
    #     args.task_bs = {k: v['bs']  for i, (k, v) in enumerate(args.task_cfg.items()) if v is not None}

    
    task_bs = {}
    task_per_dset = {}
    # reduce_classes = {}
    for i, (k, v) in enumerate(args.task_cfg.items()):
        task_bs.update({k: args.task_bs[i]})
        task_per_dset.update({k: v['task']})
        # if 'reduce_classes' in v:
        #     reduce_classes.update({k: True})
            
    # task_bs = {data: args.task_bs[i]  for i, data in enumerate(args.task_cfg.keys())}
    args.task_bs = task_bs
    args.task_per_dset = task_per_dset
    # args.reduce_classes = reduce_classes
    
    # args.task_per_dset = {data: v['task']  for i, (data, v) in enumerate(args.task_cfg.items())}
    
    # if args.lossbal:
    #     print(args.loss_ratio)
    #     # task_ratio = {k: float(r/10) for k, r in zip(list(args.task_cfg.keys()), args.loss_ratio)}
    #     # args.loss_ratio = task_ratio
    #     args.loss_ratio = {k: float() for k, r in args.loss_ratio.items()}
    #     print(args.loss_ratio)
    
    args.num_classes = {k: v['num_classes'] for k, v in args.task_cfg.items()}
    args.num_datasets = len(args.task_bs)
    args.model = ""
    if args.backbone:
        args.model += args.backbone
    
    if args.detector:
        args.model += "_" + args.detector
        
    if args.segmentor:
        args.model += "_" + args.segmentor
    
    args.dataset = ""
    n_none = list(args.task_cfg.values()).count(None)
    n_task = len(args.task_cfg) - n_none
    
    ds_list = list(args.task_cfg.keys())
    for i, ds in enumerate(ds_list):
        args.dataset += ds
        
        if not i+1 == n_task:
            args.dataset += "_"
    
    # if len(args.task_bs) == 3:
    #     args.task = [task for task in args.task_cfg.keys() if args.task is not None]
    # elif len(args.task_bs) > 3:
    #     args.task = [task for task in args.task_cfg.keys() if args.task_cfg[task] is not None]
    args.task = [task for task in args.task_cfg.keys()]
    
    args.all_data_size = {dset: 0 for dset in args.task}
    
    num_task = ""
    if args.num_datasets == 1:
        num_task = "single"
    elif args.num_datasets == 2:
        num_task = "multiple"
    elif args.num_datasets == 3:
        num_task = "triple"
    elif args.num_datasets == 4:
        num_task = "quadruple"
    elif args.num_datasets == 5:
        num_task = "quintuple"
    
    if 'gating' in args.approach:
        if args.retrain_phase:
            if args.approach == 'gating_ddp':
                args.approach = 'gating'
                for k, v in args.task_balancing.items(): args.task_balancing[k] = None
    if args.output_dir: # /root/~/exp
        # args.output_dir = os.path.join(
        #     args.output_dir, args.model, num_task, args.dataset, args.method)
        
        args.output_dir = os.path.join(
            args.output_dir, args.model, num_task, args.dataset)
        
        if torch.cuda.device_count() == 1:
            args.output_dir = os.path.join(args.output_dir, "single_GPU")
            
        args.output_dir = os.path.join(
            args.output_dir, args.approach, args.method)
        
        if args.exp_case:
            args.output_dir = os.path.join(args.output_dir, args.exp_case)
    
    return args

