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
    
    if len(args.task_bs) == 3:
        args.task = [task for task in args.task_cfg.keys() if args.task is not None]
    elif len(args.task_bs) > 3:
        args.task = [task for task in args.task_cfg.keys() if args.task_cfg[task] is not None]
    
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
    
    if args.setup == 'single_task':
        args.method = 'baseline'
    
    
    if args.output_dir: # /root/~/exp
        # args.output_dir = os.path.join(
        #     args.output_dir, args.model, num_task, args.dataset, args.method)
        
        args.output_dir = os.path.join(
            args.output_dir, args.model, num_task, args.dataset)
        
        if 'is_retrain' in args and args.is_retrain:
            args.load_trained = os.path.join(args.output_dir, "dynamic", args.approach, args.load_trained, 'ckpts', 'checkpoint.pth')
        
        args.output_dir = os.path.join(
            args.output_dir, args.method, args.approach)
        
        if args.exp_case:
            args.output_dir = os.path.join(args.output_dir, args.exp_case)
    
    return args


def calc_flops(model, input_size, task, use_gpu=True):
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement() if self.bias is not None else 0

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    multiply_adds = False
    list_conv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], []
    foo(model)
    
    if isinstance(input_size, tuple):
        input_H = input_size[0]
        input_W = input_size[1]
    
    else:
        input_H = input_size
        input_W = input_size
    
    if use_gpu:
        input = torch.cuda.FloatTensor(torch.rand(2, 3, input_H, input_W).cuda())
        
        input_dict = {task: input}
        
        model = model.cuda()
    else:
        input = torch.FloatTensor(torch.rand(2, 3, input_H, input_W))
    
    _ = model(input_dict)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))

    # print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9 / 2))
    
    return total_flops / 2


def count_params(model, task='clf', input_size=224):
    # param_sum = 0
    # with open('models.txt', 'w') as fm:
    #     fm.write(str(model))
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print('The network has {} params.'.format(params))
    
    flops = calc_flops(model, task, input_size)
    
    return params, flops


