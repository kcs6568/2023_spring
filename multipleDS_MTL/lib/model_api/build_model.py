from .modules.activations import set_activation_function


def build_model(args):
    model_args = {
        # 'state_dict': args.state_dict,
        'activation_function': set_activation_function(args.activation_function)
    }
    model_args.update({f"{k}_weight": pre_path for k, pre_path in args.state_dict.items()})
    model = None
    
    if args.setup == 'single_task':
        from .task_model.single_task import SingleTaskNetwork
        model_args.update({k: v for k, v in args.single_args.items()})
        model = SingleTaskNetwork(
            args.backbone,
            args.detector,
            args.segmentor,
            args.dataset,
            args.task_cfg[args.dataset],
            **model_args
        )
        
    elif args.setup == 'multi_task':
        if args.method == 'static':
            if args.approach == 'baseline':
                model_args.update({k: v for k, v in args.baseline_args.items()})
                from .task_model.static_mtl import StaticMTL as Model
                
                if args.detector is not None:
                    model_args.update({'use_fpn': True})
                else:
                    model_args.update({'use_fpn': False})
                    
                # model = StaticMTL(
                #     args.backbone, 
                #     args.detector,
                #     args.segmentor, 
                #     args.task_cfg, 
                #     **model_args,
                # )
                
        
        elif args.method == 'static_ddp':
            if args.approach == 'baseline':
                model_args.update({k: v for k, v in args.baseline_args.items()})
                
                from .task_model.static_mtl_ddp import DDPStatic as Model
                
                if args.detector is not None:
                    model_args.update({'use_fpn': True})
                else:
                    model_args.update({'use_fpn': False})
                    
                # model = StaticMTL(
                #     args.backbone, 
                #     args.detector,
                #     args.segmentor, 
                #     args.task_cfg, 
                #     **model_args,
                # )
        
        
        elif args.method == 'gating':
            if args.approach == 'baseline':
                args.baseline_args['gate_args']['decay_settings']['max_iter'] = args.epochs * max(args.ds_size)
                model_args.update({k: v for k, v in args.baseline_args.items()})
                from .task_model.gating import GateMTL as Model
                
                if args.detector is not None:
                    model_args.update({'use_fpn': True})
                else:
                    model_args.update({'use_fpn': False})
                    
        if args.method == 'gating_ddp':
            if args.approach == 'baseline':
                args.baseline_args['gate_args']['decay_settings']['max_iter'] = args.epochs * max(args.ds_size)
                model_args.update({k: v for k, v in args.baseline_args.items()})
                from .task_model.gating_ddp import GateMTLDDP as Model
                
                if args.detector is not None:
                    model_args.update({'use_fpn': True})
                else:
                    model_args.update({'use_fpn': False})
                    
        
        
        
        elif args.method == 'gradnorm':
            if args.approach == 'baseline':
                model_args.update({k: v for k, v in args.baseline_args.items()})
                from .task_model.gradnorm import GradNorm as Model
                
                if args.detector is not None:
                    model_args.update({'use_fpn': True})
                else:
                    model_args.update({'use_fpn': False})
                    
                # model = GradNorm(
                #     args.backbone, 
                #     args.detector,
                #     args.segmentor, 
                #     args.task_cfg, 
                #     **model_args,
                # )
                
        elif args.method == 'uw':
            if args.approach == 'baseline':
                model_args.update({k: v for k, v in args.baseline_args.items()})
                
                if "uw_args" in args:
                    model_args.update({k: v for k, v in args.uw_args.items()})
                
                from .task_model.uw import UW as Model
                
                if args.detector is not None:
                    model_args.update({'use_fpn': True})
                else:
                    model_args.update({'use_fpn': False})
                    
                # model = UW(
                #     args.backbone, 
                #     args.detector,
                #     args.segmentor, 
                #     args.task_cfg, 
                #     **model_args,
                # )

        
        elif args.method == 'dwa':
            if args.approach == 'baseline':
                model_args.update({k: v for k, v in args.baseline_args.items()})
                
                if "dwa_args" in args:
                    model_args.update({k: v for k, v in args.dwa_args.items()})
                    model_args.update({"total_epoch": args.epochs})
                
                from .task_model.dwa import DWA as Model
                
                if args.detector is not None:
                    model_args.update({'use_fpn': True})
                else:
                    model_args.update({'use_fpn': False})
                    
        elif args.method == 'main':
            if args.approach == 'baseline':
                model_args.update({k: v for k, v in args.baseline_args.items()})
                
                if "dwa_args" in args:
                    model_args.update({k: v for k, v in args.dwa_args.items()})
                    model_args.update({"total_epoch": args.epochs})
                
                from .task_model.main_method import MainMethod as Model
                
                if args.detector is not None:
                    model_args.update({'use_fpn': True})
                else:
                    model_args.update({'use_fpn': False})
                    
                    
        elif args.method == 'pcgrad':
            if args.approach == 'baseline':
                model_args.update({k: v for k, v in args.baseline_args.items()})
                
                if "pcgrad_args" in args:
                    model_args.update({k: v for k, v in args.pcgrad_args.items()})
                
                from .task_model.pcgrad import PCGradMTL as Model
                
                if args.detector is not None:
                    model_args.update({'use_fpn': True})
                else:
                    model_args.update({'use_fpn': False})
        
                    
        model = Model(
            args.backbone, 
            args.detector,
            args.segmentor, 
            args.task_cfg, 
            **model_args,
        )
        
        
    assert model is not None
    return model
        

