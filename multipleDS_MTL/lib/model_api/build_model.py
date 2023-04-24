from .modules.activations import set_activation_function


def build_model(args):
    model_args = {}
    model_args['activation_function'] = set_activation_function(args.activation_function)
    model_args.update({k: v for k, v in args.baseline_args.items()})
    if args.detector is not None:
        model_args.update({'use_fpn': True})
    else:
        model_args.update({'use_fpn': False})
        
    model_args.update({f"{k}_weight": pre_path for k, pre_path in args.state_dict.items()})
    if 'task_balancing' in args: model_args.update({k: v for k, v in args.task_balancing.items()})

    model = None
    
    arch_setup = args.setup
    learning_approach = args.approach # (ddp-style) static or (ddp-style) gating
    
    if arch_setup == 'single_task':
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
        
    else:
        if 'ddp' not in learning_approach:
            if learning_approach == 'static':
                from .task_model.static_mtl import StaticMTL as Model
                
            elif learning_approach == 'gating' or 'retrain' in learning_approach:
                model_args['decay_settings']['max_iter'] = max(args.ds_size)
                from .task_model.gating import GateMTL as Model
                
        else:
            if learning_approach == 'static_ddp':
                from .task_model.static_mtl_ddp import DDPStatic as Model
                
            elif learning_approach == 'gating_ddp':
                model_args['decay_settings']['max_iter'] = max(args.ds_size)
                from .task_model.gating_mtl_ddp import DDPGateMTL as Model
        
        
    # elif arch_setup == 'multi_task':
    #     if args.method == 'static':
    #         if learning_approach == 'baseline' or learning_approach == 'pcgrad' or learning_approach == 'uw' or learning_approach == 'dwa':
    #             model_args.update({k: v for k, v in args.baseline_args.items()})
    #             from .task_model.static_mtl import StaticMTL as Model
                
    #             if args.detector is not None:
    #                 model_args.update({'use_fpn': True})
    #             else:
    #                 model_args.update({'use_fpn': False})
                    
        
    #     elif args.method == 'gating':
    #         if (learning_approach == 'baseline' or learning_approach == 'pcgrad' or learning_approach == 'uw' 
    #             or learning_approach == 'dwa' or learning_approach == 'gradvac' or learning_approach == 'cagrad'):
    #             args.baseline_args['decay_settings']['max_iter'] = max(args.ds_size)
    #             model_args.update({k: v for k, v in args.baseline_args.items()})
    #             from .task_model.gating import GateMTL as Model
                
    #             if args.detector is not None:
    #                 model_args.update({'use_fpn': True})
    #             else:
    #                 model_args.update({'use_fpn': False})
        
        
    #     elif args.method == 'static_ddp':
    #         if (learning_approach == 'baseline' or learning_approach == 'pcgrad' or learning_approach == 'uw' 
    #             or learning_approach == 'dwa' or learning_approach == 'gradvac' or learning_approach == 'cagrad'):
    #             model_args.update({k: v for k, v in args.baseline_args.items()})
                
    #             from .task_model.static_mtl_ddp import DDPStatic as Model
                
    #             if args.detector is not None:
    #                 model_args.update({'use_fpn': True})
    #             else:
    #                 model_args.update({'use_fpn': False})
                    
        
    #     elif args.method == 'gating_ddp':
    #         if (learning_approach == 'pcgrad' or learning_approach == 'uw' 
    #             or learning_approach == 'dwa' or learning_approach == 'gradvac' or learning_approach == 'cagrad'):
    #             args.baseline_args['decay_settings']['max_iter'] = max(args.ds_size)
    #             model_args.update({k: v for k, v in args.baseline_args.items()})
    #             from .task_model.gating_mtl_ddp import DDPGateMTL as Model
                
    #             if args.detector is not None:
    #                 model_args.update({'use_fpn': True})
    #             else:
    #                 model_args.update({'use_fpn': False})
        
        
        # elif args.method == 'DP_static':
        #     if learning_approach == 'baseline' or learning_approach == 'pcgrad':
        #         model_args.update({k: v for k, v in args.baseline_args.items()})
        #         from .task_model.dp.static import DPStaticMTL as Model
                
        #         if args.detector is not None:
        #             model_args.update({'use_fpn': True})
        #         else:
        #             model_args.update({'use_fpn': False})
                    
        
        
        # elif args.method == 'DP_gating':
        #     args.baseline_args['decay_settings']['max_iter'] = max(args.ds_size)
            
        #     model_args['retrain_phase'] = args.retrain_phase
        #     model_args.update({k: v for k, v in args.baseline_args.items()})
        #     if args.detector is not None:
        #             model_args.update({'use_fpn': True})
        #     else:
        #         model_args.update({'use_fpn': False})
                    
        #     if learning_approach == 'baseline':
        #         from .task_model.dp.gating import GateMTL as Model
                    
        #     elif learning_approach == 'pcgrad':
        #         from .task_model.dp.gating import GateMTL as Model
        
        
                
                
        # elif args.method == 'gradnorm':
        #     if learning_approach == 'baseline':
        #         model_args.update({k: v for k, v in args.baseline_args.items()})
        #         from .task_model.gradnorm import GradNorm as Model
                
        #         if args.detector is not None:
        #             model_args.update({'use_fpn': True})
        #         else:
        #             model_args.update({'use_fpn': False})
                    
                # model = GradNorm(
                #     args.backbone, 
                #     args.detector,
                #     args.segmentor, 
                #     args.task_cfg, 
                #     **model_args,
                # )
                
        # elif args.method == 'uw':
        #     if learning_approach == 'baseline':
        #         model_args.update({k: v for k, v in args.baseline_args.items()})
                
        #         if "uw_args" in args:
        #             model_args.update({k: v for k, v in args.uw_args.items()})
                
        #         from .task_model.uw import UW as Model
                
        #         if args.detector is not None:
        #             model_args.update({'use_fpn': True})
        #         else:
        #             model_args.update({'use_fpn': False})
                    
        #         # model = UW(
        #         #     args.backbone, 
        #         #     args.detector,
        #         #     args.segmentor, 
        #         #     args.task_cfg, 
        #         #     **model_args,
        #         # )

        
        # elif args.method == 'dwa':
        #     if learning_approach == 'baseline':
        #         model_args.update({k: v for k, v in args.baseline_args.items()})
                
        #         if "dwa_args" in args:
        #             model_args.update({k: v for k, v in args.dwa_args.items()})
        #             model_args.update({"total_epoch": args.epochs})
                
        #         from .task_model.dwa import DWA as Model
                
        #         if args.detector is not None:
        #             model_args.update({'use_fpn': True})
        #         else:
        #             model_args.update({'use_fpn': False})
                    
        # elif args.method == 'main':
        #     if learning_approach == 'baseline':
        #         model_args.update({k: v for k, v in args.baseline_args.items()})
                
        #         if "dwa_args" in args:
        #             model_args.update({k: v for k, v in args.dwa_args.items()})
        #             model_args.update({"total_epoch": args.epochs})
                
        #         from .task_model.main_method import MainMethod as Model
                
        #         if args.detector is not None:
        #             model_args.update({'use_fpn': True})
        #         else:
        #             model_args.update({'use_fpn': False})
                    
                    
        # elif args.method == 'pcgrad':
        #     if learning_approach == 'baseline':
        #         model_args.update({k: v for k, v in args.baseline_args.items()})
                
        #         if "pcgrad_args" in args:
        #             model_args.update({k: v for k, v in args.pcgrad_args.items()})
                
        #         from .task_model.pcgrad import PCGradMTL as Model
                
        #         if args.detector is not None:
        #             model_args.update({'use_fpn': True})
        #         else:
        #             model_args.update({'use_fpn': False})
        
                    
        model = Model(
            args.backbone, 
            args.detector,
            args.segmentor, 
            args.task_cfg, 
            **model_args,
        )
        
        
    assert model is not None
    return model
        

