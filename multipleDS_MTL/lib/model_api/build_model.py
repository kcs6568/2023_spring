import torch
from .backbones.resnet import setting_resnet_args
from .backbones.mobilenet_v3 import setting_mobilenet_args


def build_model(args):
    model_args = {
        'state_dict': args.state_dict,
    }
    
    model = None
    
    if args.setup == 'single_task':
        if args.approach == 'mtan':
            model_args.update({k: v for k, v in args.single_args.items()})
            # model_args.update({
            #     'train_allbackbone': args.train_allbackbone,
            #     'freeze_bn': args.freeze_bn,
            #     'freeze_backbone': args.freeze_backbone,
            # })
            model_args.update({'use_fpn': True})
            # model_args.update({'backbone_type': 'intermediate'})
            # model_args.update({'task_per_dset': args.task_per_dset})
            model_args.update({k: v for k, v in args.mtan_kwargs.items()})
            
            from .task_model.mtan.single import MTANSingle
            
            model = MTANSingle(
                args.backbone, 
                args.detector,
                args.segmentor, 
                args.task_cfg, 
                **model_args
            )
            
        else:
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
        if args.method == 'dynamic':
            if 'sharing' not in args.loss_ratio or args.loss_ratio['sharing'] == 0 or args.loss_ratio['sharing'] is None:
                model_args.update({"use_sharing": False})
            else:
                model_args.update({"use_sharing": True})
            
            if 'disjointed' not in args.loss_ratio or args.loss_ratio['disjointed'] == 0 or args.loss_ratio['disjointed'] is None:
                model_args.update({"use_disjointed": False})
            else:
                model_args.update({"use_disjointed": True})
            
            if args.approach == 'baseline':
                model_args.update({k: v for k, v in args.baseline_args.items()})
                
                # if 'sharing' not in args.loss_ratio or args.loss_ratio['sharing'] == 0 or args.loss_ratio['sharing'] is None:
                #     model_args.update({"use_sharing": False})
                # else:
                #     model_args.update({"use_sharing": True})
                
                # if 'disjointed' not in args.loss_ratio or args.loss_ratio['disjointed'] == 0 or args.loss_ratio['disjointed'] is None:
                #     model_args.update({"use_disjointed": False})
                # else:
                #     model_args.update({"use_disjointed": True})
                    
                    
                sparsity_weight = None if args.sparsity_weight is None else args.sparsity_weight
                model_args.update({"sparsity_weight": sparsity_weight})
                
                from .task_model.multi_task import MultiTaskNetwork
                
                model_args.update({'use_fpn': True})
                # model_args.update({'is_retrain': args.is_retrain})
                # model_args.update({'pretrained_weight': args.state_dict['static_pretrained']})
                
                if model_args['decay_settings']['decay_type'] == 'simple':
                    max_iter = max(args.ds_size)
                else:
                    max_iter = args.epochs * max(args.ds_size)
                model_args.update({'max_iter': max_iter})
                
                model = MultiTaskNetwork(
                    args.backbone, 
                    args.detector,
                    args.segmentor, 
                    args.task_cfg, 
                    **model_args,
                )
                    

            elif args.approach == 'mtan':
                sparsity_weight = None if args.sparsity_weight is None else args.sparsity_weight
                model_args.update({"sparsity_weight": sparsity_weight})
                model_args.update({k: v for k, v in args.baseline_args.items()})
                model_args.update({k: v for k, v in args.mtan_kwargs.items()})
                model_args.update({'use_fpn': True})
                model_args.update({'task_per_dset': args.task_per_dset})
                
                if model_args['decay_settings']['decay_type'] == 'simple':
                    max_iter = max(args.ds_size)
                else:
                    max_iter = args.epochs * max(args.ds_size)
                model_args.update({'max_iter': max_iter})
                
                from .task_model.mtan.dynamic import MTANDynamic
                
                model = MTANDynamic(
                    args.backbone, 
                    args.detector,
                    args.segmentor, 
                    args.task_cfg, 
                    **model_args
                )
        
        
        elif args.method == 'retrain':
            if args.approach == 'baseline':
                from .task_model.multi_task_retrain import RetrainMTL
                
                model_args.update({k: v for k, v in args.baseline_args.items()})
                model_args.update({'use_fpn': True})
                
                model = RetrainMTL(
                    args.backbone, 
                    args.detector,
                    args.segmentor, 
                    args.task_cfg, 
                    **model_args,
                )
                
            elif args.approach == 'mtan':
                model_args.update({k: v for k, v in args.baseline_args.items()})
                model_args.update({'use_fpn': True})
                model_args.update({k: v for k, v in args.mtan_kwargs.items()})
                
                from .task_model.mtan.retrain import MTANRetrain
                
                model = MTANRetrain(
                    args.backbone, 
                    args.detector,
                    args.segmentor, 
                    args.task_cfg, 
                    **model_args
                )
        
        
    
        elif args.method == 'static':
            if args.approach == 'baseline':
                model_args.update({k: v for k, v in args.baseline_args.items()})
                from .task_model.static_mtl import StaticMTL
                
                model_args.update({'use_fpn': True})
                
                model = StaticMTL(
                    args.backbone, 
                    args.detector,
                    args.segmentor, 
                    args.task_cfg, 
                    **model_args,
                )

            elif args.approach == 'mtan':
                model_args.update({k: v for k, v in args.baseline_args.items()})
                # model_args.update({
                #     'train_allbackbone': args.train_allbackbone,
                #     'freeze_bn': args.freeze_bn,
                #     'freeze_backbone': args.freeze_backbone,
                # })
                model_args.update({'use_fpn': True})
                # model_args.update({'backbone_type': 'intermediate'})
                # model_args.update({'task_per_dset': args.task_per_dset})
                model_args.update({k: v for k, v in args.mtan_kwargs.items()})
                
                from .task_model.mtan.static import MTANStatic
                
                model = MTANStatic(
                    args.backbone, 
                    args.detector,
                    args.segmentor, 
                    args.task_cfg, 
                    **model_args
                )

            elif args.approach == 'multiscale':
                model_args.update({k: v for k, v in args.baseline_args.items()})
                model_args.update({k: v for k, v in args.multiscale_cfg.items()})
                model_args.update({'use_fpn': True})
                
                from .task_model.static_mtl_ms import MSStaticMTL
                model = MSStaticMTL(
                    args.backbone, 
                    args.detector,
                    args.segmentor, 
                    args.task_cfg, 
                    **model_args,
                )
            
    assert model is not None
    return model
        


# if args.method == 'policy_generator':
#     from .task_model.multi_task_policy_generator import MultiTaskNetwork
    
#     model_args.update({k: v for k, v in args.baseline_args.items()})
#     model_args.update({'use_fpn': True})
#     model_args.update({'is_retrain': args.is_retrain})
#     model_args.update({'pretrained_weight': args.state_dict['static_pretrained']})
    
#     if model_args['decay_settings']['decay_type'] == 'simple':
#         max_iter = max(args.ds_size)
#     else:
#         max_iter = args.epochs * max(args.ds_size)
#     model_args.update({'max_iter': max_iter})
    
#     model = MultiTaskNetwork(
#         args.backbone, 
#         args.detector,
#         args.segmentor, 
#         args.task_cfg, 
#         **model_args,
#     )


# elif args.method == 'test':
#     from .task_model.multi_task_test import MultiTaskNetwork
    
#     model_args.update({k: v for k, v in args.baseline_args.items()})
#     model_args.update({'use_fpn': True})
#     model_args.update({'backbone_type': 'intermediate'})
#     model_args.update({'max_iter': args.epochs * max(args.ds_size)})
#     model_args.update({'pretrained_weight': args.state_dict['static_pretrained']})
    
#     model_args.update({'use_gate': args.use_gate})
    
#     model = MultiTaskNetwork(
#         args.backbone, 
#         args.detector,
#         args.segmentor, 
#         args.task_cfg, 
#         **model_args,
#     )
    
    
# elif args.method == 'nogate':
#     from .task_model.multi_task_nogate import MTLNoGate
    
#     model_args.update({k: v for k, v in args.baseline_args.items()})
#     model_args.update({'use_fpn': True})
#     model_args.update({'backbone_type': 'intermediate'})
    
#     print(model_args)
    
#     model = MTLNoGate(
#         args.backbone, 
#         args.detector,
#         args.segmentor, 
#         args.task_cfg, 
#         **model_args,
#     )
    
# elif args.method == 'static':
#     from .task_model.static_mtl import StaticMTL
    
#     model_args.update({k: v for k, v in args.baseline_args.items()})
#     model_args.update({'use_fpn': True})
#     model_args.update({'backbone_type': 'intermediate'})
    
#     model = StaticMTL(
#         args.backbone, 
#         args.detector,
#         args.segmentor, 
#         args.task_cfg, 
#         **model_args,
#     )