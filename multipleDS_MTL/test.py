import os
import time
import math
import shutil
import datetime

import torch
import torch.utils.data

from engines import engines_test
from datasets.load_datasets import load_datasets

import lib.utils.metric_utils as metric_utils
from lib.utils.dist_utils import init_distributed_mode, get_rank, clean_dist
from lib.utils.parser import TrainParser
from lib.utils.sundries import set_args, count_params
from lib.utils.logger import TextLogger, TensorBoardLogger
# from lib.models.model_lib import general_model
# from lib.models.get_origin_models import get_origin_model
from lib.apis.warmup import get_warmup_scheduler
from lib.apis.optimization import get_optimizer
from lib.model_api.build_model import build_model

# try:
#     from torchvision import prototype
# except ImportError:
#     prototype = None


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    sch_list = args.sch_list
    
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        
        if sch_list[i] == 'multi':
            if epoch in args.lr_steps:
                lr *= 0.1 if epoch in args.lr_steps else 1.
                
        elif sch_list[i] == 'exp':
            lr = lr * 0.9
            
        param_group['lr'] = lr


def main(args):
    # args = set_args(args)
    
    metric_utils.mkdir(args.output_dir) # master save dir
    metric_utils.mkdir(os.path.join(args.output_dir, 'ckpts')) # checkpoint save dir
    
    init_distributed_mode(args)
    metric_utils.set_random_seed(args.seed)
    log_dir = os.path.join(args.output_dir, 'logs')
    metric_utils.mkdir(log_dir)
    logger = TextLogger(log_dir, print_time=False)
    
    if args.resume:
        logger.log_text("{}\n\t\t\tResume training\n{}".format("###"*40, "###"*45))
    logger.log_text(f"Experiment Case: {args.exp_case}")
    
    tb_logger = None
    if args.distributed and get_rank() == 0:
        tb_logger = TensorBoardLogger(
            log_dir = os.path.join(log_dir, 'tb_logs'),
            filename_suffix=f"_{args.exp_case}"
        )
    
    if args.seperate and args.freeze_backbone:
        logger.log_text(
        f"he seperate task is applied. But the backbone network will be freezed. Please change the value to False one of the two.",
        level='error')
        logger.log_text("Terminate process", level='error')
        
        raise AssertionError

    metric_utils.save_parser(args, path=log_dir)
    
    logger.log_text("Loading data")
    train_loaders, val_loaders, test_loaders = load_datasets(args)
    args.data_cats = {k: v['task'] for k, v in args.task_cfg.items() if v is not None}
    args.ds_size = [len(dl) for dl in train_loaders.values()]
    logger.log_text("Task list that will be trained:\n\t" \
        "Training Order: {}\n\t" \
        "Data Size: {}".format(
            list(train_loaders.keys()),
            args.ds_size
            )
        )
    
    logger.log_text("Creating model")
    logger.log_text(f"Freeze Shared Backbone Network: {args.freeze_backbone}")
    model = build_model(args)
    
    print("!!!Loading the pretrained weight....!!!")
    pretrained_ckpt = torch.load(args.state_dict["pretrained"], map_location=torch.device('cpu'))
    model.load_state_dict(pretrained_ckpt['model'], strict=False)
    print("!!!Complete to loading the weight!!!")
    
    # for n, p in model.named_parameters():
    #     print(n, p.requires_grad, p.size())
    
    
    # exit()
    
    
    model.to(args.device)
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.gpu])
        model_without_ddp = model.module
    
    logger.log_text(f"Model Configuration:\n{model}")
    metric_utils.get_params(model, logger, False)
    
    optimizer = get_optimizer(args, model)
    
    logger.log_text(f"Optimizer:\n{optimizer}")
    logger.log_text(f"Apply AMP: {args.amp}")
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    args.lr_scheduler = args.lr_scheduler.lower()
    lr_scheduler = None
    if args.lr_scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        print(lr_scheduler.gamma)
    elif args.lr_scheduler == "multi":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.gamma)
    elif args.lr_scheduler == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only MultiStepLR and CosineAnnealingLR are supported."
        )

    logger.log_text(f"Scheduler:\n{lr_scheduler}")
    
    best_results = {data: 0. for data in list(train_loaders.keys())}
    last_results = {data: 0. for data in list(train_loaders.keys())}
    best_epoch = {data: 0 for data in list(train_loaders.keys())}
    total_time = 0.
    
    if args.resume or args.resume_tmp or args.resume_file:
        logger.log_text("Load checkpoints")
        
        if args.resume_tmp:
            ckpt = os.path.join(args.output_dir, 'ckpts', "tmp_checkpoint.pth")
        elif args.resume_file is not None:
            # ckpt = os.path.join(args.output_dir, 'ckpts', args.resume_file)
            ckpt = args.resume_file
        else:
            ckpt = os.path.join(args.output_dir, 'ckpts', "checkpoint.pth")
            # ckpt = os.path.join(args.output_dir, 'ckpts', "test_ckpt.pth")

        try:
            # checkpoint = torch.load(ckpt, map_location=f'cuda:{torch.cuda.current_device()}')
            checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
            model_without_ddp.load_state_dict(checkpoint["model"])
            # model.module.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if lr_scheduler is not None:
                if args.lr_scheduler == 'step':
                    if checkpoint['lr_scheduler']['step_size'] != args.step_size:
                        checkpoint['lr_scheduler']['step_size'] = args.step_size
                        
                    if checkpoint['lr_scheduler']['gamma'] != args.gamma:
                        checkpoint['lr_scheduler']['gamma'] = args.gamma
                    # checkpoint['lr_scheduler']['_last_lr'] = args.lr * args.gamma
                    optimizer.param_groups[0]['lr'] = checkpoint['lr_scheduler']['_last_lr'][0]
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"]
            
            logger.log_text(f"Checkpoint List: {checkpoint.keys()}")
            if 'last_results' in checkpoint:
                last_results = checkpoint['last_results']
                logger.log_text(f"Performance of last epoch: {last_results}")
            
            if 'best_results' in checkpoint:
                best_results = checkpoint['best_results']
                logger.log_text(f"Best Performance so far: {best_results}")
            
            if 'epoch' in checkpoint:
                epoch = checkpoint['epoch']
                logger.log_text(f"Last epoch:", epoch)
                
            if 'best_epoch' in checkpoint:
                best_epoch = checkpoint['best_epoch']
                logger.log_text(f"Best epoch per data previous exp.:", best_epoch)
            
            if 'total_time' in checkpoint:
                total_time = checkpoint['total_time']
                
                logger.log_text(f"Previous Total Training Time:", total_time)
            
            if args.amp:
                logger.log_text("Load Optimizer Scaler for AMP")
                scaler.load_state_dict(checkpoint["scaler"])
            
        except:
            logger.log_text("The resume file is not exist")
            

    block_num = 16
    effect_type = args.effect_type
    effect_direction = args.effect_direction
    
    logger.log_text(f"First Validation: {args.validate}")
    if args.validate:
        logger.log_text("Evaluate First")
        logger.log_text(f"Type: {effect_type} / Direction: {effect_direction}")    
        for n in range(block_num):
            if effect_direction == 'bottomup' or effect_direction == 'one':
                    n = n
            elif effect_direction == 'topdown':
                n = (block_num-1) - n
                    
            results = engines_test.evaluate(
                            model, val_loaders, args.data_cats, logger, args.num_classes, n, effect_type, effect_direction)      
            line="<First Evaluation Results>\n"
            for data, v in results.items():
                line += '\t{}: Current Perf. || Last Perf.: {} || {}\n'.format(
                    data.upper(), v, last_results[data]
                )
                last_results[data] = v
            logger.log_text(line)
            
            skipped = 'all' if n == 0 else n
        
            logger.log_text(f"{skipped} block for {effect_direction} skip performance test finished")
            logger.log_text("---"*40)
            logger.log_text("---"*40)
    print(last_results)
    exit()
    
    logger.log_text("Multitask Learning Start!\n{}\n --> Method: {}".format("***"*60, args.method))
    start_time = time.time()
    
    if args.start_epoch <= args.warmup_epoch:
        if args.warmup_epoch > 1:
            args.warmup_ratio = 1
        biggest_size = len(list(train_loaders.values())[0])
        warmup_sch = get_warmup_scheduler(optimizer, args.warmup_ratio, biggest_size * args.warmup_epoch)
    else:
        warmup_sch = None
    
    logger.log_text(f"Parer Arguments:\n{args}")

    for epoch in range(args.start_epoch, args.epochs):
        # print(args.start_epoch, epoch)
        
        if (args.find_epoch is not None) and (epoch == args.find_epoch):
            logger.log_text("Finish Process early")
            break

        if args.distributed:
            for i, (dset, loader) in enumerate(train_loaders.items()):
                if 'coco' in dset:
                    loader.batch_sampler.sampler.set_epoch(epoch)
                
                else:
                    loader.sampler.set_epoch(epoch)
        
        logger.log_text("Training Start")    
        if args.num_datasets > 1:
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            time.sleep(3)
        
        if warmup_sch is not None:
            if epoch == args.warmup_epoch:
                warmup_sch = None

        # warmup_fn = get_warmup_scheduler if epoch == 0 else None
        one_training_time = engines.training(
            model, 
            optimizer, 
            train_loaders, 
            epoch, 
            logger,
            tb_logger, 
            scaler,
            args,
            warmup_sch=warmup_sch)
        total_time += one_training_time 
        logger.log_text("Training Finish\n{}".format('---'*60))

        if lr_scheduler is not None:
            if warmup_sch is None:
                lr_scheduler.step()
                
        else:
            adjust_learning_rate(optimizer, epoch, args)
            
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": args,
                "epoch": epoch
            }
        
        if lr_scheduler is not None:
            checkpoint.update({"lr_scheduler": lr_scheduler.state_dict()})
        
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        
        coco_patient = 0
        # evaluate after every epoch
        logger.log_text("Validation Start")
        time.sleep(2)
        results = engines_test.evaluate(
                model, val_loaders, args.data_cats, logger, args.num_classes
            )
        logger.log_text("Validation Finish\n{}".format('---'*60))
        
        torch.distributed.barrier()
        if get_rank() == 0:
            line = '<Compare with Best>\n'
            for data, v in results.items():
                line += '\t{}: Current Perf. || Previous Best: {} || {}\n'.format(
                    data.upper(), v, best_results[data]
                )
                
                if not math.isfinite(v):
                    logger.log_text(f"Performance of data {data} is nan.")
                    v == 0.
                    
                if v > best_results[data]:
                    best_results[data] = round(v, 2)
                    best_epoch[data] = epoch
                    
                else:
                    if 'coco' in data:
                        coco_patient += 1
                
            
            if epoch == args.epochs // 2:
                if coco_patient == 2:
                    logger.log_text(
                        "Training process will be terminated because the COCO patient is max value.", 
                        level='error')      
                    
                    import sys
                    sys.exit(1)
                
            
            logger.log_text(line)  
            logger.log_text(f"Best Epcoh per data: {best_epoch}")
            checkpoint['best_results'] = best_results
            checkpoint['last_results'] = results
            checkpoint['best_epoch'] = best_epoch
            checkpoint['total_time'] = total_time
        
            if tb_logger:
                tb_logger.update_scalars(
                    results, epoch, proc='val'
                )    
            
            # logger.log_text("Save model checkpoint...")
            # save_file = os.path.join(args.output_dir, 'ckpts', "checkpoint.pth")
            # torch.save(checkpoint, save_file)
            # logger.log_text("Complete saving checkpoint!\n")
            
            '''
            TODO
            - Timeout 발생하는 거 해결하기
                - 아무래도 모델 안에서 weight 불러오는 부분 때문에 그러지 않을까...
            '''
            # if epoch+1 == int(args.epochs):
            # if epoch % 2 == 0:
            if os.path.isfile(os.path.join(args.output_dir, 'ckpts', 'checkpoint.pth')):
                logger.log_text("Previous checkpoint is removing...")   
                os.remove(os.path.join(args.output_dir, 'ckpts', 'checkpoint.pth'))
                logger.log_text("Previous checkpoint was removed.\n")   
            logger.log_text("Save model checkpoint...")
            save_file = os.path.join(args.output_dir, 'ckpts', "checkpoint.pth")
            torch.save(checkpoint, save_file)
            # torch.save(checkpoint, os.path.join(args.output_dir, 'ckpts', "test_ckpt.pth"))
            # metric_utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'ckpts', "checkpoint.pth"))
            logger.log_text("Complete saving checkpoint!\n")
            
            if epoch in [3, 5, 7, 9, int(args.epochs)-1]:
                logger.log_text(f"Copy model at the epoch {epoch+1}...")
                epoch_save_file = os.path.join(args.output_dir, 'ckpts', f"checkpoint_epoch{int(epoch)+1}.pth")
                shutil.copyfile(save_file, epoch_save_file)
                # shutil.copyfile(os.path.join(args.output_dir, 'ckpts', "checkpoint.pth"), os.path.join(args.output_dir, 'ckpts', f"test_ckpt_epoch{int(epoch)+1}.pth"))
                # torch.savecheckpoint, os.path.join(args.output_dir, 'ckpts', f"checkpoint_epoch{int(epoch)+1}.pth"))
                logger.log_text("Complete copying checkpoint!\n")
            
            # logger.log_text("Save model checkpoint...")
            # save_file = os.path.join(args.output_dir, 'ckpts', "checkpoint.pth")
            # torch.save(checkpoint, save_file)
            # logger.log_text("Complete saving checkpoint!\n")
            
            # logger.log_text(f"Save model at the epoch {epoch+1}...")
            # epoch_save_file = os.path.join(args.output_dir, 'ckpts', f"checkpoint_epoch{int(epoch)+1}.pth")
            # shutil.copyfile(save_file, epoch_save_file)
            
            
            logger.log_text("Complete {} epoch\n{}\n\n".format(epoch+1, "###"*30))
        
        torch.distributed.barrier()
        
        '''
        TODO
        !!!Warning!!!
        - Please do not write "exit()" code --> this will occur the gpu memory
        '''
        
        torch.cuda.synchronize()
        time.sleep(2)
    # End Training -----------------------------------------------------------------------------
    
    all_train_val_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(all_train_val_time)))
    
    if get_rank() == 0:
        logger.log_text("Best Epoch for each task: {}".format(best_epoch))
        logger.log_text("Final Results: {}".format(best_results))
        logger.log_text(f"Exp Case: {args.exp_case}")
        logger.log_text(f"Save Path: {args.output_dir}")
        
        logger.log_text(f"Only Training Time: {str(datetime.timedelta(seconds=int(total_time)))}")
        logger.log_text(f"Training + Validation Time {total_time_str}")


if __name__ == "__main__":
    # args = get_args_parser().parse_args()
    args = TrainParser().args
    args = set_args(args)
    
    try:
        try:
            main(args)
            
        except Exception as ex:
            import traceback
            with open(
                os.path.join(args.output_dir, 'logs', 'learning_log.log'), 'a+') as f:
                if get_rank() == 0:
                    # args.logger.error("Suddenly the error occured!\n<Error Trace>")
                    f.write("Suddenly the error occured!\n<Error Trace>\n")
                    f.write("cuda: {} --> PID: {}\n".format(
                        torch.cuda.current_device(), os.getpid()
                    ))
                    traceback.print_exc()
                    traceback.print_exc(file=f)
                    
        finally:
            if args.distributed:
                clean_dist()
    
    except KeyboardInterrupt as K:
        if args.distributed:
            clean_dist()



  # images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
    # labels = torch.randint(1, 91, (4, 11))
    # images = list(image for image in images)
    # targets = []
    # for i in range(len(images)):
    #     d = {}
    #     d['boxes'] = boxes[i]
    #     d['labels'] = labels[i]
    #     targets.append(d)
    
    # model.train()
    # data = dict(
    #     clf=[
    #         torch.rand(1, 3, 32, 32), torch.tensor([1])
    #     ],
    #     det=[images, targets],
    #     seg=[torch.rand(1, 3, 480, 480), torch.rand(1, 480, 480)
    #     ],
    #     reload_clf=0
    # )
    
    # out = model(data)
    
    # exit()








# import os
# import time
# import math
# import datetime

# import torch
# import torch.utils.data

# from engines import engines
# from datasets.load_datasets import load_datasets

# import lib.utils.metric_utils as metric_utils
# from lib.utils.dist_utils import init_distributed_mode, get_rank, clean_dist
# from lib.utils.parser import TrainParser
# from lib.utils.sundries import set_args, count_params
# from lib.utils.logger import TextLogger, TensorBoardLogger
# # from lib.models.model_lib import general_model
# # from lib.models.get_origin_models import get_origin_model
# from lib.apis.warmup import get_warmup_scheduler
# from lib.apis.optimization import get_optimizer, get_scheduler
# from lib.model_api.build_model import build_model

# # try:
# #     from torchvision import prototype
# # except ImportError:
# #     prototype = None


# def adjust_learning_rate(optimizer, epoch, args):
#     """Decay the learning rate based on schedule"""
#     sch_list = args.sch_list
    
#     for i, param_group in enumerate(optimizer.param_groups):
#         lr = param_group['lr']
        
#         if sch_list[i] == 'multi':
#             if epoch in args.lr_steps:
#                 lr *= 0.1 if epoch in args.lr_steps else 1.
                
#         elif sch_list[i] == 'exp':
#             lr = lr * 0.9
            
#         param_group['lr'] = lr


# def main(args):
#     # args = set_args(args)
    
#     metric_utils.mkdir(args.output_dir) # master save dir
#     metric_utils.mkdir(os.path.join(args.output_dir, 'ckpts')) # checkpoint save dir
    
#     init_distributed_mode(args)
#     metric_utils.set_random_seed(args.seed)
#     log_dir = os.path.join(args.output_dir, 'logs')
#     metric_utils.mkdir(log_dir)
#     logger = TextLogger(log_dir, print_time=False)
    
#     if args.resume:
#         logger.log_text("{}\n\t\t\tResume training\n{}".format("###"*40, "###"*45))
#     logger.log_text(f"Experiment Case: {args.exp_case}")
    
#     tb_logger = None
#     if args.distributed and get_rank() == 0:
#         tb_logger = TensorBoardLogger(
#             log_dir = os.path.join(log_dir, 'tb_logs'),
#             filename_suffix=f"_{args.exp_case}"
#         )
    
#     if args.seperate and args.freeze_backbone:
#         logger.log_text(
#         f"he seperate task is applied. But the backbone network will be freezed. Please change the value to False one of the two.",
#         level='error')
#         logger.log_text("Terminate process", level='error')
        
#         raise AssertionError

#     metric_utils.save_parser(args, path=log_dir)
    
#     logger.log_text("Loading data")
#     train_loaders, val_loaders, test_loaders = load_datasets(args)
#     args.data_cats = {k: v['task'] for k, v in args.task_cfg.items() if v is not None}
#     ds_size = [len(dl) for dl in train_loaders.values()]
#     args.ds_size = ds_size
#     logger.log_text("Task list that will be trained:\n\t" \
#         "Training Order: {}\n\t" \
#         "Data Size: {}".format(
#             list(train_loaders.keys()),
#             ds_size
#             )
#         )
    
#     logger.log_text("Creating model")
#     logger.log_text(f"Freeze Shared Backbone Network: {args.freeze_backbone}")
#     model = build_model(args)
    
#     print(model)
#     # for n, p in model.named_parameters():
#     #     print(n, p.requires_grad, p.size())

#     # exit()    
    
#     model.to(args.device)
    
#     if args.distributed:
#         model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
#     model_without_ddp = model
#     if args.distributed:
#         model = torch.nn.parallel.DistributedDataParallel(model, 
#                                                           device_ids=[args.gpu])
#         model_without_ddp = model.module
    
#     logger.log_text(f"Model Configuration:\n{model}")
#     metric_utils.get_params(model, logger, False)
    
#     # exit()
    
#     optimizer = get_optimizer(args, model)
#     logger.log_text(f"Optimizer:\n{optimizer}")
#     logger.log_text(f"Apply AMP: {args.amp}")
#     scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
#     lr_scheduler = get_scheduler(args, optimizer)
    
#     logger.log_text(f"Scheduler:\n{lr_scheduler}")
    
#     best_results = {data: 0. for data in list(train_loaders.keys())}
#     last_results = {data: 0. for data in list(train_loaders.keys())}
#     best_epoch = {data: 0 for data in list(train_loaders.keys())}
#     total_time = 0.
    
#     if args.resume or args.resume_tmp or args.resume_file:
#         logger.log_text("Load checkpoints")
        
#         if args.resume_tmp:
#             ckpt = os.path.join(args.output_dir, 'ckpts', "tmp_checkpoint.pth")
#         elif args.resume_file is not None:
#             # ckpt = os.path.join(args.output_dir, 'ckpts', args.resume_file)
#             ckpt = args.resume_file
#         else:
#             ckpt = os.path.join(args.output_dir, 'ckpts', "checkpoint.pth")

#         try:
#             # checkpoint = torch.load(ckpt, map_location=f'cuda:{torch.cuda.current_device()}')
#             checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
#             model_without_ddp.load_state_dict(checkpoint["model"])
#             # model.module.load_state_dict(checkpoint["model"])
#             optimizer['main'].load_state_dict(checkpoint["main_optimizer"])
            
#             for n, sch in lr_scheduler.items():
#                 if n == 'main':
#                     sch_type = args.lr_scheduler
#                 elif n == 'gate':
#                     sch_type = args.gating_scheduler
                
#                 if sch is not None:
#                     if sch_type == 'step':
#                         if checkpoint['lr_scheduler'][n]['step_size'] != args.step_size:
#                             checkpoint['lr_scheduler'][n]['step_size'] = args.step_size
                            
#                         if checkpoint['lr_scheduler'][n]['gamma'] != args.gamma:
#                             checkpoint['lr_scheduler'][n]['gamma'] = args.gamma
#                         # checkpoint['lr_scheduler']['_last_lr'] = args.lr * args.gamma
#                         optimizer['main'].param_groups[0]['lr'] = checkpoint['lr_scheduler'][n]['_last_lr'][0]
#                     sch.load_state_dict(checkpoint["lr_scheduler"][n])
                
            
#             args.start_epoch = checkpoint["epoch"]
            
#             logger.log_text(f"Checkpoint List: {checkpoint.keys()}")
#             if 'last_results' in checkpoint:
#                 last_results = checkpoint['last_results']
#                 logger.log_text(f"Performance of last epoch: {last_results}")
            
#             if 'best_results' in checkpoint:
#                 best_results = checkpoint['best_results']
#                 logger.log_text(f"Best Performance so far: {best_results}")
            
#             if 'epoch' in checkpoint:
#                 epoch = checkpoint['epoch']
#                 logger.log_text(f"Last epoch:", epoch)
                
#             if 'best_epoch' in checkpoint:
#                 best_epoch = checkpoint['best_epoch']
#                 logger.log_text(f"Best epoch per data previous exp.:", best_epoch)
            
#             if 'total_time' in checkpoint:
#                 total_time = checkpoint['total_time']
#                 logger.log_text(f"Previous Total Training Time:", total_time)
            
#             if args.amp:
#                 logger.log_text("Load Optimizer Scaler for AMP")
#                 scaler.load_state_dict(checkpoint["scaler"])
            
#         except:
#             logger.log_text("The resume file is not exist")

#     logger.log_text(f"First Validation: {args.validate}")
#     if args.validate:
#         logger.log_text("Evaluate First")
#         results = engines.evaluate(
#             model, val_loaders, args.data_cats, logger, args.num_classes)
        
#         line="<First Evaluation Results>\n"
#         for data, v in results.items():
#             line += '\t{}: Current Perf. || Last Perf.: {} || {}\n'.format(
#                 data.upper(), v, last_results[data]
#             )
#         logger.log_text(line)
        
#     logger.log_text("Multitask Learning Start!\n{}\n --> Method: {}".format("***"*60, args.method))
#     start_time = time.time()
    
#     if args.start_epoch <= args.warmup_epoch:
#         if args.warmup_epoch > 1:
#             args.warmup_ratio = 1
#         biggest_size = len(list(train_loaders.values())[0])
#         warmup_sch = get_warmup_scheduler(optimizer, args.warmup_ratio, biggest_size * args.warmup_epoch)
#     else:
#         warmup_sch = None
    
#     logger.log_text(f"Parer Arguments:\n{args}")

#     for epoch in range(args.start_epoch, args.epochs):
#         # print(args.start_epoch, epoch)
        
#         if (args.find_epoch is not None) and (epoch == args.find_epoch):
#             logger.log_text("Finish Process early")
#             break

#         if args.distributed:
#             for i, (dset, loader) in enumerate(train_loaders.items()):
#                 if 'coco' in dset:
#                     loader.batch_sampler.sampler.set_epoch(epoch)
                
#                 else:
#                     loader.sampler.set_epoch(epoch)
        
#         logger.log_text("Training Start")    
#         if args.num_datasets > 1:
#             torch.cuda.empty_cache()
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             time.sleep(3)
        
#         if warmup_sch is not None:
#             if epoch == args.warmup_epoch:
#                 warmup_sch = None

#         # warmup_fn = get_warmup_scheduler if epoch == 0 else None
#         one_training_time = engines.training(
#             model, 
#             optimizer, 
#             train_loaders, 
#             epoch, 
#             logger,
#             tb_logger, 
#             scaler,
#             args,
#             warmup_sch=warmup_sch)
#         total_time += one_training_time 
#         logger.log_text("Training Finish\n{}".format('---'*60))
        
#         if args.output_dir:
#             checkpoint = {
#                 "model": model_without_ddp.state_dict(),
#                 "main_optimizer": optimizer['main'].state_dict(),
#                 "args": args,
#                 "epoch": epoch
#             }
            
#             if 'gate' in optimizer:
#                 checkpoint.update({"gate_optimizer": optimizer['gate'].state_dict()})
            
#             for n, sch in lr_scheduler.items():    
#                 if sch is not None:
#                     if warmup_sch is None:
#                         sch.step()
#                         checkpoint.update({f"{n}_scheduler": sch.state_dict()})
#                 else:
#                     adjust_learning_rate(optimizer, epoch, args)
        
#         torch.distributed.barrier()
        
#         if args.amp:
#             checkpoint["scaler"] = scaler.state_dict()
        
#         coco_patient = 0
#         # evaluate after every epoch
#         logger.log_text("Validation Start")
#         time.sleep(2)
#         results = engines.evaluate(
#                 model, val_loaders, args.data_cats, logger, args.num_classes
#             )
#         logger.log_text("Validation Finish\n{}".format('---'*60))
        
#         if get_rank() == 0:
#             line = '<Compare with Best>\n'
#             for data, v in results.items():
#                 line += '\t{}: Current Perf. || Previous Best: {} || {}\n'.format(
#                     data.upper(), v, best_results[data]
#                 )
                
#                 if not math.isfinite(v):
#                     logger.log_text(f"Performance of data {data} is nan.")
#                     v == 0.
                    
#                 if v > best_results[data]:
#                     best_results[data] = round(v, 2)
#                     best_epoch[data] = epoch
                    
#                 else:
#                     if 'coco' in data:
#                         coco_patient += 1
                
            
#             if epoch == args.epochs // 2:
#                 if coco_patient == 2:
#                     logger.log_text(
#                         "Training process will be terminated because the COCO patient is max value.", 
#                         level='error')      
                    
#                     import sys
#                     sys.exit(1)
                
            
#             logger.log_text(line)  
#             logger.log_text(f"Best Epcoh per data: {best_epoch}")
#             checkpoint['best_results'] = best_results
#             checkpoint['last_results'] = results
#             checkpoint['best_epoch'] = best_epoch
#             checkpoint['total_time'] = total_time
        
#             if tb_logger:
#                 tb_logger.update_scalars(
#                     results, epoch, proc='val'
#                 )    
            
#             logger.log_text("Save model checkpoint...")
#             metric_utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'ckpts', "checkpoint.pth"))
            
#             if args.lr_scheduler == 'multi':
#                 logger.log_text("Save model checkpoint before applying the lr decaying")
#                 if epoch+1 == int(args.lr_steps[0]): # if next learning rate is decayed in the first decaying step, save the model in the previous epoch.
#                     metric_utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'ckpts', f"e{epoch}_checkpoint.pth"))
            
#             logger.log_text("Complete {} epoch\n{}\n\n".format(epoch+1, "###"*30))
        
#         '''
#         TODO
#         !!!Warning!!!
#         - Please do not write "exit()" code --> this will occur the gpu memory
#         '''
        
#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         time.sleep(2)
#     # End Training -----------------------------------------------------------------------------
    
#     all_train_val_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(all_train_val_time)))
    
#     if get_rank() == 0:
#         logger.log_text("Best Epoch for each task: {}".format(best_epoch))
#         logger.log_text("Final Results: {}".format(best_results))
#         logger.log_text(f"Exp Case: {args.exp_case}")
#         logger.log_text(f"Save Path: {args.output_dir}")
        
#         logger.log_text(f"Only Training Time: {str(datetime.timedelta(seconds=int(total_time)))}")
#         logger.log_text(f"Training + Validation Time {total_time_str}")


# if __name__ == "__main__":
#     # args = get_args_parser().parse_args()
#     args = TrainParser().args
#     args = set_args(args)
    
#     try:
#         try:
#             main(args)
            
#         except Exception as ex:
#             import traceback
#             with open(
#                 os.path.join(args.output_dir, 'logs', 'learning_log.log'), 'a+') as f:
#                 if get_rank() == 0:
#                     # args.logger.error("Suddenly the error occured!\n<Error Trace>")
#                     f.write("Suddenly the error occured!\n<Error Trace>\n")
#                     f.write("cuda: {} --> PID: {}\n".format(
#                         torch.cuda.current_device(), os.getpid()
#                     ))
#                     traceback.print_exc()
#                     traceback.print_exc(file=f)
                    
#         finally:
#             if args.distributed:
#                 clean_dist()
    
#     except KeyboardInterrupt as K:
#         if args.distributed:
#             clean_dist()



#   # images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
#     # labels = torch.randint(1, 91, (4, 11))
#     # images = list(image for image in images)
#     # targets = []
#     # for i in range(len(images)):
#     #     d = {}
#     #     d['boxes'] = boxes[i]
#     #     d['labels'] = labels[i]
#     #     targets.append(d)
    
#     # model.train()
#     # data = dict(
#     #     clf=[
#     #         torch.rand(1, 3, 32, 32), torch.tensor([1])
#     #     ],
#     #     det=[images, targets],
#     #     seg=[torch.rand(1, 3, 480, 480), torch.rand(1, 480, 480)
#     #     ],
#     #     reload_clf=0
#     # )
    
#     # out = model(data)
    
#     # exit()