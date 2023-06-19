import os
import time
import math
import shutil
import datetime
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.utils.data
from torch import distributed

from engines import gating_engines
from datasets.load_datasets import load_datasets

import lib.utils.metric_utils as metric_utils
from lib.utils.dist_utils import *
from lib.utils.parser import TrainParser
from lib.utils.sundries import set_args
from lib.utils.logger import TextLogger, TensorBoardLogger

from lib.apis.warmup import get_warmup_scheduler
from lib.apis.optimization import get_optimizer, get_optimizer_for_gating, get_scheduler
from lib.model_api.build_model import build_model


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
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"
    metric_utils.mkdir(args.output_dir) # master save dir
    metric_utils.mkdir(os.path.join(args.output_dir, 'ckpts')) # checkpoint save dir
    torch.set_printoptions(linewidth=300)
    
    init_distributed_mode(args)
    
    if not args.distributed:
        rank_list = [int(device) for device in list(os.environ['CUDA_VISIBLE_DEVICES'].split(","))]
        rank_list.sort()
        first_device_id = rank_list[0]
        rank_list = [d - first_device_id for d in rank_list]
        args.seed = sum(rank_list)
        
        num_gpus = len(rank_list)
        args.num_gpus = num_gpus
        args.task_bs = {dset: bs*num_gpus for dset, bs in args.task_bs.items()}
        
        from engines import DP_gating_engines as running_engine
    else:
        from engines import gating_engines as running_engine
    
    metric_utils.set_random_seed(args.seed)
    log_dir = os.path.join(args.output_dir, 'logs')
    metric_utils.mkdir(log_dir)
    logger = TextLogger(log_dir, print_time=False)
    
    if args.resume:
        logger.log_text("{}\n\t\t\tResume training\n{}".format("###"*40, "###"*45))
    logger.log_text(f"Experiment Case: {args.exp_case}")
    logger.log_text(f"Output Path: {args.output_dir}")
    
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
    logger.log_text("All dataset size:\n\t" \
        "Training Order: {}\n\t" \
        "Data Size: {}".format(
            list(train_loaders.keys()),
            args.all_data_size
            )
        )
    
    logger.log_text("Creating model")
    logger.log_text(f"Freeze Shared Backbone Network: {args.freeze_backbone}")
    
    model = build_model(args)
    
    if model.retrain_phase:
        retrain_args = args.retrain_args
        args.epochs = retrain_args['epoch']
        
        # if 'gate_opt' in args or 'gate_opt' is not None:
        #     args.gate_opt = None
        
        if retrain_args['gated_weight'] == 'mine':
            gating_path = os.path.join(args.output_dir, 'ckpts', f"checkpoint.pth")
        else: gating_path = retrain_args['gated_weight']
        gated_weight = torch.load(gating_path, map_location=torch.device('cpu'))['model']
        
        ignore_keys = []
        if model.grad_method is None: ignore_keys.append("grad_method")
        if model.weighting_method is None: ignore_keys.append("weighting_method")
        
        if len(ignore_keys) > 0:
            new_weight = {}
            
            for n, p in gated_weight.items():
                for ign_key in ignore_keys:
                    if not ign_key in n: new_weight.update({n: p})

            model.load_state_dict(new_weight, strict=False)
            
        else:
            model.load_state_dict(gated_weight, strict=False)
                        
        
        
        # ignore_keys = ['weighting_method', 'grad_method']
        
        # if ignore_keys is not None:
        #     with torch.no_grad():
        #         for n, p in model.named_parameters():
        #             checklist = torch.zeros(len(ignore_keys)).bool()
        #             for i in range(len(ignore_keys)): checklist[i] = ignore_keys[i] in n
        #             if not torch.all(checklist): p.copy_(gated_weight[n])
        # else:
        #     model.load_state_dict(gated_weight['model'], strict=False)
            
        # for dset, gate in model.task_gating_params.items():
        #     print(dset)
        #     print(gate.t())
        #     prob = torch.softmax(gate, dim=1)
        #     print(prob.t())
        #     print()
        # exit()
        
        logger.log_text(str(model))
        model.fix_gate()
        logger.log_text(model.print_fixed_gate_info())
        
        gated_block_p, ds_param, total_param = model.compute_subnetwork_size()
        
        
        sum_ds = float(sum(ds_param))
        sum_total = float(sum(total_param)) + sum_ds
        for data, params in gated_block_p.items():
            sump = sum(params).detach().numpy()
            lines = f"{data} gated parameters:"
            lines += f" {round(sump*(1e-6) + sum_ds*(1e-6), 3)}M/{round(sum_total*(1e-6), 3)}M"
            logger.log_text(lines)
        
    if args.only_gate_train:
        learnable_params = ['gating', 'weighting']
        for n, p in model.named_parameters():
            checklist = torch.zeros(len(learnable_params)).bool()
            for i in range(len(learnable_params)): checklist[i] = not learnable_params[i] in n
            if torch.all(checklist): p.requires_grad = False
            
        setattr(model, 'only_gate_train', args.only_gate_train)
    
    
    if 'gate_opt' in args: optimizer = get_optimizer_for_gating(args, model)
    else: optimizer = get_optimizer(args, model)
    logger.log_text(f"Optimizer:\n{optimizer}")
    logger.log_text(f"Apply AMP: {args.amp}")
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    if scaler is not None:
        logger.log_text(f"Gradient Scaler for AMP: {scaler}")
    
    args.lr_scheduler = args.lr_scheduler.lower()
    
    lr_scheduler = None
    lr_scheduler = get_scheduler(args, optimizer)    
    logger.log_text(f"Scheduler:\n{lr_scheduler}")
    
    if args.distributed:
        model.to(args.device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu])
        model_without_module = model.module
        
    else:
        model = torch.nn.DataParallel(model, device_ids=rank_list)
        model = model.to("cuda")
        model_without_module = model.module
        
    logger.log_text(f"Model Configuration:\n{model}")
    metric_utils.get_params(model, logger, False)
    
    best_results = {data: 0. for data in list(train_loaders.keys())}
    last_results = {data: 0. for data in list(train_loaders.keys())}
    best_epoch = {data: 0 for data in list(train_loaders.keys())}
    total_time = 0.
    
    if args.resume or args.resume_tmp or args.resume_file:
        logger.log_text("Load checkpoints")
        if args.resume_tmp:
            ckpt = os.path.join(args.output_dir, 'ckpts', "checkpoint_tmp.pth")
        elif args.resume_file is not None:
            ckpt = args.resume_file
        else:
            ckpt = os.path.join(args.output_dir, 'ckpts', "checkpoint.pth")

        try:
            checkpoint = torch.load(ckpt, map_location="cpu")
            
            # if args.retrain_phase:
            #     checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if 'task_gating_params' not in k}
            model_without_module.load_state_dict(checkpoint["model"], strict=True)
            
            optimizer.load_state_dict(checkpoint["optimizer"])
            
            if args.lr_config['type'] in ["step", "multi"] :
                if 'gamma' in checkpoint['lr_scheduler']:
                    if checkpoint['lr_scheduler']['gamma'] != args.lr_config['gamma']:
                        checkpoint['lr_scheduler']['gamma'] = args.lr_config['gamma']
                        
                if 'milestones' in checkpoint['lr_scheduler']:
                    if args.lr_config['milestones'] != sorted(checkpoint['lr_scheduler']['milestones'].elements()):
                        checkpoint['lr_scheduler']['milestones'] = Counter(args.lr_config['milestones'])

                    tmp_lr = args.lr
                    for m in checkpoint['lr_scheduler']['milestones']:
                        if checkpoint['lr_scheduler']['last_epoch'] > m:
                            tmp_lr *= args.gamma
                            
                        elif checkpoint['lr_scheduler']['last_epoch'] == m:
                            tmp_lr *= args.gamma
                            break
                        
                    checkpoint['lr_scheduler']['_last_lr'][0] = tmp_lr
                    optimizer.param_groups[0]['lr'] = tmp_lr
            
            # elif args.lr_scheduler == 'step':
            #     if checkpoint['lr_scheduler']['step_size'] != args.step_size:
            #         checkpoint['lr_scheduler']['step_size'] = args.step_size

            #     optimizer.param_groups[0]['lr'] = checkpoint['lr_scheduler']['_last_lr'][0]
            
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            
            logger.log_text(f"Checkpoint List: {checkpoint.keys()}")
            if 'last_results' in checkpoint:
                last_results = checkpoint['last_results']
                logger.log_text(f"Performance of last epoch: {last_results}")
            
            if 'best_results' in checkpoint:
                best_results = checkpoint['best_results']
                logger.log_text(f"Best Performance so far: {best_results}")
            
            if 'epoch' in checkpoint:
                epoch = checkpoint['epoch']
                logger.log_text(f"Last epoch: {epoch}")
                
            if 'best_epoch' in checkpoint:
                best_epoch = checkpoint['best_epoch']
                logger.log_text(f"Best epoch per data previous exp.: {best_epoch}")
            
            if 'total_time' in checkpoint:
                total_time = checkpoint['total_time']
                logger.log_text(f"Previous Total Training Time: {str(datetime.timedelta(seconds=int(total_time)))}")
            
            if args.amp:
                logger.log_text("Load Optimizer Scaler for AMP")
                scaler.load_state_dict(checkpoint["scaler"])
                
            if 'temperature' in checkpoint:
                model.module.decay_function.temperature = checkpoint['temperature']
            else:
                gamma = args.baseline_args['decay_settings']['gamma']
                
                for _ in range(args.start_epoch):
                    model.module.decay_function.temperature *= gamma 
                    
            logger.log_text(f"Next Temperature for Training: {model.module.decay_function.temperature}")
            
            if 'grad_method_information' in checkpoint:
                model.module.grad_method.set_saved_information(checkpoint['grad_method_information'])
                logger.log_text(f"saved gradient method variables:\n{checkpoint['grad_method_information'].keys()}")
                # logger.log_text(f"Previous gradient managing count: {len(model.module.grad_method.surgery_count)}")
                
            if 'weighting_method_information' in checkpoint:
                model.module.weighting_method.set_saved_information(checkpoint['weighting_method_information'])
                logger.log_text(f"saved weighting method variables:\n{checkpoint['weighting_method_information'].keys()}")
            
            
        except Exception as e:
            logger.log_text(f"The resume file is not exist\n{e}")
            
    
    
    if args.validate_only:
        logger.log_text(f"Start Only Validation")
        results = running_engine.evaluate(
            model, val_loaders, args.data_cats, logger, args.num_classes)
            
        line="<First Evaluation Results>\n"
        for data in args.task:
            line += '\t{}: Current Perf. || Last Perf.: {} || {}\n'.format(
                data.upper(), results[data], last_results[data]
            )
            last_results[data] = results[data]
        logger.log_text(line)
        
        import sys
        sys.exit(1)
    

    if args.validate:
        logger.log_text(f"First Validation Start")
        results = running_engine.evaluate(
            model, val_loaders, args.data_cats, logger, args.num_classes)
        
        line="<First Evaluation Results>\n"
        for data in args.task:
            line += '\t{}: Current Perf. || Last Perf.: {} || {}\n'.format(
                data.upper(), results[data], last_results[data]
            )
            last_results[data] = results[data]
        logger.log_text(line)
        
        if args.resume_tmp:
            args.resume_tmp = False

    if args.warmup:
        if args.start_epoch <= args.warmup_epoch:
            biggest_size = len(list(train_loaders.values())[0])
            if args.warmup_epoch > 1:
                total_iters = biggest_size * args.warmup_epoch
                args.warmup_ratio = 1
            
            else:
                total_iters = args.warmup_iters
                args.warmup_epoch = 1
                
            warmup_sch = get_warmup_scheduler(optimizer, args.warmup_ratio, total_iters)
    else:
        warmup_sch = None
    
    logger.log_text(f"Parer Arguments:\n{args}")

    task_flops = {t: [] for t in args.task}
    
    loss_header = None
    task_loss = []
    task_acc = []
    
    csv_dir = os.path.join(args.output_dir, "csv_results")
    os.makedirs(csv_dir, exist_ok=True)
    
    logger.log_text("Multitask Learning Start!\n{}\n --> Method: {}".format("***"*60, args.method))
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for i, (dset, loader) in enumerate(train_loaders.items()):
                print(dset)
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

        if not args.resume_tmp:
            # warmup_fn = get_warmup_scheduler if epoch == 0 else None
            once_train_results = running_engine.training(
                model, 
                optimizer, 
                train_loaders, 
                epoch, 
                logger,
                tb_logger, 
                scaler,
                args,
                warmup_sch=warmup_sch)
            total_time += once_train_results[0]
            logger.log_text("Training Finish\n{}".format('---'*60))
            
            if warmup_sch is not None:
                warmup_sch.step()
            lr_scheduler.step()
            
            # continue
            
            if loss_header is None:
                header = once_train_results[1][-1]
                one_str_header = ''
                for i, n in enumerate(header):
                    if i == len(header)-1:
                        delim = ''
                    else:
                        delim = ', '
                    one_str_header += n + delim
                loss_header = one_str_header
                
            if len(task_loss) == 0:
                task_loss = [[l.detach().cpu().numpy() for l in loss_list] for loss_list in once_train_results[1][:-1]]
                
            else:
                detached = [[l.detach().cpu().numpy() for l in loss_list] for loss_list in once_train_results[1][:-1]]
                task_loss.extend(detached)
            
            logger.log_text(f"saved loss size in one epoch: {len(once_train_results[1][:-1])}")
            logger.log_text(f"saved size of total loss: {len(task_loss)}")
                            
            if args.output_dir:
                checkpoint = {
                    "model": model_without_module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": args,
                    "epoch": epoch
                }
            
            if lr_scheduler is not None:
                checkpoint.update({"lr_scheduler": lr_scheduler.state_dict()})
            
            if scaler is not None:
                checkpoint["scaler"] = scaler.state_dict()
            
            logger.log_text("Save model temporary checkpoint...")
            save_file = os.path.join(args.output_dir, 'ckpts', f"checkpoint_tmp.pth")
            metric_utils.save_on_master(checkpoint, save_file)
            logger.log_text("Complete saving the temporary checkpoint!\n")
            
            if args.distributed: torch.distributed.barrier()
            
        # evaluate after every epoch
        logger.log_text("Validation Start")
        time.sleep(2)
        results = running_engine.evaluate(
                model, val_loaders, args.data_cats, logger, args.num_classes
            )
        logger.log_text("Validation Finish\n{}".format('---'*60))
        if "task_flops" in results:
            for dset, mac in results["task_flops"].items():
                task_flops[dset].append(mac)
        
        #TODO modulation for validation
        task_save = {}
        tmp_acc = []
        line = '<Compare with Best>\n'
        for data in args.detailed_task:
            tmp_results = {k: res for k, res in results.items() if data in k}
            
            for k, res in tmp_results.items():
                if not torch.isnan(torch.tensor(res)):
                    res = round(res, 3)
                    
                tmp_acc.append(res)
                if k not in best_results:
                    if "low" in k: 
                        best_results[k] = 100.
                    else:
                        best_results[k] = 0.
                if k not in best_epoch:
                    best_epoch[k] = 0
                
                line += '\t{}: Current Perf. || Previous Best: {} || {}\n'.format(
                    k.upper(), res, best_results[k]
                )

                if "low" in k:
                    if res < best_results[k]:
                        best_results[k] = round(res, 3)
                        best_epoch[k] = epoch
                        task_save[k] = True
                    else:
                        task_save[k] = False
                else:
                    if res > best_results[k]:
                        best_results[k] = round(res, 3)
                        best_epoch[k] = epoch
                        task_save[k] = True
                    else:
                        task_save[k] = False
        
        task_acc.append(tmp_acc)
        
        logger.log_text(line)
        
        be_line = "Best Epcoh per data:\n"
        for task_k, b_e in best_epoch.items():
            be_line += f" - {task_k}: {b_e}\n"
        logger.log_text(be_line + "\n") 
        
        checkpoint['best_results'] = best_results
        checkpoint['last_results'] = results
        checkpoint['best_epoch'] = best_epoch
        checkpoint['total_time'] = total_time
        
        if getattr(model.module, 'grad_method') is not None:
            if hasattr(model.module.grad_method, "get_save_information"):
                checkpoint['grad_method_information'] = model.module.grad_method.get_save_information
        
        if getattr(model.module, 'weighting_method') is not None:
            if hasattr(model.module.weighting_method, "get_save_information"):
                checkpoint['weighting_method_information'] = model.module.weighting_method.get_save_information
        
        if tb_logger is not None:
            logged_data = {dset: results[dset] for dset in args.task}
            tb_logger.update_scalars(
                logged_data, epoch, proc='val'
            ) 
        
        logger.log_text("Save model checkpoint...")
        save_file = os.path.join(args.output_dir, 'ckpts', f"checkpoint.pth")
        metric_utils.save_on_master(checkpoint, save_file)
        logger.log_text("Complete saving checkpoint!\n")
        
        if args.save_all_epoch:
            save_file = os.path.join(args.output_dir, 'ckpts', f"ckpt_{epoch}e.pth")
            logger.log_text("Save per epoch checkpoint...")
            metric_utils.save_on_master(checkpoint, save_file)
            logger.log_text("Complete saving checkpoint!\n")
        
        logger.log_text(f"Current Epoch: {epoch+1} / Last Epoch: {args.epochs}\n")       
        logger.log_text("Complete {} epoch\n{}\n\n".format(epoch+1, "###"*30))
        if args.distributed: torch.distributed.barrier()
        
        if args.resume_tmp:
            args.resume_tmp = False

        if args.distributed: torch.cuda.synchronize()
        time.sleep(2)
        
        
        
    # End Training -----------------------------------------------------------------------------
    
    all_train_val_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(all_train_val_time)))
    
    if args.distributed:
        if is_main_process:
            loss_csv_path = os.path.join(csv_dir, f"allloss_result_gpu{args.gpu}.csv")
            with open(loss_csv_path, 'a') as f:
                np.savetxt(f, task_loss, delimiter=',', header=one_str_header)
                
            one_str_header = ''
            for i, dset in enumerate(args.task):
                if i == len(header)-1:
                    delim = ''
                else:
                    delim = ', '
                one_str_header += dset + delim
            task_header = one_str_header
            
            acc_csv_path = os.path.join(csv_dir, f"allacc_result_gpu{args.gpu}.csv")
            with open(acc_csv_path, 'a') as f:
                np.savetxt(f, task_acc, delimiter=',', header=task_header)
    
    else:
        loss_csv_path = os.path.join(csv_dir, f"allloss_result.csv")
        with open(loss_csv_path, 'a') as f:
            np.savetxt(f, task_loss, delimiter=',', header=one_str_header)
            
        one_str_header = ''
        for i, dset in enumerate(args.task):
            if i == len(header)-1:
                delim = ''
            else:
                delim = ', '
            one_str_header += dset + delim
        task_header = one_str_header
        
        acc_csv_path = os.path.join(csv_dir, f"allacc_result.csv")
        with open(acc_csv_path, 'a') as f:
            np.savetxt(f, task_acc, delimiter=',', header=task_header)
      
    line = "FLOPs Result:\n"        
    for dset in args.task:
        line += f"{task_flops[dset]}\n"    
    logger.log_text(line)
    logger.log_text(f"FLOPS Results:\n{task_flops}")
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
        
        except RuntimeError as re:
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