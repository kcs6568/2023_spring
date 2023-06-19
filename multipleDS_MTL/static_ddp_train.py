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

from engines import static_engines_ddp
from datasets.load_datasets import load_datasets

import lib.utils.metric_utils as metric_utils
from lib.utils.dist_utils import *
from lib.utils.parser import TrainParser
from lib.utils.sundries import set_args
from lib.utils.logger import TextLogger, TensorBoardLogger

from lib.apis.warmup import get_warmup_scheduler, create_warmup
from lib.apis.optimization import get_optimizer, get_scheduler
from lib.model_api.build_model import build_model


def adjust_learning_rate(optimizer, epoch, args, gamma_list=None):
    """Decay the learning rate based on schedule"""
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']
        
        if args.sch == 'multi':
            if epoch in args.lr_steps:
                lr *= 0.1 if epoch in args.lr_steps else 1.
                
        elif args.sch == 'exp':
            if gamma_list is None: lr = lr * 0.9
            else: lr = lr * gamma_list[epoch]
            
            
        param_group['lr'] = lr
        


def main(args):
    metric_utils.mkdir(args.output_dir) # master save dir
    metric_utils.mkdir(os.path.join(args.output_dir, 'ckpts')) # checkpoint save dir
    metric_utils.set_random_seed(args.seed)
    init_distributed_mode(args)
    log_dir = os.path.join(args.output_dir, 'logs')
    metric_utils.mkdir(log_dir)
    logger = TextLogger(log_dir, print_time=False)
    
    logger.log_text(f"Seed: {torch.seed()}")
    
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
    setattr(model, "task_bs", args.task_bs)
    
    metric_utils.get_params(model, logger, False)
    
    optimizer = get_optimizer(args, model)
    
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
        model_without_ddp = model.module
        
    else:
        model.cuda()
        
    logger.log_text(f"Model Configuration:\n{model}")
    metric_utils.get_params(model, logger, False)
    
    best_results = {data: 0. for data in list(train_loaders.keys())}
    last_results = {data: 0. for data in list(train_loaders.keys())}
    best_epoch = {}
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
            
            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            
            if 'gamma' in checkpoint['lr_scheduler']:
                if checkpoint['lr_scheduler']['gamma'] != args.gamma:
                    checkpoint['lr_scheduler']['gamma'] = args.gamma
                    
            if 'milestones' in checkpoint['lr_scheduler']:
                if args.lr_steps != sorted(checkpoint['lr_scheduler']['milestones'].elements()):
                    checkpoint['lr_scheduler']['milestones'] = Counter(args.lr_steps)
                
                tmp_lr = args.lr
                for m in checkpoint['lr_scheduler']['milestones']:
                    if checkpoint['lr_scheduler']['last_epoch'] > m:
                        tmp_lr *= args.gamma
                        
                    elif checkpoint['lr_scheduler']['last_epoch'] == m:
                        tmp_lr *= args.gamma
                        break
                    
                checkpoint['lr_scheduler']['_last_lr'][0] = tmp_lr
                optimizer.param_groups[0]['lr'] = tmp_lr
            
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
                
            if 'grad_method_information' in checkpoint:
                model.module.grad_method.set_saved_information(checkpoint['grad_method_information'])
                logger.log_text(f"saved gradient method variables:\n{checkpoint['grad_method_information'].keys()}")
                # logger.log_text(f"Previous gradient managing count: {len(model.module.grad_method.surgery_count)}")
                
            if 'weighting_method_information' in checkpoint:
                model.module.weighting_method.set_saved_information(checkpoint['weighting_method_information'])
                logger.log_text(f"saved weighting method variables:\n{checkpoint['weighting_method_information'].keys()}")
                
            if args.amp:
                logger.log_text("Load Optimizer Scaler for AMP")
                scaler.load_state_dict(checkpoint["scaler"])
            
        except Exception as e:
            logger.log_text(f"The resume file is not exist\n{e}")
    
    elif 'export_weight' in args:
        if args.export_weight is not None:
            exp_weight = torch.load(args.export_weight, map_location="cpu")
            model_without_ddp.load_state_dict(exp_weight['model'])
            logger.log_text("Export weight was loaded.")
    
    if args.validate_only:
        logger.log_text(f"Start Only Validation")
        results = static_engines_ddp.evaluate(
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
        results = static_engines_ddp.evaluate(
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

    warmup_sch = create_warmup(args, optimizer, len(list(train_loaders.values())[0]))
    
    logger.log_text(f"Parer Arguments:\n{args}")

    task_flops = {t: [] for t in args.task}
    
    loss_header = None
    task_performance = {k: [] for k in train_loaders.keys()}
    task_loss = []
    task_acc = []
    seg_scores = {dataset: {dt: [] for dt in cfg['num_classes'].keys()} for dataset, cfg in args.task_cfg.items() if cfg['task'] == "seg"}
    seg_best = {dataset: {dt: [0, 0] for dt in cfg['num_classes'].keys()} for dataset, cfg in args.task_cfg.items() if cfg['task'] == "seg"}
    
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
        
        if not args.resume_tmp:
            # warmup_fn = get_warmup_scheduler if epoch == 0 else None
            once_train_results = static_engines_ddp.training(
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

            if lr_scheduler is not None:
                if warmup_sch is not None:
                    if warmup_sch.finish:
                        lr_scheduler.step()
                else: lr_scheduler.step()

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
                
            # torch.distributed.barrier()
                            
            if args.output_dir:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": args,
                    "epoch": epoch
                }
            
            if lr_scheduler is not None:
                checkpoint.update({"lr_scheduler": lr_scheduler.state_dict()})
            
            if scaler is not None:
                checkpoint["scaler"] = scaler.state_dict()
            
            torch.distributed.barrier()
            logger.log_text("Save model temporary checkpoint...")
            save_file = os.path.join(args.output_dir, 'ckpts', f"checkpoint_tmp.pth")
            metric_utils.save_on_master(checkpoint, save_file)
            logger.log_text("Complete saving the temporary checkpoint!\n")
        
        coco_patient = 0
        # evaluate after every epoch
        logger.log_text("Validation Start")
        time.sleep(2)
        results = static_engines_ddp.evaluate(
                model, val_loaders, args.data_cats, logger, args.num_classes
            )
        logger.log_text("Validation Finish\n{}".format('---'*60))
        
        if "task_flops" in results:
            for dset, mac in results["task_flops"].items():
                task_flops[dset].append(mac)
        
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
        
        if args.setup != 'single_task':
            for dataset, cfg in args.task_cfg.items():
                if cfg['task'] == "seg":
                    seg_refer = metric_utils.load_seg_referneces(dataset)
                    
                    for dt in cfg['num_classes'].keys():
                        refer = {f"{dataset}_{dt}_{m}": v for m, v in seg_refer[dt].items()}
                        dt_results = {k: v for k, v in results.items() if f"{dataset}_{dt}" in k}
                        # print(dt_results)
                        
                        dt_scores = 0.
                        value = 0
                        for dt_k, dt_res in dt_results.items():
                            if "low" in dt_k:
                                value = (refer[dt_k] - dt_res) / refer[dt_k]
                            else:
                                value = (dt_res - refer[dt_k]) / refer[dt_k]
                            value /= len(dt_results)
                            dt_scores += value
                        seg_scores[dataset][dt].append(dt_scores)
                        
                        result_line = f"{dataset.upper()} / {dt.upper()} Relative Results:\n"
                        if epoch == 0:
                            seg_best[dataset][dt][0] = dt_scores
                            result_line += f"\t Best Score in epoch 0 : {dt_scores} ({seg_best[dataset][dt]})\n"
                            
                        else:
                            if dt_scores > seg_best[dataset][dt][0]:
                                seg_best[dataset][dt][0] = dt_scores
                                seg_best[dataset][dt][1] = epoch
                            result_line += f"\t Best Score: {dt_scores} | Best Epoch: {epoch} ({seg_best[dataset][dt]})\n"
                        logger.log_text(result_line)
                        
            logger.log_text(f"Cumulative Seg Scores:\n{seg_scores}")
        
        if getattr(model.module, 'grad_method') is not None:
            if hasattr(model.module.grad_method, "get_save_information"):
                checkpoint['grad_method_information'] = model.module.grad_method.get_save_information
        
        if getattr(model.module, 'weighting_method') is not None:
            if hasattr(model.module.weighting_method, "get_save_information"):
                checkpoint['weighting_method_information'] = model.module.weighting_method.get_save_information
            
            
        if tb_logger is not None:
            logged_data = {}
            for res_k, res_v in results.items():
                if isinstance(res_v, dict):
                    for task_k, detailed_res in res_v.items():
                        logged_data[task_k] = detailed_res
                else:
                    logged_data[res_k] = res_v
            
            tb_logger.update_scalars(
                logged_data, epoch, proc='val'
            ) 
            
        torch.distributed.barrier()
        logger.log_text("Save model checkpoint...")
        save_file = os.path.join(args.output_dir, 'ckpts', f"checkpoint.pth")
        metric_utils.save_on_master(checkpoint, save_file)
        logger.log_text("Complete saving checkpoint!\n")
        
        if args.lr_scheduler == 'multi' and epoch+1 in args.lr_steps:
            torch.distributed.barrier()
            logger.log_text("Save model checkpoint before lr decaying...")
            save_file = os.path.join(args.output_dir, 'ckpts', f"ckpt_{epoch}e.pth")
            metric_utils.save_on_master(checkpoint, save_file)
            logger.log_text("Complete saving checkpoint!\n")
        
        
        logger.log_text(f"Current Epoch: {epoch+1} / Last Epoch: {args.epochs}\n")       
        logger.log_text("Complete {} epoch\n{}\n\n".format(epoch+1, "###"*30))
        torch.distributed.barrier()
        
        if args.resume_tmp:
            args.resume_tmp = False
        
        torch.cuda.synchronize()
        time.sleep(2)
    # End Training -----------------------------------------------------------------------------
    
    all_train_val_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(all_train_val_time)))
    
    if is_main_process:
        # tensor_loss_to_np = [[l.detach().cpu().numpy() for l in loss_list] for loss_list in task_loss]
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
        
        for task, perf in task_performance.items(): logger.log_text(f"{task.upper()} all epoch performance: {perf}")
        if getattr(model.module, 'grad_method') is not None:
            logger.log_text(f"All iteration gradient managing counting: {model.module.grad_method.get_surgery_count}")
            
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
