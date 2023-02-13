import os
from sched import scheduler
import time
import math
import shutil
import datetime
import numpy as np

import torch
import torch.utils.data

from engines import engines, engines_test
from datasets.load_datasets import load_datasets

import lib.utils.metric_utils as metric_utils
from lib.utils.dist_utils import init_distributed_mode, get_rank, clean_dist, is_main_process
from lib.utils.parser import TrainParser
from lib.utils.sundries import set_args, count_params
from lib.utils.logger import TextLogger, TensorBoardLogger
# from lib.models.model_lib import general_model
# from lib.models.get_origin_models import get_origin_model
from lib.apis.warmup import get_warmup_scheduler
from lib.apis.optimization import get_optimizer, get_scheduler
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
    metric_utils.mkdir(args.output_dir) # master save dir
    metric_utils.mkdir(os.path.join(args.output_dir, 'ckpts')) # checkpoint save dir
    
    if args.distributed:
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
    
    block_p = 0
    ds_p = 0
    total_p = 0
    block_seq = 0
    
    for layer_idx, num_blocks in enumerate(model.num_per_block):
        tmp_ds_p = 0
        for p_ in model.ds[layer_idx].parameters():
            tmp_ds_p = p_.numel()
        ds_p += tmp_ds_p
        logger.log_text(f"The parameters of {layer_idx+1}th ds layer: {tmp_ds_p/1e6}M")
        
        for block_idx in range(num_blocks):
            block_param = 0.
            for p in model.blocks[layer_idx][block_idx].parameters():
                block_param += p.numel()
            block_p += block_param
            logger.log_text(f"The parameters of {block_seq+1}th block: {block_param/1e6}M")
            block_seq += 1
        
    logger.log_text(f"The total parameters of only all blocks: {block_p/1e6}M")
    logger.log_text(f"The total parameters of only all ds layer: {ds_p/1e6}M")
    logger.log_text(f"The total parameters of all blocks: {(block_p+ds_p)/1e6}M")
    
    logger.log_text("!!!Loading the pretrained weight for RE-Training....!!!")
    pretrained_ckpt = torch.load(args.load_trained, map_location=torch.device('cpu'))
    
    # assert 'final_gate_ratio' in pretrained_ckpt
    model.load_state_dict(pretrained_ckpt['model'], strict=True)
    # gate_distribution = {dset: torch.softmax(model.task_gating_params[dset], dim=1) for dset in args.task}
    
    gate_dict = {}
    all_gate = torch.zeros(sum(model.num_per_block), 2)
    for dset in args.task:
        distribution = torch.softmax(model.task_gating_params[dset], dim=1)
        task_gate = torch.tensor([torch.argmax(gate, dim=0) for gate in distribution])
        final_gate = torch.stack((1-task_gate, task_gate), dim=1)
        
        all_gate += final_gate
        model.task_gating_params[dset].data = final_gate
        model.task_gating_params[dset].requires_grad = False
        gate_dict.update({dset: 1-task_gate})

    # final_gate_ratio = pretrained_ckpt['final_gate_ratio']
    # logger.log_text("!!!Complete to getting the Pre-trained Gate Ratio\n")
    # logger.log_text(f"Pretrained Gate Ratio:\n{final_gate_ratio}")
    
    # avg_count = pretrained_ckpt['avg_count']
    # avg_count = {dset: c[-1] for dset, c in avg_count.items()}
    # logger.log_text(f"Averaged Count:\n{avg_count}")
    
    # if args.use_avg_loss_ratio:
    #     logger.log_text("Use the averaged count for loss ratio")
    #     logger.log_text(f"Previous Loss Ratio: {args.loss_ratio}")
        
    #     sum_count = sum(list(avg_count.values()))
    #     rev_sum_count = {dset: sum_count - c for dset, c in avg_count.items()}
    #     rev_sum = sum(list(rev_sum_count.values()))
    #     args.loss_ratio = {dset: round(c/rev_sum, 4) for dset, c in rev_sum_count.items()}
        
    #     logger.log_text(f"Loss Ratio after appling the averaged count: {args.loss_ratio}\n")
        
    # avg_count = {dset: round(a) for dset, a in avg_count.items()}
    # task_gate = {}
    # all_gate = torch.zeros(sum(model.num_per_block), 2)
    # for dset in args.data_cats.keys():
    #     if isinstance(final_gate_ratio[dset], np.ndarray):
    #         final_gate_ratio[dset] = torch.from_numpy(final_gate_ratio[dset])
    #     # print(final_gate_ratio)
    #     final_gate_ratio[dset] = final_gate_ratio[dset][:, 0]
    #     topk = torch.topk(final_gate_ratio[dset], avg_count[dset])[1]
    #     topk = torch.sort(topk)[0]
    #     ma = torch.tensor([1 if i in topk else 0 for i in range(len(final_gate_ratio[dset]))])
    #     gate = torch.stack((ma, 1-ma), dim=1)
        
    #     all_gate += gate
    #     model.task_gating_params[dset].data = gate
    #     model.task_gating_params[dset].requires_grad = False
    #     task_gate.update({dset: ma})
        
    #     logger.log_text(f"!!!Complete to load the gate parameters of {dset.upper()} to fixed gate!!!(Fixed Gate: {torch.sum(ma)})")
    
    logger.log_text(f"Task per gate:\n{gate_dict}")
    
    logger.log_text(f"All Gate Information:\n{all_gate}")
    
    if args.get_task_param:
        for dset in args.task:
            block_seq = 0
            for layer_idx, num_blocks in enumerate(model.num_per_block):
                for block_idx in range(num_blocks):
                    if gate_dict[dset][block_seq] == 0:
                        for p in model.blocks[layer_idx][block_idx].parameters():
                            p.requires_grad = False
                    
                    elif gate_dict[dset][block_seq] == 1:
                        for p in model.blocks[layer_idx][block_idx].parameters():
                            # if p.requires_grad == True: continue
                            p.requires_grad = True
                    
                    block_seq += 1
            
            metric_utils.get_task_params(model, dset, gate_dict[dset], logger, print_table=False)
            
            for n, p in model.named_parameters():
                if 'blocks' in n:
                    p.requires_grad = True
    
    
    if 0 in all_gate[:, 0]:
        block_seq = 0
        for layer_idx, num_blocks in enumerate(model.num_per_block):
            for block_idx in range(num_blocks):
                if all_gate[block_seq][0] == 0:
                    block_param = 0.
                    for p in model.blocks[layer_idx][block_idx].parameters():
                        block_param += p.numel()
                        p.requires_grad = False
                    # model.blocks[layer_idx][block_idx].requires_grad = False
                    logger.log_text(f"The {block_seq+1}th block was freezed. --> Freezed parameters: {block_param/1e6}M")
                
                block_seq += 1
    
    metric_utils.get_params(model, logger, False)
    
    optimizer = get_optimizer(args, model)
    
    logger.log_text(f"Optimizer:\n{optimizer}")
    logger.log_text(f"Apply AMP: {args.amp}")
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    args.lr_scheduler = args.lr_scheduler.lower()
    
    lr_scheduler = None
    lr_scheduler = get_scheduler(args, optimizer)    
    logger.log_text(f"Scheduler:\n{lr_scheduler}")
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model.to(args.device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.gpu])
        model_without_ddp = model.module
    
    logger.log_text(f"Model Configuration:\n{model}")
    best_results = {data: 0. for data in list(train_loaders.keys())}
    last_results = {data: 0. for data in list(train_loaders.keys())}
    best_epoch = {data: 0 for data in list(train_loaders.keys())}
    total_time = 0.
    
    if args.resume or args.resume_tmp or args.resume_file:
        logger.log_text("Load checkpoints")
        if args.resume_tmp:
            ckpt = os.path.join(args.output_dir, 'ckpts', "tmp_checkpoint.pth")
        elif args.resume_file is not None:
            ckpt = args.resume_file
        else:
            ckpt = os.path.join(args.output_dir, 'ckpts', "checkpoint.pth")

        try:
            checkpoint = torch.load(ckpt, map_location="cpu")
            # checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if 'policys' not in k}
            
            '''
            여기서 policys 없애고 load 하면 optimizer 생성 시 instance가 존재하지 않기 때문에 optimizer가 load 안됨
            --> 일단 strict=False로 하고 실행하면 성공함.
            
            lr scheduler key list:
            [
                'step_size', 'gamma', 'base_lrs', 
                'last_epoch', '_step_count', 'verbose', 
                '_get_lr_called_within_step', '_last_lr'
            ]
            '''

            model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            
            if 'gamma' in checkpoint['lr_scheduler']:
                if checkpoint['lr_scheduler']['gamma'] != args.gamma:
                    checkpoint['lr_scheduler']['gamma'] = args.gamma
                    
            if 'milestones' in checkpoint['lr_scheduler']:
                if args.lr_steps != sorted(checkpoint['lr_scheduler']['milestones'].elements()):
                    from collections import Counter
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
            
            elif 'step_size' in checkpoint['lr_scheduler']:
                if checkpoint['lr_scheduler']['step_size'] != args.step_size:
                    checkpoint['lr_scheduler']['step_size'] = args.step_size
                    
                if checkpoint['lr_scheduler']['gamma'] != args.gamma:
                    checkpoint['lr_scheduler']['gamma'] = args.gamma
                
                if checkpoint["lr_scheduler"]['base_lrs'][0] == checkpoint["lr_scheduler"]['_last_lr'][0]:
                    # checkpoint["lr_scheduler"]['_last_lr'][0] *= (args.gamma ** checkpoint["lr_scheduler"]['_step_count'])
                    checkpoint["lr_scheduler"]['_last_lr'][0] *= args.gamma
                    
                
                optimizer.param_groups[0]['lr'] = checkpoint['lr_scheduler']['_last_lr'][0]
                
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            
            # print(checkpoint["lr_scheduler"]['step_size'])
            # print(checkpoint["lr_scheduler"]['gamma'])
            # print(checkpoint["lr_scheduler"]['base_lrs'])
            # print(checkpoint["lr_scheduler"]['last_epoch'])
            # print(checkpoint["lr_scheduler"]['_step_count'])
            # print(checkpoint["lr_scheduler"]['_get_lr_called_within_step'])
            # print(checkpoint["lr_scheduler"]['_last_lr'])
            
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
            
        except Exception as e:
            logger.log_text(f"The resume file is not exist\n{e}")
    
    if args.validate:
        logger.log_text(f"First Validation Start")
        results = engines.evaluate_without_gate(
            model, val_loaders, args.data_cats, logger, args.num_classes)
        
        line="<First Evaluation Results>\n"
        for data in args.task:
            line += '\t{}: Current Perf. || Last Perf.: {} || {}\n'.format(
                data.upper(), results[data], last_results[data]
            )
            last_results[data] = results[data]
        logger.log_text(line)
        
    if args.validate_only:
        logger.log_text(f"First Validation Start")
        results = engines.evaluate_without_gate(
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

    if args.start_epoch <= args.warmup_epoch:
        if args.warmup_epoch > 1:
            args.warmup_ratio = 1
        biggest_size = len(list(train_loaders.values())[0])
        warmup_sch = get_warmup_scheduler(optimizer, args.warmup_ratio, biggest_size * args.warmup_epoch)
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
        
        args.cur_epoch = epoch
        # warmup_fn = get_warmup_scheduler if epoch == 0 else None
        once_train_results = engines.training(
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

        # print(optimizer.param_groups[0]['lr'])
        if lr_scheduler is not None:
            lr_scheduler.step()
            # if warmup_sch is None:
            #     lr_scheduler.step()
                
        else:
            adjust_learning_rate(optimizer, epoch, args)
        
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
        results = engines.evaluate_without_gate(
                model, val_loaders, args.data_cats, logger, args.num_classes
            )
        logger.log_text("Validation Finish\n{}".format('---'*60))
        
        if "task_flops" in results:
            for dset, mac in results["task_flops"].items():
                task_flops[dset].append(mac)
        
        task_save = {dset: False for dset in args.task}
        tmp_acc = []
        line = '<Compare with Best>\n'
        for data in args.task:
            v = results[data]
            tmp_acc.append(v)
            line += '\t{}: Current Perf. || Previous Best: {} || {}\n'.format(
                data.upper(), v, best_results[data]
            )
            
            if not math.isfinite(v):
                logger.log_text(f"Performance of data {data} is nan.")
                v == 0.
                
            if v > best_results[data]:
                best_results[data] = round(v, 2)
                best_epoch[data] = epoch
                task_save[data] = True
            else:
                task_save[data] = False
        
        task_acc.append(tmp_acc)
        logger.log_text(line)  
        logger.log_text(f"Best Epcoh per data: {best_epoch}")
        checkpoint['best_results'] = best_results
        checkpoint['last_results'] = results
        checkpoint['best_epoch'] = best_epoch
        checkpoint['total_time'] = total_time
        
        if tb_logger is not None:
            logged_data = {dset: results[dset] for dset in args.task}
            tb_logger.update_scalars(
                logged_data, epoch, proc='val'
            )   
        torch.distributed.barrier()
        logger.log_text("Save model checkpoint...")
        save_file = os.path.join(args.output_dir, 'ckpts', f"checkpoint.pth")
        metric_utils.save_on_master(checkpoint, save_file)
        logger.log_text("Complete saving checkpoint!\n")
        
        # if epoch % 4 == 0:
        #     torch.distributed.barrier()
        #     logger.log_text("Save model checkpoint...")
        #     save_file = os.path.join(args.output_dir, 'ckpts', f"checkpoint.pth")
        #     metric_utils.save_on_master(checkpoint, save_file)
        #     logger.log_text("Complete saving checkpoint!\n")
        #     for dset, is_save in task_save.items():
        #         dset = dset.upper()
        #         if is_save:
        #             save_file = os.path.join(args.output_dir, 'ckpts', f"ckpt_{dset}.pth")
        #             is_removed = metric_utils.remove_on_master(save_file)
        #             if is_removed:
        #                 logger.log_text(f"Previous checkpoint for {dset} was removed.")   
        #             # torch.save(checkpoint, save_file)
                    
        #             logger.log_text(f"Saving model checkpoint for {dset}...")
        #             metric_utils.save_on_master(checkpoint, save_file)
        #             logger.log_text(f"Complete saving checkpoint for {dset}!\n")
                
        # if epoch == args.epochs-1:
        #     torch.distributed.barrier()
        #     logger.log_text("Save model checkpoint...")
        #     save_file = os.path.join(args.output_dir, 'ckpts', f"checkpoint.pth")
        #     metric_utils.save_on_master(checkpoint, save_file)
        #     logger.log_text("Complete saving checkpoint!\n")
            
            
        logger.log_text(f"Current Epoch: {epoch+1} / Last Epoch: {args.epochs}\n")       
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
    
    if is_main_process:
        loss_csv_path = os.path.join(csv_dir, f"allloss_result_{args.gpu}.csv")
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
        
        acc_csv_path = os.path.join(csv_dir, "allacc_result.csv")
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