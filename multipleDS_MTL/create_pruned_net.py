import os
from sched import scheduler
import time
import math
import shutil
import datetime

import torch
import torch.utils.data

from engines import engines, engines_test
from datasets.load_datasets import load_datasets

import lib.utils.metric_utils as metric_utils
from lib.utils.dist_utils import *
from lib.utils.parser import TrainParser
from lib.utils.sundries import set_args, count_params
from lib.utils.logger import TextLogger, TensorBoardLogger
# from lib.models.model_lib import general_model
# from lib.models.get_origin_models import get_origin_model
from lib.apis.warmup import get_warmup_scheduler
from lib.apis.optimization import get_optimizer_for_gating, get_scheduler_for_gating
from lib.model_api.build_model import build_model

from lib.model_api.modules.disparse_api.prune_utils import disparse_prune_static


def set_ratio(ratio):
    if ratio == 90:
        keep_ratio = 0.095
    elif ratio == 70:
        keep_ratio = 0.3
    elif ratio == 50:
        keep_ratio = 0.51
    elif ratio == 30:
        keep_ratio = 0.71
    else:
        keep_ratio = (100 - ratio) / 100

    return keep_ratio
        

def main(args):
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
    args.task = [dset for dset in train_loaders.keys()]
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
    
    prune_ratio=90
    num_batches=50
    keep_ratio = set_ratio(prune_ratio)
    pruned_net = disparse_prune_static(
        model, train_loaders, num_batches, keep_ratio, logger, args
    )
    
    
    
    
    # print(model)
    # exit()
    
    if args.state_dict["static_pretrained"] is not None and args.state_dict["dynamic_pretrained"] is None:
        print("!!!Loading the static pretrained weight....!!!")
        pretrained_ckpt = torch.load(args.state_dict["static_pretrained"], map_location=torch.device('cpu'))
        model.load_state_dict(pretrained_ckpt['model'], strict=False)
        print("!!!Complete to loading the static weight!!!")

    
    elif args.state_dict["dynamic_pretrained"] is not None and not args.resume:
        print("!!!Loading the dynamic pretrained weight....!!!")
        pretrained_ckpt = torch.load(args.state_dict["dynamic_pretrained"], map_location=torch.device('cpu'))
        model.load_state_dict(pretrained_ckpt['model'], strict=False)
        print("!!!Complete to loading the dynamic weight!!!")
        
    elif args.state_dict["trained_weight"] is not None and not args.resume:
        print("!!!Loading the dynamically pretrained weight....!!!")
        pretrained_ckpt = torch.load(args.state_dict["trained_weight"], map_location=torch.device('cpu'))
        model.load_state_dict(pretrained_ckpt['model'], strict=False)
        print("!!!Complete to loading the dynamically trained weight!!!")
        
    # else:
    #     raise ValueError("the loaded weight type is not valid.")
    
    logger.log_text(f"Model Configuration:\n{model}")
    metric_utils.get_params(model, logger, False)
    
    optimizer = get_optimizer_for_gating(args, model)
    
    logger.log_text(f"Optimizer:\n{optimizer}")
    logger.log_text(f"Apply AMP: {args.amp}")
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    args.lr_scheduler = args.lr_scheduler.lower()
    
    lr_scheduler = None
    lr_scheduler = get_scheduler_for_gating(args, optimizer)    
    logger.log_text(f"Scheduler:\n{lr_scheduler}")
    
    model.to(args.device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, 
                                                          device_ids=[args.gpu])
        model_without_ddp = model.module
    
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
            model_without_ddp.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if lr_scheduler is not None:
                if args.lr_scheduler == 'step':
                    if checkpoint['lr_scheduler']['step_size'] != args.step_size:
                        checkpoint['lr_scheduler']['step_size'] = args.step_size
                        
                    if checkpoint['lr_scheduler']['gamma'] != args.gamma:
                        checkpoint['lr_scheduler']['gamma'] = args.gamma
                    optimizer.param_groups[0]['lr'] = checkpoint['lr_scheduler']['_last_lr'][0]
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
            
            if 'temperature' in checkpoint:
                model.module.temperature = checkpoint['temperature']
            else:
                gamma = args.baseline_args['decay_settings']['gamma']
                
                for _ in range(args.start_epoch):
                    model.module.temperature *= gamma 
            
        except Exception as e:
            logger.log_text(f"The resume file is not exist\n{e}")
    
    
    if args.validate:
        logger.log_text(f"First Validation Start")
        results = engines.evaluate(
            model, val_loaders, args.data_cats, logger, args.num_classes)
        
        line="<First Evaluation Results>\n"
        for data in args.task:
            line += '\t{}: Current Perf. || Last Perf.: {} || {}\n'.format(
                data.upper(), results[data], last_results[data]
            )
            last_results[data] = results[data]
        logger.log_text(line)
    
    # if args.validate_only:
    #     import numpy as np
    #     from ptflops import get_model_complexity_info
    #     flops = {data: [] for data in args.task}
        
    #     logger.log_text(f"Validation Start")
    #     task_flops = {}
        
    #     run = 3
    #     for i in range(run):
    #         for dset, loader in val_loaders.items():
    #             mac_count = 0.
    #             for i, data in enumerate(loader):
    #                 batch_set = {dset: data}
    #                 batch_set = metric_utils.preprocess_data(batch_set, args.data_cats)
    #                 macs, params, outputs = get_model_complexity_info(
    #                     model, batch_set, dset, args.data_cats[dset], as_strings=False,
    #                     print_per_layer_stat=False, verbose=False
    #                 )
    #                 mac_count += macs
                    
    #                 if dset in ['cifar10', 'stl10']:
    #                     break
    #                 if i % 50 == 0:
    #                     print(macs*1e-9)
    #                     # break
    #                     logger.log_text(f"{i}th finished")
                        
    #             mac_count = torch.tensor(mac_count).cuda()
    #             dist.all_reduce(mac_count)
    #             logger.log_text(f"All reduced MAC:{round(float(mac_count)*1e-9, 2)}")
    #             averaged_mac = mac_count/((i+1) * get_world_size())
    #             logger.log_text(f"Averaged MAC:{round(float(averaged_mac)*1e-9, 2)}\n")
                
    #             flops[dset].append(round(float(averaged_mac)*1e-9, 2))
                
    #             logger.log_text("***"*60)
        
        
    #     mean_flops = {dset: np.mean(flops[dset]) for dset in args.task}
        
    #     logger.log_text(f"FLOPs\n{flops}")
    #     logger.log_text(f"Mean FLOPs:\n{mean_flops}")
                
    #     import sys
    #     sys.exit(1)
    
    if args.validate_only:
        import numpy as np
        flops = {data: [] for data in args.task}
        acc = {data: [] for data in args.task}
        avg_count = {data: [] for data in args.task}
        
        logger.log_text(f"Validation Start")
        
        run = 1
        for i in range(run):
            results = engines.evaluate(
                model, val_loaders, args.data_cats, logger, args.num_classes)
            
            line="<Evaluation Results>\n"
            for data in args.task:
                line += '\t{}: Current Perf. || Last Perf.: {} || {}\n'.format(
                    data.upper(), results[data], last_results[data]
                )
                last_results[data] = results[data]
                
                acc[data].append(results[data])
                flops[data].append(results['task_flops'][data])
                avg_count[data].append(results['averaging_gate_counting'][data])
                
            logger.log_text(line)
        
        mean_acc = {dset: np.mean(acc[dset]) for dset in args.task}
        mean_flops = {dset: np.mean(flops[dset]) for dset in args.task}
        mean_avg_cnt = {dset: np.mean(avg_count[dset]) for dset in args.task}
        
        logger.log_text(f"Acc:\n{acc}")
        logger.log_text(f"FLOPs\n{flops}")
        logger.log_text(f"Avg Blocks\n{avg_count}")
        
        logger.log_text(f"Mean Acc:\n{mean_acc}")
        logger.log_text(f"Mean FLOPs:\n{mean_flops}")
        logger.log_text(f"Sum FLOPs:{sum(list(mean_flops.values()))} | Sum of mean FLOPs: {sum(list(mean_flops.values()))/len(mean_flops)}")
        logger.log_text(f"Mean Avg Blocks:\n{mean_avg_cnt}")
        
        
        import sys
        sys.exit(1)


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
    
    per_task_prob = {t: [] for t in args.task}
    per_task_maximum_counting = {t: [] for t in args.task}
    per_task_minimum_counting = {t: [] for t in args.task}
    per_task_averaged_counting = {t: [] for t in args.task}
    per_task_averaged_macs = {t: [] for t in args.task}
    
    loss_header = None
    task_loss = []
    task_acc = []
    
    csv_dir = os.path.join(args.output_dir, "csv_results")
    os.makedirs(csv_dir, exist_ok=True)
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
        # continue
        if lr_scheduler is not None:
            lr_scheduler.step()
                
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
                # "optimizer": optimizer.state_dict(),
                "args": args,
                "epoch": epoch
            }
        
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, dict):
                checkpoint.update({"main_scheduler": lr_scheduler['main'].state_dict()})
                checkpoint.update({"gate_scheduler": lr_scheduler['gate'].state_dict()})
            else:
                checkpoint.update({"lr_scheduler": lr_scheduler.state_dict()})
                
        if isinstance(optimizer, dict):
            checkpoint.update({"main_optimizer": optimizer['main'].state_dict()})
            checkpoint.update({"gate_optimizer": optimizer['gate'].state_dict()})
        else:
            checkpoint.update({"optimizer": optimizer.state_dict()})
        
        if args.amp:
            checkpoint["scaler"] = scaler.state_dict()
        
        # continue
        
        coco_patient = 0
        # evaluate after every epoch
        logger.log_text("Validation Start")
        time.sleep(2)
        results = engines.evaluate(
                model, val_loaders, args.data_cats, logger, args.num_classes
            )
        logger.log_text("Validation Finish\n{}".format('---'*60))
        
        # continue
        
        task_save = {dset: False for dset in args.task}
        
        if "gate_counting" in results:
            task_ratio = {}
            for d, v in results['gate_counting'].items():
                v = v.detach().cpu().numpy()
                per_task_prob[d].append(v)
                
                task_ratio.update({d: v})
                
            logger.log_text(f"Per task ratio:\n{per_task_prob}\n")
            
            if epoch+1 == args.epochs:
                logger.log_text(f"Final Gate Ratio:\n{task_ratio}\n")
                checkpoint.update({"final_gate_ratio": task_ratio})
                
        # if "maximum_counting" in results:
        #     for d, v in results['maximum_counting'].items():
        #         if isinstance(v, torch.Tensor):
        #             v = v.detach().cpu().numpy()
        #         per_task_maximum_counting[d].append(v)
        #     logger.log_text(f"Per task maximum count:\n{per_task_maximum_counting}\n")
                
        # if "minimum_counting" in results:
        #     for d, v in results['maximum_counting'].items():
        #         if isinstance(v, torch.Tensor):
        #             v = v.detach().cpu().numpy()
        #         per_task_minimum_counting[d].append(v)
        #     logger.log_text(f"Per task minimum count:\n{per_task_minimum_counting}\n")
            
        # if "averaging_gate_counting" in results:
        #     for d, v in results['averaging_gate_counting'].items():
        #         if isinstance(v, torch.Tensor):
        #             v = v.detach().cpu().numpy()
        #         per_task_averaged_counting[d].append(v)
        #     logger.log_text(f"Per task averaged count:\n{per_task_averaged_counting}\n")
            
        if "task_flops" in results:
            for d, v in results['task_flops'].items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().cpu().numpy()
                per_task_averaged_macs[d].append(v)
            logger.log_text(f"Per task averaged MACs:\n{per_task_averaged_macs}\n")
        
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
        checkpoint['avg_count'] = per_task_averaged_counting
        
        if args.baseline_args['decay_settings']['temperature'] is not None:
            checkpoint['temperature'] = model.module.temperature
        
    
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
        
        if epoch+1 in args.lr_steps:
            torch.distributed.barrier()
            logger.log_text("Save model checkpoint before lr decaying...")
            save_file = os.path.join(args.output_dir, 'ckpts', f"ckpt_{epoch}e.pth")
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
    logger.log_text(f"All Ratio:\n{per_task_prob}\n")
    
    for data in args.task:
        lines = f"<{data.upper()} counting results>:\n"
        # lines += f"<Maximum count>:\n{per_task_maximum_counting[data]}\n"
        # lines += f"<Minimum count>:\n{per_task_minimum_counting[data]}\n"
        lines += f"<Averaged count>:\n{per_task_averaged_counting[data]}\n"
        lines += f"<Averaged MACs>:\n{per_task_averaged_macs[data]}\n"
        lines += "***"*40
        logger.log_text(lines)
    
    import numpy as np
    all_train_val_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(all_train_val_time)))
    
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
    
    if is_main_process:
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