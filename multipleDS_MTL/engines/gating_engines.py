import math
import sys
import time
import datetime
from collections import OrderedDict

import torch

import lib.utils.metric_utils as metric_utils
from lib.utils.dist_utils import *
from datasets.coco.coco_eval import CocoEvaluator
from datasets.coco.coco_utils import get_coco_api_from_dataset

# BREAK=True
BREAK=False

def newline_print(msg):
    print(msg); print()


class LossCalculator:
    def __init__(self, type, data_cats, loss_ratio, task_weights=None, weighting_method=None) -> None:
        self.type = type
        self.weighting_method = weighting_method
        self.data_cats = data_cats
        
        if self.type == 'balancing':
            assert loss_ratio is not None
            self.loss_ratio = loss_ratio
            
            self.loss_calculator = self.balancing_loss
        
        elif self.type == 'general':
            self.loss_calculator = self.general_loss
            
    
    def balancing_loss(self, output_losses):
        assert isinstance(output_losses, dict)
        losses = 0.
        balanced_losses = dict()
        for data in self.data_cats:
            data_loss = sum(loss for k, loss in output_losses.items() if data in k)
            data_loss *= self.loss_ratio[data]
            balanced_losses.update({f"bal({str(self.loss_ratio[data])})_{self.data_cats[data]}_{data}": data_loss})
            losses += data_loss
        return losses, balanced_losses
    
    
    def general_loss(self, output_losses):
        assert isinstance(output_losses, dict)
        
        # task_losses = {data: sum(loss for k, loss in output_losses.items() if data in k) for data in self.data_cats}
        # logging_dict = self.weighting_method(task_losses)
        # losses = sum(loss for loss in logging_dict.values())
        
        logging_dict = None
        if self.weighting_method is None: losses = sum(loss for loss in output_losses.values())
        else:
            task_losses = {data: sum(loss for k, loss in output_losses.items() if data in k) for data in self.data_cats}
            logging_dict = self.weighting_method(task_losses)
            
            if logging_dict is not None: 
                losses = sum(loss for loss in logging_dict.values())
            else:
                losses = sum(loss for loss in task_losses.values())
            
        return losses, logging_dict 


def training(model, optimizer, data_loaders, 
          epoch, logger, 
          tb_logger, scaler, args,
          warmup_sch=None):
    model.train()
    module = model.module
    
    prt = newline_print
    
    import pytz
    seoul = pytz.timezone('Asia/Seoul')

    metric_logger = metric_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("main_lr", metric_utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    
    input_dicts = OrderedDict()
    
    datasets = list(data_loaders.keys())
    loaders = list(data_loaders.values())
    
    biggest_datasets, biggest_dl = datasets[0], loaders[0]
    biggest_size = len(biggest_dl)
    
    others_dsets = [None] + datasets[1:]
    others_size = [None] + [len(ld) for ld in loaders[1:]]
    others_iterator = [None] + [iter(dl) for dl in loaders[1:]]
    
    load_cnt = {k: 1 for k in datasets}
    header = f"Epoch: [{epoch+1}/{args.epochs}]"
    iter_time = metric_utils.SmoothedValue(fmt="{avg:.4f}")
    metric_logger.largest_iters = biggest_size
    metric_logger.epohcs = args.epochs
    metric_logger.set_before_train(header)
    
    if warmup_sch:
        logger.log_text(f"Warmup Iteration: {int(warmup_sch.total_iters)}/{biggest_size}")
    else:
        logger.log_text("No Warmup Training")
    
    if not module.retrain_phase:
        logger.log_text(f"Current Sparsity Lambda: {args.baseline_args['gate_args']['lambda_sparsity']}")
        logger.log_text(f"Different Block Weighting: {args.baseline_args['sparsity_weighting']}\n({module.sparsity_weight})")
    
    if module.retrain_phase:
        logger.log_text(module.print_fixed_gate_info())
    
    weighting_method = None if module.weighting_method is None else module.weighting_method
    grad_method = None if module.grad_method is None else module.grad_method
    
    if args.lossbal:
        loss_calculator = LossCalculator(
            'balancing', args.task_per_dset, args.loss_ratio, weighting_method=weighting_method)
    elif args.general:
        loss_calculator = LossCalculator(
            'general', args.task_per_dset, args.loss_ratio, weighting_method=weighting_method)
    
    
    other_args = {"task_list": args.task_per_dset, "current_epoch": epoch}
    loss_for_save = None
    
    all_iter_losses = []
    
    start_time = time.time()
    end = time.time()
    for i, b_data in enumerate(biggest_dl):
        input_dicts.clear()
        input_dicts[biggest_datasets] = b_data
        
        try:
            for n_dset in range(1, len(others_iterator)):
                input_dicts[others_dsets[n_dset]] = next(others_iterator[n_dset])
            
        except StopIteration:
            logger.log_text("occur StopIteration")
            for j, (it, size) in enumerate(zip(others_iterator, others_size)):
                if it is None:
                    continue
                
                if it._num_yielded == size:
                    logger.log_text("reloaded dataset:", datasets[j])
                    logger.log_text("currnet iteration:", i)
                    logger.log_text("yielded size:", it._num_yielded)
                    others_iterator[j] = iter(loaders[j])
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    load_cnt[datasets[j]] += 1
                    logger.log_text(f"Reloading Count: {load_cnt}\n")
                    
            for n_task in range(1, len(others_iterator)):
                if not others_dsets[n_task] in input_dicts.keys():
                    input_dicts[others_dsets[n_task]] = next(others_iterator[n_task])

            
        if args.return_count:
            input_dicts.update({'load_count': load_cnt})
        
        input_set = metric_utils.preprocess_data(input_dicts, args.task_per_dset)
        
        sparsity_loss = {}
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            other_args.update({'cur_iter': i})
            loss_dict = model(input_set, other_args)
            if not module.retrain_phase:
                for l_k in list(loss_dict.keys()):
                    if "sparsity" in l_k: sparsity_loss.update({l_k: loss_dict.pop(l_k)})
        
        losses, logging_dict = loss_calculator.loss_calculator(loss_dict)
        if not module.retrain_phase:
            if len(sparsity_loss) > 0:
                losses += sum(s for s in sparsity_loss.values())
                loss_dict.update(sparsity_loss)
        
        if logging_dict is not None: loss_dict.update(logging_dict)
        
        loss_dict_reduced = metric_utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            logger.log_text(f"Loss is {loss_value}, stopping training\n\t{loss_dict_reduced}", level='error')
            sys.exit(1)
        
        # print("aa")
        list_losses = list(loss_dict_reduced.values())
        list_losses.append(losses_reduced)
        all_iter_losses.append(list_losses)
        
        optimizer.zero_grad(set_to_none=args.grad_to_none)
        if scaler is not None:
            # optimizer.zero_grad(set_to_none=args.grad_to_none)
            scaler.scale(losses).backward()
            
            if args.filtering_method: model.module.compute_sim()
            
            if args.grad_clip_value is not None:
                scaler.unscale_(optimizer) # this must require to get clipped gradients.
                torch.nn.utils.clip_grad_norm_( # clip gradient values to maximum 1.0
                    [p for p in model.parameters() if p.requires_grad],
                    args.grad_clip_value)
                
            scaler.step(optimizer)
            scaler.update()
        
        else:
            losses.backward()
            
            if grad_method is not None:
                module.compute_newgrads(total_mean_grad=args.total_mean_grad)
            
            if args.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip_value)
            optimizer.step()
            
        # for n, p in model.named_parameters():
        #     if p.grad is None:
        #         print(f"{n} has no grad")
        # exit()
        
        if warmup_sch is not None:
            warmup_sch.step()
        
        dist.all_reduce(losses)
        metric_logger.update(main_lr=optimizer.param_groups[0]["lr"])
        if len(optimizer.param_groups) > 1:
            metric_logger.update(gate_lr=optimizer.param_groups[1]["lr"])
        metric_logger.update(loss=losses, **loss_dict_reduced)
        iter_time.update(time.time() - end) 
        
        if BREAK:
            args.print_freq = 10
        
        if (i % args.print_freq == 0 or i == (biggest_size - 1)) and get_rank() == 0:
            curtime = str(seoul.localize(datetime.datetime.now()))[:19]
            logger.log_text(f"Current {seoul.zone} Local Time: {curtime}")
            metric_logger.log_iter(
                iter_time.global_avg,
                args.epochs-epoch,
                logger,
                i
            )
            
            if not model.module.retrain_phase:
                logger.log_text(f"Intermediate Gate Logits (up: use / down: no use)\n{str(model.module)}")
                logger.log_text(f"Temperature: {model.module.decay_function.temperature}\n")

            if weighting_method is not None:
                logger.log_text(f"{weighting_method.name}_Params: {str(weighting_method)}")
                
            if grad_method is not None:
                if grad_method.print_str: logger.log_text(str(grad_method))
                if hasattr(grad_method, 'weighting_method'): logger.log_text(str(grad_method.weighting_method))
            
            logger.log_text(
                f"Middle Processing Time: {str(datetime.timedelta(seconds=int(time.time() - start_time)))}\n\n")
            
        if tb_logger:
            tb_logger.update_scalars(loss_dict_reduced, i)   
            # if not model.module.retrain_phase:
            # if weighting_method is not None: tb_logger.update_scalars(wm.logsigma, i)

            '''
            If the break block is in this block, the gpu memory will stuck (bottleneck)
            '''
            
        # if BREAK and i == args.print_freq:
        if BREAK and i == 10:
            print("BREAK!!")
            # torch.cuda.synchronize()
            break
            
        end = time.time()
        
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize(torch.cuda.current_device)
        
        # logger.log_text(f"{i} iter finished\n")
        # torch.cuda.synchronize()
    
    if i+1 == module.decay_function.max_iter:
        module.decay_function.set_temperature(i+1)
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.log_text(f"{header} Total time: {total_time_str} ({total_time / biggest_size:.4f} s / it)")
    
    if grad_method is not None:
        if hasattr(grad_method, "after_iter"):
            grad_method.after_iter()
    
    if weighting_method is not None:
        if hasattr(weighting_method, "after_iter"):
            weighting_method.after_iter()
    
    del data_loaders
    torch.cuda.empty_cache()
    time.sleep(3)
    
    loss_keys = list(loss_dict_reduced.keys())
    loss_keys.append("sum_loss")
    all_iter_losses.append(loss_keys)
    
    return [total_time, all_iter_losses]

def _get_iou_types(task):
    iou_types = ["bbox"]
    if task == 'seg':
        iou_types.append("segm")

    return iou_types

@torch.inference_mode()
def evaluate(model, data_loaders, data_cats, logger, num_classes):
    assert isinstance(num_classes, dict) or isinstance(num_classes, OrderedDict)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    
    def _validate_classification(outputs, targets, start_time):
        # print("######### entered clf validate")
        accuracy = metric_utils.accuracy(outputs['outputs'].data, targets, topk=(1, 5))
        eval_endtime = time.time() - start_time
        metric_logger.update(
            top1=accuracy[0],
            top5=accuracy[1],
            eval_time=eval_endtime)
        

    def _metric_classification():
        # print("######### entered clf metric")
        metric_logger.synchronize_between_processes()
        # print("######### synchronize finish")
        top1_avg = metric_logger.meters['top1'].global_avg
        top5_avg = metric_logger.meters['top5'].global_avg
        
        logger.log_text("<Current Step Eval Accuracy>\n --> Top1: {}% || Top5: {}%".format(
            top1_avg, top5_avg))
        torch.set_num_threads(n_threads)
        
        return top1_avg
        
        
    def _validate_detection(outputs, targets, start_time):
        # print("######### entered det validate")
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - start_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        
    
    def _metric_detection():
        logger.log_text("Validation result accumulate and summarization")
        if torch.cuda.is_available():
            torch.cuda.synchronize(torch.cuda.current_device)
        logger.log_text("Metric logger synch start")
        metric_logger.synchronize_between_processes()
        logger.log_text("Metric logger synch finish\n")
        logger.log_text("COCO evaluator synch start")
        coco_evaluator.synchronize_between_processes()
        logger.log_text("COCO evaluator synch finish\n")

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        logger.log_text("Finish accumulation")
        coco_evaluator.summarize()
        logger.log_text("Finish summarization")
        coco_evaluator.log_eval_summation()
        torch.set_num_threads(n_threads)
        
        return coco_evaluator.coco_eval['bbox'].stats[0] * 100.
    
    
    def _validate_segmentation(outputs, targets, start_time=None):
        # print("######### entered seg validate")
        confmat.update(targets.flatten(), outputs['outputs'].argmax(1).flatten())
        
        
    def _metric_segmentation():
        # print("######### entered seg metirc")
        confmat.reduce_from_all_processes()
        logger.log_text("<Current Step Eval Accuracy>\n{}".format(confmat))
        return confmat.mean_iou
    
    
    def _select_metric_fn(task, datatype):
        if task == 'clf':
            return _metric_classification
        
        elif task == 'det':
            if 'coco' in datatype:
                return _metric_detection
            elif 'voc' in datatype:
                pass
            
        elif task == 'seg':
            if 'coco' in datatype:
                pass
            elif ('voc' in datatype) \
                or ('cityscapes' in datatype):
                    return _metric_segmentation

                
    def _select_val_fn(task, datatype):
        if task == 'clf':
            return _validate_classification
        elif task == 'det':
            if 'coco' in datatype:
                return _validate_detection
            elif datatype == 'voc':
                pass
            
        elif task == 'seg':
            if 'coco' in datatype:
                pass
            
            elif ('voc' in datatype) \
                or ('cityscapes' in datatype):
                return _validate_segmentation
    
    
    final_results = dict()
    task_flops = {}
    task_total_time = {}
    task_avg_time = {}
    
    dense_shape = {}
    is_dense = False
    
    from lib.utils.flop_counters.ptflops import get_model_complexity_info
    for dataset, taskloader in data_loaders.items():
        if 'coco' in dataset or 'voc' in dataset:
            dense_shape.update({dataset: []})
            is_dense = True
        else:
            is_dense = False
        
        task = data_cats[dataset]
        dset_classes = num_classes[dataset]
        
        if 'coco' in dataset:
            coco = get_coco_api_from_dataset(taskloader.dataset)
            iou_types = _get_iou_types(task)
            coco_evaluator = CocoEvaluator(coco, iou_types)
            coco_evaluator.set_logger_to_pycocotools(logger)
        
        val_function = _select_val_fn(task, dataset)
        metric_function = _select_metric_fn(task, dataset)
        metric_logger = metric_utils.MetricLogger(delimiter="  ")
        
        assert val_function is not None
        assert metric_function is not None
        
        confmat = None
        if task == 'seg':
            confmat = metric_utils.ConfusionMatrix(dset_classes)
        
        header = "Validation - " + dataset.upper() + ":"
        iter_time = metric_utils.SmoothedValue(fmt="{avg:.4f}")
        metric_logger.largest_iters = len(taskloader)
        metric_logger.set_before_train(header)
        
        mac_count = 0.
        batch_size = 0
        
        total_start_time = time.time()
        for i, data in enumerate(taskloader):
            batch_set = {dataset: data}
            
            '''
            batch_set: images(torch.cuda.tensor), targets(torch.cuda.tensor)
            '''
            task_data = metric_utils.preprocess_data(batch_set, data_cats)
            
            if i == 0: batch_size = len(batch_set[dataset])
            # imgs, labels = task_data[dataset][0], task_data[dataset][1]
            
            iter_start_time = time.time()
            macs, _, outputs = get_model_complexity_info(
                model, task_data, dataset, task, as_strings=False,
                print_per_layer_stat=False, verbose=False
            )
            
            torch.cuda.synchronize()
            iter_time.update(time.time() - iter_start_time) 
            mac_count += macs
            
            # val_function(outputs, labels, iter_start_time)
            val_function(outputs, task_data[dataset][1], iter_start_time)
            
            
            torch.cuda.synchronize()
            if ((i % 50 == 0) or (i == len(taskloader) - 1)) and get_rank() == 0:
                metric_logger.log_iter(
                    iter_time.global_avg,
                    1,
                    logger,
                    i
                )
                
                logger.log_text(str(model.module))
                
            if BREAK and i == 2:
                print("BREAK!!!")
                break
            
            torch.cuda.synchronize()
        total_end_time = time.time() - total_start_time
        
        all_time_str = str(datetime.timedelta(seconds=int(total_end_time)))
        logger.log_text(f"{dataset.upper()} Total Evaluation Time: {all_time_str}")
        task_total_time.update({dataset: all_time_str})
        
        # avg_time = round(total_end_time/((i+1) * get_world_size()), 2)
        avg_time = total_end_time/(i+1)
        avg_time_str = str(round(avg_time, 2))
        logger.log_text(f"{dataset.upper()} Averaged Evaluation Time: {avg_time_str}")
        task_avg_time.update({dataset: avg_time_str})
        
        mac_count = torch.tensor(mac_count/batch_size).cuda()
        dist.all_reduce(mac_count)
        logger.log_text(f"All reduced MAC:{round(float(mac_count)*1e-9, 2)}")
        averaged_mac = mac_count/((i+1) * get_world_size())
        logger.log_text(f"Averaged MAC:{round(float(averaged_mac)*1e-9, 2)}\n")
        
        task_flops.update({dataset: round(float(averaged_mac)*1e-9, 2)})
        
        torch.distributed.barrier()
        
        time.sleep(2)
        if torch.cuda.is_available():
            torch.cuda.synchronize(torch.cuda.current_device)
        eval_result = metric_function()
        final_results[dataset] = eval_result
        
        del taskloader
        time.sleep(1)
        torch.cuda.empty_cache()
    
    time.sleep(3)        
    final_results.update({"task_flops": task_flops})
    
    return final_results
    

def classification_for_cm(model, data_loaders, data_cats, output_dir):
    model.eval()
    
    y_pred = []
    y_true = []
    with torch.no_grad():
        for dataset, taskloader in data_loaders.items():
            task = data_cats[dataset]
            
            task_kwargs = {dataset: task} 
            for i, data in enumerate(taskloader):
                batch_set = {dataset: data}
                batch_set = metric_utils.preprocess_data(batch_set, data_cats)
                outputs = model(batch_set[dataset][0], task_kwargs)['outputs']
                
                _, predicted = outputs.max(1)

                y_pred.extend(predicted.cpu().detach().numpy())
                y_true.extend(batch_set[dataset][1].cpu().detach().numpy())

    if 'cifar10' in dataset:
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif 'stl10' in dataset:
        classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']
    
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sn
    import numpy as np
    import pandas as pd
    import os
    
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True, cbar=False)
    plt.savefig(
        os.path.join(output_dir, "cls_cm.png"),
        dpi=600    
    )

    