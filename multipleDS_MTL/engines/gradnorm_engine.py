import math
import sys
import time
import datetime
import numpy as np
from copy import deepcopy
from collections import OrderedDict

import torch
import torchviz

import lib.utils.metric_utils as metric_utils
from lib.utils.dist_utils import *
from datasets.coco.coco_eval import CocoEvaluator
from datasets.coco.coco_utils import get_coco_api_from_dataset

# BREAK=True
BREAK=False


class LossCalculator:
    def __init__(self, type, data_cats, loss_ratio, task_weights=None, method='multi_task') -> None:
        self.type = type
        self.method = method
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
        # balanced_losses = dict()
        for data in self.data_cats:
            data_loss = sum(loss for k, loss in output_losses.items() if data in k)
            data_loss *= self.loss_ratio[data]
            # balanced_losses.update({f"bal_{self.data_cats[data]}_{data}": data_loss})
            losses += data_loss
        
        return losses
    
    
    def general_loss(self, output_losses):
        assert isinstance(output_losses, dict)
        losses = sum(loss for loss in output_losses.values())
        
        return losses


def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


def training(model, optimizer, data_loaders, 
          epoch, logger, 
          tb_logger, scaler, args,
          warmup_sch=None):
    model.train()
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        module = model.module
    else:
        module = model
    
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
    
    if args.lossbal:
        loss_calculator = LossCalculator(
            'balancing', args.task_per_dset, args.loss_ratio, method=args.method)
    elif args.general:
        loss_calculator = LossCalculator(
            'general', args.task_per_dset, args.loss_ratio, method=args.method)
    
    start_time = time.time()
    end = time.time()
    
    other_args = {"task_list": args.task_per_dset, "current_epoch": epoch}
    loss_for_save = None
    
    all_iter_losses = []
    
    # loss_weights = module.loss_weights
    alpha = args.alpha
    
    # weight_opt = torch.optim.AdamW([w for w in loss_weights.values()], lr=0.01)
    
    # init_weight_sum = sum(list(loss_weights.values())).detach()
    init_loss = {}
    init_loss_sum = 0.
    for i, b_data in enumerate(biggest_dl):
        # print("iteration ",i)
        input_dicts.clear()
        input_dicts[biggest_datasets] = b_data
        # logger.log_text(f"start iteration {i}")
        
        try:
            for n_dset in range(1, len(others_iterator)):
                input_dicts[others_dsets[n_dset]] = next(others_iterator[n_dset])
                # logger.log_text(f"{n_dset} next")
            # torch.cuda.synchronize()
            
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
        
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = module(input_set, other_args)
            # print(loss_dict)
        losses = loss_calculator.loss_calculator(loss_dict)
        logged_loss = {}
        
        dist.all_reduce(losses)
        loss_dict_reduced = metric_utils.reduce_dict(loss_dict, average=False)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            logger.log_text(f"Loss is {loss_value}, stopping training\n\t{loss_dict_reduced}", level='error')
            sys.exit(1)
        
        list_losses = list(loss_dict_reduced.values())
        list_losses.append(losses_reduced)
        all_iter_losses.append(list_losses)
        logged_loss.update(loss_dict_reduced)
        ###########################################
        
        ################ GradNorm #################
        loss = []
        for data in datasets:
            loss.append(sum(loss for k, loss in loss_dict.items() if data in k))
        loss = torch.stack(loss)
        if i == 0:
            weights = torch.ones_like(loss)
            weights = torch.nn.Parameter(weights, requires_grad=True)
            T = weights.sum().detach()
            optimizer2 = torch.optim.AdamW([weights], lr=0.01)
            l_0 = loss.detach()
        
        print(weights.grad)
        weighted_loss = weights @ loss
        optimizer.zero_grad()  
        
        # weighted_loss.backward(retain_graph=True)
        
        # for n, p in module.named_parameters():
        #     if p.grad is not None:
        #         print(n, p.grad.size(), p.grad_fn)
        # exit()
        
        # shared_layer = deepcopy(module.get_last_shared_module())
        shared_layer = module.get_last_shared_module()
        
        # exit()
        
        # weights.grad.data = torch.zeros_like(weights.grad.data).to(args.device)
        weights = weights.detach()
        
        gw = []
        for i in range(len(loss)):
            dl = torch.autograd.grad(
                weights[i]*loss[i], shared_layer.parameters(), 
                retain_graph=True, create_graph=True)[0]
            gw.append(torch.norm(dl).detach())
        
        # for p in shared_layer.parameters(): p.detach()
        
        gw = torch.stack(gw)
        loss_ratio = loss.detach() / l_0
        rt = loss_ratio / loss_ratio.mean()
        gw_avg = gw.mean().detach().cpu().numpy()
        constant = torch.tensor(gw_avg * (rt.detach().cpu().numpy() ** alpha), requires_grad=False).to(args.device)
        # constant = (gw_avg * rt ** alpha).detach()
        gradnorm_loss = torch.abs(gw - constant).sum()
        
        # optimizer2.zero_grad()
        gradnorm_loss.backward()
        
        weights.grad = torch.autograd.grad(gradnorm_loss, weights)[0]
        
        
        
        optimizer.step()
        optimizer2.step()
        
        weights = (weights / weights.sum() * T).detach()
        weights = torch.nn.Parameter(weights)
        optimizer2 = torch.optim.AdamW([weights], lr=0.01)
        
        
        exit()
        
        # dataset_loss = {}
        # for data in datasets:
        #     dataset_loss.update({data: sum(loss for k, loss in loss_dict.items() if data in k)})
        # if i == 0:
        #     init_loss.update({k: v.clone().detach() for k, v in dataset_loss.items()})
        
        # # weighted_loss = {dset: torch.mul(loss_weights[dset], dataset_loss[dset]) for dset in datasets}
        # # weighted_loss_sum = sum(list(weighted_loss.values()))
        
        # loss_sum = sum(list(dataset_loss.values()))
        # if scaler is not None:
        #     # optimizer.zero_grad(set_to_none=args.grad_to_none)
        #     scaler.scale(loss_sum).backward(retain_graph=True)   
            
        # else:
        #     # optimizer.zero_grad(set_to_none=args.grad_to_none)
        #     loss_sum.backward(retain_graph=True)
        
        # # exit()
        
        # # if tb_logger:
        # #     for n, p in module.named_parameters():
        # #         if p.grad is not None:
        # #             tb_logger.add_histogram(n, p.grad, i)
        
        
        # # last_module_grad = module.get_last_shared_module().weight.grad.data
        # # l2norm_grad = {d: (last_module_grad * loss_weights[d] * l).norm(p=2).detach().clone() for d, l in dataset_loss.items()}
        
        
        # last_weight = module.get_last_shared_module()
        # # l2norm_grad = {d: (torch.autograd.grad(l, last_weight.parameters(), retain_graph=True)[0] * loss_weights[d] * l).norm(p=2).detach().clone() for d, l in dataset_loss.items()}
        # l2norm_grad = {d: (torch.autograd.grad(
        #     loss_weights[d] * l, last_weight.parameters(), retain_graph=True, create_graph=True)[0]).norm(p=2) for d, l in dataset_loss.items()}
        
        # for n, p in module.named_parameters():
        #     if p.grad is not None:
        #         print(n, "has grad", p.grad_fn)
        
        # mean_grad = (sum(list(l2norm_grad.values())) / len(l2norm_grad)).detach()
        # inverse_ratio = {d: (l / init_loss[d]) for d, l in dataset_loss.items()}
        # mean_inverse_ratio = (sum(list(inverse_ratio.values())) / len(inverse_ratio))
        # rel_inverse_ratio = {d: l / mean_inverse_ratio for d, l in inverse_ratio.items()}
        
        # grad_gradnorm = 0.
        # loss_gradnorm = 0.
        # for dset in datasets:
        #     mean_grad_task = (mean_grad * (rel_inverse_ratio[dset] ** alpha)).detach()
                
        #     loss_gradnorm += torch.abs(l2norm_grad[dset] - mean_grad_task)
        #     # grad_gradnorm = torch.autograd.grad(loss_gradnorm, loss_weights[dset], create_graph=True)[0]
        
        # loss_gradnorm.backward(retain_graph=True)
        
        # # for n, p in module.named_parameters():
        # #     if p.grad is None:
        # #         print(n, "has not grad")
        # weight_opt.step()
        
        # for d in datasets:
        #     print(d, loss_weights[d].grad)
            
        # if scaler is not None:
        #     optimizer.zero_grad(set_to_none=args.grad_to_none)
        #     # for idx, dset in enumerate(datasets):
        #     #     loss_weights[dset].grad = weight_grad[idx]
        #     #     logged_loss.update({f"{dset}_loss_weights": loss_weights[dset].data})
            
        #     scaler.step(optimizer)
        #     scaler.update()
            
        # else:
        #     optimizer.zero_grad(set_to_none=args.grad_to_none)
        #     # for idx, dset in enumerate(datasets):
        #     #     loss_weights[dset].grad = weight_grad[idx]
        #     #     logged_loss.update({f"{dset}_loss_weights": loss_weights[dset].data})
            
        #     for d in datasets:
        #         print(d, loss_weights[d].data)
            
        #     optimizer.step() 
            
        #     weight_opt.zero_grad(set_to_none=args.grad_to_none)
        #     weight_opt.step()
            
        #     for d in datasets:
        #         print(d, loss_weights[d].data)
        #     exit()
        
        
        
        
        # # for dset in datasets:
        # #     inverse_ratio = (dataset_loss[dset] / mean_loss) ** alpha
        # #     dset_mean_grad = mean_grad_norm * float(inverse_ratio)
            
        #     # loss_gradnorm = torch.abs()
        
        # # for data, origin_task_loss in dataset_loss.items():
        # #     final_layer_grad = torch.autograd.grad(
        # #         origin_task_loss, last_module_grad.parameters(), retain_graph=True)
            
        
        # # for dset in datasets:
        # #     loss_weights[dset].grad.data *= 0.0
        
        # # last_module = module.get_last_shared_module()
        # # norms = []
        # # for data, origin_task_loss in dataset_loss.items():
        # #     final_layer_grad = torch.autograd.grad(
        # #         origin_task_loss, last_module.parameters(), retain_graph=True)
        # #     # print(loss_weights[data].grad.data)
        # #     norms.append(torch.norm(torch.mul(loss_weights[data], final_layer_grad[0])))
        
        # # norms = torch.stack(norms)
        # # loss_ratio = {dset: loss.detach().data / init_loss[dset].data for dset, loss in dataset_loss.items()}
        # # mean_ratio = torch.stack(list(loss_ratio.values())).mean()
        # # inverse_train_rate = {dset: ratio / mean_ratio for dset, ratio in loss_ratio.items()}
        
        # # mean_norm = norms.mean().detach()
        # # constant_target_grad = {dset: (mean_norm * (ratio ** alpha)).detach() for dset, ratio in inverse_train_rate.items()}
        # # grad_norm_loss = {dset: torch.abs(norms[i] - target) for i, (dset, target) in enumerate(constant_target_grad.items())}
        # # logged_loss.update({f"{dset}_grad_norm_loss": norm_loss for dset, norm_loss in grad_norm_loss.items()})
        
        # # grad_norm_loss_sum = sum(list(grad_norm_loss.values()))
        # # logged_loss.update({"total_grad_norm_loss": grad_norm_loss_sum})
        # # weight_grad = torch.autograd.grad(grad_norm_loss_sum, loss_weights.parameters())
        
        # # if scaler is not None:
        # #     optimizer.zero_grad(set_to_none=args.grad_to_none)
        # #     scaler.scale(losses).backward()
        # #     for idx, dset in enumerate(datasets):
        # #         loss_weights[dset].grad = weight_grad[idx]
        # #         logged_loss.update({f"{dset}_loss_weights": loss_weights[dset].data})
            
        # #     scaler.step(optimizer)
        # #     scaler.update()
            
        # # else:
        # #     optimizer.zero_grad(set_to_none=args.grad_to_none)
        # #     # if the retain_graph is True, the gpu memory will be increased, consequently occured OOM
        # #     losses.backward()
        # #     for idx, dset in enumerate(datasets):
        # #         loss_weights[dset].grad = weight_grad[idx]
        # #         logged_loss.update({f"{dset}_loss_weights": loss_weights[dset].data})
            
        # #     optimizer.step()    
        
        
        
        # normalized_coeff = sum(list(loss_weights.values()))
        # for dset in datasets:
        #     loss_weights[dset].data = (loss_weights[dset].data / normalized_coeff * len(datasets))
        #     print(loss_weights[dset].data)
        #     logged_loss.update({})
        # exit()
        
        
        
        ###########################################
        
        # for n, p in module.named_parameters():
        #     if p.grad is None:
        #         print(f"{n} has no grad")
        
        if warmup_sch is not None:
            warmup_sch.step()
        
        metric_logger.update(main_lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=losses, **logged_loss)
        iter_time.update(time.time() - end) 
        
        if BREAK:
            args.print_freq = 10
        
        if (i % args.print_freq == 0 or i == (biggest_size - 1)) and get_rank() == 0:
            metric_logger.log_iter(
                iter_time.global_avg,
                args.epochs-epoch,
                logger,
                i
            )
            
            str_loss_weights = ""
            for dset in datasets:
                str_loss_weights += f"{dset}: {loss_weights[dset].data}\n"
            
            logger.log_text(f"Normalizing Weight:\n {str_loss_weights}")
            
        # if tb_logger:
        #     tb_logger.update_scalars(loss_dict_reduced, i)
        #     grad_norm_loss_reduced = metric_utils.reduce_dict(grad_norm_loss, average=False)
        #     tb_logger.update_scalars(grad_norm_loss_reduced, i)   

            '''
            If the break block is in this block, the gpu memory will stuck (bottleneck)
            '''
            
        # if BREAK and i == args.print_freq:
        if BREAK and i == 2:
            print("BREAK!!")
            torch.cuda.synchronize()
            break
            
        end = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize(torch.cuda.current_device)
        
        logger.log_text(f"{i} iter finished\n")
        torch.cuda.synchronize()
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.log_text(f"{header} Total time: {total_time_str} ({total_time / biggest_size:.4f} s / it)")
    
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
    
    # from ptflops import get_model_complexity_info
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
        # metric_logger.epohcs = args.epochs
        metric_logger.set_before_train(header)
        
        # task_kwargs = {'dtype': dataset, 'task': task}
        # task_kwargs = {dataset: task} 
        task_kwargs = {"task_list": {dataset: task}}
        mac_count = 0.
        total_eval_time = 0
        
        
        total_start_time = time.time()
        for i, data in enumerate(taskloader):
            batch_set = {dataset: data}
            '''
            batch_set: images(torch.cuda.tensor), targets(torch.cuda.tensor)
            '''
            batch_set = metric_utils.preprocess_data(batch_set, data_cats)

            iter_start_time = time.time()
            macs, _, outputs = get_model_complexity_info(
                model, batch_set, dataset, task, as_strings=False,
                print_per_layer_stat=False, verbose=False
            )
            
            torch.cuda.synchronize()
            iter_time.update(time.time() - iter_start_time) 
            mac_count += macs
            
            val_function(outputs, batch_set[dataset][1], iter_start_time)
            torch.cuda.synchronize()
            if ((i % 50 == 0) or (i == len(taskloader) - 1)) and get_rank() == 0:
                metric_logger.log_iter(
                    iter_time.global_avg,
                    1,
                    logger,
                    i
                )
            
            # if tb_logger:
            #     tb_logger.update_scalars(loss_dict_reduced, i)   
                
            end = time.time()
            if BREAK and i == 2:
                print("BREAK!!!")
                break
            
            # if i == 19:
            #     break
            
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
        
        mac_count = torch.tensor(mac_count).cuda()
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

    