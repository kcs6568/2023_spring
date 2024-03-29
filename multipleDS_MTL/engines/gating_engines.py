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
    def __init__(self, data_cats, loss_ratio, task_weights=None, weighting_method=None) -> None:
        self.weighting_method = weighting_method
        self.data_cats = data_cats
        self.loss_ratio = loss_ratio
        
    
    def merge_all_losses(self, output_losses):
        assert isinstance(output_losses, dict)
        logging_dict = None
        
        if self.weighting_method is None: 
            if self.loss_ratio is not None:
                assert isinstance(self.loss_ratio, dict)
                for key, ratio in self.loss_ratio.items():
                    output_losses.update({k: v*ratio for k, v in output_losses.items() if key in k})    
            losses = sum(loss for loss in output_losses.values())
            
        else:
            task_losses = {}

            for data in self.data_cats:
                data_loss = sum(loss for k, loss in output_losses.items() if data in k)
                if self.loss_ratio is not None:
                    data_loss*= self.loss_ratio[data]
                task_losses[data] = data_loss
             
            logging_dict = self.weighting_method(task_losses)
            if logging_dict is None:
                losses = sum(loss for loss in output_losses.values())
            else:
                losses = sum(loss for loss in logging_dict.values())
                
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
        logger.log_text(f"Warmup Iteration: {int(warmup_sch.total_iters)}")
    else:
        logger.log_text("No Warmup Training")
    
    if not module.retrain_phase:
        logger.log_text(f"Current Sparsity Lambda: {args.baseline_args['gate_args']['lambda_sparsity']}")
        logger.log_text(f"Different Block Weighting: {args.baseline_args['sparsity_weighting']}\n({module.sparsity_weight})")
    
    if module.retrain_phase:
        logger.log_text(module.print_fixed_gate_info())
    
    weighting_method = None if module.weighting_method is None else module.weighting_method
    grad_method = None if module.grad_method is None else module.grad_method
    
    loss_calculator = LossCalculator(
            args.task_per_dset, args.loss_ratio, weighting_method=weighting_method)
    
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
        
        losses, logging_dict = loss_calculator.merge_all_losses(loss_dict)
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
    
    ##############################################################################################################################
    def _validate_segmentation(outputs, targets, start_time=None):
        confmat.update(outputs.argmax(1).flatten(), targets.flatten())
        # confmat.update(targets.flatten(), outputs['outputs'].argmax(1).flatten())
    
    def _validate_depth(outputs, targets):
        depth_metrics.update(outputs, targets)
        
    def _validate_surface_normal(outputs, targets):
        surface_normal_metrics.update(outputs, targets)
        
    def _validate_keypoint(outputs, targets):
        keypoint_metrics.update(outputs, targets)
        
    def _validate_edge(outputs, targets):
        edge_metrics.update(outputs, targets)
    ##############################################################################################################################  
    
    
    ##############################################################################################################################
    def _metric_segmentation():
        # print("######### entered seg metirc")
        # confmat.reduce_from_all_processes()
        # logger.log_text("<Current Step Eval Accuracy>\n{}".format(confmat))
        
        confmat.compute()
        # confmat.reduce_from_all_processes()
        logger.log_text("<Current Step Eval Accuracy>\n{}".format(confmat))
        
        return {'mIoU': confmat.mean_iou.item(), 'pixel_acc': confmat.pixelAcc.item()}
    
    def _metric_depth():
        depth_metrics.compute()
        depth_metrics.reduce_from_all_processes()
        logger.log_text(f"<Depth Estimation Results>:\n{str(depth_metrics)}")
        
        # mean_results = sum(v for v in depth_metrics.val_metrics.values()) / len(depth_metrics.val_metrics)
        depth_metrics.reset
        
        # return {"avg": mean_results, "val_metrics": depth_metrics.val_metrics}
        return depth_metrics.val_metrics
    
    def _metric_surface_normal():
        surface_normal_metrics.compute()
        surface_normal_metrics.reduce_from_all_processes()
        logger.log_text(f"<Surface Normal Estimation Results>:\n{str(surface_normal_metrics)}")
        
        # mean_results = sum(v for v in surface_normal_metrics.val_metrics.values()) / len(surface_normal_metrics.val_metrics)
        surface_normal_metrics.reset
        
        # return {"avg": mean_results, "val_metrics": surface_normal_metrics.val_metrics}
        return surface_normal_metrics.val_metrics
    
    def _metric_keypoint():
        keypoint_metrics.compute()
        keypoint_metrics.reduce_from_all_processes()
        logger.log_text(f"<2D Keypoint Estimation Results>:\n{str(keypoint_metrics)}")
        
        # mean_results = sum(v for v in depth_metrics.val_metrics.values()) / len(depth_metrics.val_metrics)
        keypoint_metrics.reset
        
        # return {"avg": mean_results, "val_metrics": depth_metrics.val_metrics}
        return keypoint_metrics.val_metrics
    
    def _metric_edge():
        edge_metrics.compute()
        edge_metrics.reduce_from_all_processes()
        logger.log_text(f"<2D Edge Estimation Results>:\n{str(edge_metrics)}")
        
        # mean_results = sum(v for v in depth_metrics.val_metrics.values()) / len(depth_metrics.val_metrics)
        edge_metrics.reset
        
        # return {"avg": mean_results, "val_metrics": depth_metrics.val_metrics}
        return edge_metrics.val_metrics
    
    ##############################################################################################################################
    
    
    ##############################################################################################################################
    def _select_metric_fn(task, datatype, detail_type=None):
        if task == 'clf':
            return _metric_classification
        
        elif task == 'det':
            if 'coco' in datatype:
                return _metric_detection
            elif 'voc' in datatype:
                pass
            
        elif task == 'seg':
            assert detail_type is not None
            if detail_type == "sseg":
                return _metric_segmentation
            elif detail_type == "depth":
                return _metric_depth
            elif detail_type == "sn":
                return _metric_surface_normal
            elif detail_type == "keypoint":
                return _metric_keypoint
            elif detail_type == "edge":
                return _metric_edge
            
            # if 'coco' in datatype:
            #     pass
            # elif 'voc' in datatype:
            #     return _metric_segmentation
            
            # elif 'cityscapes' in datatype:
            #     assert detail_type is not None
            #     if detail_type == "sseg":
            #         return _metric_segmentation
            #     elif detail_type == "depth":
            #         return _metric_depth
    ##############################################################################################################################
             
             
    ##############################################################################################################################   
    def _select_val_fn(task, datatype, detail_type=None):
        if task == 'clf':
            return _validate_classification
        elif task == 'det':
            if 'coco' in datatype:
                return _validate_detection
            elif datatype == 'voc':
                pass
            
        elif task == 'seg':
            assert detail_type is not None
            if detail_type == "sseg":
                return _validate_segmentation
            elif detail_type == "depth":
                return _validate_depth
            elif detail_type == "sn":
                return _validate_surface_normal
            elif detail_type == "keypoint":
                return _validate_keypoint
            elif detail_type == "edge":
                return _validate_edge
            
            # if 'coco' in datatype:
            #     pass
            # elif 'voc' in datatype:
            #     return _validate_segmentation
            
            # elif 'cityscapes' in datatype:
            #     assert detail_type is not None
            #     if detail_type == "sseg":
            #         return _validate_segmentation
            #     elif detail_type == "depth":
            #         return _validate_depth
    ##############################################################################################################################
    
    
    
    final_results = dict()
    task_flops = {}
    task_total_time = {}
    task_avg_time = {}
    
    dense_shape = {}
    is_dense = False
    
    # from ptflops import get_model_complexity_info
    from lib.utils.flop_counters.ptflops import get_model_complexity_info
    for dataset, taskloader in data_loaders.items():
        # if not dataset in ["voc", "nyuv2", "cityscapes"]: continue
        
        task = data_cats[dataset]
        dset_classes = num_classes[dataset]
        
        if 'coco' in dataset:
            coco = get_coco_api_from_dataset(taskloader.dataset)
            iou_types = _get_iou_types(task)
            coco_evaluator = CocoEvaluator(coco, iou_types)
            coco_evaluator.set_logger_to_pycocotools(logger)
        
        confmat = None
        if task == 'seg':
            val_function = {detail_type: _select_val_fn(task, dataset, detail_type) for detail_type in dset_classes.keys()}
            metric_function = {detail_type: _select_metric_fn(task, dataset, detail_type) for detail_type in dset_classes.keys()}
            
            if 'sseg' in dset_classes:
                confmat = metric_utils.ConfusionMatrix(dset_classes['sseg'])
            if "depth" in dset_classes:
                depth_metrics = metric_utils.DepthMetric(dataset)
            if "sn" in dset_classes:
                surface_normal_metrics = metric_utils.SurfaceNormalMetric()
            if "keypoint" in dset_classes:
                keypoint_metrics = metric_utils.KeypointMetric()
            if "edge" in dset_classes:
                edge_metrics = metric_utils.EdgeMetric()
        
        else:
            val_function = _select_val_fn(task, dataset)
            metric_function = _select_metric_fn(task, dataset)
        metric_logger = metric_utils.MetricLogger(delimiter="  ")
        
        assert val_function is not None
        assert metric_function is not None
        
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
        
        batch_size = 0
        
        total_start_time = time.time()
        for i, data in enumerate(taskloader):
            batch_set = {dataset: data}
            '''
            batch_set: images(torch.cuda.tensor), targets(torch.cuda.tensor)
            '''
            batch_set = metric_utils.preprocess_data(batch_set, data_cats)
            
            # if i == 0: batch_size = len(batch_set[dataset])
            batch_size += len(batch_set[dataset])

            iter_start_time = time.time()
            macs, _, outputs = get_model_complexity_info(
                model, batch_set, dataset, task, as_strings=False,
                print_per_layer_stat=False, verbose=False
            )
            
            torch.cuda.synchronize()
            iter_time.update(time.time() - iter_start_time) 
            mac_count += macs
            
            if task == 'seg':
                 for detail_type, val_func in val_function.items():
                    val_func(outputs[detail_type], batch_set[dataset][1][detail_type])
                    if detail_type == "sseg":
                        confmat.total_batch_size.append(len(batch_set[dataset][0]))
                    elif detail_type == "depth":
                        depth_metrics.total_batch_size.append(len(batch_set[dataset][0]))
                    elif detail_type == "keypoint":
                        keypoint_metrics.total_batch_size.append(len(batch_set[dataset][0]))
                    elif detail_type == "edge":
                        edge_metrics.total_batch_size.append(len(batch_set[dataset][0]))
                    
            else:
                val_function(outputs, batch_set[dataset][1], iter_start_time)
            
            # if i == 9: break
            
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
        
        if task == "seg":
            for detail_type, met_func in metric_function.items():
                eval_result = met_func()
                name = f"{dataset}_{detail_type}"
                
                # final_results[name] = eval_result
                
                if detail_type in ["depth", "sn", "keypoint", "edge"]:
                    if detail_type == "depth":
                        lower_metrics = depth_metrics.lower_better
                    elif detail_type == "sn":
                        lower_metrics = surface_normal_metrics.lower_better
                    elif detail_type == "keypoint":
                        lower_metrics = keypoint_metrics.lower_better
                    elif detail_type == "edge":
                        lower_metrics = edge_metrics.lower_better    
                    
                    for m_type in eval_result.keys():
                        if m_type in lower_metrics:
                            deg_name = f"{name}_{m_type}_low"
                        else:
                            deg_name = f"{name}_{m_type}"
                        final_results[deg_name] = eval_result[m_type]
                
                else:                    
                    if isinstance(eval_result, dict):
                        final_results.update({f"{name}_{k}": v for k, v in eval_result.items()})
                    else:
                        final_results[name] = eval_result
                
        else:
            eval_result = metric_function()
            final_results[dataset] = eval_result
        
        del taskloader
        time.sleep(1)
        torch.cuda.empty_cache()
    
    time.sleep(3)        
    final_results.update({"task_flops": task_flops})
    
    return final_results
    