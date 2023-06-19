import argparse
import datetime
import errno
import os
import time
import numpy as np
from collections import defaultdict, deque
from collections import OrderedDict
import torch
from .dist_utils import *

torch.set_printoptions(linewidth=1000)

class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
        self.mean_iou = 0.
        self.filter_cats = None
        self.pixel_acc = []
        self.acc = 0.
        self.total_batch_size = []
        
        
    def update(self, pred, targets):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=targets.device)
        with torch.inference_mode():
            k = (targets >= 0) & (targets < n)
            inds = n * targets[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)
            
            label = targets < n
            gt = targets[label].int()
            
            pred = pred[label].int()
            pixel_acc = (gt == pred).float().mean().cpu().numpy()
            self.pixel_acc.append(pixel_acc)
            

    def reset(self):
        self.mat.zero_()
        self.pixel_acc = []

    # def compute(self):
    #     h = self.mat.float()
    #     acc_global = torch.diag(h).sum() / h.sum()
    #     acc = torch.diag(h) / h.sum(1)
    #     iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        
    #     self.pixelAcc = (self.pixel_acc * torch.stack(self.total_batch_size)).sum() / sum(
    #         self.total_batch_size)
        
    #     return acc_global, acc, iu, self.pixelAcc

    def compute(self):
        torch.distributed.all_reduce(self.mat)
        h = self.mat.float()
        self.acc_global = torch.diag(h).sum() / h.sum()
        self.acc = torch.diag(h) / h.sum(1)
        self.iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        self.mean_iou = self.iu.mean()
        
        self.pixel_acc = torch.from_numpy(np.array(self.pixel_acc)).cuda()
        self.total_batch_size = torch.tensor(self.total_batch_size).cuda()
        torch.distributed.all_reduce(self.pixel_acc)
        torch.distributed.all_reduce(self.total_batch_size)
        self.pixelAcc = (self.pixel_acc * self.total_batch_size).sum() / sum(
            self.total_batch_size) / get_world_size()
        

    # def reduce_from_all_processes(self):
    #     if not torch.distributed.is_available():
    #         return
    #     if not torch.distributed.is_initialized():
    #         return
    #     torch.distributed.barrier()
    #     torch.distributed.all_reduce(self.mat)
                
    
    def __str__(self):
        return ("- Global Correct: {:.1f}\n\n- Average Row Correct: {}\n\n- IoU: {}\n- mean IoU: {:.1f}\n- Pixel Acc: {:.2f}").format(
            self.acc_global.item() * 100,
            [f"{i:.1f}" for i in (self.acc * 100).tolist()],
            [f"{i:.1f}" for i in (self.iu * 100).tolist()],
            self.mean_iou.item() * 100,
            self.pixelAcc * 100
        )
     

    # def __str__(self):
    #     self.acc_global, self.acc, iu, self.pixelAcc = self.compute()
    #     self.mean_iou = iu.mean().item() 
    #     return ("- Global Correct: {:.1f}\n\n- Average Row Correct: {}\n\n- IoU: {}\n- mean IoU: {:.1f}\n- Pixel Acc: {:.2f}").format(
    #         self.acc_global.item() * 100,
    #         [f"{i:.1f}" for i in (self.acc * 100).tolist()],
    #         [f"{i:.1f}" for i in (iu * 100).tolist()],
    #         self.mean_iou * 100,
    #         self.pixelAcc * 100
    #     )


class EdgeMetric:
    def __init__(self):
        self.abs_err = []
        self.total_batch_size = []
        self.lower_better = ['abs_err']
    
    
    def update(self, pred, targets):
        binary_mask = (targets != 255)
        edge_output_true = pred.masked_select(binary_mask)
        edge_gt_true = targets.masked_select(binary_mask)
        abs_err = torch.abs(edge_output_true - edge_gt_true).mean()
        self.abs_err.append(abs_err.cpu().numpy())
        
        
    @property
    def reset(self):
        self.abs_err = []
        
    def compute(self):
        self.abs_err = np.stack(self.abs_err, axis=0)
        
        val_metrics = {}
        val_metrics['abs_err'] = (np.array(self.abs_err) * np.array(self.total_batch_size)).sum() / sum(self.total_batch_size)
        self.val_metrics = val_metrics
        
    
    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        
        torch.distributed.barrier()
        for k, value in self.val_metrics.items():
            torch.distributed.all_reduce(torch.from_numpy(np.expand_dims(value, axis=0)).cuda())
    

    def __str__(self):
        info = ""
        for mtype, value in self.val_metrics.items():
            info += f" - (Lower Better) {mtype.upper()}: {round(float(value), 3)}\n"
        return info



class KeypointMetric:
    def __init__(self):
        self.abs_err = []
        self.total_batch_size = []
        self.lower_better = ['abs_err']
        
    
    def update(self, pred, targets):
        binary_mask = (targets != 255)
        keypoint_output_true = pred.masked_select(binary_mask)
        keypoint_gt_true = targets.masked_select(binary_mask)
        abs_err = torch.abs(keypoint_output_true - keypoint_gt_true).mean()
        self.abs_err.append(abs_err.cpu().numpy())
        
        
    @property
    def reset(self):
        self.abs_err = []
        self.total_batch_size = []
        
    def compute(self):
        self.abs_err = np.stack(self.abs_err, axis=0)
        
        val_metrics = {}
        val_metrics['abs_err'] = (np.array(self.abs_err) * np.array(self.total_batch_size)).sum() / sum(self.total_batch_size)
        self.val_metrics = val_metrics
        
    
    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        
        torch.distributed.barrier()
        for k, value in self.val_metrics.items():
            torch.distributed.all_reduce(torch.from_numpy(np.expand_dims(value, axis=0)).cuda())
    

    def __str__(self):
        info = ""
        for mtype, value in self.val_metrics.items():
            info += f" - (Lower Better) {mtype.upper()}: {round(float(value), 3)}\n"
        return info     


class DepthMetric:
    def __init__(self, dataset):
        self.abs_err = []
        self.rel_err = []
        self.sq_rel_err = []
        self.ratio = []
        self.rms = []
        self.rms_log = []
        self.total_batch_size = []
        self.dataset = dataset
        
        self.lower_better = ['abs_err', 'rel_err', 'sq_rel_err']
        self.higher_better = ['sigma_1.25', 'sigma_1.25^2', 'sigma_1.25^3']
    
    
    def update(self, pred, targets):
        if self.dataset in ["nyuv2", "cityscapes"]:
            binary_mask = (torch.sum(targets, dim=1) > 3 * 1e-5).unsqueeze(1).cuda()
        elif self.dataset == 'taskonomy' and "mask" in targets:
            assert targets["mask"] is not None
            binary_mask = (targets["gt"] != 255) * (targets["mask"].int() == 1)
            targets = targets["gt"]
        
        depth_output_true = pred.masked_select(binary_mask)
        depth_gt_true = targets.masked_select(binary_mask)
        abs_err = torch.abs(depth_output_true - depth_gt_true)
        rel_err = torch.abs(depth_output_true - depth_gt_true) / depth_gt_true
        sq_rel_err = torch.pow(depth_output_true - depth_gt_true, 2) / depth_gt_true
        abs_err = torch.sum(abs_err) / torch.nonzero(binary_mask).size(0)
        rel_err = torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)
        sq_rel_err = torch.sum(sq_rel_err) / torch.nonzero(binary_mask).size(0)
        # calcuate the sigma
        term1 = depth_output_true / depth_gt_true
        term2 = depth_gt_true / depth_output_true
        ratio = torch.max(torch.stack([term1, term2], dim=0), dim=0)
        # calcualte rms
        rms = torch.pow(depth_output_true - depth_gt_true, 2)
        rms_log = torch.pow(torch.log10(depth_output_true + 1e-7) - torch.log10(depth_gt_true + 1e-7), 2)
        
        self.abs_err.append(abs_err.cpu().numpy())
        self.rel_err.append(rel_err.cpu().numpy())
        self.sq_rel_err.append(sq_rel_err.cpu().numpy())
        self.ratio.append(ratio[0].cpu().numpy())
        self.rms.append(rms.cpu().numpy())
        self.rms_log.append(rms_log.cpu().numpy())
        
        
    @property
    def reset(self):
        self.abs_err = []
        self.rel_err = []
        self.sq_rel_err = []
        self.ratio = []
        self.rms = []
        self.rms_log = []
        self.total_batch_size = []
        
    def compute(self):
        self.abs_err = np.stack(self.abs_err, axis=0)
        self.rel_err = np.stack(self.rel_err, axis=0)
        # self.sq_rel_err = np.stack(self.sq_rel_err, axis=0)
        self.ratio = np.concatenate(self.ratio, axis=0)
        self.rms = np.concatenate(self.rms, axis=0)
        self.rms_log = np.concatenate(self.rms_log, axis=0)
        self.rms_log = self.rms_log[~np.isnan(self.rms_log)]
        
        val_metrics = {}
        val_metrics['abs_err'] = (self.abs_err * np.array(self.total_batch_size)).sum() / sum(self.total_batch_size)
        val_metrics['rel_err'] = (self.rel_err * np.array(self.total_batch_size)).sum() / sum(self.total_batch_size)
        # val_metrics['sq_rel_err'] = (self.sq_rel_err * np.array(self.total_batch_size)).sum() / sum(self.total_batch_size)
        val_metrics['sigma_1.25'] = np.mean(np.less_equal(self.ratio, 1.25)) * 100
        val_metrics['sigma_1.25^2'] = np.mean(np.less_equal(self.ratio, 1.25 ** 2)) * 100
        val_metrics['sigma_1.25^3'] = np.mean(np.less_equal(self.ratio, 1.25 ** 3)) * 100
        # val_metrics['rms'] = (np.sum(self.rms) / len(self.rms)) ** 0.5
        # val_metrics['rms_log'] = (np.sum(self.rms_log) / len(self.rms_log)) ** 0.5
        
        self.val_metrics = val_metrics
        
    
    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        
        
        torch.distributed.barrier()
        for k, value in self.val_metrics.items():
            torch.distributed.all_reduce(torch.from_numpy(np.expand_dims(value, axis=0)).cuda())
    

    def __str__(self):
        max_len = 0
        for k in self.val_metrics.keys():
            if len(k) > max_len: max_len = len(k)
        
        info = ""
        for mtype, value in self.val_metrics.items():
            name = mtype.upper() + " " * (max_len - len(mtype))
            if mtype in self.higher_better:
                degree_type = "(Hihger Better)"
            elif mtype in self.lower_better:
                degree_type = "( Lower Better)"
            
            info += f" - {degree_type} {name}: {round(float(value), 3)}\n"
        
        return info            
            

class SurfaceNormalMetric:
    def __init__(self):
        self.cos_sim = []
        self.cosine_similarity = torch.nn.CosineSimilarity()
        self.normalize = torch.nn.functional.normalize
        
        self.lower_better = ['angle_mean', 'angle_median']
        self.higher_better = ['angle_11.25', 'angle_22.5', 'angle_30']
    
    
    def update(self, pred, targets):
        prediction = pred.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        gt = targets.permute(0, 2, 3, 1).contiguous().view(-1, 3)

        labels = gt.max(dim=1)[0] != 255
        # if hasattr(self, 'normal_mask'):
        #     gt_mask = self.normal_mask.permute(0, 2, 3, 1).contiguous().view(-1, 3)
        #     labels = labels and gt_mask.int() == 1

        gt = gt[labels]
        prediction = prediction[labels]

        gt = self.normalize(gt.float(), dim=1)
        prediction = self.normalize(prediction, dim=1)

        cos_similarity = self.cosine_similarity(gt, prediction)
        self.cos_sim.append(cos_similarity.cpu().numpy())
        
    @property
    def reset(self):
        self.cos_sim = []
        
        
    def compute(self):
        val_metrics = {}
        overall_cos = np.clip(np.concatenate(self.cos_sim), -1, 1)

        angles = np.arccos(overall_cos) / np.pi * 180.0
        # val_metrics['cosine_similarity'] = overall_cos.mean()
        val_metrics['angle_mean'] = np.mean(angles)
        val_metrics['angle_median'] = np.median(angles)
        # val_metrics['Angle RMSE'] = np.sqrt(np.mean(angles ** 2))
        val_metrics['angle_11.25'] = np.mean(np.less_equal(angles, 11.25)) * 100
        val_metrics['angle_22.5'] = np.mean(np.less_equal(angles, 22.5)) * 100
        val_metrics['angle_30'] = np.mean(np.less_equal(angles, 30.0)) * 100
        # val_metrics['Angle 45'] = np.mean(np.less_equal(angles, 45.0)) * 100
        
        self.val_metrics = val_metrics
        
    
    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        
        torch.distributed.barrier()
        for k, value in self.val_metrics.items():
            torch.distributed.all_reduce(torch.from_numpy(np.expand_dims(value, axis=0)).cuda())
    

    def __str__(self):
        max_len = 0
        for k in self.val_metrics.keys():
            if len(k) > max_len: max_len = len(k)
        
        info = ""
        for mtype, value in self.val_metrics.items():
            name = mtype.upper() + " " * (max_len - len(mtype))
            if mtype in self.higher_better:
                degree_type = "(Hihger Better)"
            elif mtype in self.lower_better:
                degree_type = "( Lower Better)"
                
            info += f" - {degree_type} {name}: {round(float(value), 3)}\n"
        
        return info            




class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
        
        if isinstance(value, int):
            self.total = int(self.total)
            
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=False):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        dist.all_reduce(values) # bottleneck occur
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="  "):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.metrics = ""
        self.n = 0
        self.val = 0.
        self.best_acc = 0.
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def set_before_train(self, header):
        space_fmt = ":" + str(len(str(self.largest_iters))) + "d"
        if torch.cuda.is_available():
            self.metrics = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "total_eta: {total_eta}",
                    "{meters}",
                ]
            )
    
    
    def log_iter(self, global_time, epochs, logger, iters):
        eta_seconds = global_time * (self.largest_iters - iters)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        total_eta = str(datetime.timedelta(
            seconds=int((global_time * (self.largest_iters * epochs - iters)))))
        
        if torch.cuda.is_available():
            logger.log_text(
                self.metrics.format(
                    iters,
                    self.largest_iters,
                    eta=eta_string,
                    total_eta=total_eta,
                    meters=str(self)
                )
            )
            
        else:
            logger.log_text(
                self.delimiter.format(
                    iters, self.largest_iters, eta=eta_string, 
                    meters=str(self), time=str(self.iter_time)
                )
            )
    
    
    def log_every(self, loaders, print_freq, logger, epochs=1, header=None, train_mode=True, return_count=False):
        largest_iters = len(list(loaders.values())[0])
        
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        # space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        space_fmt = ":" + str(len(str(largest_iters))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "total_eta: {total_eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        
        for obj in extract_batch(loaders, train_mode=train_mode, return_count=return_count):
            data_time.update(time.time() - end)
            yield obj # stop and return obj and come back.
            # break
            iter_time.update(time.time() - end)
            
            if i % print_freq == 0 or i == largest_iters - 1:
                global_time = iter_time.global_avg
                # eta_seconds = global_time * (len(iterable) - i)
                eta_seconds = global_time * (largest_iters - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                total_eta = str(datetime.timedelta(seconds=int((global_time * (largest_iters * epochs - i)))))
                
                if torch.cuda.is_available():
                    logger.log_text(
                        log_msg.format(
                            i,
                            largest_iters,
                            eta=eta_string,
                            total_eta=total_eta,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.log_text(
                        log_msg.format(
                            i, largest_iters, eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
            # if 'Validation' in header:
            #     break
            
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # logger.log_text(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")
        logger.log_text(f"{header} Total time: {total_time_str} ({total_time / largest_iters:.4f} s / it)")



def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        

def remove_on_master(path):
    if is_main_process():
        if os.path.isfile(path):
            os.remove(path)
            return True
        else:
            return False
            
def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def extract_batch(loaders, train_mode=True, return_count=False):
    return_dicts = OrderedDict()
    iter_data = OrderedDict()
    
    loader_lists = list(loaders.values())
    ds_keys = list(loaders.keys())
    if train_mode:
        if len(loader_lists) > 1:
            
            loader_size = [len(ld) for ld in loader_lists]
            iterator_lists = [iter(loader) for loader in loader_lists]
            load_cnt = {k: 1 for k in ds_keys}
            
            torch.cuda.empty_cache()
            for i in range(len(loader_lists[0])):
                return_dicts.clear()
                
                try:
                    print(f"###### {i}th mini-batch getter")
                    for dl, k in zip(iterator_lists, ds_keys):
                        return_dicts[k] = next(dl)
                    
                except StopIteration:
                    for i, (it, size) in enumerate(zip(iterator_lists, loader_size)):
                        if it._num_yielded == size:
                            iterator_lists[i] = iter(loader_lists[i])
                            load_cnt[ds_keys[i]] += 1
                    return_dicts.update({k: next(iterator_lists[i]) for i, k in enumerate(ds_keys) if not k in return_dicts.keys()})
                    
                    time.sleep(2)
                    
                finally:
                    iter_data.update(return_dicts)
                    if return_count:
                        iter_data.update({'load_count': load_cnt})
                    print(f"###### {i}th mini-batch will be yielded")
                    yield iter_data
                    
                    
        else:
            for data in loader_lists[0]:
                yield {ds_keys[0]: data}
                
    else:
        for data in loader_lists[0]:
            yield {ds_keys[0]: data}
    

def preprocess_data(batch_set, tasks, device="cuda"):
    def general_preprocess(batches):
        return batches[0].to(device), batches[1].to(device)
    
    
    def multitask_preprocess(batches):
        for task in batches[1].keys():
            if isinstance(batches[1][task], dict):
                batches[1][task] = {t_name: t.to(device) for t_name, t in batches[1][task].items()}
            else:
                batches[1][task] = batches[1][task].to(device)
        return batches[0].to(device), batches[1]
        
        
    def coco_preprocess(batches):
        # images = list(image.cuda() for image in batches[0])
        images = list(image.to(device) for image in batches[0])
        targets = [{k: v.to(device) for k, v in t.items()} for t in batches[1]]
        return images, targets
        
    data_dict = OrderedDict()
    
    for dset, data in batch_set.items():
        task = tasks[dset]
        if task == 'clf':
            data_dict[dset] = general_preprocess(data)
        
        elif task =='det':
            if 'coco' in dset:
                data_dict[dset] = coco_preprocess(data)
        
        elif task == 'seg':
            data_dict[dset] = multitask_preprocess(data)
            
    return data_dict


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
        
    return res


def get_params(model, logger, print_table=False):
    from prettytable import PrettyTable
    
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0

    for name, parameter in model.named_parameters():
        # print(name, parameter.numel())
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    
    param_log = "<Model Learnable Parameter>"
    
    if print_table:
        param_log += f"\n{table}\t"
    param_log += f" ---->> Total Trainable Params: {total_params/1e6}M"
    
    logger.log_text(param_log)
    

def get_task_params(model, task, gate, logger, only_backbone=True, print_table=False):
    from prettytable import PrettyTable
    
    table = PrettyTable(["Modules", "Parameters"])
    
    task_params = 0
    not_used_params = 0
    not_used_params_dict = {}
    
    fpn = False
    if task == 'minicoco':
        fpn = True
        
    name_list= []
    for name, parameter in model.named_parameters():
        if 'task_gating_params' in name: continue
            
        if only_backbone:
            if 'head' in name or 'stem' in name or 'fpn' in name:
                continue
            if parameter.requires_grad:
                task_params += parameter.numel() 
            else:
                params = parameter.numel()
                not_used_params += params
                not_used_params_dict.update({name: str(params/1e6)+"M"})
        else:
            if not parameter.requires_grad:
                params = parameter.numel()
                not_used_params += params
                not_used_params_dict.update({name: str(params/1e6)+"M"})
            # if not parameter.requires_grad: continue
            else:
                if 'head' in name or 'stem' in name:
                    if task in name:
                        task_params += parameter.numel()
                        # print(name, parameter.numel())
                        # name_list.append(name)
                elif fpn:
                    task_params += parameter.numel()
                    # print(name, parameter.numel())
                    # name_list.append(name)
                else:
                    if 'fpn' not in name and 'blocks' in name:
                        task_params += parameter.numel()
                        # print(name, parameter.numel())
                        # name_list.append(name)
                    elif 'ds' in name:
                        task_params += parameter.numel()
                    # print(name, parameter.numel())
                    # name_list.append(name)
        # task_params += parameter.numel()
        # print(name, parameter.numel())
        
        # if 'cifar10' in name:
        #     print("cifar10", name, parameter.numel())
        #     params = parameter.numel()
        
        # else:
        #     if 'blocks' in name and 'fpn' not in name:
        #         print("blocks", name, parameter.numel())
        #         params = parameter.numel()
                
        #     elif 'ds' in name and 'head' not in name:
        #         print("ds", name, parameter.numel())
        #         params = parameter.numel()
        
        # if 'stem' in name:
        #     if task in name:
        #         print("stem", name, parameter.numel())
        #         params = parameter.numel()
        # elif 'head' in name:
        #     if task in name:
        #         print("head", name, parameter.numel())
        #         params = parameter.numel()
        # else:
        #     if 'blocks' in name and not 'fpn' in name:
        #         print("block", name, parameter.numel())
        #         params = parameter.numel()
        #     elif 'ds' in name:
        #         print("ds", name, parameter.numel())
        #         params = parameter.numel()
                
        
        
        
        # if task in name:
        #     print("task", name, parameter.numel())
        #     params = parameter.numel()
        
        # else:
        #     if 'ds' in name:
        #         if 'head' not in name: # only ds module
        #             print("not head ds", name, parameter.numel())
        #             params = parameter.numel()
                    
        #         elif 'head' in name:
        #             if task in name:
        #                 print("head ds", name, parameter.numel())
        #                 params = parameter.numel()
                
        #     elif 'blocks' in name:
        #         if not 'fpn' in name:
        #             print("blocks not fpn", name, parameter.numel()) 
        #             params = parameter.numel()
        #         else:
        #             if fpn:
        #                 print("blocks in fpn", name, parameter.numel())
        #                 params = parameter.numel()
        #             else:
        #                 print("continue", name, parameter.numel())
        #                 continue
        
        # table.add_row([name, params])
        # task_params+=params
    
    param_log = f"<Learnable Parameter for {task.upper()}>"
    
    if print_table:
        param_log += f"\n{table}\t"
    param_log += f" ---->> Total Trainable Params: {task_params/1e6}M"
    param_log += f" (Noe Used Params: {not_used_params/1e6}M)"
    param_log += f"\n Noe Used Param List: {not_used_params_dict}M)"
    # param_log += f" ---->> Total Trainable Params: {task_params}M"
    
    logger.log_text(param_log)
    # name_list= []
    # for name, parameter in model.named_parameters():
    #     if not parameter.requires_grad: continue
        
    #     if 'head' in name or 'stem' in name:
    #         if 'cifar10' not in name:
    #             task_params -= parameter.numel()
    #             print(name, parameter.numel(), task_params/1e6)
    #             name_list.append(name)
                
    #     elif 'fpn' in name:
    #         task_params -= parameter.numel()
    #         print(name, parameter.numel(), task_params/1e6)
    #         name_list.append(name)
    
    # print(task_params/1e6)
    # print(name_list)
        

def save_parser(args, path, filename='parser.json', format='json'):
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    
    save_file = os.path.join(path, filename)
    
    if format == 'json':
        import json
        with open(save_file, 'w') as f:
            json.dump(args, f, indent=2)


def set_random_seed(seed=None, deterministic=False, device='cuda'):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    # import random
    # import numpy as np
        
    # rank, world_size = get_rank(), get_world_size()
    # # if seed is None:
    # #     seed = np.random.randint(2**31)
    # # else:
    # #     seed = seed + rank
    # #     # seed = seed
    # # seed = seed + rank 
    
    # seed = sum(s for s in range(world_size))
    
    # if world_size == 1:
    #     return seed
    # print(seed)
    
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # # if deterministic:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    # return seed
    
    seed = seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    

def reduce_across_processes(val):
    if not is_dist_avail_and_initialized():
        # nothing to sync, but we still convert to tensor for consistency with the distributed case.
        return torch.tensor(val)

    t = torch.tensor(val, device="cuda")
    torch.distributed.barrier()
    torch.distributed.all_reduce(t)
    return t


def get_mtl_performance(single, multi):
    assert len(single) == len(multi)
    T = len(single)  
    
    total = 0.
    for i in range(T):
        total += (multi[i] - single[i]) / single[i]

    delta_perf = total / T
    
    return delta_perf

    
def load_seg_referneces(dataset):   
    seg_references = {
        'voc':{
            "sseg":{
                "mIoU": 0.901,
                "pixel_acc": 0.979
            }},
        
        'nyuv2': {
            "sseg":{
                "mIoU": 0.363,
                "pixel_acc": 0.669
                },
            'depth': {
                # 'abs_err_low': 0.54,
                # 'rel_err_low': 0.21,
                # 'sigma_1.25': 64.2,
                # 'sigma_1.25^2': 89.7,
                # 'sigma_1.25^3': 97.7},
                
                # scores from training with pretrained weight
                'abs_err_low': 0.507,
                'rel_err_low': 0.202,
                'sigma_1.25': 66.83,
                'sigma_1.25^2': 91.71,
                'sigma_1.25^3': 98.3},
                
            'sn': {
                # 'angle_mean_low': 0.151,
                # 'angle_median_low': 0.116,
                # 'angle_11.25': 48.6,
                # 'angle_22.5': 76.8,
                # 'angle_30': 88.3}
                
                # encBias_bs4_scrat200E_SN_warm12_BN
                'angle_mean_low': 15.7,
                'angle_median_low': 12.7,
                'angle_11.25': 45.0,
                'angle_22.5': 76.0,
                'angle_30': 88.8}
            }
        }

    assert dataset in seg_references
    
    return seg_references[dataset]