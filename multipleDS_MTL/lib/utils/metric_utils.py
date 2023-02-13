import argparse
import datetime
import errno
import os
import time
from collections import defaultdict, deque
from collections import OrderedDict
import torch
from .dist_utils import *


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
        self.mean_iou = 0.
        self.filter_cats = None
        self.global_acc = 0.
        self.acc = 0.
        
        
    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.inference_mode():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        if self.filter_cats:
            mask = []
            for i in range(self.num_classes):
                if not i in self.filter_cats:
                    mask.append(i)
            # mask = torch.tensor(mask).cuda()
            mask = torch.tensor(mask).to("cuda")
            h = torch.index_select(h, 1, mask)
            h = torch.index_select(h, 0, mask)
            
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        self.acc_global, self.acc, iu = self.compute()
        self.mean_iou = iu.mean().item() * 100
        return ("- Global Correct: {:.1f}\n\n- Average Row Correct: {}\n\n- IoU: {}\n- mean IoU: {:.1f}").format(
            self.acc_global.item() * 100,
            [f"{i:.1f}" for i in (self.acc * 100).tolist()],
            [f"{i:.1f}" for i in (iu * 100).tolist()],
            self.mean_iou,
        )



class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
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
        # print(self.total, self.count)
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
            # print(k, v, self.meters[k].count)

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
        
        elif task =='det' or task == 'seg':
            if 'coco' in dset:
                data_dict[dset] = coco_preprocess(data)
                
            else:
                data_dict[dset] = general_preprocess(data)
                
    
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


def set_random_seed(seed, deterministic=False, device='cuda'):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    import random
    import numpy as np
    def _set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        if seed is None:
            seed = torch.initial_seed()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
    
    rank, world_size = get_rank(), get_world_size()
    if seed is None:
        seed = np.random.randint(2**31)
    else:
        # seed = seed + rank
        seed = seed
        
    if world_size == 1:
        return seed
        
    random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    # dist.broadcast(random_num, src=0)
    seed = random_num.item()
    _set_seed(seed)
    
    # return seed

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



    
    # a = 0
    # print("here")
    # if len(ds_keys) > 1:
    #     print("here2")
    #     for data in loader_lists[0]:
    #         print("here3")
    #         return_dict[ds_keys[0]] = data
            
    #         for data_list in zip(*loader_lists[1:]):
    #             try:
    #                 dicts = {ds_keys[i+1]:data for (i, data) in enumerate(data_list)}
                    
    #             except StopIteration:
    #                 print("StopIteration")
    #                 print("---"*60)
    #                 print(dicts)
    #                 for k, v in dicts.items():
    #                     if v is None:
    #                         key_idx = ds_keys.index(k)
    #                         loader_lists[key_idx] = iter(loader_lists[key_idx])
    #                         dicts[k] = next(loader_lists[key_idx])
                            
    #                 print(dicts)
    #                 exit()
                    

    #             return_dict.update(dicts)
    #             yield return_dict

    # if len(ds_keys) == 1 and (not train_mode):
    #     task = keys.pop()
    #     for data in loaders[task]:
    #         yield dict(task=data)
    
    # if len(ds_keys) == 2:
    #     if 'clf' in keys and 'det' in keys:
    #         for clf_, det_ in zip(loaders['clf'], loaders['det']):
    #             yield dict(clf=clf_, det=det_)
