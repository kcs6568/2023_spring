from torch import normal
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

try:
    from torchvision import prototype
except ImportError:
    prototype = None

from lib.presets import det_presets, seg_presets


def is_multiple_dataset(task_cfg):
    '''
        Desc: check the replicated dataset in detection and segmentation task
    '''
    valid_task = []
    ds_list = []
    
    for k, v in task_cfg.items():
        if v is not None:
            valid_task.append(k)
            ds_list.append(v['type'])
    
    if len(valid_task) <= 2:
        if 'clf' in valid_task:
            return False
    
    elif len(valid_task) > 2:
        if 'det' in valid_task and 'seg' in valid_task:
            if ds_list.count(task_cfg['det']['type']) > 1:
                return True
            
            else:
                return False
            
            
def return_sampler(train_dset, test_dset, world_size, rank, seed):
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True, seed=seed
    )
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True, seed=seed
    )
    
    return train_sampler, test_sampler


def get_train_dataloader(train_dset, train_sampler, args, bs, collate_fn=None):
    '''
    if segmentation dataset loader, the args.no_pin_memory is False, other datasets is True
    '''
    train_loader = torch.utils.data.DataLoader(
        train_dset, batch_size=bs, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=args.pin_memory, sampler=train_sampler, collate_fn=collate_fn, drop_last=True)
    
    return train_loader


def get_test_dataloader(test_dset, test_sampler, args, bs, collate_fn=None):
    '''
    if segmentation dataset loader, the args.no_pin_memory is False, other datasets is True
    '''
    
    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=args.pin_memory, sampler=test_sampler, collate_fn=collate_fn)
    
    return test_loader


def get_det_transform(train, args):
    if train:
        return det_presets.DetectionPresetTrain(args.data_augmentation)
    elif not args.prototype:
        return det_presets.DetectionPresetEval()
    else:
        if args.weights:
            weights = prototype.models.get_weight(args.weights)
            return weights.transforms()
        else:
            return prototype.transforms.CocoEval()
        
        
def get_seg_transform(train, args):
    if train:
        return seg_presets.SegmentationPresetTrain(base_size=520, crop_size=480)
    elif args.weights and args.test_only:
        from torchvision.transforms import functional as F
        
        weights = torchvision.models.get_weight(args.weights)
        trans = weights.transforms()

        def preprocessing(img, target):
            img = trans(img)
            size = F.get_dimensions(img)[1:]
            target = F.resize(target, size, interpolation=transforms.InterpolationMode.NEAREST)
            return img, F.pil_to_tensor(target)

        return preprocessing
    else:
        return seg_presets.SegmentationPresetEval(base_size=520)    


def get_transforms(type, train, args):
    # if multiple:
    #     t = dict(
    #         det=get_det_transform(train, args),
    #         seg=get_seg_transform(train, args)
    #     )
        
    #     return t
        
    if type == 'det':
        return get_det_transform(train, args)
    
    elif type == 'seg':
        return get_seg_transform(train, args)