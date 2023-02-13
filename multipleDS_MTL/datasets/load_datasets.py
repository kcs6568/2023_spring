import os
from collections import OrderedDict
from random import shuffle

import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from datasets.coco.coco_utils import get_coco, coco_collate_fn
from datasets.voc.voc_utils import *
from datasets.data_utils import *
from lib.transforms.shared_transforms import get_mean, get_std


def load_datasets(args, only_val=True):
    train_loaders = OrderedDict()
    val_loaders = OrderedDict()
    test_loaders = OrderedDict()
    
    for data, cfg in args.task_cfg.items():
        args.batch_size = args.task_bs[data]
        if 'clf' in cfg['task']:
            print(data)
            if data == 'cifar10':
                test_ld = None
                train_ld, val_ld = load_cifar10(args, data, '/root/data/pytorch_datasets')
            
            elif data == 'cifar100':
                test_ld = None
                train_ld, val_ld = load_cifar100(args, data, '/root/data/pytorch_datasets')
            
            if data == 'stl10':
                test_ld = None
                train_ld, val_ld = load_stl10(args, data, '/root/data/pytorch_datasets', input_size=cfg['input_size'])
            
            elif data == 'imagenet1k':
                test_ld = None
                train_ld, val_ld = load_imagenet1k(args, data, path='/root/data/img_type_datasets/ImageNet-1K', only_val=only_val)
                
            elif data == 'omniglot':
                test_ld = None
                train_ld, val_ld = load_omniglot(args, data, path='/root/data/pytorch_datasets/')
                
            elif data == 'usps':
                test_ld = None
                train_ld, val_ld = load_usps(args, data, path='/root/data/pytorch_datasets/USPS')
            
            elif data == 'mnist':
                test_ld = None
                train_ld, val_ld = load_mnist(args, data, path='/root/data/pytorch_datasets')
            
            elif data == 'fashion':
                test_ld = None
                train_ld, val_ld = load_fashion(args, data, path='/root/data/pytorch_datasets')
            

        elif 'det' in cfg['task']:
            if 'coco' in data:
                train_ld, val_ld, test_ld = load_coco(args, data, "/root/data/mmdataset/coco", trs_type='det')
                
            elif 'voc' in data:
                train_ld, val_ld, test_ld = load_voc(args, data, cfg['task'], cfg['task_cfg'])
            
        elif 'seg' in cfg['task']:
            if 'coco' in data:
                train_ld, val_ld, test_ld = load_coco(args, data, "/root/data/mmdataset/coco", trs_type='det')
                
            elif 'voc' in data:
                train_ld, val_ld, test_ld = load_voc(args, data, cfg['task'], cfg['task_cfg'])
                
            elif 'cityscapes' in data:
                train_ld, val_ld, test_ld = load_cityscape(args, cfg['task'], data, "/root/data/img_type_datasets/cityscapes")
           
        train_loaders[data] = train_ld
        val_loaders[data] = val_ld
        
        if test_ld:
            test_loaders[data] = test_ld
    
    # print(train_loaders)
    # exit()
    dataset_size = {data: len(dl) for data, dl in train_loaders.items()}
    largest_size = max(list(dataset_size.values()))
    largest_dataset = sorted([d for d, l in dataset_size.items() if l == largest_size])[0]

    
    train_loaders.move_to_end(largest_dataset, last=False)
    val_loaders.move_to_end(largest_dataset, last=False)
        
    return train_loaders, val_loaders, test_loaders



download=False
    
def load_cifar10(args, data, path):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if args.no_hflip:
        transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])
        
    else:
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    train_dataset = datasets.CIFAR10(
        path,
        transform=transform,
        download=download
    )
    
    test_dataset = datasets.CIFAR10(
        path,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        download=download,
        train=False
    )
    
    data_size = get_train_dataloader(train_dataset, None, args, 1)
    args.all_data_size[data] = len(data_size)
    del data_size
    
    train_sampler = None
    test_sampler = None
    
    if args.distributed:
        train_sampler, test_sampler = return_sampler(train_dataset, test_dataset, args.world_size, args.gpu, args.seed)

    args.pin_memory = True
    # train_loader, test_loader = get_dataloader(train_dataset, test_dataset, train_sampler, test_sampler, 
    #                                             args, args.batch_size)
    
    train_loader = get_train_dataloader(train_dataset, train_sampler, args, args.batch_size)
    test_loader = get_test_dataloader(test_dataset, test_sampler, args, args.batch_size)
    
    return train_loader, test_loader


def load_cifar100(args, path):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if args.no_hflip:
        transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])
        
    else:
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    train_dataset = datasets.CIFAR100(
        path,
        transform=transform,
        download=download
    )
    
    test_dataset = datasets.CIFAR100(
        path,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ]),
        download=download,
        train=False
    )

    train_sampler = None
    test_sampler = None
    
    if args.distributed:
        train_sampler, test_sampler = return_sampler(train_dataset, test_dataset, args.world_size, args.gpu, args.seed)

    args.pin_memory = False
    # train_loader, test_loader = get_dataloader(train_dataset, test_dataset, train_sampler, test_sampler, 
    #                                             args, args.batch_size)
    
    train_loader = get_train_dataloader(train_dataset, train_sampler, args, args.batch_size)
    test_loader = get_test_dataloader(test_dataset, test_sampler, args, args.batch_size)
    
    return train_loader, test_loader
    

def load_stl10(args, data, path, input_size=96):
    train_dataset = datasets.STL10(
        root=path,
        split="train",
        transform=None,
        download=download
    )
    test_dataset = datasets.STL10(
        root=path,
        split="test",
        transform=None,
        download=download
    )
    
    if not 'color_jitter' in args:
        args.color_jitter = False
    
    normalize = transforms.Normalize(
        mean = [0.4914, 0.4822, 0.4465], std = [0.2023, 0.1994, 0.2010])
    
    if args.color_jitter:
        train_transform = transforms.Compose([
            transforms.RandomCrop(input_size, 4),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(input_size, 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    data_size = get_train_dataloader(train_dataset, None, args, 1)
    args.all_data_size[data] = len(data_size)
    del data_size
    
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform
    
    train_sampler = None
    test_sampler = None
    
    if args.distributed:
        train_sampler, test_sampler = return_sampler(train_dataset, test_dataset, args.world_size, args.gpu, args.seed)
    
    # if args.num_datasets != 1:
    #     args.pin_memory = False
    args.pin_memory = False
    
    # train_loader, test_loader = get_dataloader(train_dataset, test_dataset, train_sampler, test_sampler, 
    #                                             args, args.batch_size)
    
    train_loader = get_train_dataloader(train_dataset, train_sampler, args, args.batch_size)
    test_loader = get_test_dataloader(test_dataset, test_sampler, args, args.batch_size)
    
    return train_loader, test_loader


def load_imagenet1k(args, path, img_size=224, only_val=True):
    ratio = 224.0 / float(img_size)
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    traindir = os.path.join(path, 'ILSVRC2012_img_train')
    valdir = os.path.join(path, 'ILSVRC2012_img_val')
    
    train_sampler = None
    val_sampler = None
    
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # ColorAugmentation(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(int(256 * ratio)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
    ]))
    
    # train_loader, val_loader = get_dataloader(train_dataset, val_dataset, train_sampler, val_sampler, 
    #                                             args, args.batch_size)
    
    train_loader = get_train_dataloader(train_dataset, args, args.batch_size)
    val_loader = get_test_dataloader(val_dataset, val_sampler, args, args.batch_size)
    
    if only_val:
        return None, val_loader
    else:
        return train_loader, val_loader
    
    
def load_omniglot(args, data, path):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    if args.no_hflip:
        transform=transforms.Compose([
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])
        
    else:
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    train_dataset = datasets.Omniglot(
        path,
        background=True,
        transform=transform,
        download=False
    )
    
    test_dataset = datasets.Omniglot(
        path,
        background=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
        download=True,
    )
    
    data_size = get_train_dataloader(train_dataset, None, args, 1)
    args.all_data_size[data] = len(data_size)
    del data_size
    
    train_sampler = None
    test_sampler = None
    
    if args.distributed:
        train_sampler, test_sampler = return_sampler(train_dataset, test_dataset, args.world_size, args.gpu, args.seed)

    args.pin_memory = True
    # train_loader, test_loader = get_dataloader(train_dataset, test_dataset, train_sampler, test_sampler, 
    #                                             args, args.batch_size)
    
    train_loader = get_train_dataloader(train_dataset, train_sampler, args, args.batch_size)
    test_loader = get_test_dataloader(test_dataset, test_sampler, args, args.batch_size)
    
    return train_loader, test_loader


def load_usps(args, data, path):
    usps_transforms = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # getitem에서 PIL color space를 강제로 rgb로 변경했음
    train_dataset = datasets.USPS(
        path,
        transform=usps_transforms,
        download=download
    )
    
    test_dataset = datasets.USPS(
        path,
        transform=usps_transforms,
        download=download,
        train=False
    )
    
    data_size = get_train_dataloader(train_dataset, None, args, 1)
    args.all_data_size[data] = len(data_size)
    del data_size
    
    train_sampler = None
    test_sampler = None
    
    if args.distributed:
        train_sampler, test_sampler = return_sampler(train_dataset, test_dataset, args.world_size, args.gpu, args.seed)

    args.pin_memory = True
    # train_loader, test_loader = get_dataloader(train_dataset, test_dataset, train_sampler, test_sampler, 
    #                                             args, args.batch_size)
    
    train_loader = get_train_dataloader(train_dataset, train_sampler, args, args.batch_size)
    test_loader = get_test_dataloader(test_dataset, test_sampler, args, args.batch_size)
    
    return train_loader, test_loader


def load_mnist(args, data, path):
    mnist_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # getitem에서 PIL color space를 강제로 rgb로 변경했음
    train_dataset = datasets.MNIST(
        path,
        transform=mnist_transforms,
        download=download
    )
    
    test_dataset = datasets.MNIST(
        path,
        transform=mnist_transforms,
        download=download,
        train=False
    )
    
    data_size = get_train_dataloader(train_dataset, None, args, 1)
    args.all_data_size[data] = len(data_size)
    del data_size
    
    train_sampler = None
    test_sampler = None
    
    if args.distributed:
        train_sampler, test_sampler = return_sampler(train_dataset, test_dataset, args.world_size, args.gpu, args.seed)

    args.pin_memory = True
    # train_loader, test_loader = get_dataloader(train_dataset, test_dataset, train_sampler, test_sampler, 
    #                                             args, args.batch_size)
    
    train_loader = get_train_dataloader(train_dataset, train_sampler, args, args.batch_size)
    test_loader = get_test_dataloader(test_dataset, test_sampler, args, args.batch_size)
    
    return train_loader, test_loader


def load_fashion(args, data, path):
    mnist_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # getitem에서 PIL color space를 강제로 rgb로 변경했음
    train_dataset = datasets.FashionMNIST(
        path,
        transform=mnist_transforms,
        download=download
    )
    
    test_dataset = datasets.FashionMNIST(
        path,
        transform=mnist_transforms,
        download=download,
        train=False
    )
    
    data_size = get_train_dataloader(train_dataset, None, args, 1)
    args.all_data_size[data] = len(data_size)
    del data_size
    
    train_sampler = None
    test_sampler = None
    
    if args.distributed:
        train_sampler, test_sampler = return_sampler(train_dataset, test_dataset, args.world_size, args.gpu, args.seed)

    args.pin_memory = True
    # train_loader, test_loader = get_dataloader(train_dataset, test_dataset, train_sampler, test_sampler, 
    #                                             args, args.batch_size)
    
    train_loader = get_train_dataloader(train_dataset, train_sampler, args, args.batch_size)
    test_loader = get_test_dataloader(test_dataset, test_sampler, args, args.batch_size)
    
    return train_loader, test_loader


# ----------------CLASSIFICATION----------------------    
    
    
def load_coco(args, data, data_path, trs_type='det'):
    import lib.utils.metric_utils as metric_utils
    from lib.apis.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
    
    train_type = 'train'
    if args.use_minids:
        train_type = 'mini' + train_type
    
    train_transforms = get_transforms(trs_type, True, args)
    val_transforms = get_transforms(trs_type, False, args)
    
    train_ds = get_coco(data_path, train_type, train_transforms)
    val_ds = get_coco(data_path, 'val', val_transforms)
    
    data_size = get_train_dataloader(train_ds, None, args, 1)
    args.all_data_size[data] = len(data_size)
    del data_size
    
    # train_ds = get_coco(data_path, train_type, get_transform(True, args))
    # val_ds = get_coco(data_path, 'val', get_transform(False, args))
        
    if args.distributed:
        train_sampler, val_sampler = return_sampler(train_ds, val_ds, args.world_size, args.gpu, args.seed)
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_ds)
        val_sampler = torch.utils.data.SequentialSampler(val_ds)
    
    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(train_ds, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    
    args.pin_memory = False
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_sampler=train_batch_sampler, num_workers=args.workers, collate_fn=coco_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, sampler=val_sampler, num_workers=args.workers, collate_fn=coco_collate_fn
    )
    
    test_loader = None
    if args.use_testset:
        test_transforms = get_transforms(trs_type, False, args)
        test_ds = get_coco(data_path, 'test', test_transforms)
        
        if args.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_ds)
        else:
            test_sampler = torch.utils.data.SequentialSampler(test_ds)
            
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=1, sampler=test_sampler, num_workers=args.workers, collate_fn=metric_utils.collate_fn
        )

    return train_loader, val_loader, test_loader



def load_voc(args, data, task_type, task_cfg):
    assert task_type in ['det', 'seg', 'aug']
    
    task_cfg['train'].update(dict(transform=get_transforms(task_type, True, args))) 
    task_cfg['test'].update(dict(transform=get_transforms(task_type, False, args))) 
    
    train_ds = get_voc_dataset(task_type, task_cfg['train'])
    val_ds = get_voc_dataset(task_type, task_cfg['test'])
    
    data_size = get_train_dataloader(train_ds, None, args, 1)
    args.all_data_size[data] = len(data_size)
    del data_size
    
    if args.distributed:
        train_sampler, val_sampler = return_sampler(train_ds, val_ds, args.world_size, args.gpu, args.seed)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_ds)
        val_sampler = torch.utils.data.SequentialSampler(val_ds)

    test_loader = None
    
    # args.pin_memory = False
    # if len(args.task_bs) == 1:
    #     args.pin_memory = True
    
    args.pin_memory = False
    
    # args.pin_memory = True  
    
    train_loader = get_train_dataloader(
        train_ds, train_sampler, args, args.batch_size, collate_fn=voc_collate_fn(task_type))
    
    val_loader = get_test_dataloader(
       val_ds, val_sampler, args, bs=1, collate_fn=voc_collate_fn(task_type))
    
    return train_loader, val_loader, test_loader


def load_cityscape(args, task_type, data, path):
    from .cityscapes.cityscapes_dataset import CityScapes
    # from lib.transforms import seg_transforms, shared_transforms
    '''
    Reference of used transform below:
        https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/main.py
    '''
    
    train_transform = get_transforms(task_type, True, args)
    test_transform = get_transforms(task_type, False, args)
    
    # crop_size = args.task_cfg['cityscapes']['crop_size']
    # train_transform = shared_transforms.Compose([
    #         seg_transforms.RandomCrop(size=crop_size),
    #         seg_transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    #         seg_transforms.RandomHorizontalFlip(flip_prob=0.5),
    #         seg_transforms.PILToTensor(),
    #         shared_transforms.ConvertImageDtype(torch.float),
    #         seg_transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                         std=[0.229, 0.224, 0.225]),
    #     ])

    # val_transform = shared_transforms.Compose([
    #     seg_transforms.PILToTensor(),
    #     shared_transforms.ConvertImageDtype(torch.float),
    #     seg_transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                     std=[0.229, 0.224, 0.225]),
    # ])
    
    train_ds = CityScapes(
        root=path,
        split='train',
        transform=train_transform
    )
    
    val_ds = CityScapes(
        root=path,
        split='val',
        transform=test_transform
    )
    
    data_size = get_train_dataloader(train_ds, None, args, 1)
    args.all_data_size[data] = len(data_size)
    del data_size
    
    if args.distributed:
        train_sampler, val_sampler = return_sampler(train_ds, val_ds, args.world_size, args.gpu, args.seed)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_ds)
        val_sampler = torch.utils.data.SequentialSampler(val_ds)

    test_loader = None
    
    # args.pin_memory = False
    # if len(args.task_bs) == 1:
    #     args.pin_memory = True
    
    args.pin_memory = False
    
    # args.pin_memory = True  
    
    train_loader = get_train_dataloader(
        train_ds, train_sampler, args, args.batch_size)
    
    val_loader = get_test_dataloader(
       val_ds, val_sampler, args, bs=1)
    
    return train_loader, val_loader, test_loader