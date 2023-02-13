from .voc_dataset import VOCDetection, VOCSegmentation

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs


def voc_collate_fn(task):
    def seg_collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


    def det_collate_fn(batch):
        return tuple(zip(*batch))


    if task in ['seg', 'aug']:
        return seg_collate_fn

    else:
        return det_collate_fn


def get_voc_dataset(task, cfg):
    if task == 'det':
        return VOCDetection(**cfg)
    
    elif task == 'seg':
        return VOCSegmentation(**cfg)