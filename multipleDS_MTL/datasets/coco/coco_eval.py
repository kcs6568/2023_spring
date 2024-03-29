import copy
import io
from contextlib import redirect_stdout

import numpy as np
import pycocotools.mask as mask_util
import torch
import lib.utils.metric_utils as metric_utils
import lib.utils.dist_utils as dist
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        
    
    def set_logger_to_pycocotools(self, logger):
        for eval_module in self.coco_eval.values():
            setattr(eval_module, "logger", logger)


    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            with redirect_stdout(io.StringIO()):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            img_ids, eval_imgs = evaluate(coco_eval)
            self.eval_imgs[iou_type].append(eval_imgs)
        
            
    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            
            if dist.is_dist_avail_and_initialized():
                create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])
            else:
                DP_create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])
            

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()
            
    def log_eval_summation(self):
        for iou_type, coco_eval in self.coco_eval.items():
            results = coco_eval.stats
            area_type = ['small', 'medium', 'large']
            iou_rng = ['0.50', '0.75', '0.50:0.95']
            
            line = "<Evaluation Results>\n" \
                "IoU metric: {}\n".format(self.iou_types[0])
            
            for i in range(len(results)):
                titleStr = 'Average Precision' if i < 6 else 'Average Recall'
                typeStr = '(AP)' if i < 6 else '(AR)'
                areaRng = 'all' if i % 6 < 3 else area_type[i % 3]
                
                if i == 1 or i == 2:
                    iou_type = iou_rng[i-1]
                else:
                    iou_type = iou_rng[2]
                
                line += "  {:<18} {} @[ IoU={:<9} | area={:>6s} ] = {:0.3f}\n".format(
                    titleStr, typeStr, iou_type, areaRng, results[i]
                )
                
            coco_eval.logger.log_text(line)
        
            
    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        if iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        if iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        raise ValueError(f"Unknown iou type {iou_type}")


    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue
            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            
            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
            
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)



def merge_DP(img_ids, eval_imgs):
    eval_imgs = [eval_imgs]
    merged_img_ids = np.array(img_ids)
    merged_eval_imgs = np.concatenate(eval_imgs, 2)

    # print(len(merged_img_ids), len(merged_eval_imgs))
    
    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def DP_create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    # print(len(img_ids), len(eval_imgs))
    img_ids, eval_imgs = merge_DP(img_ids, eval_imgs)
    # print("***"*60)
    # print(len(img_ids), len(eval_imgs))
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())
    # print(len(img_ids), len(eval_imgs))

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

    # exit()




def merge(img_ids, eval_imgs):
    # print(len(img_ids), len(eval_imgs))
    all_img_ids = metric_utils.all_gather(img_ids)
    all_eval_imgs = metric_utils.all_gather(eval_imgs)
    # print(len(all_img_ids), len(all_eval_imgs))

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    # print(len(merged_img_ids), len(merged_eval_imgs[0]), len(merged_eval_imgs[1]))

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs

def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    # print("***"*60)
    # print(len(img_ids), len(eval_imgs))
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())
    # print(len(img_ids), len(eval_imgs))

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
    # exit()


def evaluate(imgs):
    with redirect_stdout(io.StringIO()):
        imgs.evaluate()
    return imgs.params.imgIds, np.asarray(imgs.evalImgs).reshape(-1, len(imgs.params.areaRng), len(imgs.params.imgIds))
