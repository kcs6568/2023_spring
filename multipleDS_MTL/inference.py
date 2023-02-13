import os
import cv2
import json
from matplotlib.image import _ImageBase
from matplotlib.style import available
import yaml
import numpy as np
import pandas as pd
import colorsys
from PIL import Image
from cv2 import transform
import matplotlib.pyplot as plt
from matplotlib import colors
from copy import deepcopy

import torch
import torchvision.transforms.functional as tv_F
from torch.nn import functional as F
from torchvision import datasets, models, transforms


from lib.utils.parser import InferenceParser
from lib.model_api.build_model import build_model


'''
automobile = car

'''

cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
stl_classes = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

# 91 classes
coco_classes_91 = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
    'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window',
    'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush'
)

coco_classes_80 = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)

voc_classes = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

'''
background, 
aeroplane, bicycle, bird, boat, bottle,
bus, car, cat, chair, cow, 
diningtable, dog, horse, motorbike, person, 
pottedplant, sheep, sofa, train, tvmonitor



boat
diningtable

'''



# 검정
# 보라, 진녹, 갈색, 
# 사람(분홍), 의자(갈색)
VOC_COLORMAP = [[0, 0, 0], 
                [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], 
                [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
                [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]


def visualize_classification(prediction, output_dir, save_name):
    results = F.softmax(prediction['outputs'], dim=1)
    print(results)
    if isinstance(results, torch.Tensor):
        results = results.cpu().detach().numpy()
        
    cmap = plt.cm.YlGn
    # norm = colors.Normalize(vmin=1.5, vmax=4.5)
    # c = np.random.rand(len(results))*3+1.5
    
    im = plt.imshow(results.reshape(1, 10), cmap=cmap)
    plt.xticks(np.arange(10), labels=cifar_classes, rotation=90, fontsize=23)
    ax = plt.gca()
    ax.get_yaxis().set_visible(False)
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    plt.colorbar(im, cax)
    plt.show()    
    plt.savefig(
        os.path.join(output_dir, f"{save_name}_prob.png"),
        dpi=600)
    

def visualize_detection(image, prediction, output_dir, save_name):
    def _nms(scores, threshold=0.95):
        available_idx = np.ndarray(scores.shape)
        
        for i, s in enumerate(scores):
            if s >= threshold:
                available_idx[i] = 1
            else:
                available_idx[i] = 0
                
        return available_idx
    
    
    from torchvision.utils import draw_bounding_boxes, save_image
    
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    boxes = prediction[0]['boxes'].cpu().detach().numpy()
    scores = prediction[0]['scores'].cpu().detach().numpy()
    labels = prediction[0]['labels'].cpu().detach().numpy()
    boxes = boxes.astype(np.int32)
    
    available_boxes_idx = _nms(prediction[0]['scores'])
    
    # boxes = prediction[0]['boxes'].to(torch.uint8)
    # labels = [str(l.cpu().detach()) for l in prediction['labels']]
    # 11(12), 25(26), 28(29), 29(30), 44(45), 65(66), 67(68), 68(69), 70(71), 82(83), 90(91)
    # (0, 0, 142) --> (175, 116, 175)
    # (0, 80, 100) --> (0, 152, 182)

    
    color = [
        (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), 
        (0, 60, 100), (0, 152, 182), (0, 0, 70), (0, 0, 192), (250, 170, 30), 
        (100, 170, 30), 
        (), # 11(12)
        
        (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
        (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
        (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
        (), # 25(26)
        
        (255, 179, 240), (0, 125, 92), 
        (), (), # 28(29), 29(30)
        
        (209, 0, 151), (188, 208, 182),
        (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
        (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
        (134, 134, 103), (145, 148, 174), (255, 208, 186),
        (197, 226, 255),
        (), # 44(45),
        (171, 134, 1), (109, 63, 54), (207, 138, 255),
        (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
        (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
        (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
        (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
        (119, 0, 170),  
        (), # 65(66)
        (0, 182, 199), 
        (), (), # 67(68), 68(69)
        (0, 165, 120), 
        (), # 70(71)
        (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
        (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
        (127, 167, 115), (59, 105, 106), 
        (), # 82(83)
        (142, 108, 45), (196, 172, 0),
        (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
        (191, 162, 208),
        () # 90(91)
        ]
    
    font = [
        cv2.FONT_ITALIC,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX
    ]
    
    # print(prediction)
    # exit()
    for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
        if available_boxes_idx[i] == 1:
            # print(boxes[i], scores[i], labels[i], coco_classes_91[labels[i]-1])
            image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color[labels[i]-1], 3)
            
            label = coco_classes_91[labels[i]-1]
            score = str(round(scores[i], 2))
            text = f"{label}|{score}"
            
            # cv2.putText(image, text, (xmin, ymin), font[1], 0.8, (255, 255, 255), 2)
            
    # exit()
    # output_dir = "/root/volume/coco_det_all_infer"
    # output_dir = "/root/volume/voc_det_all_infer"
    output_dir = "/root/volume/coco_det_temp"
    
    save_path = os.path.join(output_dir, f"{save_name}.png")
    cv2.imwrite(save_path, image)
    # exit()
    
    # drawn_boxes = draw_bounding_boxes(
    #     image, 
    #     boxes,
    #     fill=True)
    #     # labels)
    
    # print(drawn_boxes)
    # print(drawn_boxes.size())
    
    # drawn_boxes = drawn_boxes.permute(2, 1, 0)
    # print(drawn_boxes.size())
    # drawn_boxes = drawn_boxes.cpu().detach().numpy()
    # save_path = os.path.join(output_dir, f"{save_name}_drawn.png")
    # plt.imshow(drawn_boxes)
    # plt.savefig(
    #     save_path, dpi=600
    # )
    
    
    
    # save_image(drawn_boxes, save_path)
    
    
    
    pass

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    # fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = tv_F.to_pil_image(img)
        plt.imshow(np.asarray(img))
        plt.axis("off")
        plt.savefig(f"/root/test{i}.png")
        
        # axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        # exit()
        


def visualize_segmentation(image, torch_image, prediction, output_dir, save_name, threshold=0.8):
    from torchvision.utils import draw_segmentation_masks, save_image
    image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    prediction = prediction['outputs']
    
    out = prediction.squeeze(0)
    out = torch.softmax(out, dim=0)
    out_max = torch.argmax(out, dim=0)
    out_max = out_max.cpu().detach().numpy()
    
    H, W = out_max.shape
    black_mask = np.zeros((H, W, 3))
    # black_mask = cv2.cvtColor(black_mask, cv2.COLOR_BGR2RGB)
    
    # for i in out_max:
    #     print(i)
    # print(save_name)
    # exit()
        
    
    for ids in range(21):
        black_mask[out_max==ids, :] = VOC_COLORMAP[ids]

    # black_mask = cv2.cvtColor(black_mask, cv2.COLOR_RGB2BGR)
    
    split_name = save_name.split(".")[0]
    
    dirs_for_coco_to_seg = f"/root/volume/coco_seg_all_infer/{split_name}.png"
    # dirs_for_voc_seg = f"/root/volume/voc_all_infer/{split_name}.png"
    
    cv2.imwrite(dirs_for_coco_to_seg, black_mask)
    # cv2.imwrite(dirs_for_voc_seg, black_mask)
    
    # exit()
    
    
    
    
    
    
    # # print(prediction.size())
    
    # # print(prediction[0][:, 0,0])
    # # print(torch.softmax(prediction[0][:, 0,0], dim=0))
    
    # # exit()
    # # masks = torch.nn.functional.softmax(prediction, dim=1)
    # # m = masks[0].cpu().detach().numpy()
    
    # # # image_copy_for_mask = deepcopy(image)
    # # for i, c in enumerate(VOC_COLORMAP):
    # #     image_copy = deepcopy(image)
    # #     image_copy_for_mask = deepcopy(image)
    # #     m_ = cv2.threshold(m[i], 0.3, 255, cv2.THRESH_BINARY)[1]
    # #     image_copy_for_mask[m_==255] = c
    # #     # transparented_result = cv2.addWeighted(image_copy, 0.5, image_copy_for_mask, 0.7, 0)
    # # # bool_masks = [m > threshold for m in masks[0]]
    
    # #     cv2.imwrite(f"/root/volume/seg_multi_res_{i}.png", image_copy_for_mask)
    
    # # # # mask = bool_masks[0].cpu().detach().numpy()
    # # # mask = masks[0][0].cpu().detach().numpy()
    # # # dst = cv2.addWeighted(image, 0.8, mask, 0.2, 0)
    
    # # # cv2.imwrite("ttt.png", dst)
    # # exit()
    
    
    # # result = [m for m in masks[0]]
    # # print(result[0].unsqueeze(0).size())
    # # exit()
    # # show(result)
    # # exit()
    
    
    
    
    
    # # masks = torch.nn.functional.softmax(prediction, dim=1)
    # # print(masks.size())
    # # exit()

    # # torch_image = torch_image.to(torch.uint8)
    
    
    # # re = prediction.argmax(1).flatten()
    # # print(re.size())
    # # exit()
    
    
    # # a = masks[0][0][0]
    # # print(a.size())
    # # for p in masks:
    # #     print(p)
    # # exit()
    
    
    # cls = torch.argmax(prediction, dim=1)
    # print(cls.size())
    # print(max(cls))
    # exit()
    
    
    
    # cls = cls.to(torch.uint8)
    # a = draw_segmentation_masks(torch_image[0], cls)
    # exit()
    
    # image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # prediction = prediction['outputs']
    # cls = torch.argmax(prediction, dim=1)
    # cls = cls.cpu().detach().numpy()
    # # canvus = np.zeros(cls.shape, np.uint8)
    
    # cv2.imwrite("./test.png", cls)
    
    # # print(cls)
    # prediction = prediction.cpu().detach().numpy()
    # pred_imgs = [prediction[p] for p in cls]

    # print(pred_imgs)
    # print(pred_imgs.shape)
    
    # # for i, pred_img in enumerate(pred_imgs):
    # #     plt.imshow(pred_img)
    # #     plt.savefig(
    # #         f"./test_{i}", dpi=600
    # #     )
    
    # exit()
    
    
    # masks = torch.nn.functional.softmax(prediction, dim=1)
    
    
    # bool_masks = masks > threshold
    # # bool_masks = bool_masks.permute(2, 1, 0)
    # # print(bool_masks.size())
    
    
    
    
    # return
    # color = [
    #     (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (36,255,12)
    # ]
    
    # font = [
    #     cv2.FONT_ITALIC,
    #     cv2.FONT_HERSHEY_COMPLEX,
    #     cv2.FONT_HERSHEY_COMPLEX_SMALL,
    #     cv2.FONT_HERSHEY_DUPLEX,
    #     cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
    #     cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     cv2.FONT_HERSHEY_TRIPLEX
    # ]
    
    
    
    
    # save_path = os.path.join(output_dir, f"{save_name}.png")
    # cv2.imwrite(save_path, image)


from datasets.voc.voc_dataset import VOCSegmentation


task_setting = {
    'cifar10': 'clf',
    'stl10': 'clf',
    'minicoco': 'det',
    'voc': 'seg'
}

def main(args):
    with open(args.yaml_cfg, 'r') as f:
        configs = yaml.safe_load(f)
    
    for i, j in configs.items():
        setattr(args, i, j)
        
    with open(args.cfg, 'r') as f:
        configs = yaml.safe_load(f)
    
    for i, j in configs.items():
        setattr(args, i, j)
    
    model = build_model(args)
    ckpt = os.path.join(args.ckpt)
    checkpoint = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    best_result = checkpoint['best_results']
    print("best result:", best_result)
    torch.cuda.set_device(0)
    model.cuda()
    model.eval()
    
    dataset = args.dataset
    task = task_setting[dataset]
    
    ds = VOCSegmentation('2007', 'test', transform=transforms.Compose([transforms.ToTensor()]))
    
    for i, img in enumerate(ds):
        task_cfg = {dataset: task}
        prediction = model(img, task_cfg)
        
        if task == 'clf':
            visualize_classification(prediction, args.outdir, save_name)
        elif task == 'det':
            visualize_detection(image, prediction, args.outdir, save_name)
        elif task == 'seg':
            visualize_segmentation(image, torch_image, prediction, args.outdir, save_name)
        print(f"{i}th image ({save_name}) inference finished")    
    print("\nInference Finish\n\n")

if __name__ == "__main__":
    args = InferenceParser().args
    main(args)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # x = np.arange(10)
    # y = np.random.rand(len(x))
    # c = np.random.rand(len(x))*3+1.5
    # df = pd.DataFrame({"x":x,"y":y,"c":c})

    # cmap = plt.cm.YlGn
    # norm = colors.Normalize(vmin=1.5, vmax=4.5)

    # # plt.barh(y, x, color=cmap(norm(df.c.values)))
    
    # # plt.yticks([0, 0.5, 1])
    # # plt.xticks(['a', 'b', 'c','d','e','f','g','h','i','j'])
    
    # fig, ax = plt.subplots()
    # hbars = ax.barh(y, x, color=cmap(norm(c)))
    # # ax.set_xticks([0, 0.5, 1])
    # ax.set_yticks(y, labels=classes)
    # ax.invert_yaxis()
    # ax.set_xlim(right=1)

    # # 
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # only needed for matplotlib < 3.1
    # fig.colorbar(sm)


    # cmap = plt.cm.YlGn
    # norm = colors.Normalize(vmin=1.5, vmax=4.5)
    # people = classes
    # y_pos = np.arange(len(people))
    # c = np.random.rand(len(people))*3+1.5
    # performance = np.array([9.9990e-01, 3.2414e-10, 8.3237e-05, 2.4076e-06, 5.4104e-08, 4.2238e-09,
    #      1.4520e-08, 1.2475e-10, 1.9153e-05, 7.9884e-10])
    
    # fig, ax = plt.subplots()

    # hbars = ax.barh(y_pos, performance, align='center', color=cmap(norm(c)))
    # ax.set_yticks(y_pos, labels=people)
    # ax.invert_yaxis()  # labels read top-to-bottom
    # ax.set_xlabel('Performance')
    # ax.set_xticks([0, 0.5, 1])
    # ax.set_title('How fast do you want to go today?')

    # # Label with specially formatted floats
    # # ax.bar_label(hbars, fmt='%.2f')
    # ax.set_xlim(right=0.1)  # adjust xlim to fit labels


    # c = ax1.pcolor(Z, edgecolors='k', linewidths=4)
    # ax1.set_title('thick edges')

    
    
    # color = [192, 64, 1]
    # ratio = np.array([1/3, 1/2, 1/1, 1/5, 1/7, 1/8, 1/2, 1/2, 1/1, 1/3])
    
    # r, g, b = 192, 64, 1
    # r, g, b = [x/255.0 for x in [r, g, b]]
    # h, l, s = colorsys.rgb_to_hls(r, g*1/3, b)
    # r, g, b = colorsys.hls_to_rgb(h, l, s)
    # r, g, b = [x*255.0 for x in [r, g, b]]
    
    # print(r, g, b)
    # exit()
    
    
    # data = [scale_lightness(color, 1/scale) for scale in ratio]
    # print(data)
    
    
    
    
    # # exit()
    
    
    # # rgb_ratio = colors.hsv_to_rgb(hsv_ratio)
    # # print(rgb_ratio)
    # # # print(hsv_ratio)
    # # exit()
    # # colors = [255, 255, 0]
    
    # # data = (np.array(ratio) * 60).round().astype(int)
    # ratios_int = (np.array(ratio) * 60).round().astype(int)
    # plt.imshow(
    #     np.repeat(np.arange(len(data)), ratios_int).reshape(1, -1),
    #     cmap=colors.ListedColormap(data),
    #     aspect=ratios_int.sum()/10
    # )
    
    # plt.axis('off')
    
    
    # ax = plt.subplots()
    # im = ax.imshow(np.array(10).reshape(1,10), cmap=)
    

    
    
    
