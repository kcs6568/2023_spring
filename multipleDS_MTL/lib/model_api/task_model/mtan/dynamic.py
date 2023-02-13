from dataclasses import replace
import numpy as np
from scipy.special import softmax
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ...modules.get_detector import build_detector, DetStem
from ...modules.get_backbone import build_backbone
from ...modules.get_segmentor import build_segmentor, SegStem
from ...modules.get_classifier import build_classifier, ClfStem
from ...backbones.resnet import IdentityBottleneck, conv1x1
from ....apis.warmup import set_decay_fucntion
from ....apis.loss_lib import disjointed_gate_loss


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        nn.init.constant_(m.bias, 0)


# class MTANStatic(nn.Module):
#     def __init__(self,
#                  backbone,
#                  detector,
#                  segmentor,
#                  task_cfg,
#                  **kwargs
#                  ) -> None:
#         super().__init__()
#         backbone_net = build_backbone(
#             backbone, detector, segmentor, kwargs)
#         attention_ch = kwargs['attention_channels']
#         self.task_per_dset = {
#             'cifar10': 'clf', 'stl10': 'clf',
#             'minicoco': 'det', 'voc': 'seg'
#         }
        
#         self.shared_layer1_b = backbone_net.body.layer1[:-1] 
#         self.shared_layer1_t = backbone_net.body.layer1[-1]

#         self.shared_layer2_b = backbone_net.body.layer2[:-1]
#         self.shared_layer2_t = backbone_net.body.layer2[-1]

#         self.shared_layer3_b = backbone_net.body.layer3[:-1]
#         self.shared_layer3_t = backbone_net.body.layer3[-1]

#         self.shared_layer4_b = backbone_net.body.layer4[:-1]
#         self.shared_layer4_t = backbone_net.body.layer4[-1]
        
#         self.fpn = backbone_net.fpn
        
#         self.stem_dict = nn.ModuleDict()
#         self.head_dict = nn.ModuleDict()
        
#         stem_weight = kwargs['state_dict']['stem']
#         for data, cfg in task_cfg.items():
#             task = cfg['task']
#             num_classes = cfg['num_classes']
            
#             if task == 'clf':
#                 stem = ClfStem(**cfg['stem'])
#                 head = build_classifier(
#                     backbone, num_classes, cfg['head'])
#                 stem.apply(init_weights)
                
                
#             elif task == 'det':
#                 stem = DetStem(**cfg['stem'])
                
#                 head_kwargs = {'num_anchors': len(backbone_net.body.return_layers)+1}
#                 head = build_detector(
#                     backbone, detector, 
#                     backbone_net.fpn_out_channels, num_classes, **head_kwargs)
#                 if stem_weight is not None:
#                     ckpt = torch.load(stem_weight)
#                     stem.load_state_dict(ckpt, strict=False)
#                     print("!!!Load weights for detection stem layer!!!")
            
#                 # stem = DetStem()
#                 # head = build_detector(detector, backbone.fpn_out_channels, 
#                 #                       cfg['num_classes'])
#                 # if stem_weight is not None:
#                 #     ckpt = torch.load(stem_weight)
#                 #     stem.load_state_dict(ckpt)
            
#             elif task == 'seg':
#                 stem = SegStem(**cfg['stem'])
#                 head = build_segmentor(segmentor, num_classes=num_classes, cfg_dict=cfg['head'])
#                 if stem_weight is not None:
#                     ckpt = torch.load(stem_weight)
#                     stem.load_state_dict(ckpt, strict=False)
#                     print("!!!Load weights for segmentation stem layer!!!")
                    
#                 # stem = SegStem(**cfg['stem'])
#                 # head = build_segmentor(segmentor, cfg['head'])
#                 # if stem_weight is not None:
#                 #     ckpt = torch.load(stem_weight)
#                 #     stem.load_state_dict(ckpt)
            
#             head.apply(init_weights)
#             self.stem_dict.update({data: stem})
#             self.head_dict.update({data: head})
        
#         self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
#         self.att_encoder1 = nn.ModuleDict({k: self.att_layer(attention_ch[0], attention_ch[0]//8, attention_ch[0]) for k in task_cfg.keys()})
#         self.att_encoder2 = nn.ModuleDict({k: self.att_layer(2 * attention_ch[1], attention_ch[1]//8, attention_ch[1]) for k in task_cfg.keys()})
#         self.att_encoder3 = nn.ModuleDict({k: self.att_layer(2 * attention_ch[2], attention_ch[2]//8, attention_ch[2]) for k in task_cfg.keys()})
#         self.att_encoder4 = nn.ModuleDict({k: self.att_layer(2 * attention_ch[3], attention_ch[3]//8, attention_ch[3]) for k in task_cfg.keys()})
        
#         self.encoder_block_att1 = self.att_block_layer(attention_ch[0], attention_ch[1] // 4)
#         self.encoder_block_att2 = self.att_block_layer(attention_ch[1], attention_ch[2] // 4)
#         self.encoder_block_att3 = self.att_block_layer(attention_ch[2], attention_ch[3] // 4)
        
    
#     def att_layer(self, in_channel, intermediate_channel, out_channel):
#         return nn.Sequential(
#             nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
#             nn.BatchNorm2d(intermediate_channel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
#             nn.BatchNorm2d(out_channel),
#             nn.Sigmoid())
                

#     def att_block_layer(self, in_channel, out_channel):
#         downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
#                                    nn.BatchNorm2d(4 * out_channel))
#         # return the same channel of each resnet block: 256, 512, 1024, 2048
#         return IdentityBottleneck(in_channel, out_channel, downsample=downsample)

    
#     def _generate_features(self, data_dict, tasks):
#         mode = 'train' if self.training else 'val'
#         det_feats = OrderedDict()
#         seg_feats = OrderedDict()
        
#         # ttt = True if 'voc' in data_dict else False
        
#         stem_feats = OrderedDict(
#             {dset: self.stem_dict[dset](data[0]) for dset, data in data_dict.items()}
#         )
        
        
#         ################################# layer 1 #################################
#         shared_backbone_feat1 = OrderedDict(
#             {dset: self.shared_layer1_b(data) for dset, data in stem_feats.items()}
#         )
#         shared_last_feat1 = OrderedDict(
#             {dset: self.shared_layer1_t(data) for dset, data in shared_backbone_feat1.items()}
#         )
#         ##########################################################################
        
#         ################################# layer 2 #################################
#         shared_backbone_feat2 = OrderedDict(
#             {dset: self.shared_layer2_b(data) for dset, data in shared_last_feat1.items()}
#         )
#         shared_last_feat2 = OrderedDict(
#             {dset: self.shared_layer2_t(data) for dset, data in shared_backbone_feat2.items()}
#         )
#         ##########################################################################
        
#         ################################# layer 3 #################################
#         shared_backbone_feat3 = OrderedDict(
#             {dset: self.shared_layer3_b(data) for dset, data in shared_last_feat2.items()}
#         )
#         shared_last_feat3 = OrderedDict(
#             {dset: self.shared_layer3_t(data) for dset, data in shared_backbone_feat3.items()}
#         )
#         ##########################################################################
        
#         ################################# layer 4 #################################
#         shared_backbone_feat4 = OrderedDict(
#             {dset: self.shared_layer4_b(data) for dset, data in shared_last_feat3.items()}
#         )
#         shared_last_feat4 = OrderedDict(
#             {dset: self.shared_layer4_t(data) for dset, data in shared_backbone_feat4.items()}
#         ) 
#         ##########################################################################
        
        
#         a_1_mask = {dset: self.att_encoder1[dset](back_feat) for dset, back_feat in shared_backbone_feat1.items()}  
#         a_1 = {dset: a_1_mask_i * shared_last_feat1[dset] for dset, a_1_mask_i in a_1_mask.items()}  
#         det_feats.update({'0': f for t, f in a_1.items() if self.task_per_dset[t] == 'det'})
#         a_1 = {dset: self.down_sampling(self.encoder_block_att1(a_1_i)) for dset, a_1_i in a_1.items()} # 256 -> 512
        
        
#         a_2_mask = {dset: self.att_encoder2[dset](torch.cat(
#             (back_feat, a_1[dset]), dim=1)) for dset, back_feat in shared_backbone_feat2.items()}    
#         a_2 = {dset: a_2_mask_i * shared_last_feat2[dset] for dset, a_2_mask_i in a_2_mask.items()}  
#         det_feats.update({'1': f for t, f in a_2.items() if self.task_per_dset[t] == 'det'})
#         a_2 = {dset: self.down_sampling(self.encoder_block_att2(a_2_i)) for dset, a_2_i in a_2.items()} # 512 -> 1024
        
        
#         a_3_mask = {dset: self.att_encoder3[dset](torch.cat(
#             (back_feat, a_2[dset]), dim=1)) for dset, back_feat in shared_backbone_feat3.items()} 
#         a_3 = {dset: a_3_mask_i * shared_last_feat3[dset] for dset, a_3_mask_i in a_3_mask.items()}  
#         det_feats.update({'2': f for t, f in a_3.items() if self.task_per_dset[t] == 'det'})
#         seg_feats.update({'2': f for t, f in a_3.items() if self.task_per_dset[t] == 'seg'})
#         a_3 = {dset: self.encoder_block_att3(a_3_i) for dset, a_3_i in a_3.items()} # 1024 -> 2048
        
#         a_4_mask = {dset: self.att_encoder4[dset](torch.cat( # 2048 + 2048
#             (back_feat, a_3[dset]), dim=1)) for dset, back_feat in shared_backbone_feat4.items()} 
#         a_4 = {dset: a_4_mask_i * shared_last_feat4[dset] for dset, a_4_mask_i in a_4_mask.items()} # 2048  
#         det_feats.update({'3': f for t, f in a_4.items() if self.task_per_dset[t] == 'det'})
#         seg_feats.update({'3': f for t, f in a_4.items() if self.task_per_dset[t] == 'seg'})
        
#         total_losses = OrderedDict()
        
        
#         for dset, att_feats in a_4.items():
#             # task = tasks[dset]
#             task = self.task_per_dset[dset]
#             print(task)
#             print(dset)
#             head = self.head_dict[dset]
#             targets = data_dict[dset][1]
            
#             if task == 'clf':
#                 out = head(att_feats, targets)
                
#                 if mode == 'val':
#                     return dict(outputs=out)
                
#             elif task == 'det':
#                 fpn_feats = self.fpn(det_feats)
                
#                 out = head(data_dict[dset][0], fpn_feats, 
#                                        origin_targets=targets, 
#                                        trs_fn=self.stem_dict[dset].transform)
                
#                 if mode == 'val':
#                     return out
                
#             elif task == 'seg':
#                 out = head(
#                     seg_feats, targets, input_shape=data_dict[dset][0].shape[-2:])
                
#                 if mode == 'val':
#                     return dict(outputs=out)
#             print("aa")
#             total_losses.update({f"{dset}_{k}": l for k, l in out.items()})
#             print("bb")
            
        
#         return total_losses
        
    
#     def forward(self, data_dict, kwargs):
#         if not self.training:
#             if not hasattr(data_dict, 'items'):
#                 data_dict = {list(kwargs.keys())[0]: [data_dict, None]}
        
#         return self._generate_features(data_dict, kwargs)
    



class MTANDynamic(nn.Module):
    def __init__(self,
                 backbone,
                 detector,
                 segmentor,
                 task_cfg,
                 **kwargs
                 ) -> None:
        super().__init__()
        backbone_network = build_backbone(
            backbone, detector, segmentor, kwargs)
        
        self.label_smoothing_alpha = kwargs['label_smoothing_alpha']
        kwargs['decay_settings'].update({'max_iter': kwargs['max_iter']})
        self.temp_decay = set_decay_fucntion(kwargs['decay_settings'])
        self.temperature = kwargs['decay_settings']['temperature']
    
        self.current_iter = 0
        self.is_hardsampling = kwargs['is_hardsampling']
        self.same_loss_weight = kwargs['same_loss_weight']
        self.sparsity_weight = kwargs['sparsity_weight']
        
        self.blocks = []
        self.ds = []
        self.num_per_block = []
        self.channel_per_block = []
        for _, p in backbone_network.body.named_children():
            block = []
            self.num_per_block.append(len(p))
            for m, q in p.named_children():
                if m == '0':
                    self.ds.append(q.downsample)
                    q.downsample = None
                    
                block.append(q)
                
            self.blocks.append(nn.ModuleList(block))
        
        self.blocks = nn.ModuleList(self.blocks)
        self.ds = nn.ModuleList(self.ds)
        self.fpn = backbone_network.fpn
        
        self.stem_dict = nn.ModuleDict()
        self.head_dict = nn.ModuleDict()
        
        self.return_layers = {}
        data_list = []
        
        stem_weight = kwargs['state_dict']['stem']
        for data, cfg in task_cfg.items():
            data_list.append(data)
            self.return_layers.update({data: cfg['return_layers']})
            
            task = cfg['task']
            num_classes = cfg['num_classes']
            if task == 'clf':
                stem = ClfStem(**cfg['stem'])
                head = build_classifier(
                    backbone, num_classes, cfg['head'])
                stem.apply(init_weights)
                
            elif task == 'det':
                stem = DetStem(**cfg['stem'])
                
                head_kwargs = {'num_anchors': len(backbone_network.body.return_layers)+1}
                head = build_detector(
                    backbone, detector, 
                    backbone_network.fpn_out_channels, num_classes, **head_kwargs)
                if stem_weight is not None:
                    ckpt = torch.load(stem_weight)
                    stem.load_state_dict(ckpt, strict=False)
                    print("!!!Load weights for detection stem layer!!!")
            
            elif task == 'seg':
                stem = SegStem(**cfg['stem'])
                head = build_segmentor(segmentor, num_classes=num_classes, cfg_dict=cfg['head'])
                if stem_weight is not None:
                    ckpt = torch.load(stem_weight)
                    stem.load_state_dict(ckpt, strict=False)
                    print("!!!Load weights for segmentation stem layer!!!")
            
            head.apply(init_weights)
            self.stem_dict.update({data: stem})
            self.head_dict.update({data: head})
        
        attention_ch = kwargs['attention_channels']
        
        self.dilation_count = 2
        self.is_downsample_for_next_feat = [True, True, False, False] # layer 1, 2, 3: layer 4 ignore because last layer was applied the dilated convolution.
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.att_encoder1 = nn.ModuleDict({k: self.att_layer(    attention_ch[0], attention_ch[0]//4, attention_ch[0]) for k in task_cfg.keys()})
        self.att_encoder2 = nn.ModuleDict({k: self.att_layer(2 * attention_ch[1], attention_ch[1]//4, attention_ch[1]) for k in task_cfg.keys()})
        self.att_encoder3 = nn.ModuleDict({k: self.att_layer(2 * attention_ch[2], attention_ch[2]//4, attention_ch[2]) for k in task_cfg.keys()})
        self.att_encoder4 = nn.ModuleDict({k: self.att_layer(2 * attention_ch[3], attention_ch[3]//4, attention_ch[3]) for k in task_cfg.keys()})
        
        self.is_global_attetion = [True, True, True, False]
        self.global_attention = nn.ModuleList([
            self.global_att_layer(attention_ch[idx], attention_ch[idx+1] // 4) for idx in range(len(attention_ch)-1)])
        
        self.make_gate_logits(data_list, len(task_cfg))    
        self.policys = {dset: torch.zeros(gate.size()).float() for dset, gate in self.task_gating_params.items()}
        
        self.use_sharing = kwargs['use_sharing']
        self.use_disjointed = kwargs['use_disjointed']
        
    
    def make_gate_logits(self, data_list, num_task):
        logit_dict = {}
        for t_id in range(num_task):
            task_logits = Variable((0.5 * torch.ones(sum(self.num_per_block), 2)), requires_grad=True)
            
            logit_dict.update(
                {data_list[t_id]: nn.Parameter(task_logits, requires_grad=True)})
            
        self.task_gating_params = nn.ParameterDict(logit_dict)
        
    
    def train_sample_policy(self):
        policys = {}
        for dset, prob in self.task_gating_params.items():
            policy = F.gumbel_softmax(prob, self.temperature, hard=self.is_hardsampling)
            # policy = torch.softmax(prob, dim=1)
            
            policys.update({dset: policy.float()})
            
        return policys
    
    
    def test_sample_policy(self, dset):
        task_policy = []
        task_logits = self.task_gating_params[dset]
        
        if self.is_hardsampling:
            hard_gate = torch.argmax(task_logits, dim=1)
            policy = torch.stack((1-hard_gate, hard_gate), dim=1).cuda()
            
        else:
            logits = softmax(task_logits.detach().cpu().numpy(), axis=-1)
            for l in logits:
                sampled = np.random.choice((1, 0), p=l)
                policy = [sampled, 1 - sampled]
                # policy = [1, 0]
                task_policy.append(policy)
            
            policy = torch.from_numpy(np.array(task_policy)).cuda()
        
        return {dset: policy}
        
    
    def decay_temperature(self):
        self.temperature = self.temp_decay.decay_temp(self.current_iter)
    
    
    
    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel))
            # nn.Sigmoid())
                

    def global_att_layer(self, in_channel, out_channel):
        channel_downsample = nn.Sequential(conv1x1(in_channel, 4 * out_channel, stride=1),
                                   nn.BatchNorm2d(4 * out_channel))
        return IdentityBottleneck(in_channel, out_channel, downsample=channel_downsample)
    
    
    def _extract_stem_feats(self, data_dict):
        stem_feats = OrderedDict()
        
        for dset, (images, _) in data_dict.items():
            stem_feats.update({dset: self.stem_dict[dset](images)})
        return stem_feats
    
    
    def get_features(self, data_dict, other_hyp):
        # print("1")
        backbone_feats = OrderedDict({dset: {} for dset in data_dict.keys()})
        # attention_mask = OrderedDict({dset: {} for dset in data_dict.keys()})
        # global_attention = OrderedDict({dset: {} for dset in data_dict.keys()})
        
        att_feat = None
        att_mask = None
        
        last_feat = None
        last_sec_feat = None
        
        data = self._extract_stem_feats(data_dict)
        
        if self.training:
            self.policys = self.train_sample_policy()
        else:
            self.policys = self.test_sample_policy(
                dset=list(other_hyp["task_list"].keys())[0])
            
        self.current_iter += 1
        self.decay_temperature()
        
        for dset, feat in data.items():
            block_count=0
            # print(f"{dset} geration")
            for layer_idx, num_blocks in enumerate(self.num_per_block):
                for block_idx in range(num_blocks):
                    # print(dset, layer_idx, block_idx)
                    # if leaky relu function is not in-place, process will be stucked in specific one gpu randomly
                    identity = self.ds[layer_idx](feat) if block_idx == 0 else feat
                    feat = F.leaky_relu_(self.blocks[layer_idx][block_idx](feat) + identity) 
                    feat = feat * self.policys[dset][block_count, 0] + identity * self.policys[dset][block_count, 1]
                    
                    block_count += 1
                    
                    if block_idx == num_blocks - 2:
                        last_sec_feat = feat
                    
                    elif block_idx == num_blocks - 1:
                        last_feat = feat
                    
                    
                att_feat = feat if layer_idx == 0 else torch.cat((last_sec_feat, att_mask), dim=1)
                # print(layer_idx, feat.size())
                # att_feat = feat if layer_idx == 0 else att_mask
                att_feat = getattr(self, f"att_encoder{layer_idx+1}")[dset](att_feat).sigmoid_()
                # print(att_feat.size())
                
                att_mask = att_feat * last_feat
                
                # print(att_mask.size())
                if block_idx == (num_blocks - 1):
                    if str(layer_idx) in self.return_layers[dset]:
                        backbone_feats[dset].update({str(layer_idx): att_mask})
                
                # if block_idx == (num_blocks - 1):
                #     if str(layer_idx) in self.return_layers[dset]:
                #         backbone_feats[dset].update({str(layer_idx): att_mask})
                
                
                if self.is_global_attetion[layer_idx]:
                    att_mask = self.global_attention[layer_idx](att_mask)
                # print(att_mask.size())
                if self.is_downsample_for_next_feat[layer_idx]:
                    att_mask = self.down_sampling(att_mask) 
                    
                # print(att_mask.size())
                # print()
                
            # exit()

        return backbone_feats
    
    
    def forward_train(self, origin_data, backbone_features, other_hyp):
        total_losses = OrderedDict()
        # print("4")
        
        for dset, back_feats in backbone_features.items():
            # print("eval", dset)
            task = other_hyp["task_list"][dset]
            head = self.head_dict[dset]
            
            targets = origin_data[dset][1]
            
            if task == 'clf':
                losses = head(back_feats, targets)
                
            elif task == 'det':
                fpn_feat = self.fpn(back_feats)
                losses = head(origin_data[dset][0], fpn_feat,
                                        self.stem_dict[dset].transform, 
                                    origin_targets=targets)
                
            elif task == 'seg':
                # print("here")
                losses = head(
                    back_feats, targets, input_shape=targets.shape[-2:])
            
            losses = {f"feat_{dset}_{k}": l for k, l in losses.items()}
            total_losses.update(losses)
            
            
        task_list = list(backbone_features.keys())
        
        if self.use_disjointed:     
            # print(self.sparsity_weight)           
            disjointed_loss = disjointed_gate_loss(
                self.task_gating_params, 
                task_list, 
                sum(self.num_per_block), 
                smoothing_alpha=self.label_smoothing_alpha,
                sparsity_weight=self.sparsity_weight)
            total_losses.update({"disjointed": disjointed_loss})
            
        return total_losses
    
    
    def forward_val(self, origin_data, backbone_features, other_hyp):
        dset = list(other_hyp["task_list"].keys())[0]
        task = list(other_hyp["task_list"].values())[0]
        head = self.head_dict[dset]
        
        back_feats = backbone_features[dset]
        
        if task == 'det':
            fpn_feat = self.fpn(back_feats)
            predictions = head(origin_data[dset][0], fpn_feat, self.stem_dict[dset].transform)
            
        else:
            if task == 'seg':
                predictions = head(
                    back_feats, input_shape=origin_data[dset][0].shape[-2:])
        
            else:
                predictions = head(back_feats)
            
            predictions = dict(outputs=predictions)
        
        return predictions
    
    
    def forward(self, data_dict, hyp):
        shared_features = self.get_features(data_dict, hyp)
        
        if self.training:
            return self.forward_train(data_dict, shared_features, hyp)
        else:
            return self.forward_val(data_dict, shared_features, hyp)