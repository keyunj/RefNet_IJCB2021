import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import *


class SingleAnchor(nn.Module):
    def __init__(self, args):
        super(SingleAnchor, self).__init__()
        self.ori_chns = 180 // args.ori_stride
        self.ori_stride = args.ori_stride

        self.normaliza_module = NormalizeModule(0, 1)
        self.backbone = BackboneFeature(
            args.in_chns,
            args.feat_filters[:3],
            args.feat_layers[:3],
            args.feat_resblks,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=args.norm,
            act=args.act,
        )
        self.aspp = OriSegASPPModule(
            args.aspp_filters,
            self.ori_chns,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=args.dilation,
            norm=args.norm,
            act=args.act,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # initializing all parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def forward(self, sample):
        x = self.normaliza_module(sample["data"])
        backbone = self.backbone(x)

        tar_size = backbone.size()
        anchor_ang = transform_anchor(
            sample["anchors_ang"], sample["pose_trans"], sample["pose_theta"], tar_size=tar_size, angle=True
        )
        anchor_rmsd = transform_anchor(
            sample["anchors_rmsd"], sample["pose_trans"], sample["pose_theta"], tar_size=tar_size
        )
        # transform anchor
        anchor = torch.cat(
            (binary_orientation(anchor_ang, ang_stride=self.ori_stride), anchor_rmsd), dim=1
        )

        # generate prediction
        ori_fusion, seg_fusion = self.aspp(backbone, anchor)

        # orientation
        ori_peak = torch.softmax(ori_fusion, dim=1)
        ori_peak = orientation_highest_peak(ori_peak, ang_stride=self.ori_stride)
        ori_peak = ori_peak / ori_peak.sum(dim=1, keepdim=True)

        # segmentation
        seg_sigmoid = torch.sigmoid(seg_fusion)

        return ori_peak, seg_sigmoid, anchor_ang


class MultiAnchor(nn.Module):
    def __init__(self, args):
        super(MultiAnchor, self).__init__()
        self.ori_chns = 180 // args.ori_stride
        self.ori_stride = args.ori_stride
        self.n_anchors = args.n_anchors

        self.normaliza_module = NormalizeModule(0, 1)
        self.backbone = BackboneFeature(
            args.in_chns,
            args.feat_filters[:3],
            args.feat_layers[:3],
            0,
            kernel_size=3,
            stride=1,
            padding=1,
            norm=args.norm,
            act=args.act,
        )
        self.combine_aligned = CombineModule(
            args.feat_filters[3], self.ori_chns + 1, kernel_size=3, stride=1, padding=1, norm=args.norm, act=args.act,
        )
        self.aspp = nn.Sequential(
            ResConvModule(
                args.feat_filters[3],
                args.feat_resblks,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=args.norm,
                act=args.act,
            ),
            ASPPModule(
                args.aspp_filters[0],
                args.aspp_filters,
                args.feat_filters[3:],
                args.feat_layers[3:],
                [self.ori_chns, 1],
                kernel_size=3,
                stride=1,
                padding=1,
                dilation=args.dilation,
                norm=args.norm,
                act=args.act,
            ),
        )

        self.base_angle = torch.arange(args.ori_stride / 2, 180, args.ori_stride).float().view(1, 1, -1, 1, 1) - 90
        self.initialize_weights()

    def initialize_weights(self):
        # initializing all parameters
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Linear)
                or isinstance(m, nn.BatchNorm1d)
                or isinstance(m, nn.BatchNorm2d)
            ):
                init_weights(m, init_type="kaiming")

    def forward(self, sample, use_gt=False):
        x = self.normaliza_module(sample["data"])
        backbone = self.backbone(x)

        delta_pred_lst = []
        ori_pred_lst = []
        seg_pred_lst = []
        type_pred_lst = []
        for ii in range(self.n_anchors):
            anchor_ang = sample["base_angs"][:, ii : ii + 1]
            anchor_rmsd = sample["base_rmsds"][:, ii : ii + 1]
            anchor = torch.cat((binary_orientation(anchor_ang, self.ori_stride), anchor_rmsd), dim=1)
            # prediction
            delta_pred, seg_pred, type_pred = self.aspp(self.combine_aligned([backbone, anchor]))
            # orientation
            delta_pred = torch.softmax(delta_pred, dim=1)
            delta_pred = orientation_highest_peak(delta_pred, ang_stride=self.ori_stride)
            delta_pred = delta_pred / delta_pred.sum(dim=1, keepdim=True)
            ori_pred = transform_angle(delta_pred, anchor_ang, self.ori_stride)
            # segmentation
            seg_pred = torch.sigmoid(seg_pred)
            # list
            delta_pred_lst.append(delta_pred)
            ori_pred_lst.append(ori_pred)
            seg_pred_lst.append(seg_pred)
            type_pred_lst.append(type_pred)

        ori_pred = torch.stack(ori_pred_lst, dim=1)
        seg_pred = torch.cat(seg_pred_lst, dim=1)
        type_pred = torch.softmax(torch.cat(type_pred_lst, dim=1), dim=1)

        type_weight = sample["type_wt"] if use_gt else type_pred
        ori_pred = (ori_pred * type_weight.view(-1, self.n_anchors, 1, 1, 1)).sum(dim=1)
        seg_pred = (seg_pred * type_weight.view(-1, self.n_anchors, 1, 1)).sum(dim=1, keepdim=True)

        return ori_pred, seg_pred, type_pred, delta_pred_lst
