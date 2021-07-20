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
        self.base_feat = BaseFeature(
            args.in_chns,
            args.feat_filters[:3],
            args.feat_layers[:3],
            kernel_size=3,
            stride=1,
            padding=1,
            norm=args.norm,
            act=args.act,
        )
        self.combine = nn.Sequential(
            CombineModule(
                args.feat_filters[3],
                self.ori_chns + 1,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=args.norm,
                act=args.act,
            ),
            ResConvModule(
                args.feat_filters[3],
                args.feat_resblks,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=args.norm,
                act=args.act,
            ),
        )
        self.aspp = ASPPModule(
            args.feat_filters[3],
            args.aspp_filters,
            args.feat_filters[3:],
            args.feat_layers[3:],
            [self.ori_chns, 1],
            1,
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
        base_feat = self.base_feat(x)

        tar_size = base_feat.size()

        # transform anchor
        # anchor_ang = transform_anchor(
        #     sample["anchors_ang"], sample["pose_trans"], sample["pose_theta"], tar_size=tar_size, angle=True
        # )
        # anchor_rmsd = transform_anchor(
        #     sample["anchors_rmsd"], sample["pose_trans"], sample["pose_theta"], tar_size=tar_size
        # )
        align = [tar_size[3] * 1.0 / sample["anchors_ang"].size(3), tar_size[2] * 1.0 / sample["anchors_ang"].size(2)]
        grid = generate_rigid_grid(tar_size, sample["pose_trans"], sample["pose_theta"], align=align)
        anchor_ang = transform_anchor_angle(sample["anchors_ang"], grid)
        anchor_ang = anchor_ang + compute_local_rotation_angle(grid.permute(0, 3, 1, 2))
        anchor_rmsd = F.grid_sample(sample["anchors_rmsd"], grid, padding_mode="zeros", align_corners=False)
        anchor = torch.cat((binary_orientation(anchor_ang, ang_stride=self.ori_stride), anchor_rmsd), dim=1)

        ori_pred, seg_pred, type_pred = self.aspp(self.combine([base_feat, anchor]))

        # orientation
        ori_pred = torch.softmax(ori_pred, dim=1)
        ori_pred = orientation_highest_peak(ori_pred, ang_stride=self.ori_stride)
        ori_pred = ori_pred / ori_pred.sum(dim=1, keepdim=True)

        # segmentation
        seg_pred = torch.sigmoid(seg_pred)

        # type
        type_pred = torch.sigmoid(type_pred).flatten()

        return ori_pred, seg_pred, type_pred, anchor_ang
