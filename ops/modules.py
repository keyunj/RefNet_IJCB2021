import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal

from .SwitchNorm import SwitchNorm2d


class NormalizeModule(nn.Module):
    def __init__(self, m0, var0, eps=1e-6):
        super(NormalizeModule, self).__init__()
        self.m0 = m0
        self.var0 = var0
        self.eps = eps

    def forward(self, x):
        x_m = x.mean(dim=(1, 2, 3), keepdim=True)
        x_var = x.var(dim=(1, 2, 3), keepdim=True)
        y = (self.var0 * (x - x_m) ** 2 / x_var.clamp_min(self.eps)).sqrt()
        y = torch.where(x > x_m, self.m0 + y, self.m0 - y)
        return y


class EnhanceModule(nn.Module):
    def __init__(self, factor=8, enh_ksize=25, ori_stride=1, sigma=4, Lambda=9, psi=0, gamma=1):
        super(EnhanceModule, self).__init__()
        self.factor = factor
        self.padding = enh_ksize // 2
        self.gb_sin, self.gb_cos = gabor_bank(enh_ksize, ori_stride, sigma, Lambda, psi, gamma)

    def forward(self, img, ori_p, mask=None):
        img_real = F.conv2d(img, self.gb_cos.type_as(img), padding=self.padding)
        # img_real = F.pad(img_real, pad=(self.padding, self.padding, self.padding, self.padding))
        img_real = (img_real * F.interpolate(ori_p, scale_factor=self.factor)).sum(dim=1, keepdim=True)

        if mask is None:
            return img_real
        else:
            return img_real * F.interpolate(mask, scale_factor=self.factor)


class BackboneFeature(nn.Module):
    def __init__(
        self,
        in_chns,
        nb_filters=[64, 128, 256],
        nb_layers=[2, 2, 2],
        nb_resblks=4,
        kernel_size=3,
        stride=1,
        padding=1,
        norm="b",
        act="p",
    ):
        super(BackboneFeature, self).__init__()

        layers = []

        cur_channels = in_chns
        for out_chns, cur_layers in zip(nb_filters, nb_layers):
            layers.append(
                SingleConv(
                    cur_channels, out_chns, kernel_size=kernel_size, stride=stride, padding=padding, norm=norm, act=act,
                )
            )
            cur_channels = out_chns
            for _ in range(cur_layers - 1):
                layers.append(
                    SingleConv(
                        cur_channels,
                        cur_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        norm=norm,
                        act=act,
                    )
                )
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        for _ in range(nb_resblks):
            layers.append(ResBlock(cur_channels, kernel_size, stride, padding, dilation=1, norm=norm, act=act))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class CombineModule(nn.Module):
    def __init__(self, feat_chns=256, cat_chns=180, kernel_size=3, stride=1, padding=1, norm="b", act="p"):
        super(CombineModule, self).__init__()
        self.combine = SingleConv(feat_chns + cat_chns, feat_chns, kernel_size, stride, padding, norm=norm, act=act)

    def forward(self, x):
        return self.combine(torch.cat(x, dim=1))


class ResConvModule(nn.Module):
    def __init__(self, feat_chns=256, feat_resblks=2, kernel_size=3, stride=1, padding=1, norm="b", act="p"):
        super(ResConvModule, self).__init__()
        layers = []
        for _ in range(feat_resblks):
            layers.append(ResBlock(feat_chns, kernel_size, stride, padding, norm=norm, act=act))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ClassifierModule(nn.Module):
    def __init__(self, feat_chns=256, out_chns=1, kernel_size=3, stride=1, padding=1, norm="b", act="p"):
        super(ClassifierModule, self).__init__()
        self.conv = SingleConv(feat_chns, 512, kernel_size, stride, padding, norm=norm, act=act)
        self.classifier = nn.Sequential(nn.Linear(512, 512), nn.ReLU(True), nn.Linear(512, out_chns))

    def forward(self, x):
        y = self.conv(x)
        y = F.adaptive_avg_pool2d(y, (1, 1)).flatten(start_dim=1)
        y = self.classifier(y)
        return y


class ASPPModule(nn.Module):
    def __init__(
        self,
        in_chns=256,
        aspp_filters=[256, 128],
        feat_filters=[256, 256],
        feat_layers=[1, 2],
        out_chns=[1],
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=[1, 4, 8],
        norm="b",
        act="p",
    ):
        super(ASPPModule, self).__init__()
        self.n_out = len(out_chns)

        self.aspp = nn.ModuleList()
        for d in dilation:
            p = d * (kernel_size // 2)
            self.aspp.append(
                ASPPSingleModule(in_chns, aspp_filters, out_chns, kernel_size, stride, p, d, norm=norm, act=act)
            )

        self.ftype = nn.Sequential(
            BackboneFeature(in_chns, feat_filters, feat_layers, 0, kernel_size, stride, padding, norm=norm, act=act),
            ClassifierModule(feat_filters[-1], 1, kernel_size, stride, padding, norm=norm, act=act),
        )

    def forward(self, x):
        # predict orientation and segmentation
        out_fusion = [0 for _ in range(self.n_out)]
        for module in self.aspp:
            out = module(x)
            out_fusion = [y1 + y2 for y1, y2 in zip(out_fusion, out)]

        type_pred = self.ftype(x)
        return (*out_fusion, type_pred)


class ASPPSingleModule(nn.Module):
    def __init__(
        self,
        in_chns=256,
        aspp_filters=[256, 128],
        out_chns=[1],
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        norm="b",
        act="p",
    ):
        super(ASPPSingleModule, self).__init__()
        self.conv = SingleConv(in_chns, aspp_filters[0], kernel_size, stride, padding, dilation, norm=norm, act=act)
        self.out_list = nn.ModuleList()
        for cur_out_chns in out_chns:
            self.out_list.append(
                nn.Sequential(
                    SingleConv(aspp_filters[0], aspp_filters[1], norm=norm, act=act),
                    nn.Conv2d(aspp_filters[1], cur_out_chns, kernel_size=1),
                )
            )

    def forward(self, x):
        conv = self.conv(x)
        out_list = []
        for module in self.out_list:
            out_list.append(module(conv))
        return out_list


class OriSegASPPModule(nn.Module):
    def __init__(
        self,
        feat_channels=[256, 128],
        ori_channels=90,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=[1, 4, 8],
        norm="b",
        act="p",
    ):
        super(OriSegASPPModule, self).__init__()
        self.aspp = nn.ModuleList()
        for d in dilation:
            p = d * (kernel_size // 2)
            self.aspp.append(
                OriSegSingleModule(
                    feat_channels, ori_channels, kernel_size, stride, padding=p, dilation=d, norm=norm, act=act
                )
            )

    def forward(self, x, anchor):
        # predict orientation and segmentation
        ori_fusion = 0
        seg_fusion = 0
        for module in self.aspp:
            ori, seg = module(x, anchor)
            ori_fusion += ori
            seg_fusion += seg
        return ori_fusion, seg_fusion


class OriSegSingleModule(nn.Module):
    def __init__(
        self,
        feat_channels=[256, 128],
        ori_channels=90,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        norm="b",
        act="p",
    ):
        super(OriSegSingleModule, self).__init__()
        in_chns = feat_channels[0] + ori_channels + 1
        self.conv = SingleConv(
            in_chns,
            feat_channels[0],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            norm=norm,
            act=act,
        )
        self.ori_layers = nn.Sequential(
            SingleConv(feat_channels[0], feat_channels[1], norm=norm, act=act),
            nn.Conv2d(feat_channels[1], ori_channels, kernel_size=1),
        )
        self.seg_layers = nn.Sequential(
            SingleConv(feat_channels[0], feat_channels[1], norm=norm, act=act),
            nn.Conv2d(feat_channels[1], 1, kernel_size=1),
        )

    def forward(self, x, anchor=None):
        if anchor is not None:
            conv = self.conv(torch.cat((x, anchor), dim=1))
        else:
            raise (ValueError("Anchor is required to predict orientation"))
        return self.ori_layers(conv), self.seg_layers(conv)


class ResBlock(nn.Module):
    def __init__(self, chns, kernel_size=3, stride=1, padding=1, dilation=1, norm="b", act="p"):
        super(ResBlock, self).__init__()
        self.conv1 = SingleConv(chns, chns, kernel_size, stride, padding, dilation, norm, act)
        self.conv2 = SingleConv(chns, chns, kernel_size, stride, padding, dilation, norm, act)

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return y + x


class SingleConv(nn.Module):
    def __init__(
        self, in_chns, out_chns, kernel_size=3, stride=1, padding=1, dilation=1, norm="b", act="p", order="cbr"
    ):
        super(SingleConv, self).__init__()
        layers = []
        for c in order:
            if c == "c":
                layers.append(
                    nn.Conv2d(
                        in_chns,
                        out_chns,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        bias=norm is None,
                    )
                )
            elif c == "b":
                if norm == "b":
                    layers.append(nn.BatchNorm2d(out_chns))
                elif norm == "s":
                    layers.append(SwitchNorm2d(out_chns))
                elif norm == "i":
                    layers.append(nn.InstanceNorm2d(out_chns))
                elif norm is not None:
                    raise ValueError(f"Unsupport normalization type {norm}")
            elif c == "r":
                if act == "r":
                    layers.append(nn.ReLU(inplace=True))
                elif act == "l":
                    layers.append(nn.LeakyReLU(inplace=True))
                elif act == "p":
                    layers.append(nn.PReLU(out_chns, init=0.0))
                elif act is not None:
                    raise ValueError(f"Unsupport activation type {act}")
            else:
                raise ValueError(f"Unsupported layer type {c}")

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def transform_angle(ori_p, anchor_ang, ori_stride):
    # type 1
    B, C, H, W = ori_p.size()
    grid_c, grid_h, grid_w = torch.meshgrid(
        torch.linspace(-1, 1, C).type_as(ori_p),
        torch.linspace(-1, 1, H).type_as(ori_p),
        torch.linspace(-1, 1, W).type_as(ori_p),
    )
    grid_c = (grid_c[None] - 2 * anchor_ang / (180 - ori_stride) + 1) % 2 - 1
    grid = torch.stack((grid_w[None].expand(B, -1, -1, -1), grid_h[None].expand(B, -1, -1, -1), grid_c), dim=-1)
    ori_p = F.grid_sample(ori_p[:, None], grid, padding_mode="border", align_corners=False).squeeze(1)

    return ori_p


def transform_anchor(anchor, trans, theta, tar_size, angle=False):
    sin_theta = torch.sin(theta * np.pi / 180.0)
    cos_theta = torch.cos(theta * np.pi / 180.0)
    affine_rotate = torch.stack(
        (torch.cat((cos_theta, -sin_theta), dim=1), torch.cat((sin_theta, cos_theta), dim=1)), dim=1
    )
    affine_trans = torch.bmm(affine_rotate, -trans.view(-1, 2, 1))
    affine_matrix = torch.cat((affine_rotate, affine_trans), dim=2)  # B x 2 x 3
    indices = F.affine_grid(affine_matrix, tar_size, align_corners=False)

    if angle:
        new_anchor = anchor - theta.view(-1, 1, 1, 1)
        new_anchor = transform_anchor_ang(new_anchor, indices)
    else:
        new_anchor = F.grid_sample(anchor, indices, padding_mode="border", align_corners=False)

    return new_anchor


def transform_anchor_ang(anchor, grid):
    cos_2angle = torch.cos(anchor * np.pi / 90)
    sin_2angle = torch.sin(anchor * np.pi / 90)
    cos_2angle = F.grid_sample(cos_2angle, grid, padding_mode="border", align_corners=False)
    sin_2angle = F.grid_sample(sin_2angle, grid, padding_mode="border", align_corners=False)
    angle = torch.atan2(sin_2angle, cos_2angle) * 90 / np.pi
    return angle


def compute_local_rotation_angle(grid_pred, eps=1e-8):
    B, C, H, W = grid_pred.size()
    grid_h, grid_w = torch.meshgrid(
        torch.linspace(-1, 1, H).type_as(grid_pred), torch.linspace(-1, 1, W).type_as(grid_pred)
    )
    lower_grid = torch.stack((grid_w[None].expand(B, -1, -1) - 1.0 / W, grid_h[None].expand(B, -1, -1)), dim=-1)
    higher_grid = torch.stack((grid_w[None].expand(B, -1, -1) + 1.0 / W, grid_h[None].expand(B, -1, -1)), dim=-1)
    lower_pred = F.grid_sample(grid_pred, lower_grid, padding_mode="border", align_corners=False)
    higher_pred = F.grid_sample(grid_pred, higher_grid, padding_mode="border", align_corners=False)
    delta_pred = higher_pred - lower_pred
    # add offset when the delta is 0
    R2 = (delta_pred ** 2).sum(dim=1)
    vec0 = torch.tensor(2.0 / W).type_as(delta_pred)
    delta_pred[:, 0] = torch.where(R2 <= eps, vec0, delta_pred[:, 0])
    # compute rotation angle
    local_angle = torch.atan2(-delta_pred[:, 1], delta_pred[:, 0])[:, None] * 180 / np.pi
    return local_angle


def weighted_filter(blk_size=5):
    grid_x, grid_y = np.meshgrid(np.arange(blk_size), np.arange(blk_size))
    grid_x -= blk_size // 2
    grid_y -= blk_size // 2
    blk = 1 / np.sqrt(grid_x ** 2 + grid_y ** 2).clip(0.001, None)
    blk = blk / blk.sum()
    return blk


np_avg5_filter = weighted_filter(blk_size=5)
np_avg3_filter = weighted_filter(blk_size=3)


def expectation_pred(input, stride=2, angle=False, double=True, smooth=False, logits=False):
    if logits:
        p = torch.softmax(input, dim=1)
    else:
        p = input

    view_shape = (1, -1,) + (1,) * (p.ndim - 2)
    if angle:
        double_angle = 90 if double else 180
        base_angle = torch.arange(stride / 2, 180, stride).float().view(*view_shape) - 90
        sin_angle = torch.sin(base_angle * np.pi / double_angle).type_as(p)
        cos_angle = torch.cos(base_angle * np.pi / double_angle).type_as(p)
        sin_angle = (p * sin_angle).sum(dim=1, keepdim=True)
        cos_angle = (p * cos_angle).sum(dim=1, keepdim=True)
        ex = torch.atan2(sin_angle, cos_angle) * double_angle / np.pi
    else:
        coord = torch.linspace(0, 1, 180 // stride).view(*view_shape).type_as(p)
        ex = (p * coord).sum(dim=1, keepdim=True)

    if smooth:
        avg_filter = torch.tensor(np_avg3_filter).type_as(ex).view(1, 1, 3, 3)
        if angle:
            double_angle = 90 if double else 180
            sin_ex = F.conv2d(torch.sin(ex * np.pi / double_angle), avg_filter, padding=1)
            cos_ex = F.conv2d(torch.cos(ex * np.pi / double_angle), avg_filter, padding=1)
            ex = torch.atan2(sin_angle, cos_angle) * double_angle / np.pi
        else:
            ex = F.conv2d(ex, avg_filter, padding=1)

    return ex


def anchors_fusion(anchor, types_wt, angle=False):
    view_shape = (anchor.size(0), -1) + (1,) * (anchor.ndim - 2)
    if angle:
        cos_2ori = (torch.cos(anchor * np.pi / 90) * types_wt.view(*view_shape)).sum(dim=1, keepdim=True)
        sin_2ori = (torch.sin(anchor * np.pi / 90) * types_wt.view(*view_shape)).sum(dim=1, keepdim=True)
        new_anchor = torch.atan2(sin_2ori, cos_2ori) * 90 / np.pi
    else:
        new_anchor = (anchor * types_wt.view(*view_shape)).sum(dim=1, keepdim=True)
    return new_anchor


def minus_orientation(ori, anchor):
    ori = ori - anchor
    ori = torch.where(ori >= 90, ori - 180, ori)
    ori = torch.where(ori < -90, ori + 180, ori)
    return ori


def binary_orientation(ori, ang_stride=2):
    gaussian_pdf = torch.tensor(signal.windows.gaussian(361, 5)).type_as(ori)
    new_shape = (1, -1,) + (1,) * (ori.ndim - 2)
    coord = torch.arange(ang_stride // 2, 180, ang_stride).view(*new_shape).type_as(ori) - 90
    delta = (ori - coord).abs()
    delta = torch.min(delta, 180 - delta) + 180  # [0,180)
    return gaussian_pdf[delta.long()]


def cosine_similarity(delta):
    delta = delta.abs()
    delta = torch.min(delta, 180 - delta)
    return torch.cos(delta * np.pi / 180)


def gabor_bank(enh_ksize=25, ori_stride=1, sigma=4, Lambda=9, psi=0, gamma=1):
    grid_theta, grid_x, grid_y = torch.meshgrid(
        torch.arange(0, 180, ori_stride), torch.arange(enh_ksize), torch.arange(enh_ksize)
    )
    grid_theta = -(grid_theta - 90) * np.pi / 180.0
    grid_x = grid_x - enh_ksize // 2
    grid_y = grid_y - enh_ksize // 2

    cos_theta = torch.cos(grid_theta)
    sin_theta = torch.sin(grid_theta)
    x_theta = grid_y * sin_theta + grid_x * cos_theta
    y_theta = grid_y * cos_theta - grid_x * sin_theta
    # gabor filters
    exp_fn = torch.exp(-0.5 * (x_theta ** 2 + gamma ** 2 * y_theta ** 2) / sigma ** 2)
    gb_cos = exp_fn * torch.cos(2 * np.pi * x_theta / Lambda + psi)
    gb_sin = exp_fn * torch.sin(2 * np.pi * x_theta / Lambda + psi)

    return gb_sin[:, None], gb_cos[:, None]


def gabor_fn(enh_ksize, sigma, theta, Lambda, psi, gamma):
    # indices
    half = enh_ksize // 2
    x, y = np.meshgrid(np.arange(enh_ksize) - half, np.arange(enh_ksize) - half, indexing="ij")
    # rotation
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_theta = x * cos_theta + y * sin_theta
    y_theta = -x * sin_theta + y * cos_theta
    # gabor filter
    exp_fn = np.exp(-0.5 * (x_theta ** 2 + gamma ** 2 * y_theta ** 2) / sigma ** 2)
    gb_cos = exp_fn * np.cos(2 * np.pi * x_theta / Lambda + psi)
    gb_sin = exp_fn * np.sin(2 * np.pi * x_theta / Lambda + psi)
    return gb_cos, gb_sin


def orientation_highest_peak(x, ang_stride=2):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    filter_weight = cycle_gaussian_weight(ang_stride=ang_stride).type_as(x)
    return F.conv2d(x, filter_weight, stride=1, padding=0)


def select_max_orientation(x):
    x = x / x.max(dim=1, keepdim=True).clamp_min(0.001)
    x = torch.where(x > 0.999, x, 0)
    x = x / x.sum(dim=1, keepdim=True).clamp_min(0.001)
    return x


def cycle_gaussian_weight(ang_stride=2, to_tensor=True):
    gaussian_pdf = signal.windows.gaussian(181, 3)
    coord = np.arange(ang_stride // 2, 180, ang_stride)
    delta = np.abs(coord.reshape(1, -1, 1, 1) - coord.reshape(-1, 1, 1, 1))
    delta = np.minimum(delta, 180 - delta) + 90
    if to_tensor:
        return torch.tensor(gaussian_pdf[delta]).float()
    else:
        return gaussian_pdf[delta].astype(np.float32)


def init_weights(net, init_type="normal"):
    if init_type == "normal":
        net.apply(weights_init_normal)
    elif init_type == "xavier":
        net.apply(weights_init_xavier)
    elif init_type == "kaiming":
        net.apply(weights_init_kaiming)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError("initialization method [%s] is not implemented" % init_type)


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Conv") != -1:
        nn.init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("Linear") != -1:
        nn.init.orthogonal_(m.weight.data, gain=1)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
