import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CELoss(nn.Module):
    def __init__(self, use_cuda=False, eps=1e-6):
        super(CELoss, self).__init__()
        self.eps = eps

    def forward(self, log_pt, mask=None):
        loss = -log_pt
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp_min(self.eps)
        else:
            loss = loss.mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, use_cuda=False, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.eps = eps
        self.gamma = gamma

    def forward(self, log_pt, pt, mask=None, weight=None):
        loss = -torch.pow(1 - pt, self.gamma) * log_pt
        if weight is not None:
            loss = loss * weight
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum().clamp_min(self.eps)
        else:
            loss = loss.mean()
        return loss


class SmoothLoss(nn.Module):
    def __init__(self, use_cuda=False):
        super(SmoothLoss, self).__init__()
        self.smooth_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).view(1, 1, 3, 3) / 8.0
        if use_cuda:
            self.smooth_kernel = self.smooth_kernel.cuda()

    def forward(self, input):
        # smoothness
        loss = F.conv2d(input, self.smooth_kernel.type_as(input), padding=1).abs().mean()
        return loss


class CoherenceLoss(nn.Module):
    def __init__(self, ang_stride=2, use_cuda=False, eps=1e-6):
        super(CoherenceLoss, self).__init__()
        self.eps = eps
        ang_kernel = torch.arange(ang_stride / 2, 180, ang_stride).view(1, -1, 1, 1) / 90.0 * torch.tensor(np.pi)
        self.cos2angle = torch.cos(ang_kernel)
        self.sin2angle = torch.sin(ang_kernel)
        self.coh_kernel = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).view(1, 1, 3, 3) / 9.0
        if use_cuda:
            self.cos2angle = self.cos2angle.cuda()
            self.sin2angle = self.sin2angle.cuda()
            self.coh_kernel = self.coh_kernel.cuda()

    def ori2angle(self, ori):
        cos2angle_ori = (ori * self.cos2angle).sum(dim=1, keepdim=True)
        sin2angle_ori = (ori * self.sin2angle).sum(dim=1, keepdim=True)
        modulus_ori = (sin2angle_ori ** 2 + cos2angle_ori ** 2).sqrt()
        return sin2angle_ori, cos2angle_ori, modulus_ori

    def forward(self, pt, mask=None):
        sin2angle_ori, cos2angle_ori, modulus_ori = self.ori2angle(pt)
        cos2angle = F.conv2d(cos2angle_ori, self.coh_kernel, padding=1)
        sin2angle = F.conv2d(sin2angle_ori, self.coh_kernel, padding=1)
        modulus = F.conv2d(modulus_ori, self.coh_kernel, padding=1)
        coherence = (sin2angle ** 2 + cos2angle ** 2).sqrt() / modulus.clamp_min(self.eps)
        if mask is not None:
            loss = mask.sum() / (coherence * mask).sum().clamp_min(self.eps) - 1
        else:
            loss = 1 / coherence.mean().clamp_min(self.eps) - 1
        return loss


class WeightedSegLoss(nn.Module):
    def __init__(self, smooth=1.0, use_cuda=False, eps=1e-6):
        super(WeightedSegLoss, self).__init__()
        self.eps = eps
        self.smooth = smooth
        self.loss_bce = CELoss(eps=eps, use_cuda=False)
        self.loss_smooth = SmoothLoss(use_cuda=use_cuda)

    def forward(self, input, target, logits=True):
        if logits:
            p = torch.sigmoid(input)
        else:
            p = input.clamp(self.eps, 1.0 - self.eps)
        # weighted CE
        N_total = target.numel()
        N_pos = target.sum()
        w_pos = 0.5 * N_total / N_pos if N_pos > 0 else 0.5
        w_neg = 1 / (2 - 1 / w_pos)
        log_pt = w_pos * target * torch.log(p) + w_neg * (1.0 - target) * torch.log(1.0 - p)
        loss_bce = self.loss_bce(log_pt)
        # smooth loss
        loss_smooth = self.loss_smooth(input)
        return loss_bce + self.smooth * loss_smooth


class FocalSegLoss(nn.Module):
    def __init__(self, gamma=2.0, smooth=1.0, use_cuda=False, eps=1e-6):
        super(FocalSegLoss, self).__init__()
        self.eps = eps
        self.gamma = gamma
        self.smooth = smooth
        self.loss_bce = FocalLoss(eps=eps, gamma=gamma, use_cuda=use_cuda)
        self.loss_smooth = SmoothLoss(use_cuda=use_cuda)

    def forward(self, input, target, logits=True):
        if logits:
            p = torch.sigmoid(input)
        else:
            p = input.clamp(self.eps, 1.0 - self.eps)

        pt = target * p + (1.0 - target) * (1.0 - p)
        log_pt = target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p)
        # focal loss
        loss_bce = self.loss_bce(log_pt, pt)
        # smooth loss
        loss_smooth = self.loss_smooth(input)
        return loss_bce + self.smooth * loss_smooth


class WeightedOriLoss(nn.Module):
    def __init__(self, ang_stride=2, coherence=1.0, use_cuda=False, eps=1e-6):
        super(WeightedOriLoss, self).__init__()
        self.eps = eps
        self.coherence = coherence
        self.loss_bce = CELoss(eps=eps, use_cuda=use_cuda)
        self.loss_coh = CoherenceLoss(ang_stride=ang_stride, use_cuda=use_cuda)

    def forward(self, input, target, mask=None, logits=True):
        if logits:
            p = torch.softmax(input, dim=1)
        else:
            p = input.clamp(self.eps, 1.0 - self.eps)

        # weighted CE
        log_pt = (target * torch.log(p)).sum(dim=1, keepdim=True)
        loss_bce = self.loss_bce(log_pt, mask=mask)
        # coherence loss, nearby ori should be as near as possible
        loss_coh = self.loss_coh(p, mask)
        return loss_bce + self.coherence * loss_coh


class FocalOriLoss(nn.Module):
    def __init__(self, ang_stride=2, gamma=2, coherence=1.0, use_cuda=False, eps=1e-6):
        super(FocalOriLoss, self).__init__()
        self.eps = eps
        self.gamma = gamma
        self.coherence = coherence
        self.loss_bce = FocalLoss(eps=eps, gamma=gamma, use_cuda=use_cuda)
        self.loss_coh = CoherenceLoss(ang_stride=ang_stride, use_cuda=use_cuda)

    def forward(self, input, target, mask=None, weight=None, logits=True):
        if logits:
            p = torch.softmax(input, dim=1)
        else:
            p = input.clamp(self.eps, 1.0 - self.eps)

        pt = (target * p).sum(dim=1, keepdim=True)
        log_pt = (target * torch.log(p)).sum(dim=1, keepdim=True)

        # focal loss
        loss_bce = self.loss_bce(log_pt, pt, mask=mask, weight=weight)
        # coherence loss, nearby ori should be as near as possible
        loss_coh = self.loss_coh(p, mask=mask)
        return loss_bce + self.coherence * loss_coh


class FocalTypeLoss(nn.Module):
    def __init__(self, gamma=2.0, use_cuda=False, eps=1e-6):
        super(FocalTypeLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.loss_bce = FocalLoss(eps=eps, gamma=gamma, use_cuda=use_cuda)

    def forward(self, input, target, logits=True):
        if logits:
            p = torch.softmax(input, dim=1)
        else:
            p = input.clamp(self.eps, 1.0 - self.eps)

        onehot = torch.zeros_like(input).scatter(1, target[:, None], 1)
        pt = (onehot * p).sum(dim=1, keepdim=True)
        log_pt = (onehot * torch.log(p)).sum(dim=1, keepdim=True)
        loss = self.loss_bce(log_pt, pt)
        return loss


class MSELoss(nn.Module):
    def __init__(self, use_cuda=False, eps=1e-6):
        super(MSELoss, self).__init__()
        self.eps = eps

    def forward(self, input, target, mask=None):
        delta = (input - target) ** 2
        if mask is not None:
            delta = (delta * mask).sum(dim=(-1, -2)) / mask.sum(dim=(-1, -2)).clamp_min(self.eps)
        loss = delta.mean()
        return loss


class FocalBCELoss(nn.Module):
    def __init__(self, gamma=2.0, use_cuda=False, eps=1e-6):
        super(FocalBCELoss, self).__init__()
        self.eps = eps
        self.loss_bce = FocalLoss(gamma=gamma, use_cuda=use_cuda, eps=eps)

    def forward(self, input, target, logits=True):
        if logits:
            p = torch.sigmoid(input)
        else:
            p = input.clamp(self.eps, 1.0 - self.eps)

        pt = target * p + (1.0 - target) * (1.0 - p)
        log_pt = target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p)

        loss_bce = self.loss_bce(log_pt, pt)
        return loss_bce
