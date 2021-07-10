from __future__ import print_function, absolute_import
import torch
import numpy as np

from .misc import AverageMeter


def calc_ori_delta(input, target, mask=None):
    eps = 1e-3
    if torch.is_tensor(input):
        input_var = input.to(torch.float32)
        diff = (input_var - target).abs()
        diff = torch.min(diff, 180.0 - diff)
    else:
        input_var = input.astype(np.float32)
        diff = np.abs(input_var - target)
        diff = np.minimum(diff, 180.0 - diff)

    if mask is not None:
        if mask.sum() == 0:
            return None
        diff = (diff ** 2 * mask).sum() / (mask.sum() + eps)
    else:
        diff = (diff ** 2).mean()

    if torch.is_tensor(diff):
        return diff.sqrt()
    else:
        return np.sqrt(diff)


def get_bound_around_minutia(center, psize, ori_shape):
    lower = center - psize // 2
    higher = center + psize - psize // 2
    lower = np.where(lower < 0, 0, lower)
    higher = np.where(higher > ori_shape, ori_shape, higher)
    return lower.astype(np.int), higher.astype(np.int)


def calc_angle_difference_around_minutia(input, target, mnt, mask=None, factor=8, psize=5):
    angdiffs = AverageMeter()
    for ii in range(len(mnt)):
        center = np.rint(mnt[ii][:2] * 1.0 / factor)
        ori_shape = np.array(input.shape)
        lower, higher = get_bound_around_minutia(center, psize, ori_shape)
        if mask is not None:
            angdiffs.update(
                calc_ori_delta(
                    input[..., lower[0] : higher[0], lower[1] : higher[1]],
                    target[..., lower[0] : higher[0], lower[1] : higher[1]],
                    mask=mask[..., lower[0] : higher[0], lower[1] : higher[1]],
                )
            )
        else:
            angdiffs.update(
                calc_ori_delta(
                    input[..., lower[0] : higher[0], lower[1] : higher[1]],
                    target[..., lower[0] : higher[0], lower[1] : higher[1]],
                )
            )

    return angdiffs.avg, len(mnt)


def calc_seg_iou(input, target):
    smooth = 1e-3
    if torch.is_tensor(input):
        input_var = input.to(torch.float32)
    else:
        input_var = input.astype(np.float32)
    target_cal = 1.0 * (target > 0.5)
    input_cal = 1.0 * (input_var > 0.5)
    if target_cal.sum() == 0:
        return None

    intersect = input_cal * target_cal
    iou = (intersect.sum() + smooth) / (input_cal.sum() + target_cal.sum() - intersect.sum() + smooth)
    return iou


def calc_qua_difference(input, target, mask=None):
    eps = 1e-3
    if torch.is_tensor(input):
        input_var = input.to(torch.float32)
        diff = (input_var - target).abs()
    else:
        input_var = input.astype(np.float32)
        diff = np.abs(input_var - target)

    if mask is not None:
        if mask.sum() == 0:
            return None
        diff = (diff * mask).sum() / (mask.sum() + eps)
    else:
        diff = diff.mean()

    return diff


def cal_all_metric(input, target):
    """
    calculate all metric used in segmentation task
    """
    smooth = 1
    if torch.is_tensor(input):
        input_var = input.to(torch.float32)
    else:
        input_var = input.astype(np.float32)
    intersection_sum = (input_var * target).sum()
    input_sum = input_var.sum()
    target_sum = target.sum()
    dice = (2 * intersection_sum + smooth) / (input_sum + target_sum + smooth)
    precision = (intersection_sum + smooth) / (input_sum + smooth)
    recall = (intersection_sum + smooth) / (target_sum + smooth)
    return (dice, precision, recall)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
