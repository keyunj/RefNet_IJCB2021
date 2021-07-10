"""
This file (calc_metrics.py) is designed for:
    calculate metrics including orientation difference and segmentation overlap
Copyright (c) 2020, Yongjie Duan. All rights reserved.
"""
import os
import os.path as osp
import numpy as np
from glob import glob
import argparse
import imageio

from utils import Logger, AverageMeter, calc_seg_iou, calc_ori_delta, calc_angle_difference_around_minutia


def calc_metrics(prefix, res_dir, factor, mask):
    data_name_lst = glob(osp.join(res_dir, "ori", "*.png"))
    data_name_lst = [osp.basename(x).strip(".png") for x in data_name_lst]
    data_name_lst.sort()

    angdiffs = [AverageMeter() for _ in range(2)]
    segious = AverageMeter()

    logger = Logger(osp.join(res_dir, "0-results.txt"))
    logger.set_names(["Name", "SegIoU", "Angdiff_all", "Angdiff_mnt", "Num_mnt"])

    for idx, data_name in enumerate(data_name_lst):
        # print(data_name)
        ori_true = imageio.imread(osp.join(prefix, "ori", f"{data_name}.png")).astype(np.float32)
        seg_true = imageio.imread(osp.join(prefix, "seg", f"{data_name}.png")) > 0
        mnt = np.loadtxt(osp.join(prefix, "minutia", f"{data_name}.txt"))

        ori_pred = imageio.imread(osp.join(res_dir, "ori", f"{data_name}.png")).astype(np.float32)
        seg_pred = imageio.imread(osp.join(res_dir, "seg", f"{data_name}.png")) > 0

        if mask == "local":
            seg_local = imageio.imread(osp.join(res_dir.replace(args.ckp, "localdict"), "seg", f"{data_name}.png")) > 0
            angdiff_all = calc_ori_delta(ori_pred, ori_true, mask=seg_true * seg_local)
            angdiff_mnt, num_mnt = calc_angle_difference_around_minutia(
                ori_pred, ori_true, mnt, mask=seg_true * seg_local, factor=factor
            )
            angdiffs[0].update(angdiff_all)
            angdiffs[1].update(angdiff_mnt)
        elif mask == "ex":
            seg_ex = imageio.imread(osp.join(res_dir.replace(args.ckp, "exsearch"), "seg", f"{data_name}.png")) > 0
            angdiff_all = calc_ori_delta(ori_pred, ori_true, mask=seg_true * seg_ex)
            angdiff_mnt, num_mnt = calc_angle_difference_around_minutia(
                ori_pred, ori_true, mnt, mask=seg_true * seg_ex, factor=factor
            )
            angdiffs[0].update(angdiff_all)
            angdiffs[1].update(angdiff_mnt)
        elif mask == "auto":
            angdiff_all = calc_ori_delta(ori_pred, ori_true, mask=seg_true * seg_pred)
            angdiff_mnt, num_mnt = calc_angle_difference_around_minutia(
                ori_pred, ori_true, mnt, mask=seg_true * seg_pred, factor=factor
            )
            angdiffs[0].update(angdiff_all)
            angdiffs[1].update(angdiff_mnt)
        else:
            angdiff_all = calc_ori_delta(ori_pred, ori_true, mask=seg_true)
            angdiff_mnt, num_mnt = calc_angle_difference_around_minutia(
                ori_pred, ori_true, mnt, mask=seg_true, factor=factor
            )
            angdiffs[0].update(angdiff_all)
            angdiffs[1].update(angdiff_mnt)
        segious.update(calc_seg_iou(seg_pred, seg_true))

        logger.append([data_name, segious.val, angdiffs[0].val, angdiffs[1].val, num_mnt])

    logger.close()

    name_lst = np.array([int(x) for x in data_name_lst])

    angdiff_all = np.asarray(logger.numbers["Angdiff_all"])
    good_rmsd_all = angdiff_all[name_lst <= 88].mean()
    bad_rmsd_all = angdiff_all[(name_lst > 88) & (name_lst <= 173)].mean()
    ugly_rmsd_all = angdiff_all[name_lst > 173].mean()

    angdiff_mnt = np.asarray(logger.numbers["Angdiff_mnt"])
    num_mnt = np.asarray(logger.numbers["Num_mnt"])
    good_rmsd_mnt = (angdiff_mnt[name_lst <= 88] * num_mnt[name_lst <= 88]).sum() / num_mnt[name_lst <= 88].sum()
    bad_rmsd_mnt = (
        angdiff_mnt[(name_lst > 88) & (name_lst <= 173)] * num_mnt[(name_lst > 88) & (name_lst <= 173)]
    ).sum() / num_mnt[(name_lst > 88) & (name_lst <= 173)].sum()
    ugly_rmsd_mnt = (angdiff_mnt[name_lst > 173] * num_mnt[name_lst > 173]).sum() / num_mnt[name_lst > 173].sum()

    print_str = [f"evaluation(mean+/-std) mask-{mask}:\n"]
    print_str.append(f"seg iou: {segious.avg:.3f} +/- {segious.std:.3f}\n")
    print_str.append(f"for the whole region:\n")
    print_str.append(f"  angle difference for all region: {angdiffs[0].avg:.3f} +/- {angdiffs[0].std:.3f}\n")
    print_str.append(f"  good: {good_rmsd_all:.3f}, bad: {bad_rmsd_all:.3f}, ugly: {ugly_rmsd_all:.3f}\n")
    print_str.append(f"for regions around minutia:\n")
    print_str.append(f"  angle difference for minutia region: {angdiffs[1].avg:.3f} +/- {angdiffs[1].std:.3f}\n")
    print_str.append(f"  good: {good_rmsd_mnt:.3f}, bad: {bad_rmsd_mnt:.3f}, ugly: {ugly_rmsd_mnt:.3f}\n")

    print(" ".join(print_str))
    with open(osp.join(res_dir, "summary.txt"), "a") as fp:
        fp.write("-------------------------\n")
        fp.write(" ".join(print_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckp", default="singleresanchor_mixhard_20200901/manual", type=str)
    parser.add_argument("--mask", default=None, type=str)
    args = parser.parse_args()

    prefix = "/home/dyj/disk1/data/finger/NIST27/search"
    res_dir = osp.join(prefix, "estimation", args.ckp)
    factor = 16

    calc_metrics(prefix, res_dir, factor, args.mask)
