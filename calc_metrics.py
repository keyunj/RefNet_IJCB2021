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

from utils import Logger, AverageMeter, calc_ori_delta, calc_seg_iou


def calc_metrics(prefix, res_dir, ori_ck, mask):
    data_name_lst = glob(osp.join(res_dir, ori_ck, "*.png"))
    data_name_lst = [osp.basename(x).strip(".png") for x in data_name_lst]
    data_name_lst.sort()

    angdiffs = AverageMeter()
    segious = AverageMeter()
    logger = Logger(osp.join(res_dir, "0-results.txt"))
    logger.set_names(["Name", "Angdiff", "SegIoU"])

    for idx, data_name in enumerate(data_name_lst):
        # print(data_name)
        ori_true = imageio.imread(osp.join(prefix, "ori", f"{data_name}.png")).astype(np.float32)
        seg_true = imageio.imread(osp.join(prefix, "seg", f"{data_name}.png")) > 0

        ori_pred = imageio.imread(osp.join(res_dir, ori_ck, f"{data_name}.png")).astype(np.float32)
        seg_pred = imageio.imread(osp.join(res_dir, "seg", f"{data_name}.png")) > 0

        if mask == "local":
            seg_local = imageio.imread(osp.join(res_dir.replace(args.ckp, "localdict"), "seg", f"{data_name}.png")) > 0
            angdiffs.update(calc_ori_delta(ori_pred, ori_true, mask=seg_true * seg_manual))
        elif mask == "ex":
            seg_ex = imageio.imread(osp.join(res_dir.replace(args.ckp, "exsearch"), "seg", f"{data_name}.png")) > 0
            angdiffs.update(calc_ori_delta(ori_pred, ori_true, mask=seg_true * seg_ex))
        elif mask == "auto":
            angdiffs.update(calc_ori_delta(ori_pred, ori_true, mask=seg_true * seg_pred))
        else:
            angdiffs.update(calc_ori_delta(ori_pred, ori_true, mask=seg_true))
        segious.update(calc_seg_iou(seg_pred, seg_true))

        logger.append([data_name, angdiffs.val, segious.val])

    logger.close()

    angdiff = np.asarray(logger.numbers["Angdiff"])
    name_lst = np.array([int(x) for x in data_name_lst])
    good_rmsd = angdiff[name_lst <= 88].mean()
    bad_rmsd = angdiff[(name_lst > 88) & (name_lst <= 173)].mean()
    ugly_rmsd = angdiff[name_lst > 173].mean()

    print_str = [f"evaluation(mean+/-std) mask-{mask} ori-{ori_ck}:"]
    print_str.append(f"angle difference: {angdiffs.avg:.3f} +/- {angdiffs.std:.3f},")
    print_str.append(f"seg iou: {segious.avg:.3f} +/- {segious.std:.3f},")
    print_str.append(f"good: {good_rmsd:.3f}, bad: {bad_rmsd:.3f}, ugly: {ugly_rmsd:.3f}")
    print("\n".join(print_str))
    with open(osp.join(res_dir, "summary.txt"), "a") as fp:
        fp.write(" ".join(print_str))
        fp.write("\n-------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckp", default="fuseoffset_mixhard_20200711/manual", type=str)
    parser.add_argument("--mask", default=None, type=str)
    parser.add_argument("--smooth", dest="smooth", action="store_true", default=False)
    args = parser.parse_args()

    prefix = "/home/dyj/disk1/data/finger/NIST27/search"
    res_dir = osp.join(prefix, "estimation", args.ckp)

    ori_ck = "ori_smooth" if args.smooth else "ori"

    calc_metrics(prefix, res_dir, ori_ck, args.mask)
