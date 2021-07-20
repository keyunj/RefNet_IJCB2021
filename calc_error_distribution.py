"""
This file (calc_error_distribution.py) is designed for:
    calculate distribution of angle error
Copyright (c) 2020, Yongjie Duan. All rights reserved.
"""
import os
import os.path as osp
import numpy as np
from glob import glob
import cv2
import argparse
import imageio
from scipy.ndimage import map_coordinates, zoom
import matplotlib.pyplot as plt


def transform_latent(arr, pose, factor=8, order=3, cval=0, angle=False):
    tar_shape = np.array(arr.shape[-2:])

    x, y, theta = pose
    src_center = np.array([y, x], np.float32) / factor
    tar_center = tar_shape // 2
    sin_theta = np.sin(theta * np.pi / 180)
    cos_theta = np.cos(theta * np.pi / 180)
    mat = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    indices = np.stack(np.meshgrid(*[np.arange(x) for x in arr.shape], indexing="ij")).astype(np.float32)
    coord_indices = indices[-2:].reshape(2, -1)
    coord_indices = np.dot(mat, coord_indices - tar_center[:, None]) + src_center[:, None]
    indices[-2:] = coord_indices.reshape(2, *tar_shape)
    if angle:
        new_arr = arr + theta
        cos_2angle = map_coordinates(np.cos(2 * new_arr * np.pi / 180), indices, cval=cval, mode="constant")
        sin_2angle = map_coordinates(np.sin(2 * new_arr * np.pi / 180), indices, cval=cval, mode="constant")
        new_arr = np.arctan2(sin_2angle, cos_2angle) * 180 / np.pi / 2
    else:
        new_arr = map_coordinates(arr, indices, order=order, cval=cval)
    return new_arr


def calc_error_distribution(prefix, res_dir, factor=8, filters="*.png"):
    name_lst = glob(osp.join(res_dir, "ori", filters))
    name_lst = [osp.basename(x).strip(".png") for x in name_lst]
    name_lst.sort()
    if len(name_lst) > 8100:
        name_lst = name_lst[8100:8100+2000]

    err_arr = 0
    std_arr = 0
    cnt_arr = 0
    for img_idx, img_name in enumerate(name_lst):
        pose = np.loadtxt(osp.join(prefix, "pose", "manual", f"{img_name}.txt"), delimiter=",").astype(np.float32)
        ori_true = imageio.imread(osp.join(prefix, "ori", f"{img_name}.png")).astype(np.float32) - 90
        seg_true = imageio.imread(osp.join(prefix, "seg", f"{img_name}.png")) > 0
        ori_pred = imageio.imread(osp.join(res_dir, "ori", f"{img_name}.png")).astype(np.float32) - 90

        cur_err = np.abs(
            transform_latent(ori_true, pose, factor=factor, angle=True)
            - transform_latent(ori_pred, pose, factor=factor, angle=True)
        )
        cur_err = np.minimum(cur_err, 180 - cur_err)
        cur_cnt = transform_latent(seg_true, pose, factor=factor, order=0)

        err_arr = (cnt_arr * err_arr + cur_cnt * cur_err) / (cnt_arr + cur_cnt).clip(1, None)
        std_arr = (cnt_arr * std_arr + cur_cnt * cur_err ** 2) / (cnt_arr + cur_cnt).clip(1, None)
        cnt_arr = cnt_arr + cur_cnt

        print(f"=> {img_name} done.")

    # err_arr = np.sqrt(err_arr)
    std_arr = np.sqrt((std_arr - err_arr ** 2).clip(0, None))

    # save
    min_hw = min(err_arr.shape)
    err_arr = zoom(err_arr, 256.0 / min_hw)
    std_arr = zoom(std_arr, 256.0 / min_hw)
    cnt_arr = zoom(cnt_arr, 256.0 / min_hw)
    print(err_arr.max())
    print(std_arr.max())

    err_arr = np.where(cnt_arr > (0.1 * len(name_lst)), err_arr, 0)
    std_arr = np.where(cnt_arr > (0.1 * len(name_lst)), std_arr, 0)

    imageio.imwrite(osp.join(res_dir, "error_distribution.png"), np.rint(err_arr).astype(np.uint8))
    imageio.imwrite(osp.join(res_dir, "error_distribution_std.png"), np.rint(err_arr).astype(np.uint8))
    imageio.imwrite(osp.join(res_dir, "error_foreground.png"), np.rint(cnt_arr / cnt_arr.max() * 255).astype(np.uint8))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax0 = axes[0].imshow(err_arr, vmin=0, vmax=25)
    ax1 = axes[1].imshow(std_arr, vmin=0, vmax=25)
    axes[0].set_title("error mean")
    axes[1].set_title("error standard")
    axes[0].axis("off")
    axes[1].axis("off")
    plt.colorbar(ax1, ax=[axes[0], axes[1]])
    plt.savefig(osp.join(res_dir, "error_distribution_plot.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckp", default="singleresanchor_mixhard_20200901/manual", type=str)
    args = parser.parse_args()

    factor = 16
    prefix = "/home/dyj/disk1/data/finger/NIST27/search"
    filters = "*.png"

    # factor = 8
    # prefix = "/home/dyj/disk1/data/finger/NIST14"
    # filters = "F*.png"

    res_dir = osp.join(prefix, "estimation", args.ckp)

    calc_error_distribution(prefix, res_dir, factor=factor, filters=filters)
