"""
This file (display_with_orientation.py) is designed for:
    display orientation field with different indicating the error, it is for paper usage
Copyright (c) 2021, Yongjie Duan. All rights reserved.
"""
import os
import sys

# os.chdir(sys.path[0])
import os.path as osp
import numpy as np
from glob import glob
import argparse
import imageio
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

thresh = 15
# rgb = np.linspace(0, 1, thresh + 1)
# rgb = np.stack((rgb, 1 - rgb, np.zeros_like(rgb)), axis=1)
rgb = ((1, 0, 0), (0, 1, 0))


def draw_orientation(ori, mask=None, factor=8, stride=16):
    ori = ori * np.pi / 180
    for ii in range(stride // factor // 2, ori.shape[0], stride // factor):
        for jj in range(stride // factor // 2, ori.shape[1], stride // factor):
            if mask is not None and mask[ii, jj] == 0:
                continue
            x, y, o, r = jj, ii, ori[ii, jj], stride * 0.75
            plt.plot(
                [x * factor - 0.5 * r * np.cos(o), x * factor + 0.5 * r * np.cos(o)],
                [y * factor - 0.5 * r * np.sin(o), y * factor + 0.5 * r * np.sin(o)],
                "k-",
                linewidth=1.2,
            )


def display_with_orientation(plot_dir, img_name, img, seg, ori, factor=8, stride=16):
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    draw_orientation(ori, mask=seg, factor=factor, stride=stride)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(osp.join(plot_dir, f"{img_name}.png"), bbox_inches="tight")
    plt.close(fig)


def display_rmsd(plot_dir, img_name, img):
    cmap = plt.get_cmap()
    rgba_img = (cmap(img / np.max(img)) * 255).astype(np.uint8)
    rgba_img = zoom(rgba_img, (8, 8, 1), order=1)
    rgba_img[:, :, 3] = zoom(img > 0, 8, order=0) * 255

    fig = plt.figure(figsize=(5, 5), facecolor="white")
    plt.imshow(rgba_img)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(osp.join(plot_dir, f"{img_name}.png"), bbox_inches="tight")
    plt.close(fig)


def display_with_orientation_errors(
    plot_dir, img_name, img, seg, ori_true, ori_pred, thresh=15, factor=8, stride=16, vmin=None, vmax=None,
):
    error = np.abs(ori_pred - ori_true)
    error = np.rint(np.minimum(error, 180 - error)).astype(np.int)
    # error = np.clip(error, 0, thresh)
    error = np.where(error >= thresh, 0, 1)

    fig = plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    draw_orientation(ori, mask=seg, factor=factor, stride=stride, error=error)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(osp.join(plot_dir, f"{img_name}.png"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", "-n", default=5, type=int)
    args = parser.parse_args()

    prefix = "manual"

    for ii in range(args.num):
        ori = imageio.imread(osp.join(prefix, f"{args.num}", f"meanfield{ii}.png")).astype(np.float32) - 90
        seg = imageio.imread(osp.join(prefix, f"{args.num}", f"mask{ii}.png")) > 0
        rmsd = imageio.imread(osp.join(prefix, f"{args.num}", f"rmsd{ii}.png")).astype(np.float32)

        size = [8 * x for x in rmsd.shape]

        img = rmsd * seg
        display_rmsd(osp.join(prefix, f"{args.num}"), f"plot_rmsd{ii}", img)

        img = np.ones(size) * 255
        display_with_orientation(osp.join(prefix, f"{args.num}"), f"plot_meanfield{ii}", img, seg, ori, stride=24)
        print(f"=> plot {ii}")

