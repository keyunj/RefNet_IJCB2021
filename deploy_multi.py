"""
This file (test.py) is designed for:
    predict orientation and segmentation of fingerprint (mix of NIST14, which is synthesized as latent finger)
Copyright (c) 2020, Yongjie Duan. All rights reserved.
"""
import os
import os.path as osp
import numpy as np
from glob import glob
import imageio
import time
import json
import random
import argparse
import shutil
from scipy import signal
from collections import namedtuple
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn

from ops.losses import *
from ops.modules import expectation_pred
from dataset import rescale_trans
from settings import *
from model import generate_model, load_ckp
from utils import re_mkdir, mkdir, AverageMeter, zoom, zoom_orientation, draw_orientation


def run_on_image(img, base_angs, base_rmsds, pose, args, net):
    net.eval()

    img_shape = np.array(img.shape[:2])
    new_img_shape = img_shape // 8 * 8
    grid_x, grid_y = np.meshgrid(*[np.linspace(-1, 1, x) for x in new_img_shape[::-1] // 8])

    pose_trans, pose_theta = pose[:2], pose[2]
    pose_trans = rescale_trans(pose_trans, new_img_shape[::-1])
    pose_theta = np.where(pose_theta >= 180, pose_theta - 180, pose_theta)
    pose_theta = np.where(pose_theta < -180, pose_theta + 180, pose_theta)

    batch_sample = {}
    batch_sample["data"] = img[: new_img_shape[0], : new_img_shape[1]]
    batch_sample["data"] = torch.tensor(batch_sample["data"][None, None].astype(np.float32))
    batch_sample["grid"] = torch.tensor(np.stack((grid_x, grid_y), axis=0)[None].astype(np.float32))
    batch_sample["base_angs"] = torch.tensor(base_angs[None].astype(np.float32))
    batch_sample["base_rmsds"] = torch.tensor(base_rmsds[None].astype(np.float32))
    batch_sample["pose_trans"] = torch.tensor(pose_trans[None].astype(np.float32))
    batch_sample["pose_theta"] = torch.tensor(pose_theta[None, None].astype(np.float32))
    if args.use_cuda:
        batch_sample["data"] = batch_sample["data"].cuda()
        batch_sample["grid"] = batch_sample["grid"].cuda()
        batch_sample["base_angs"] = batch_sample["base_angs"].cuda()
        batch_sample["base_rmsds"] = batch_sample["base_rmsds"].cuda()
        batch_sample["pose_trans"] = batch_sample["pose_trans"].cuda()
        batch_sample["pose_theta"] = batch_sample["pose_theta"].cuda()

    # transform anchor
    tar_size = (1, 1, new_img_shape[0] // 8, new_img_shape[1] // 8)
    base_angs = transform_anchor(
        sample["base_angs"], sample["pose_trans"], sample["pose_theta"], tar_size=tar_size, angle=True
    )
    base_rmsds = transform_anchor(sample["base_rmsds"], sample["pose_trans"], sample["pose_theta"], tar_size=tar_size)

    with torch.no_grad():
        ori_pred, seg_pred, type_pred, _, = net(batch_sample)

    ori_angle = expectation_pred(ori_pred, args.ori_stride, angle=True, smooth=True, logits=False)

    if args.use_cuda:
        ori_angle = ori_angle.squeeze().cpu().numpy()
        seg_pred = seg_pred.squeeze().cpu().numpy()
        type_pred = type_pred.squeeze(0).cpu().numpy()
    else:
        ori_angle = ori_angle.squeeze().numpy()
        seg_pred = seg_pred.squeeze().numpy()
        type_pred = type_pred.squeeze(0).numpy()

    # build orientation and segmentation results (fused)
    seg_pred = seg_pred > 0.5
    type_pred = np.argmax(type_pred)

    if "NIST27" in args.deploy_set:
        ori_angle = zoom_orientation(ori_angle, scale=0.5)
        seg_pred = zoom(seg_pred, zoom=0.5, order=0)

    return ori_angle, seg_pred, type_pred


def deploy(args):
    # cuda
    net = generate_model(args)
    print("=> Total params: {:.2f}M".format(sum(p.numel() for p in net.parameters()) / 1.0e6))

    if args.use_cuda:
        net = torch.nn.DataParallel(net).cuda()
        cudnn.benchmark = True

    # load trained parameters
    checkpoint = load_ckp(args.logdir)
    if not checkpoint:
        print("[!] Failed to load checkpoint!")
    else:
        net.load_state_dict(checkpoint["state_dict"])

    total_time = AverageMeter()

    # output
    tar_draw_dir = osp.join(args.output, "draw")
    tar_ori_dir = osp.join(args.output, "ori")
    tar_seg_dir = osp.join(args.output, "seg")
    re_mkdir(args.output)
    mkdir(tar_draw_dir)
    mkdir(tar_ori_dir)
    mkdir(tar_seg_dir)

    # images name list
    print(f"dataset directory: {args.deploy_set}")
    img_path_lst = glob(osp.join(args.deploy_set, args.filter))
    img_path_lst.sort()

    # anchor
    base_angs = (
        np.stack(
            [
                np.asarray(
                    imageio.imread(osp.join("data", "anchors", f"{args.n_anchors}", f"meanfield{x}.png")), np.float32
                )
                for x in range(args.n_anchors)
            ]
        )
        - 90
    )
    base_rmsds = np.stack(
        [
            np.asarray(imageio.imread(osp.join("data", "anchors", f"{args.n_anchors}", f"rmsd{x}.png")), np.float32)
            for x in range(args.n_anchors)
        ]
    )

    gaussian_pdf = signal.windows.gaussian(361, 3)

    total_cnt = len(img_path_lst)
    if args.draw == "all":
        args.draw = total_cnt
    else:
        args.draw = int(args.draw)

    name_lst = []
    type_cls = []
    for data_idx, img_path in enumerate(img_path_lst):
        eval_name = osp.basename(img_path).split(".")[0]
        # load data
        img = np.asarray(imageio.imread(img_path), np.float32) / 255.0
        pose = np.loadtxt(osp.join(args.pose_set, f"{eval_name}.txt"), delimiter=",").astype(np.float32)

        end = time.time()
        ori_angle, seg_pred, type_pred = run_on_image(img, base_angs, base_rmsds, pose, args, net)
        eval_time = time.time() - end

        name_lst.append(eval_name)
        type_cls.append(type_pred)

        # save
        if data_idx < args.draw:
            factor = 16 if "NIST27" in args.deploy_set else 8
            # seg_true = imageio.imread(osp.join(args.deploy_set.replace("image", "seg"), f"{eval_name}.png")) > 0
            save_ori_on_img(img, ori_angle, seg_pred, osp.join(tar_draw_dir, f"{eval_name}.png"), factor)

        if args.mask_ori:
            ori_angle = np.where(seg_pred > 0, ori_angle, 91)
        imageio.imwrite(osp.join(tar_ori_dir, f"{eval_name}.png"), (ori_angle + 90).astype(np.uint8))
        imageio.imwrite(osp.join(tar_seg_dir, f"{eval_name}.png"), (seg_pred * 255).astype(np.uint8))

        total_time.update(eval_time)
        print(f"({data_idx+1}/{total_cnt}) name: {eval_name} time: {eval_time:.3f}s")

    print_str = f"=> {args.logdir}\t\ttime: {total_time.avg:.3f}\n"
    print(print_str)
    with open(osp.join(args.output, "summary.txt"), "a") as fp:
        fp.write(print_str)
    np.savetxt(
        osp.join(args.output, "type_cls.txt"),
        np.stack((np.array(name_lst), np.array(type_cls, np.int32)), axis=1),
        fmt="%s",
    )
    # save config
    with open(osp.join(args.output, "config.json"), "w") as fp:
        json.dump(args.__dict__, fp, indent=4)


def save_ori_on_img(img, ori, seg, fname, factor=8, stride=16):
    fig = plt.figure()
    plt.imshow(img, cmap="gray")
    draw_orientation(ori, seg, factor=factor, stride=stride)
    plt.axis([0, img.shape[1], img.shape[0], 0])
    plt.axis("off")
    plt.savefig(fname, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", default=[2], type=int, nargs="+", help="gpu device(s) used")
    parser.add_argument("--ckp", default="multiresanchor_mixhard_20201005", type=str)
    parser.add_argument("--pose-type", default="manual", type=str)
    parser.add_argument("--draw", default="16", type=str, help="numbers of draw orientation")
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--mask-ori", dest="mask_ori", action="store_true", default=False)
    args = parser.parse_args()

    logdir = args.ckp
    args.model_name = logdir.split("_")[0]
    args.dataset = logdir.split("_")[1]
    args.id = logdir.split("_")[2]
    args.phase = "test"
    args.logdir = osp.join("checkpoints", logdir)

    args.filter = "*.bmp"
    pose_type = args.pose_type
    args.deploy_set = "/home/dyj/disk1/data/finger/NIST27/search/image"
    args.pose_set = f"/home/dyj/disk1/data/finger/NIST27/search/pose/{pose_type}"
    args.output = osp.join(osp.dirname(args.deploy_set), "estimation", logdir, pose_type)

    load_config(osp.join(args.logdir, "config.json"), args)

    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id).strip("[]")

    deploy(args)
