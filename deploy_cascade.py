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
from ops.modules import expectation_pred, transform_angle, EnhanceModule
from dataset import rescale_trans
from settings import *
from model import generate_model, load_ckp
from utils import re_mkdir, mkdir, AverageMeter, zoom, zoom_orientation, draw_orientation, draw_pose

all_metrics = namedtuple("all_metrics", ["angdiff", "segiou"])


def run_on_image(img, anchors_ang, anchors_rmsd, pose, args, net, enhance):
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
    batch_sample["anchors_ang"] = torch.tensor(anchors_ang[None, None].astype(np.float32))
    batch_sample["anchors_rmsd"] = torch.tensor(anchors_rmsd[None, None].astype(np.float32))
    batch_sample["pose_trans"] = torch.tensor(pose_trans[None].astype(np.float32))
    batch_sample["pose_theta"] = torch.tensor(pose_theta[None, None].astype(np.float32))
    if args.use_cuda:
        batch_sample["data"] = batch_sample["data"].cuda()
        batch_sample["anchors_ang"] = batch_sample["anchors_ang"].cuda()
        batch_sample["anchors_rmsd"] = batch_sample["anchors_rmsd"].cuda()
        batch_sample["pose_trans"] = batch_sample["pose_trans"].cuda()
        batch_sample["pose_theta"] = batch_sample["pose_theta"].cuda()

    with torch.no_grad():
        ori_pred, seg_pred, type_pred, anchors_ang = net(batch_sample)
        # enhancing
        ori_bin = transform_angle(ori_pred, anchors_ang, args.ori_stride)
        batch_sample["data"] = enhance(batch_sample["data"], ori_bin)
        ori_pred, _, _, anchors_ang = net(batch_sample)

    # update metric
    ori_angle = expectation_pred(ori_pred, args.ori_stride, angle=True, smooth=True, logits=False)
    ori_angle = ori_angle + anchors_ang
    ori_angle = torch.where(ori_angle >= 90, ori_angle - 180, ori_angle)
    ori_angle = torch.where(ori_angle < -90, ori_angle + 180, ori_angle)

    if args.use_cuda:
        ori_angle = ori_angle.squeeze().cpu().numpy()
        seg_pred = seg_pred.squeeze().cpu().numpy()
        type_pred = type_pred.squeeze().cpu().numpy()
    else:
        ori_angle = ori_angle.squeeze().numpy()
        seg_pred = seg_pred.squeeze().numpy()
        type_pred = type_pred.squeeze().numpy()

    return ori_angle, seg_pred, type_pred


def deploy(args):
    if args.use_cuda:
        cudnn.benchmark = True

    nets = {}
    # cuda
    for ii in range(args.n_anchors):
        net = generate_model(args)
        print("=> No.{:d} network Total params: {:.2f}M".format(ii, sum(p.numel() for p in net.parameters()) / 1.0e6))

        if args.use_cuda:
            net = torch.nn.DataParallel(net).cuda()

        # load trained parameters
        checkpoint = load_ckp(osp.join(args.logdir, f"{ii}"))
        if not checkpoint:
            print("[!] Failed to load checkpoint!")
        else:
            net.load_state_dict(checkpoint["state_dict"])

        nets[ii] = net

    enhance = EnhanceModule(8, ori_stride=args.ori_stride)
    if args.use_cuda:
        enhance = torch.nn.DataParallel(enhance).cuda()

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
    anchors_ang = (
        np.stack(
            [
                imageio.imread(
                    osp.join("data", "anchors", args.pose_type, f"{args.n_anchors}", f"meanfield{x}.png")
                ).astype(np.float32)
                for x in range(args.n_anchors)
            ]
        )
        - 90
    )
    anchors_rmsd = np.stack(
        [
            imageio.imread(osp.join("data", "anchors", args.pose_type, f"{args.n_anchors}", f"rmsd{x}.png")).astype(
                np.float32
            )
            for x in range(args.n_anchors)
        ]
    )

    total_cnt = len(img_path_lst)
    if args.draw == "all":
        args.draw = total_cnt
    else:
        args.draw = int(args.draw)
    for data_idx, img_path in enumerate(img_path_lst):
        eval_name = osp.basename(img_path).split(".")[0]
        # load data
        img = np.asarray(imageio.imread(img_path), np.float32) / 255.0
        pose = np.loadtxt(osp.join(args.pose_set, f"{eval_name}.txt"), delimiter=",").astype(np.float32)

        end = time.time()
        ori_lst = []
        seg_lst = []
        type_lst = []
        type_thresh = 0
        for ii in range(args.n_anchors):
            ori_angle, seg_pred, type_pred = run_on_image(
                img, anchors_ang[ii], anchors_rmsd[ii], pose, args, nets[ii], enhance
            )
            ori_lst.append(ori_angle)
            seg_lst.append(seg_pred)
            type_lst.append(type_pred)

        ori_angle, seg_pred = merging_results(type_lst, ori_lst, seg_lst, mode="mean")
        eval_time = time.time() - end

        # build orientation and segmentation results (fused)
        seg_pred = seg_pred > 0.5
        if "NIST27" in args.deploy_set:
            ori_angle = zoom_orientation(ori_angle, scale=0.5)
            seg_pred = zoom(seg_pred, zoom=0.5, order=0)

        # save
        if data_idx < args.draw:
            factor = 16 if "NIST27" in args.deploy_set else 8
            save_ori_on_img(
                img, ori_angle, seg_pred, pose[:2], pose[2], osp.join(tar_draw_dir, f"{eval_name}.png"), factor
            )

        imageio.imwrite(osp.join(tar_ori_dir, f"{eval_name}.png"), (ori_angle + 90).astype(np.uint8))
        imageio.imwrite(osp.join(tar_seg_dir, f"{eval_name}.png"), (seg_pred * 255).astype(np.uint8))

        total_time.update(eval_time)
        print(f"({data_idx+1}/{total_cnt}) name: {eval_name} time: {eval_time:.3f}s")

    print_str = f"=> {args.logdir}\t\ttime: {total_time.avg:.3f}\n"
    print(print_str)
    with open(osp.join(args.output, "summary.txt"), "a") as fp:
        fp.write(print_str)
    shutil.copy(osp.join(osp.join(args.logdir, "0", "config.json")), osp.join(args.output, ""))


def merging_results(type_lst, ori_lst, seg_lst, mode="max"):
    type_lst = np.stack(type_lst)
    ori_lst = np.stack(ori_lst)
    seg_lst = np.stack(seg_lst)

    if mode == "max":
        argmax = np.argmax(type_lst, axis=0)
        return ori_lst[argmax], seg_lst[argmax]
    elif mode == "mean":
        weight = type_lst / type_lst.sum(axis=0, keepdims=True).clip(0.001, None)
        return ori_average(ori_lst, weight), (seg_lst * type_lst.reshape(-1, 1, 1)).sum(axis=0)
    else:
        raise ValueError("Not supported merging mode")


def ori_average(ori_lst, weight):
    sin2_ori = (np.sin(ori_lst * np.pi / 90) * weight.reshape(-1, 1, 1)).sum(axis=0)
    cos2_ori = (np.cos(ori_lst * np.pi / 90) * weight.reshape(-1, 1, 1)).sum(axis=0)
    ori_mean = np.arctan2(sin2_ori, cos2_ori) * 90 / np.pi
    return ori_mean


def save_ori_on_img(img, ori, seg, trans, theta, fname, factor=8, stride=16):
    fig = plt.figure()
    plt.imshow(img, cmap="gray")
    draw_orientation(ori, seg, factor=factor, stride=stride)
    draw_pose(trans, theta)
    plt.axis([0, img.shape[1], img.shape[0], 0])
    plt.axis("off")
    plt.savefig(fname, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckp", default="singleresanchor_mixhard_20200901", type=str)
    parser.add_argument("--pose-dir", default="yin", type=str)
    parser.add_argument("--gpu-id", default=[2], type=int, nargs="+", help="gpu device(s) used")
    parser.add_argument("--draw", default="16", type=str, help="numbers of draw orientation")
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false", default=True)
    args = parser.parse_args()

    logdir = args.ckp
    args.model_name = logdir.split("_")[0]
    args.dataset = logdir.split("_")[1]
    args.id = logdir.split("_")[2]
    args.phase = "test"
    args.pose_type = "manual"

    args.logdir = osp.join("checkpoints", logdir)

    # args.filter = "*.bmp"
    # args.deploy_set = "/home/dyj/disk1/data/finger/NIST27/search/image"
    # args.pose_set = f"/home/dyj/disk1/data/finger/NIST27/search/pose/{args.pose_dir}"
    # args.filter = "*.png"
    # args.deploy_set = "/home/dyj/disk1/data/finger/NIST14/image_mixhard"
    # args.pose_set = f"/home/dyj/disk1/data/finger/NIST14/pose/{args.pose_dir}"
    # args.filter = "*.bmp"
    # args.deploy_set = "/home/dyj/disk1/data/finger/OldFpTHU/image"
    # args.pose_set = f"/home/dyj/disk1/data/finger/OldFpTHU/pose/{args.pose_dir}"
    args.filter = "*.bmp"
    args.deploy_set = "/home/dyj/disk1/data/finger/Hisign/latent/image"
    args.pose_set = f"/home/dyj/disk1/data/finger/Hisign/latent/pose/{args.pose_dir}"

    args.output = osp.join(
        osp.dirname(args.deploy_set), "estimation", f"{args.model_name}cascade_{args.dataset}_{args.id}", args.pose_dir
    )

    load_config(osp.join(args.logdir, "0", "config.json"), args)

    # random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id).strip("[]")

    deploy(args)
