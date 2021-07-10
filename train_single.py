import os
import time
import random
import numpy as np
import imageio

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import dataset
from ops.losses import *
from ops.modules import minus_orientation, binary_orientation, expectation_pred
from settings import *
from model import generate_model, save_ckp, load_ckp
from utils import (
    AverageMeter,
    Bar,
    Logger,
    savefig,
    draw_on_image,
    calc_ori_delta,
    calc_seg_iou,
)


def run_epoch(data_loader, net, phase="train", optimizer=None, criterian=None, epoch=0, args=None, tb_writer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    seg_ious = AverageMeter()
    ang_diffs = AverageMeter()
    end = time.time()

    batch_N = len(data_loader)
    bar = Bar(f"{phase} {epoch}", max=batch_N)
    for batch_idx, (batch_sample, batch_name) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        end = time.time()

        # to cuda device
        if args.use_cuda:
            for key in batch_sample.keys():
                batch_sample[key] = batch_sample[key].cuda()

        # output
        if phase == "train":
            torch.autograd.set_detect_anomaly(True)
            ori_pred, seg_pred, anchors_ang = net(batch_sample)
        else:
            with torch.no_grad():
                ori_pred, seg_pred, anchors_ang = net(batch_sample)

        # update metric
        ori_angle = expectation_pred(ori_pred, args.ori_stride, angle=True, smooth=True, logits=False)
        ori_angle = ori_angle + anchors_ang
        ori_angle = torch.where(ori_angle >= 90, ori_angle - 180, ori_angle)
        ori_angle = torch.where(ori_angle < -90, ori_angle + 180, ori_angle)

        # for loss function
        anchor_delta = minus_orientation(batch_sample["ori_ang"], anchors_ang)
        anchor_delta = binary_orientation(anchor_delta, args.ori_stride)

        # loss
        loss_ori = criterian["ori"][1](ori_pred, anchor_delta.detach(), batch_sample["seg"], logits=False)
        loss_seg = criterian["seg"][1](seg_pred, batch_sample["seg"], logits=False)
        loss = criterian["ori"][0] * loss_ori + criterian["seg"][0] * loss_seg

        # compute gradient and do SGD step
        if phase == "train":
            with torch.autograd.detect_anomaly():
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(net.parameters(), 20)
                optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        ang_diff = calc_ori_delta(ori_angle.detach(), batch_sample["ori_ang"], batch_sample["seg"])
        seg_iou = calc_seg_iou(seg_pred.detach(), batch_sample["seg"])

        losses.update(loss.data, batch_sample["data"].size(0))
        ang_diffs.update(ang_diff, batch_sample["data"].size(0))
        seg_ious.update(seg_iou, batch_sample["data"].size(0))

        if phase == "train" and tb_writer is not None and batch_idx % 200 == 0 and ang_diff is not None:
            # scalar
            tb_writer.add_scalar(f"{phase}_loss", losses.val, epoch * batch_N + batch_idx)
            tb_writer.add_scalar(f"{phase}_ang_diff", ang_diffs.val, epoch * batch_N + batch_idx)
            tb_writer.add_scalar(f"{phase}_seg_iou", seg_ious.val, epoch * batch_N + batch_idx)
            # image with orientation and segmentation
            tb_writer.add_figure(
                f"{phase}_prediction",
                draw_on_image(
                    batch_sample["data"],
                    ori_angle.detach(),
                    seg_pred.detach(),
                    batch_sample["ori_ang"],
                    batch_sample["seg"],
                    anchors_ang,
                    ang_diff,
                    seg_iou,
                    batch_name,
                ),
                epoch * batch_N + batch_idx,
            )

        # if batch_idx % 1 == 0:
        #     print_str = f"epoch: {epoch} batch: {batch_idx} | "
        #     print_str += f"Lori: {loss_ori.item():.3f} Lseg: {loss_seg.item():.3f} | "
        #     print_str += f"Dori: {ang_diffs.avg:.3f} Sseg: {seg_ious.avg:.3f}"
        #     print(print_str)

        # plot progress
        suffix = f"Td+b:{data_time.avg + batch_time.avg:.3f} Tt:{bar.elapsed_td.total_seconds()//60} ETA:{bar.eta_td.total_seconds()//60} | "
        suffix += f"Loss:{losses.avg:.3f} | "
        suffix += f"IOU:{seg_ious.avg*100:.2f} Angle:{ang_diffs.avg:.2f}"
        bar.suffix = f"({batch_idx+1}/{batch_N}) " + suffix
        bar.next()

    bar.finish()
    return (losses.avg, ang_diffs.avg, seg_ious.avg)


if __name__ == "__main__":
    args = parse_args()
    args = args_train(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # model
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id).strip("[]")
    net = generate_model(args)

    # cuda
    if args.use_cuda:
        net = torch.nn.DataParallel(net).cuda()
    cudnn.benchmark = True

    print(
        "=> Total params: {:.2f}M".format(
            sum(p.numel() for p in filter(lambda p: p.requires_grad, net.parameters())) / 1.0e6
        )
    )

    # optimizer
    if args.optim == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args.lr,
            momentum=0.5,
            nesterov=True,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.weight_decay,
        )
    elif args.optim == "rmsprop":
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"unsupport optimizer {args.optim}")

    # schedule
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)

    # resume
    if args.resume:
        checkpoint = load_ckp(args.logdir)
        if not checkpoint:
            raise ValueError("Failed to load checkpoint")
        else:
            args.start_epoch = checkpoint["epoch"]
            net.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            print(f"resume checkpoint: '{args.logdir}', start from epoch {args.start_epoch}")

    logger = Logger(os.path.join(args.logdir, "log.txt"), resume=args.resume is not None)
    logger.set_names(
        [
            "Learning Rate",
            "Train Loss",
            "Valid Loss",
            "Train AngleDiff",
            "Valid AngleDiff",
            "Train SegIoU",
            "Valid SegIoU",
        ]
    )

    # criterian
    criterian = {}
    for k, v in args.loss_lst.items():
        if k == "wseg":
            criterian["seg"] = [v, WeightedSegLoss(use_cuda=args.use_cuda)]
        elif k == "fseg":
            criterian["seg"] = [v, FocalSegLoss(use_cuda=args.use_cuda)]
        elif k == "wori":
            criterian["ori"] = [v, WeightedOriLoss(ang_stride=args.ori_stride, use_cuda=args.use_cuda)]
        elif k == "fori":
            criterian["ori"] = [v, FocalOriLoss(ang_stride=args.ori_stride, use_cuda=args.use_cuda)]
        elif k != "wtype" or k != "ftype":
            raise ValueError(f"Unsupport loss type {k}")

    # dataloader
    train_transform = transforms.Compose(
        [
            dataset.GaussianNoise(mean=0, std=0.05),
            dataset.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            dataset.RandomHorizontalFlip(p=0.5),
            dataset.RandomAffine(degrees=30, translate=(0.1, 0.1)),
        ]
    )
    train_set = dataset.SingleDataSet(args, "train", transforms=train_transform)
    valid_set = dataset.SingleDataSet(args, "valid", transforms=None)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.use_cuda,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=args.use_cuda
    )

    if args.debug:
        tb_writer = None
    else:
        tb_writer = SummaryWriter(args.logdir, purge_step=args.start_epoch * (len(train_loader)))

    # train and val
    for epoch in range(args.start_epoch, args.max_epochs):
        net.train()
        train_loss, train_ang_diff, train_seg_iou = run_epoch(
            train_loader,
            net,
            phase="train",
            optimizer=optimizer,
            criterian=criterian,
            epoch=epoch,
            args=args,
            tb_writer=tb_writer,
        )

        net.eval()
        val_loss, val_ang_diff, val_seg_iou = run_epoch(
            valid_loader, net, phase="valid", criterian=criterian, epoch=epoch, args=args, tb_writer=tb_writer
        )

        lr = optimizer.param_groups[-1]["lr"]
        logger.append(
            [lr, train_loss, val_loss, train_ang_diff, val_ang_diff, train_seg_iou, val_seg_iou,]
        )
        save_ckp(net, optimizer, epoch + 1, args, lr_scheduler)

        lr_scheduler.step(val_loss)
        if lr <= (args.lr / 100):
            break

    logger.close()
