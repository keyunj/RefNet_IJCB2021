import os
import os.path as osp
import json
import argparse
import copy
from datetime import datetime
from utils import re_mkdir, mkdir


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Orientation Prediction Trainging")

    # sharing
    parser.add_argument("-d", "--dataset", default="mixhard", help="dataset directory")
    parser.add_argument("-p", "--phase", default="train", help="train/eval/play")
    parser.add_argument("-w", "--workers", default=16, type=int, help="number of workers (default: 16)")
    parser.add_argument("--model-name", default="singleresanchor", type=str, help="model name")
    parser.add_argument("--gpu-id", default=[2], type=int, nargs="+", help="gpu device(s) used")
    parser.add_argument("--no-cuda", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--in-chns", default=1, type=int, help="input channels")
    parser.add_argument("--dilation", default=[1, 4, 8], type=int, help="dilation")
    parser.add_argument("--norm", default="b", type=str, help="normalization type ('b','s','i', or None)")
    parser.add_argument("--act", default="p", type=str, help="activation type ('r','l','p', or None)")
    parser.add_argument(
        "--feat-filters", default=[64, 128, 256, 256, 256], type=int, nargs="+", help="feature channels"
    )
    parser.add_argument("--feat-layers", default=[2, 2, 2, 1, 2], type=int, nargs="+", help="feature layers")
    parser.add_argument("--feat-resblks", default=4, type=int, help="number of res-blocks")
    parser.add_argument("--aspp-filters", default=[256, 128], type=int, nargs="+", help="feature channels")
    parser.add_argument("--type-filters", default=[256, 512], type=int, nargs="+", help="feature channels")
    parser.add_argument("--type-layers", default=[2, 2], type=int, nargs="+", help="feature layers")
    parser.add_argument("--type-resblks", default=1, type=int, help="number of res-blocks")
    parser.add_argument("--ori-stride", default=1, type=int, help="angle stride")
    parser.add_argument("--n-anchors", default=5, type=int, help="number of anchors")

    # only for training
    parser.add_argument("--id", default="work", type=str, help="directory of checkpoint")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--optim", default="adam", type=str, help="optimizer used")
    parser.add_argument("--resume", default=None, type=str, help="resume model checkpoint path")
    parser.add_argument("--batch-size", default=4, type=int, help="batch size in train")
    parser.add_argument("--mini", dest="mini_test", action="store_true", default=False)
    parser.add_argument("--seed", default=2020, type=int, help="mannual seed")
    parser.add_argument("--weight-decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch id")
    parser.add_argument("--loss-lst", default=["fseg", "fori"], type=str, nargs="+", help="loss")
    parser.add_argument("--loss-weight", default=[10.0, 0.5], type=float, nargs="+", help="loss weights")
    parser.add_argument("--joint-epoch", default=10, type=int, help="number of epoch to jointly training")
    parser.add_argument("--max-epochs", default=200, type=int, help="number of total epochs")

    # only for testing
    parser.add_argument("--logdir", default=None, type=str, help="directory of checkpoint")
    parser.add_argument("--debug", dest="debug", action="store_true", default=False, help="debug")
    parser.add_argument("--save", dest="is_save", action="store_true", default=False, help="save result")

    args = parser.parse_args()

    if args.batch_size == 1 and args.phase == "train":
        args.norm = None
    args.workers = min(args.workers, args.batch_size, 16)
    if args.debug:
        arg.workers = 0

    return args


def load_config(cfg_path, args):
    with open(cfg_path, "r") as fp:
        cfgs = json.load(fp)
    for k, v in cfgs.items():
        if not hasattr(args, k):
            setattr(args, k, v)


def args_test(args):
    # model name
    args.model_name = args.logdir.split("_")[0]

    if args.debug:
        args.output = osp.join("outputs", "debug")
    else:
        args.output = osp.join("outputs", args.logdir, args.dataset)
    args.logdir = osp.join("checkpoints", args.logdir)

    # load config
    args = load_config(osp.join(args.logdir, config.json), args)

    # save
    if args.is_save:
        re_mkdir(args.output)
        mkdir(osp.join(args.output, "seg"))
        mkdir(osp.join(args.output, "ori"))

    print(f"test checkpoint: {args.logdir}")
    print(f"output directory: {args.output}")


def args_train(args_original):
    args = copy.deepcopy(args_original)

    assert len(args.loss_lst) == len(args.loss_weight)
    args.loss_lst = {k: v for k, v in zip(args.loss_lst, args.loss_weight)}

    # debug
    if args.debug:
        args.logdir = osp.join("checkpoints", "debug")
    else:
        args.logdir = osp.join("checkpoints", f"{args.model_name}_{args.dataset}_{args.id}")

    # resume
    if args.resume is not None:
        args.resume = osp.join("checkpoints", args.resume)
    else:
        re_mkdir(args.logdir)
    # save config
    with open(osp.join(args.logdir, "config.json"), "w") as fp:
        json.dump(args_original.__dict__, fp, indent=4)

    print(f"train checkpoint: {args.logdir}")

    return args


if __name__ == "__main__":
    args = parse_args()
    args = args_train(args)
