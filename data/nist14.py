import os
import json
import random
import argparse
import os.path as osp
from glob import glob
import numpy as np


def save_jsons(train_samples, valid_samples, test_samples, prefix, args, outdir):
    if args.use_all:
        outdir = f"{outdir}all"

    outdir = osp.join(outdir, args.fp_type)

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    seg_name = "seg" if args.subdir != "plain" else "seg_plain"

    for stage, p_samples in zip(["train", "valid", "test"], [train_samples, valid_samples, test_samples]):
        samples = {"meta": {"seed": args.seed}}
        samples["image"] = [x[0] for x in p_samples]
        samples["ori"] = [osp.join(prefix, "ori", osp.basename(x[0]).split(".")[0] + ".png") for x in p_samples]
        samples["seg"] = [osp.join(prefix, seg_name, osp.basename(x[0]).split(".")[0] + ".png") for x in p_samples]
        # samples["type5"] = [int(x[1]) for x in p_samples]
        samples["pose"] = [
            osp.join(prefix, "pose", args.pose_type, osp.basename(x[0]).split(".")[0] + ".txt") for x in p_samples
        ]
        samples["meta"]["number"] = len(samples["image"])
        with open(f"{outdir}/{stage}.json", "w") as fp:
            j_obj = json.dumps(samples, indent=4)
            fp.write(j_obj)


# def random_split_dataset(img_path_lst, img_type_lst, prefix, args):
def random_split_dataset(img_path_lst, prefix, args):
    indices = np.arange(len(img_path_lst))
    random.seed(args.seed)
    random.shuffle(indices)

    if args.use_all:
        valid_N = len(indices) // 10
        train_N = len(indices) - valid_N
        train_samples = [[img_path_lst[ii]] for ii in indices[:train_N]]
        valid_samples = [[img_path_lst[ii]] for ii in indices[-valid_N:]]
        test_samples = []
    else:
        valid_N = len(indices) // 5
        train_N = len(indices) - 2 * valid_N
        train_samples = [[img_path_lst[ii]] for ii in indices[:train_N]]
        valid_samples = [[img_path_lst[ii]] for ii in indices[train_N : train_N + valid_N]]
        test_samples = [[img_path_lst[ii]] for ii in indices[-valid_N:]]

    # sort
    train_samples.sort(key=lambda x: x[0])
    valid_samples.sort(key=lambda x: x[0])
    test_samples.sort(key=lambda x: x[0])

    save_jsons(train_samples, valid_samples, test_samples, prefix, args, args.subdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=601, type=int, help="manual seed")
    parser.add_argument("--subdir", default="mixhard", type=str, help="sub-directory")
    parser.add_argument("--pose-type", default="manual", type=str, help="sub-directory")
    parser.add_argument("--fp-type", default="0", type=str, help="fingerprint type")
    parser.add_argument("--all", dest="use_all", default=False, action="store_true", help="use all dataset")
    args = parser.parse_args()

    prefix = "/home/dyj/disk1/data/finger/NIST14"

    img_path_lst = glob(osp.join(prefix, f"image_{args.subdir}", "*.png"))
    img_path_lst.sort()

    img_type_lst = np.loadtxt(osp.join(prefix, "fingertype_scores.txt"))
    img_type_lst = np.argmax(img_type_lst, axis=1)

    img_path_lst = [x for x, t in zip(img_path_lst, img_type_lst) if t == int(args.fp_type)]

    # if args.use_all:
    total_N = len(img_path_lst)
    # else:
    #     # randomly select 8100 + 2700 + 2700 images
    #     total_N = 8100 * 5 // 3

    random.seed(args.seed)
    indices = np.arange(len(img_path_lst)).tolist()
    indices = random.sample(indices, total_N)
    img_path_lst = np.array(img_path_lst)[indices].tolist()
    # img_type_lst = img_type_lst[indices]

    # random_split_dataset(img_path_lst, img_type_lst, prefix, args)

    random_split_dataset(img_path_lst, prefix, args)
