import os
import json
import random
import argparse
import os.path as osp
from glob import glob
import numpy as np


def save_jsons(img_path_lst, args):
    outdir = "nist27"

    samples = {"meta": {}}
    samples["image"] = img_path_lst
    samples["ori"] = [x.replace("image", "ori") for x in img_path_lst]
    samples["seg"] = [x.replace("image", "seg") for x in img_path_lst]
    samples["meta"]["number"] = len(samples["image"])

    for sub in ["whole", "parts192"]:
        if not os.path.isdir(osp.join(outdir, sub)):
            os.makedirs(osp.join(outdir, sub))
        with open(osp.join(outdir, sub, "test.json"), "w") as fp:
            j_obj = json.dumps(samples, indent=4)
            fp.write(j_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=601, type=int, help="manual seed")
    args = parser.parse_args()

    prefix = "/home/dyj/disk1/data/finger/NIST27/search"

    img_path_lst = glob(osp.join(prefix, "align", "image", "*.png"))
    img_path_lst.sort()

    save_jsons(img_path_lst, args)
