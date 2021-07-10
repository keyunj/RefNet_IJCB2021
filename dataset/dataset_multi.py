import os
import os.path as osp
import json
import random
import numpy as np
import imageio
from scipy import signal
from torchvision import transforms
from torch.utils.data import Dataset

from .transforms import rescale_trans


class MultiDataSet(Dataset):
    def __init__(self, args, prefix, transforms=None):
        super(MultiDataSet, self).__init__()
        self.prefix = prefix
        self.transforms = transforms
        self.dataset = args.dataset
        self.n_anchors = args.n_anchors
        self.KEYS = ["image", "ori", "seg", "type", "pose"]
        self.gaussian_pdf = signal.windows.gaussian(361, 3)

        self.anchors_ang = np.stack(
            [
                np.asarray(
                    imageio.imread(osp.join("data", "anchors", f"{self.n_anchors}", f"meanfield{x}.png")), np.float32,
                )
                for x in range(self.n_anchors)
            ]
        )
        self.anchors_rmsd = np.stack(
            [
                np.asarray(imageio.imread(osp.join("data", "anchors", f"{self.n_anchors}", f"rmsd{x}.png")), np.float32)
                for x in range(self.n_anchors)
            ]
        )

        json_path = osp.join("data", self.dataset, f"{self.prefix}.json")
        print(f"dataset: {json_path}")

        # images name list
        with open(json_path, "r") as fp:
            self.data_lst = json.load(fp)

        # ===============================================
        # shuffle first
        if self.prefix == "train":
            for key in self.KEYS:
                random.seed(args.seed)
                if key == "type":
                    random.shuffle(self.data_lst[f"{key}{self.n_anchors}"])
                else:
                    random.shuffle(self.data_lst[key])
        # ===============================================
        # mini-dataset test
        if args.mini_test:
            for key in self.KEYS:
                self.data_lst[key] = self.data_lst[key][: args.batch_size * 15]
        # ===============================================

    def __len__(self):
        return len(self.data_lst["image"])

    def __getitem__(self, index):
        # data load
        sample = {}
        sample["data"] = np.asarray(imageio.imread(self.data_lst["image"][index]), np.float32)  # [0,255]
        sample["seg"] = np.asarray(imageio.imread(self.data_lst["seg"][index]) > 0, np.uint8)  # {0,1}
        sample["ori_ang"] = np.asarray(imageio.imread(self.data_lst["ori"][index]), np.float32) - 90  # [0,180)
        sample["type"] = np.array(self.data_lst[f"type{self.n_anchors}"][index]) % self.n_anchors

        pose = np.loadtxt(self.data_lst["pose"][index], delimiter=",").astype(np.float32)
        sample["pose_trans"], sample["pose_theta"] = pose[:2], pose[2]
        img_size = np.array(sample["data"].shape[-1:-3:-1]).astype(np.float32)
        sample["pose_trans"] = rescale_trans(sample["pose_trans"], img_size)  # (x, y)
        sample["pose_theta"] = np.where(sample["pose_theta"] >= 180, sample["pose_theta"] - 180, sample["pose_theta"])
        sample["pose_theta"] = np.where(sample["pose_theta"] < -180, sample["pose_theta"] + 180, sample["pose_theta"])

        if self.transforms is not None:
            sample = self.transforms(sample)

        # others
        sample["data"] = sample["data"][None].astype(np.float32) / 255.0
        sample["seg"] = sample["seg"][None].astype(np.float32)
        sample["type_wt"] = self.binary_type(sample["type"]).astype(np.float32)
        sample["soft_bin"] = self.softbinary(sample["type"]).astype(np.float32)
        sample["anchors_ang"] = self.anchors_ang.astype(np.float32) - 90
        sample["anchors_rmsd"] = self.anchors_rmsd.astype(np.float32)
        sample["ori_ang"] = sample["ori_ang"][None].astype(np.float32)
        sample["pose_trans"] = sample["pose_trans"].astype(np.float32)
        sample["pose_theta"] = sample["pose_theta"][None].astype(np.float32)

        return sample, osp.basename(self.data_lst["image"][index])

    def binary_type(self, type):
        return np.eye(self.n_anchors)[type]

    def softbinary(self, type):
        soft_bin = np.ones(self.n_anchors) * 0.5 / (self.n_anchors - 1)
        soft_bin[type] = 0.5
        return soft_bin
