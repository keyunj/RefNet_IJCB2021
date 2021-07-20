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


class SingleDataSet(Dataset):
    def __init__(self, args, prefix, transforms=None):
        super(SingleDataSet, self).__init__()
        self.prefix = prefix
        self.transforms = transforms
        self.dataset = args.dataset
        self.n_anchors = args.n_anchors
        self.model_type = int(args.model_type)
        self.pose_type = args.pose_type
        self.KEYS = ["image", "ori", "seg", "pose", "type"]
        self.gaussian_pdf = signal.windows.gaussian(361, 3)
        self.anchors_ang = (
            np.asarray(
                imageio.imread(
                    osp.join("data", "anchors", self.pose_type, f"{self.n_anchors}", f"meanfield{self.model_type}.png")
                ),
                np.float32,
            )
            - 90
        )
        self.anchors_rmsd = np.asarray(
            imageio.imread(
                osp.join("data", "anchors", self.pose_type, f"{self.n_anchors}", f"rmsd{self.model_type}.png")
            ),
            np.float32,
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
        sample["type"] = np.array(self.data_lst[f"type{self.n_anchors}"][index])

        pose = np.loadtxt(self.data_lst["pose"][index].replace("manual", self.pose_type), delimiter=",").astype(
            np.float32
        )
        sample["pose_trans"], sample["pose_theta"] = pose[:2], pose[2]
        img_size = np.array(sample["data"].shape[-1:-3:-1]).astype(np.float32)
        sample["pose_trans"] = rescale_trans(sample["pose_trans"], img_size)  # (x, y)
        sample["pose_theta"] = np.where(sample["pose_theta"] >= 90, sample["pose_theta"] - 180, sample["pose_theta"])
        sample["pose_theta"] = np.where(sample["pose_theta"] < -90, sample["pose_theta"] + 180, sample["pose_theta"])

        if self.transforms is not None:
            sample = self.transforms(sample)

        # others
        sample["data"] = sample["data"][None].astype(np.float32) / 255.0
        sample["seg"] = sample["seg"][None].astype(np.float32)
        sample["type"] = (sample["type"] == self.model_type).astype(np.float32)
        sample["anchors_ang"] = self.anchors_ang[None].astype(np.float32)
        sample["anchors_rmsd"] = self.anchors_rmsd[None].astype(np.float32)
        sample["ori_ang"] = sample["ori_ang"][None].astype(np.float32)
        sample["pose_trans"] = sample["pose_trans"].astype(np.float32)
        sample["pose_theta"] = sample["pose_theta"][None].astype(np.float32)

        return sample, osp.basename(self.data_lst["image"][index])
