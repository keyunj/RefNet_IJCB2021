"""
This file (transforms.py) is designed for:
    transformations self-defined
Copyright (c) 2020, Yongjie Duan. All rights reserved.
"""
import random
import numpy as np
import math
import numbers
from PIL import Image
from scipy.ndimage import map_coordinates
from torchvision import transforms


def rescale_trans(pose_trans, img_size):
    # rescale to [-1, 1]
    pose_trans = 2 * (1.0 * pose_trans / img_size - 0.5)
    return pose_trans


class SamplePose(object):
    def __init__(self, std=(0.05, 5)):
        self.std = std

    def __call__(self, sample):
        # random sampling pose
        sample["pose_trans"] = sample["pose_trans"] + np.random.randn(2) * self.std[0]
        sample["pose_theta"] = sample["pose_theta"] + np.random.randn() * self.std[1]
        return sample

    def __repr__(self):
        return self.__class__.__name__ + f"(std={self.std})"


class GaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.std = std
        self.mean = mean

    def __call__(self, sample):
        img_shape = sample["data"].shape
        sample["data"] = sample["data"] + np.random.randn(*(img_shape)) * self.std * 255 + self.mean
        sample["data"] = sample["data"].clip(0, 255)
        return sample

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class ColorJitter(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0):
        super(ColorJitter, self).__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=0)

    def __call__(self, sample):
        trans = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        img = Image.fromarray(sample["data"].astype(np.uint8))
        img = trans(img)
        sample["data"] = np.asarray(img)
        return sample


def hflip_pose(pose_trans, pose_theta):
    pose_trans[0] = -pose_trans[0]
    pose_theta = -pose_theta
    return pose_trans, pose_theta


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__(p=p)

    def __call__(self, sample):
        if random.random() < self.p:
            for key in ["data", "ori_ang", "seg"]:
                sample[key] = np.flip(sample[key], -1).copy()
            # for orientation
            sample["ori_ang"] = -sample["ori_ang"]
            sample["pose_trans"], sample["pose_theta"] = hflip_pose(sample["pose_trans"], sample["pose_theta"])
            if "type" in sample and (sample["type"] == 2 or sample["type"] == 3):
                sample["type"] = 5 - sample["type"]
        return sample


def get_affine_matrix(angle, translate, scale, shear):
    angle = math.radians(angle)
    if isinstance(shear, (tuple, list)) and len(shear) == 2:
        shear = [math.radians(s) for s in shear]
    elif isinstance(shear, numbers.Number):
        shear = math.radians(shear)
        shear = [shear, 0]
    else:
        raise ValueError(
            "Shear should be a single value or a tuple/list containing " + "two values. Got {}".format(shear)
        )

    matrix = [
        math.cos(angle + shear[0]) * scale,
        -math.sin(angle + shear[0]) * scale,
        translate[0],
        math.sin(angle + shear[1]) * scale,
        math.cos(angle + shear[1]) * scale,
        translate[1],
    ]

    return np.array(matrix).reshape(2, 3)


def affine_pose(pose_trans, pose_theta, M, angle):
    pose_trans = np.matmul(M[:2, :2], pose_trans) + M[:2, 2]
    pose_theta = pose_theta - angle
    return pose_trans, pose_theta


class RandomAffine(transforms.RandomAffine):
    def __init__(self, degrees=0, translate=0):
        super(RandomAffine, self).__init__(degrees=degrees, translate=translate)

    def __call__(self, sample):
        angle, translations, scale, shear = self.get_params(
            self.degrees, self.translate, self.scale, self.shear, sample["seg"].shape[::-1]
        )
        # image
        cval = int(sample["data"].min())
        img_translations = [x * 8 for x in translations]
        img = Image.fromarray(sample["data"].astype(np.uint8))
        img = transforms.functional.affine(
            img, angle, img_translations, scale, shear, resample=Image.BILINEAR, fillcolor=cval,
        )
        sample["data"] = np.asarray(img)

        # segmentation and orientation
        for key in ["ori_ang", "seg"]:
            if sample[key].ndim == 2:
                data = Image.fromarray(sample[key])
                data = transforms.functional.affine(
                    data, angle, translations, scale, shear, resample=Image.NEAREST, fillcolor=0
                )
                sample[key] = np.asarray(data)
            else:
                for ii in range(sample[key].shape[0]):
                    data = Image.fromarray(sample[key][ii])
                    data = transforms.functional.affine(
                        data, angle, translations, scale, shear, resample=Image.NEAREST, fillcolor=0
                    )
                    sample[key][ii] = np.asarray(data)

        # for orientation
        sample["ori_ang"] = sample["ori_ang"] + angle
        sample["ori_ang"] = np.where(sample["ori_ang"] >= 90, sample["ori_ang"] - 180, sample["ori_ang"])
        sample["ori_ang"] = np.where(sample["ori_ang"] < -90, sample["ori_ang"] + 180, sample["ori_ang"])
        # for pose
        pose_translations = [2.0 * x / s for x, s in zip(translations, sample["seg"].shape[::-1])]
        M = get_affine_matrix(angle, pose_translations, scale, shear)
        sample["pose_trans"], sample["pose_theta"] = affine_pose(sample["pose_trans"], sample["pose_theta"], M, angle)
        return sample
