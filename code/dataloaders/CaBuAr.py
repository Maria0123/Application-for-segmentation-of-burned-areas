import os
import cv2
import torch
import random
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt
from PIL import Image


class CaBuAr(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        else:
            h5f = h5py.File(self._base_dir + "/data/slices/{}.h5".format(case), "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        sample = {"image": image, "label": label}
        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            elif self.transform:
                sample = self.transform(sample)
        sample["idx"] = idx
        return sample

class RandomFlip(object):
    def __make_transform__(self, image, transform, angle = -1):
        img = torch.tensor([])
        for i in range(image.shape[0]):
            img_PIL = transforms.ToPILImage()(image[i])
            img_t = transform(img_PIL) if angle == -1 else transform(img_PIL, angle)
            img_t = transforms.ToTensor()(img_t)
            img = torch.cat((img, img_t), dim = 0)
        
        return img

        
    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if torch.rand(1) < 0.5:
            transform = transforms.functional.hflip
            image2 = image.copy()
            image2 = self.__make_transform__(image2, transform)
            image = self.__make_transform__(image, transform)
            assert image == image2
            print("pass")
            label = self.__make_transform__(label, transform)

        if torch.rand(1) < 0.5:
            transform = transforms.functional.vflip
            image = self.__make_transform__(image, transform)
            label = self.__make_transform__(label, transform)

        if torch.rand(1) < 0.5:
            angle = torch.randint(0, 360, (1,)).item()  
            transform = transforms.functional.rotate
            image = self.__make_transform__(image, transform, angle)
            label = self.__make_transform__(label, transform, angle)

        return {'image': image, 'label': label}

class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        if type(image) is torch.Tensor:
            return sample
        image = image.reshape(
            image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}
