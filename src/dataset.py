import cv2
import os
from os.path import join
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from src.transforms import SimpleDepthTransform
from src.config import TEST_DIR


def get_samples(train_folds_path, folds):
    images_lst = []
    target_lst = []
    depth_lst = []

    train_folds_df = pd.read_csv(train_folds_path)

    for i, row in train_folds_df.iterrows():
        if row.fold not in folds:
            continue

        image = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found {row.image_path}")
        mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found {row.mask_path}")
        images_lst.append(image)
        target_lst.append(mask)
        depth_lst.append(row.z)

    return images_lst, target_lst, depth_lst


class SaltDataset(Dataset):
    def __init__(self, train_folds_path, folds,
                 transform=None,
                 depth_transform=None):
        super().__init__()
        self.train_folds_path = train_folds_path
        self.folds = folds
        self.transform = transform
        if depth_transform is None:
            self.depth_transform = SimpleDepthTransform()
        else:
            self.depth_transform = depth_transform

        self.images_lst, self.target_lst, self.depth_lst = \
            get_samples(train_folds_path, folds)

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        image = self.images_lst[idx]
        depth = self.depth_lst[idx]
        target = self.target_lst[idx]

        input = self.depth_transform(image, depth)

        if self.transform is not None:
            input, target = self.transform(input, target)

        return input, target


def get_test_samples(test_images_dir):
    images_lst = []
    depth_lst = []

    for image_name in os.listdir(test_images_dir):
        image_path = join(test_images_dir, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if np.sum(image):  # skip black images
            images_lst.append(image)
            depth_lst.append(0)  # TODO: load depth

    return images_lst, depth_lst


class SaltTestDataset(Dataset):
    def __init__(self, test_dir,
                 transform=None,
                 depth_transform=None):
        super().__init__()
        self.test_dir = test_dir
        self.transform = transform
        if depth_transform is None:
            self.depth_transform = SimpleDepthTransform()
        else:
            self.depth_transform = depth_transform

        self.images_lst, self.depth_lst = \
            get_test_samples(test_dir)

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        image = self.images_lst[idx]
        depth = self.depth_lst[idx]

        input = self.depth_transform(image, depth)

        if self.transform is not None:
            input = self.transform(input)

        return input
