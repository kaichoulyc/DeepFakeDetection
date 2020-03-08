import os

import jpeg4py as jpeg
import pandas as pd
import torch
from albumentations import Compose, HorizontalFlip, Normalize, Resize
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def transformation(mode, side_size):

    transforms = []
    if mode == 'train':
        transforms.extend([HorizontalFlip()])
    transforms.extend([Resize(side_size, side_size), Normalize()])
    return Compose(transforms)


def get_dataloader(path: str,
                   mode: str,
                   dataset: str,
                   side_size: int,
                   batch_size: int,
                   num_workers: int,
                   ddp: bool = False,
                   df_path: str = None):
    if dataset == 'FaceForensics':
        some_dataset = FaceForensics(path, mode, side_size)
    elif dataset == 'Facebook':
        some_dataset = FacebookFakes(path, df_path, mode, side_size)
    if ddp:
        sampler = DistributedSampler(some_dataset)
        some_dataloader = DataLoader(some_dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     sampler=sampler)
    else:
        some_dataloader = DataLoader(some_dataset,
                                     batch_size=batch_size,
                                     num_workers=num_workers)

    return some_dataloader


class FacebookFakes(Dataset):

    def __init__(self, path, df_path, mode, side_size):

        self.path = path
        self.df = pd.read_csv(df_path)
        self.transforms = transformation(mode, side_size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        image_data = self.df.iloc[index]
        image = jpeg.JPEG(os.path.join(self.path, image_data['image_name'])).decode()
        transed = self.transforms(image=image)
        image = transed['image']
        image = torch.tensor(image.transpose(2, 0, 1)).float()

        target = image_data['label']
        if target == 'real':
            target = 0
        elif target == 'fake':
            target = 1

        return image, target


class FaceForensics(Dataset):

    def __init__(self, path, mode, side_size):

        self.path = path
        self.files = os.listdir(path)
        self.trasforms = transformation(mode, side_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image = jpeg.JPEG(os.path.join(self.path, self.files[index])).decode()
        transed = self.trasforms(image=image)
        image = transed['image']
        image = torch.tensor(image.transpose(2, 0, 1)).float()

        target = int(self.files[index].split('_')[0])

        return image, target
