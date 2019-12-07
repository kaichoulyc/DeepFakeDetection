import os

import cv2
import torch
from albumentations import Compose, HorizontalFlip, Normalize, Resize
from torch.utils.data import DataLoader, Dataset


def transformation(mode):

    transforms = []
    if mode == 'train':
        transforms.extend([HorizontalFlip()])
    transforms.extend([Resize(256, 256), Normalize()])
    return Compose(transforms)


def get_dataloader(path, mode, binary, batch_size, num_workers):

    some_dataset = FakeDataset(path, mode, binary)
    some_dataloader = DataLoader(some_dataset, batch_size=batch_size,
                                 num_workers=num_workers)

    return some_dataloader


class FakeDataset(Dataset):

    def __init__(self, path, mode, binary):

        self.path = path
        self.files = os.listdir(path)
        self.trasfroms = transformation(mode)
        self.binary = binary

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        image = cv2.imread(os.path.join(self.path, self.files[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transed = self.trasfroms(image=image)
        image = transed['image']
        image = torch.tensor(image.transpose(2, 0, 1)).float()

        target = int(self.files[index].split('_')[0])
        if self.binary:
            target = 1 if target else 0

        return image, target
