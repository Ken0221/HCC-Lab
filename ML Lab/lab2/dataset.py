import os

import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image

# digits ['0' ~ '9'] -> [0, 9]
CODES_TO_INT = {
    chr(v): code for code, v in enumerate(range(48, 48 + 10))}
# Upper Letter ['A' ~ 'Z'] -> [10, 35]
CODES_TO_INT.update({
    chr(v): 10 + code for code, v in enumerate(range(65, 65 + 26))})
# Inverse [0, 35] -> ['0' ~ '9' 'A' ~ 'Z']
INT_TO_CODE = {v: k for k, v in CODES_TO_INT.items()}
HEIGHT = 48
WIDTH = 128


class LicensePlateDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None):
        self.transform = transform
        csv = pd.read_csv(csv_path)
        root = os.path.dirname(csv_path)
        self.paths = csv['image_path'].apply(
            lambda p: os.path.join(root, p)).values
        self.labels = csv['lp'].apply(
            lambda lp: [CODES_TO_INT[c] for c in lp]).values
        for label in self.labels:
            assert len(label) == 7

    def __len__(self):           # return the size of dataset
        return len(self.paths)

    def __getitem__(self, idx):  # return i-th element in dataset
        image = Image.open(self.paths[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx])


class TrainDataset(LicensePlateDataset):
    def __init__(self, csv_path='./dataset/train/trainValNew.csv'):
        transform = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.RandomRotation(3),
            transforms.RandomResizedCrop(
                (HEIGHT, WIDTH), scale=(0.7, 1.0), ratio=[2., 2.8]),
            transforms.ToTensor()
        ])
        super().__init__(csv_path, transform)


class ValDataset(LicensePlateDataset):
    def __init__(self, csv_path='./dataset/val/trainValNew.csv'):
        '''
        Checkpoint 1: Resize image to (HEIGHT, WIDTH)
        Hint: Add resize transformation to the begining of transfrom
        '''
        transform = transforms.Compose([
            ???
            transforms.ToTensor()
        ])
        super().__init__(csv_path, transform)


if __name__ == '__main__':
    from torchvision.utils import save_image
    dataset = TrainDataset()
    img, label = dataset[0]
    save_image(img, "train.jpg")
    print("train.jpg:", label)

    dataset = ValDataset()
    img, label = dataset[0]
    save_image(img, "val.jpg")
    print("val.jpg:", label)
