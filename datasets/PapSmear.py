import os
from PIL import Image
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

class BreakHis(Dataset):

    def __init__(self, data_path: str, setname:str, backbone: str, augment: bool):
        d = os.path.join(data_path, setname)
        dirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]

        data = []
        label = []
        lb = -1

        for d in dirs:
            lb += 1
            for image_name in os.listdir(d):
                path = os.path.join(d, image_name)
                data.append(path)
                label.append(lb)

        self.data = data
        self.label = label

        mean = [x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
        std = [x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
        normalize = transforms.Normalize(mean=mean, std=std)

        self.image_size = 84
        if augment and setname == 'train':
            transforms_list = [
                transforms.Resize((self.image_size,self.image_size), antialias=True),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]

        self.transform = transforms.Compose(
            transforms_list + [normalize]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label, path
