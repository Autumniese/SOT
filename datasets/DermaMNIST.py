import os
from PIL import Image
import pandas as pd
import numpy as np
import random

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

class DermaMNIST(Dataset):

    def __init__(self, data_path: str, setname:str, backbone: str, augment: bool):
        # Load the dataset
        d = np.load(data_path)
        
        # Extract samples and labels
        data = d[f'{setname}_images']
        labels = d[f'{setname}_labels']

        self.data = data
        self.labels = labels

        # mean = [x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
        # std = [x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)

        if augment and setname == 'train':
            transforms_list = [
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.ToTensor(),
            ]

        self.transform = transforms.Compose(
            transforms_list + [normalize]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, label = self.data[i], self.labels[i].astype(int)
        image = Image.fromarray(image)
        image = self.transform(image)
        return image, label
 