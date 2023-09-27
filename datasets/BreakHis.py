import os
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

class BreakHis(Dataset):

    class_id_map = dict(
        adenosis = 0,
        tubular_adenoma = 1,
        phyllodes_tumor = 2,
        fibroadenoma = 3,
        papillary_carcinoma = 4,
        ductal_carcinoma = 5,
        mucinous_carcinoma = 6,
        lobular_carcinoma = 7
    )
    
    def __init__(self, data_path: str, setname: str, backbone: str, augment: bool, img_size=84, img_ext='png', transform=None, target_transform=None):

        self.csv_path = os.path.join(
            data_path,
            data_path.rsplit('/',1)[-1] + '_' + setname + '.csv'
        )

        self.csv_df = pd.read_csv(self.csv_path)

        self.img_base_path = data_path

        self.img_concrete_path = os.path.join(
            self.img_base_path,
            setname
        )

        self.img_frmt_ext = img_ext

        # data = []
        # label = []
        # lb = -1

        # self.wnids = []
        # for l in lines:
        #     name, wnid = l.split(',')
        #     path = os.path.join(data_path,setname,name)
        #     if wnid not in self.wnids:
        #         self.wnids.append(wnid)
        #         lb += 1
        #     data.append(path)
        #     label.append(lb)

        # self.data = data
        # self.label = label

        self.label = list(map(
            lambda idx: self.get_sparse_label(self.csv_df, idx),
            range(len(self.csv_df))
        ))

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean,std=std)

        self.image_size = img_size

        if augment:
            transforms_list = [
                transforms.Resize((self.image_size,self.image_size), antialias=True),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ]

        self.transform = transforms.Compose(
            transforms_list + [normalize]
        )
    
    def __len__(self):
        return len(self.csv_df)
    
    def __getitem__(self, idx):
        # path, label = self.data[i], self.label[i]
        # orig = self.transform(Image.open(path).convert('RGB'))
        # return orig, label

        """
        Generates a single (img, label) pair

        NOTE: Shuffling is taken care of by the DataLoader wrapper!
        """

        if not isinstance(idx,int):
            idx = idx.item()

        img_path = os.path.join(
            self.img_concrete_path,
            "{0}.{1}".format(
                self.csv_df.iloc[idx,3],
                self.img_frmt_ext
            )
        )

        # read data
        try:
            img_data = Image.open(img_path).convert('RGB')
        except Exception as e:
            print("Error when trying to read data file:", e)
            return None
        
        # apply transforms:
        if self.transform is not None:
            img_data = self.transform(img_data)
        if self.target_transform is not None:
            img_label = self.target_transform(img_label)

        # return data, label and path
        return img_data, img_label, img_path

    @staticmethod
    def get_sparse_label(csv_df, idx):

        """ 
        Convert labels (from across the Dataframe columns)
        to a single sparse label format
        """
        try:
            return BreakHis.class_id_map.get(csv_df.iloc[idx, 4])
        except:
            return -1
