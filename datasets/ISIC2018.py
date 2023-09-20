import os
from PIL import Image
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

class ISIC2018(Dataset):

    class_id_map = dict(
        MEL = 0,
        NV = 1,
        BCC = 2,
        AKIEC = 3,
        BKL = 4,
        DF = 5,
        VASC = 6
    )

    def __init__(self, data_path: str, setname: str, backbone: str, augment: bool, img_size=224, img_ext='jpg', transform=None, target_transform=None):
        
        """
        Initialize the Dataset
        - size: Image Size
        - mode: Determines the CSV file to use (ISIC18_T3_<mode>.csv)
                Also determined the subdirectory for data split
                Valid values: <subdirectory names within the data root>

        NOTE: CSV must be present in the [ROOT_PATH]/data 
        """

        super(ISIC2018, self).__init__()

        if setname == "train":
            setname = "Training"

        # Fetch CSV
        self.csv_path = os.path.join(
            data_path,
            f"ISIC2018_Task3_{setname}_GroundTruth.csv"
        )
        assert os.path.isfile(self.csv_path), f"CSV file was not found at {self.csv_path}"

        # Store CSV as Dataframe
        self.csv_df = pd.read_csv(self.csv_path)

        # Ensure data path validity
        self.img_base_path = data_path
        assert os.path.isfile(self.csv_path), f"CSV file was not found at {self.csv_path}"
        self.img_concrete_path = os.path.join(
            self.img_base_path,
            f"ISIC2018_Task3_{setname}_Input/ISIC2018_Task3_{setname}_Input/"
        )
        assert os.path.isdir(self.img_concrete_path), f"Could not find valid data path at {self.img_concrete_path}"

        self.img_frmt_ext = img_ext
        self.transform = transform
        self.target_transform = target_transform

        self.num_classes = len(self.class_id_map)
        self.class_names = list(self.class_id_map.keys())
        # All `target` values of the dataset
        self.label = list(map(
            lambda idx: self.get_sparse_label(self.csv_df, idx),
            range(len(self.csv_df))
        ))
        print(f"Getting '{setname}' data from {self.img_concrete_path}")

        if setname == 'Training':
            setname = 'train'
            
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean,std=std)
        
        self.image_size=img_size
        if augment and setname=='train':
            transforms_list = [
                # transforms.ToPILImage(),
                transforms.RandomResizedCrop((self.image_size,self.image_size), antialias=True),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                # transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ]

        self.transform = transforms.Compose(
            transforms_list + [normalize]
        )

    def __len__(self):
        return len(self.csv_df)
    
    def __getitem__(self,idx):

        """
        Generates a single (img, label) pair

        NOTE: Shuffling is taken care of by the DataLoader wrapper!
        """

        if not isinstance(idx,int):
            idx = idx.item()

        img_path = os.path.join(
            self.img_concrete_path,
            "{0}.{1}".format(
                self.csv_df.iloc[idx,0],
                self.img_frmt_ext
            )
        )
        img_label = self.get_sparse_label(self.csv_df, idx)

        # read data
        try:
            # img_data = torchvision.io.read_image(img_path).float()
            img_data = Image.open(img_path)
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
    def get_class_id(class_names):
        return[
            ISIC2018.class_id_map.get(x)
            for x in class_names
        ]
    
    @staticmethod
    def return_tensor(func):
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            return torch.Tensor(result)
        
        return wrapped

    @staticmethod
    def get_sparse_label(csv_df, idx):
        
        """ 
        Convert one-hot-encoded labels (from across the Dataframe columns)
        to a single sparse label format
        """

        one_hot_label = csv_df.iloc[idx, 1:]
        for classlabel, value in one_hot_label.items():
            if value == 1:
                return ISIC2018.class_id_map.get(classlabel)
        return -1
    
# def get_transform(img_size: int, split_name: str):
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     normalize = transforms.Normalize(mean=mean, std=std)

#     if split_name == 'train':
#         return transforms.Compose([
#             transforms.RandomResizedCrop(size=(img_size, img_size), antialias=True),
#             transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             normalize
#         ])

#     else:
#         return transforms.Compose([
#             transforms.Resize((img_size, img_size)),
#             transforms.ToTensor(),
#             normalize
#         ])
