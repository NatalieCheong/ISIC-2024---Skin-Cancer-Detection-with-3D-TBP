import numpy as np
import pandas as pd
import h5py
import cv2
import gc
from tqdm import tqdm
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
import random
import warnings
warnings.filterwarnings('ignore')

# The code below have copied from: https://github.com/ilyanovo/isic-2024/blob/main/src/datasets.py and make some adjustment.

class ISICDatasetSampler(Dataset):
    def __init__(self, meta_df, transforms=None, process_target: bool=False, n_classes:int=3, do_augmentations: bool=True, *args, **kwargs):
        self.df_positive = meta_df[meta_df["target"] == 1].reset_index()
        self.df_negative = meta_df[meta_df["target"] == 0].reset_index()
        self.file_names_positive = self.df_positive['path'].values
        self.file_names_negative = self.df_negative['path'].values
        self.targets_positive = self.df_positive['target'].values
        self.targets_negative = self.df_negative['target'].values
        self.transforms = transforms
        self.n_classes = n_classes
        self.process_target = process_target
        self.do_augmentations = do_augmentations

    def __len__(self):
         return len(self.df_positive) * 2


    def __getitem__(self, index):
        if random.random() >= 0.5:
            df = self.df_positive
            file_names = self.file_names_positive
            targets = self.targets_positive
        else:
            df = self.df_negative
            file_names = self.file_names_negative
            targets = self.targets_negative
        index = index % df.shape[0]

        if isinstance(file_names[0], str) and file_names[0].endswith('.hdf5'):
            # Handle HDF5 file
            with h5py.File(file_names[0], 'r') as hf:
                 img_bytes = hf[df['isic_id'].iloc[index]][()]
                 img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Handle regular image files
            img_path = file_names[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = targets[index]

        if self.transforms and self.do_augmentations:
            img = self.transforms(image=img)["image"]

        if self.process_target:
            target_pr = np.zeros(shape=(self.n_classes,))
            target_pr[int(target)] += 1.0
            target = target_pr

        return {
            'image': img,
            'target': target
        }


class ISICDatasetSimple(Dataset):
    def __init__(self, meta_df, targets=None, transforms=None, process_target: bool=False, n_classes:int=3, do_augmentations: bool=True, *args, **kwargs):
        self.meta_df = meta_df
        self.targets = targets
        self.transforms = transforms
        self.n_classes = n_classes
        self.process_target = process_target
        self.do_augmentations = do_augmentations


    def __len__(self):
        return self.meta_df.shape[0]

    def __getitem__(self, idx):
        target = self.meta_df.iloc[idx].target
        path = self.meta_df.iloc[idx].path

        if isinstance(path, str) and path.endswith('.hdf5'):
            # Handle HDF5 file
            with h5py.File(path, 'r') as hf:
                img_bytes = hf[self.meta_df.iloc[idx].isic_id][()]
                img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Handle regular image files
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms and self.do_augmentations:
            transformed = self.transforms(image=img)
            img = transformed['image']

        if self.process_target:
            target_pr = np.zeros(shape=(self.n_classes,))
            target_pr[int(target)] += 1.0
            target = target_pr

        return {
            'image': img,
            'target': target
        }



def prepare_loaders(df_train, df_valid, CONFIG, data_transforms, data_loader_base=ISICDatasetSampler, weight_adg=1, num_workers=10):

    train_dataset = data_loader_base(df_train, transforms=data_transforms["train"], weight_adg=weight_adg)
    valid_dataset = ISICDatasetSimple(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['train_batch_size'],
                              num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'],
                              num_workers=num_workers, shuffle=False, pin_memory=True)

    return train_loader, valid_loader
