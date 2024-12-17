import numpy as np
import pandas as pd
import h5py
import cv2

class MemoryEfficientDataset(Dataset):
    def __init__(self, meta_df, hdf5_path, transforms=None):
        self.meta_df = meta_df
        self.hdf5_path = hdf5_path
        self.transforms = transforms
        self.hdf5_file = None

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')

        img_id = self.meta_df.iloc[idx].isic_id
        img_bytes = self.hdf5_file[img_id][()]
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return {
            'image': img,
            'target': self.meta_df.iloc[idx].target
        }
