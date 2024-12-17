import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_efficient_augmentations(CONFIG):
    data_transforms = {
        "train": A.Compose([
            A.RandomResizedCrop(CONFIG['img_size'], CONFIG['img_size'], scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7)),
                A.GaussNoise(),
                A.MedianBlur()
            ], p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ]),
        "valid": A.Compose([
            A.Resize(CONFIG['img_size'], CONFIG['img_size']),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    }
    return data_transforms
               
                   
