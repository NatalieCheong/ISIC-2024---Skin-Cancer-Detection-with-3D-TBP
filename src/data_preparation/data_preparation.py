import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from image_preprocessing import ISICDatasetSampler, ISICDatasetSimple, prepare_loaders
from augmentations import get_augmentations
import h5py

def prepare_data():
    # Load metadata - this is fast
    meta_df = pd.read_csv('/kaggle/input/isic-2024-challenge/train-metadata.csv', low_memory=False)

    # Add path column - pointing to HDF5 file
    with h5py.File('/kaggle/input/isic-2024-challenge/train-image.hdf5', 'r') as hf:
        # Get all valid image IDs
        valid_ids = list(hf.keys())

    # Filter metadata to only include images that exist in HDF5
    meta_df = meta_df[meta_df['isic_id'].isin(valid_ids)].reset_index(drop=True)

    # Add path column pointing to HDF5 file
    meta_df['path'] = '/kaggle/input/isic-2024-challenge/train-image.hdf5'

    print("Total samples:", len(meta_df))

    # Split data - this is fast
    train_df, valid_df = train_test_split(
        meta_df,
        test_size=0.2,
        random_state=42,
        stratify=meta_df['target']
    )
    print(f"Train samples: {len(train_df)}")
    print(f"Valid samples: {len(valid_df)}")

    # Setup config
    CONFIG = {
        'img_size': 128,
        'train_batch_size': 32,
        'valid_batch_size': 64
    }

    # Get transforms
    data_transforms = get_augmentations(CONFIG)

    # Create data loaders
    train_loader, valid_loader = prepare_loaders(
        df_train=train_df,
        df_valid=valid_df,
        CONFIG=CONFIG,
        data_transforms=data_transforms,
        num_workers=4  # Reduced number of workers to prevent memory issues
    )

    return train_loader, valid_loader

# Run preparation
train_loader, valid_loader = prepare_data()
print("Data loaders ready!")
