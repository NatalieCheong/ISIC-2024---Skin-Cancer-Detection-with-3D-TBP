import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from image_preprocessing import ISICDatasetSampler, ISICDatasetSimple, prepare_loaders
from augmentations import get_augmentations
import h5py

def prepare_data(hdf5_path, metadata_path):
    """Prepare train and validation datasets"""
    print("Loading metadata...")
    meta_df = pd.read_csv(metadata_path)
    
    # Split data
    train_df, valid_df = train_test_split(
        meta_df,
        test_size=0.2,
        random_state=42,
        stratify=meta_df['target']
    )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Valid samples: {len(valid_df)}")
    
    # Get transforms
    transforms = get_efficient_augmentations(CONFIG)
    
    # Create datasets
    train_dataset = MemoryEfficientDataset(
        train_df,
        hdf5_path,
        transforms=transforms['train']
    )
    
    valid_dataset = MemoryEfficientDataset(
        valid_df,
        hdf5_path,
        transforms=transforms['valid']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['train_batch_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        shuffle=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CONFIG['valid_batch_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        shuffle=False
    )
    
    return train_loader, valid_loader
       
