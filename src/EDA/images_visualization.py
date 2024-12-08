import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from PIL import Image
import random
import io

def decode_image(img_bytes):
    """
    Decode image from bytes to numpy array
    """
    try:
        # Convert bytes to image
        img = Image.open(io.BytesIO(img_bytes))
        # Convert to numpy array
        return np.array(img)
    except Exception as e:
        print(f"Error decoding image: {str(e)}")
        return None

def load_data(hdf5_path, metadata_path):
    """
    Load image data from HDF5 file and metadata from CSV with proper error handling
    """
    print("Loading data...")
    # Load metadata with low_memory=False to handle mixed types
    metadata_df = pd.read_csv(metadata_path, low_memory=False)
    print(f"Metadata shape: {metadata_df.shape}")

    # Load HDF5 file
    with h5py.File(hdf5_path, 'r') as hf:
        # Get list of keys
        image_keys = list(hf.keys())
        print(f"Total images in HDF5: {len(image_keys)}")

        # Sample 20 random images
        sample_keys = random.sample(image_keys, min(20, len(image_keys)))
        sample_images = []

        print("\nLoading sample images...")
        for i, key in enumerate(sample_keys):
            try:
                # Read the dataset
                img_bytes = hf[key][()]
                # Decode image
                if isinstance(img_bytes, bytes) or isinstance(img_bytes, np.ndarray):
                    img = decode_image(img_bytes)
                    if img is not None:
                        sample_images.append((key, img))
                        if (i + 1) % 5 == 0:
                            print(f"Loaded {i + 1}/{len(sample_keys)} images")
            except Exception as e:
                print(f"Error loading image {key}: {str(e)}")
                continue

    return sample_images, metadata_df

def visualize_sample_images(sample_images, metadata_df):
    """
    Visualize sample images with their metadata
    """
    if not sample_images:
        print("No images to visualize!")
        return

    print("\nVisualizing sample images...")
    rows = (len(sample_images) + 4) // 5  # Ceiling division by 5
    plt.figure(figsize=(20, 4*rows))

    for idx, (img_id, img) in enumerate(sample_images):
        plt.subplot(rows, 5, idx + 1)

        # Print image shape and value range for debugging
        print(f"\nImage {idx+1} shape: {img.shape}, dtype: {img.dtype}")
        print(f"Value range: [{np.min(img)}, {np.max(img)}]")

        plt.imshow(img)

        # Get metadata for this image
        img_meta = metadata_df[metadata_df['isic_id'] == img_id]
        if not img_meta.empty:
            img_meta = img_meta.iloc[0]
            # Create title with relevant information
            title = f"ID: {img_id[:8]}...\n"
            if 'target' in img_meta:
                title += f"{'Malignant' if img_meta['target'] == 1 else 'Benign'}\n"
            if 'anatom_site_general' in img_meta:
                site = str(img_meta['anatom_site_general'])
                if len(site) > 10:
                    site = site[:10] + '..'
                title += f"{site}"
        else:
            title = f"ID: {img_id[:8]}..."

        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def print_dataset_statistics(metadata_df):
    """
    Print basic statistics about the dataset
    """
    print("\nDataset Statistics:")
    print("-" * 50)

    try:
        # Print column names
        print("\nAvailable columns:")
        print(metadata_df.columns.tolist())

        if 'target' in metadata_df.columns:
            print("\nClass Distribution:")
            class_dist = metadata_df['target'].value_counts(normalize=True)
            print(metadata_df['target'].value_counts())
            print(f"Benign: {class_dist.get(0, 0):.2%}")
            print(f"Malignant: {class_dist.get(1, 0):.2%}")

        if 'anatom_site_general' in metadata_df.columns:
            print("\nAnatomical Site Distribution:")
            site_dist = metadata_df['anatom_site_general'].value_counts().head()
            print(metadata_df['anatom_site_general'].value_counts())

        if 'age_approx' in metadata_df.columns:
            print("\nAge Statistics:")
            print(metadata_df['age_approx'].describe())

        # Print sample of null values
        print("\nMissing Values (top 10 columns):")
        null_counts = metadata_df.isnull().sum()
        print(null_counts[null_counts > 0].head(10))

        # Print first few rows of key columns
        print("\nMetadata Sample (key columns):")
        columns_to_show = ['isic_id', 'target', 'anatom_site_general', 'age_approx', 'sex']
        columns_to_show = [col for col in columns_to_show if col in metadata_df.columns]
        print(metadata_df[columns_to_show].head())

    except Exception as e:
        print(f"Error in statistics calculation: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    # Define paths
    train_hdf5_path = 'train-image.hdf5'
    train_metadata_path = 'train-metadata.csv'

    # Load data
    try:
        sample_images, metadata_df = load_data(train_hdf5_path, train_metadata_path)

        if sample_images:
            # Visualize images
            visualize_sample_images(sample_images, metadata_df)

            # Print statistics
            print_dataset_statistics(metadata_df)
        else:
            print("No images were successfully loaded")
            print("\nLet's examine the HDF5 file structure:")
            with h5py.File(train_hdf5_path, 'r') as hf:
                # Print first key and its attributes
                first_key = list(hf.keys())[0]
                print(f"\nFirst key: {first_key}")
                print(f"Dataset type: {type(hf[first_key])}")
                print(f"Dataset shape: {hf[first_key].shape}")
                print(f"Dataset dtype: {hf[first_key].dtype}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
