import os
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def apply_windowing(img, window_center, window_width):
    """
    Apply windowing to better visualize different tissues
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = ((img - img_min) / (window_width) * 255.0)
    return np.clip(img, 0, 255).astype('uint8')

def load_dicom(path):
    """Load and preprocess DICOM image with better windowing"""
    try:
        # Read DICOM file
        dicom = pydicom.dcmread(path)

        # Get pixel array
        img = dicom.pixel_array

        # Convert to Hounsfield Units (HU) if possible
        if hasattr(dicom, 'RescaleIntercept') and hasattr(dicom, 'RescaleSlope'):
            img = img * float(dicom.RescaleSlope) + float(dicom.RescaleIntercept)

        # Apply different window settings for different tissues
        # Soft tissue window
        img_soft = apply_windowing(img.copy(), 40, 400)  # Standard abdominal window
        # Bone window
        img_bone = apply_windowing(img.copy(), 400, 1800)
        # Combine windows (you can adjust the weights)
        img = cv2.addWeighted(img_soft, 0.7, img_bone, 0.3, 0)

        return img

    except Exception as e:
        print(f"Error loading {path}: {str(e)}")
        return None

def get_sample_images(train_df, image_level_df, base_path, injury_type, n_samples=5):
    """Get sample images for a specific injury type"""
    images = []
    patient_ids = []

    try:
        if injury_type in ['bowel', 'extravasation']:
            # For binary cases
            samples = image_level_df[image_level_df['injury_name'] == injury_type].head(n_samples)
            for _, row in samples.iterrows():
                path = os.path.join(base_path, str(row['patient_id']),
                                  str(row['series_id']),
                                  f"{row['instance_number']}.dcm")
                patient_ids.append(row['patient_id'])
                if os.path.exists(path):
                    img = load_dicom(path)
                    if img is not None:
                        images.append(img)
        else:
            # For multi-level cases (kidney, liver, spleen)
            injury_patients = train_df[train_df[f'{injury_type}_high'] == 1]['patient_id'].head(n_samples)
            for patient_id in injury_patients:
                patient_path = os.path.join(base_path, str(patient_id))
                if os.path.exists(patient_path):
                    series_folders = os.listdir(patient_path)
                    if series_folders:
                        series_path = os.path.join(patient_path, series_folders[0])
                        dcm_files = sorted([f for f in os.listdir(series_path) if f.endswith('.dcm')])
                        if dcm_files:
                            middle_slice = len(dcm_files) // 2
                            path = os.path.join(series_path, dcm_files[middle_slice])
                            img = load_dicom(path)
                            if img is not None:
                                images.append(img)
                                patient_ids.append(patient_id)

        return images, patient_ids

    except Exception as e:
        print(f"Error processing {injury_type} images: {str(e)}")
        return [], []

def main():
    # Set paths - Update this for Kaggle path
    base_path = "/kaggle/input/rsna-2023-abdominal-trauma-detection/train_images"

    # Load dataframes
    train_df = pd.read_csv("/kaggle/input/rsna-2023-abdominal-trauma-detection/train_2024.csv")
    image_level_df = pd.read_csv("/kaggle/input/rsna-2023-abdominal-trauma-detection/image_level_labels_2024.csv")

    # Injury types to visualize
    injury_types = ['kidney', 'bowel', 'liver', 'extravasation']

    # Create figure
    fig, axes = plt.subplots(len(injury_types), 5, figsize=(20, 16))
    plt.suptitle("Sample Images of Different Injury Types", fontsize=16)

    for i, injury_type in enumerate(injury_types):
        print(f"Processing {injury_type} images...")
        images, patient_ids = get_sample_images(train_df, image_level_df, base_path, injury_type)

        for j in range(5):
            if j < len(images):
                # Display image with proper contrast
                axes[i, j].imshow(images[j], cmap='gray')
                axes[i, j].set_title(f"{injury_type.capitalize()}\nPatient {patient_ids[j]}")
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.savefig('injury_samples.png')
    plt.show()

    print("Visualization complete!")

if __name__ == "__main__":
    main()
