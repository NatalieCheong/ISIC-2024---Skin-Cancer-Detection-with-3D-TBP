import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import h5py
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'img_size': 96,
    'batch_size': 32,
    'num_workers': 4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
class ImprovedSkinLesionModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features
        
        # Remove original classifier
        self.model.classifier = nn.Identity()
        
        # Create new classifier
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)
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
        
        # Store original image before transforms
        original_img = cv2.resize(img.copy(), (CONFIG['img_size'], CONFIG['img_size']))
        
        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed["image"]
            
        return {
            'image': img,
            'target': self.meta_df.iloc[idx].target,
            'image_id': img_id,
            'original_image': original_img
        }
def get_test_transforms():
    return A.Compose([
        A.Resize(CONFIG['img_size'], CONFIG['img_size']),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    image_ids = []
    original_images = []
    
    for batch in tqdm(loader, desc="Getting predictions"):
        images = batch['image'].to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        
        predictions.extend(probs)
        targets.extend(batch['target'].numpy())
        image_ids.extend(batch['image_id'])
        original_images.extend(batch['original_image'])
    
    return (np.array(predictions), 
            np.array(targets), 
            image_ids, 
            np.array(original_images))
def plot_roc_curve(targets, predictions):
    plt.figure(figsize=(10, 10))
    fpr, tpr, _ = roc_curve(targets, predictions)
    auc_score = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_prediction_distribution(predictions, targets):
    plt.figure(figsize=(10, 6))
    
    # Plot benign predictions in blue
    sns.histplot(predictions[targets == 0], 
                label='Benign', 
                color='blue',
                alpha=0.5, 
                bins=50)
    
    # Plot malignant predictions in red
    sns.histplot(predictions[targets == 1], 
                label='Malignant', 
                color='red',
                alpha=0.5, 
                bins=50)
    
    plt.title('Prediction Score Distribution', fontsize=12, pad=15)
    plt.xlabel('Prediction Score', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    
    # Add a more visible legend
    plt.legend(fontsize=10, 
              bbox_to_anchor=(1.05, 1), 
              loc='upper left')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()
    
    # Print distribution statistics
    print("\nDistribution Statistics:")
    print(f"Benign predictions - Mean: {predictions[targets == 0].mean():.3f}, "
          f"Std: {predictions[targets == 0].std():.3f}")
    print(f"Malignant predictions - Mean: {predictions[targets == 1].mean():.3f}, "
          f"Std: {predictions[targets == 1].std():.3f}")
def visualize_predictions(original_images, predictions, targets, image_ids, num_samples=10):
    # Select a mix of correct and incorrect predictions
    preds_binary = predictions.squeeze() > 0.5
    correct_mask = preds_binary == targets
    incorrect_mask = ~correct_mask
    
    # Get indices for correct and incorrect predictions
    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]
    
    # Select samples
    num_correct = min(num_samples // 2, len(correct_indices))
    num_incorrect = min(num_samples - num_correct, len(incorrect_indices))
    
    selected_correct = np.random.choice(correct_indices, num_correct, replace=False)
    selected_incorrect = np.random.choice(incorrect_indices, num_incorrect, replace=False)
    selected_indices = np.concatenate([selected_correct, selected_incorrect])
    
    # Plot images
    fig = plt.figure(figsize=(20, 4 * ((len(selected_indices) + 4) // 5)))
    for idx, img_idx in enumerate(selected_indices):
        plt.subplot(((len(selected_indices) + 4) // 5), 5, idx + 1)
        
        # Get image and predictions
        img = original_images[img_idx]
        pred_score = predictions[img_idx]
        true_label = targets[img_idx]
        img_id = image_ids[img_idx]
        
        # Plot image
        plt.imshow(img)
        
        # Color code based on prediction accuracy
        color = 'green' if (pred_score > 0.5) == true_label else 'red'
        
        # Create title with prediction information
        title = f'ID: {img_id[:8]}...\n'
        title += f'True: {"M" if true_label else "B"}\n'
        title += f'Pred: {pred_score[0]:.3f}'
        
        plt.title(title, color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
def main():
    # Set paths
    model_path = '/kaggle/working/best_model.pth'
    hdf5_path = '/kaggle/input/isic-2024-challenge/train-image.hdf5'
    metadata_path = '/kaggle/input/isic-2024-challenge/train-metadata.csv'
    
    # Load metadata
    print("Loading metadata...")
    meta_df = pd.read_csv(metadata_path)
    
    # Create test dataset and loader
    test_dataset = MemoryEfficientDataset(
        meta_df,
        hdf5_path,
        transforms=get_test_transforms()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        shuffle=False
    )
    
    # Load model
    print("Loading model...")
    model = ImprovedSkinLesionModel()
    model.load_state_dict(torch.load(model_path))
    model = model.to(CONFIG['device'])
    
    # Get predictions
    print("Getting predictions...")
    predictions, targets, image_ids, original_images = get_predictions(model, test_loader, CONFIG['device'])
    
    # Plot ROC curve
    print("Plotting ROC curve...")
    plot_roc_curve(targets, predictions)
    
    # Plot prediction distribution
    print("Plotting prediction distribution...")
    plot_prediction_distribution(predictions, targets)
    
    # Visualize sample predictions
    print("Visualizing sample predictions...")
    visualize_predictions(original_images, predictions, targets, image_ids)

if __name__ == "__main__":
    main()
