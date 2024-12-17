import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from PIL import Image

def plot_sample_predictions(model, loader, num_samples=8):
    """
    Plot sample predictions with probability distribution bars next to each image
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get batch of samples
    batch = next(iter(loader))
    images = batch['image'].to(device)
    targets = batch['target'].numpy()
    original_images = batch['original_image']
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
    
    # Plot samples
    fig = plt.figure(figsize=(20, 4 * ((num_samples + 1) // 2)))
    
    for idx in range(min(num_samples, len(images))):
        # Plot original image
        plt.subplot(((num_samples + 1) // 2), 4, 2*idx + 1)
        img = original_images[idx]
        plt.imshow(img)
        true_label = 'Malignant' if targets[idx] == 1 else 'Benign'
        pred_label = 'Malignant' if probs[idx] > 0.5 else 'Benign'
        color = 'green' if (probs[idx] > 0.5) == targets[idx] else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}', color=color)
        plt.axis('off')
        
        # Plot prediction probabilities
        plt.subplot(((num_samples + 1) // 2), 4, 2*idx + 2)
        classes = ['Benign', 'Malignant']
        class_probs = [1 - probs[idx][0], probs[idx][0]]  # Benign and Malignant probabilities
        colors = ['green' if targets[idx] == i else 'red' for i in range(2)]
        
        # Create bar plot
        bars = plt.bar(classes, class_probs, color=colors, alpha=0.6)
        plt.ylim(0, 1)
        plt.title('Predictions')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
    
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
        batch_size=8,  # Small batch size for visualization
        num_workers=4,
        shuffle=True  # Shuffle to get random samples
    )
    
    # Load model
    print("Loading model...")
    model = ImprovedSkinLesionModel()
    model.load_state_dict(torch.load(model_path))
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Visualize sample predictions
    print("Visualizing sample predictions...")
    plot_sample_predictions(model, test_loader, num_samples=8)

if __name__ == "__main__":
    main()
                
