import torch
import matplotlib.pyplot as plt
from model import ImprovedSkinLesionModel

def load_specific_epoch_model(model_path='/kaggle/working/best_improved_model.pth'):
    """
    Load model weights from the best epoch (epoch 5)
    """
    model = ImprovedSkinLesionModel('efficientnet_b0')
    # Load the specific epoch weights
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    return model

def visualize_predictions(model, loader, num_images=10):
    """
    Visualize model predictions with probability distribution using epoch 5 metrics
    """
    model.eval()
    all_images = []
    all_probs = []
    all_targets = []

    # Epoch 5 metrics
    epoch5_sensitivity = 0.8861
    epoch5_specificity = 0.7860
    epoch5_threshold = 0.5  # adjust if needed based on epoch 5 performance

    with torch.no_grad():
        for batch in loader:
            images = batch['image']
            targets = batch['target']
            outputs = torch.sigmoid(model(images.cuda())).cpu()

            all_images.extend(images)
            all_probs.extend(outputs)
            all_targets.extend(targets)

            if len(all_images) >= num_images:
                break

    plt.figure(figsize=(15, 6))
    for idx in range(num_images):
        plt.subplot(2, 5, idx+1)

        # Plot image
        img = all_images[idx].permute(1,2,0).numpy()
        # Proper denormalization using actual mean and std
        mean = np.array([0.4815, 0.4578, 0.4082])
        std = np.array([0.2686, 0.2613, 0.2758])
        img = img * std[None, None, :] + mean[None, None, :]
        img = np.clip(img, 0, 1)

        plt.imshow(img)
        plt.axis('off')

        # Add title with probabilities using epoch 5 metrics
        prob = all_probs[idx].item()
        true_label = all_targets[idx].item()

        # Color coding based on prediction accuracy
        color = 'green' if (prob > epoch5_threshold) == true_label else 'red'

        # Add sensitivity/specificity info for relevant cases
        if true_label == 1:
            metric_info = f'\nSens: {epoch5_sensitivity:.3f}'
        else:
            metric_info = f'\nSpec: {epoch5_specificity:.3f}'

        plt.title(f'True: {"M" if true_label==1 else "B"}\nPred: {prob:.2f}{metric_info}',
                 color=color)

    plt.suptitle('Model Predictions (Based on Best Epoch 5)', y=1.02)
    plt.tight_layout()
    plt.show()

# Load model and run visualization with epoch 5 weights
model = load_specific_epoch_model('/kaggle/working/best_improved_model.pth')
visualize_predictions(model, valid_loader)
