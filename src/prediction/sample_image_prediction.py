import torch
import matplotlib.pyplot as plt
import numpy as np
from model import ImprovedSkinLesionModel

def plot_sample_predictions(model, loader, num_samples=10):
    """
    Plot sample predictions with probability distribution bars
    """
    model.eval()
    classes = ['Benign', 'Malignant']

    with torch.no_grad():
        for batch in loader:
            images = batch['image']
            targets = batch['target']

            # Get predictions
            outputs = torch.sigmoid(model(images.cuda())).cpu()

            # Plot samples
            for idx in range(min(num_samples, len(images))):
                plt.figure(figsize=(10, 4))

                # Plot image
                plt.subplot(1, 2, 1)
                img = images[idx].permute(1,2,0).numpy()
                img = (img * 0.5) + 0.5  # denormalize
                plt.imshow(img)
                plt.title('Original Image')
                plt.axis('off')

                # Plot probability bars
                plt.subplot(1, 2, 2)
                probs = [1 - outputs[idx].item(), outputs[idx].item()]
                colors = ['green' if targets[idx] == 0 else 'red' for _ in range(2)]
                bars = plt.bar(classes, probs, color=colors, alpha=0.6)
                plt.ylim(0, 1)

                # Add title
                true_class = classes[int(targets[idx])]
                pred_class = classes[1] if outputs[idx] > 0.5 else classes[0]
                plt.title(f"Predictions\nTrue: {true_class}\n" +
                         f"Predicted: {pred_class}\n" +
                         f"Confidence: {max(probs):.2f}")

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}',
                            ha='center', va='bottom')

                plt.tight_layout()
                plt.show()

            break  # Only process first batch

# Run sample predictions visualization
plot_sample_predictions(model, valid_loader)
