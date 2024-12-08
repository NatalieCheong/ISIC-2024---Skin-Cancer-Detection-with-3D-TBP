import torch
import torch.nn as nn
import timm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc
import gc
import os
from torch.cuda.amp import autocast, GradScaler

# Improved model architecture
class ImprovedSkinLesionModel(nn.Module):
    def __init__(self, model_name='efficientnet_b0'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.classifier.in_features

        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Dropout(0.4),
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        targets = batch['target'].float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    for batch in tqdm(loader, desc='Validating'):
        images = batch['image'].to(device)
        target = batch['target'].float().to(device)

        output = model(images).squeeze()
        loss = criterion(output, target)

        total_loss += loss.item()
        predictions.extend(torch.sigmoid(output).cpu().numpy())
        targets.extend(target.cpu().numpy())

    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    preds = (predictions > 0.5).astype(int)

    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets, preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (preds == targets).mean()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'predictions': predictions,
        'targets': targets,
        'confusion_matrix': cm,  # Added confusion matrix
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }


def calculate_pauc(targets, predictions, min_tpr=0.8):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(targets, predictions)
    idx = tpr >= min_tpr
    return auc(fpr[idx], tpr[idx])

def main():
    # Configuration
    CONFIG = {
        'img_size': 128,
        'train_batch_size': 32,
        'valid_batch_size': 64,
        'epochs': 10,
        'learning_rate': 2e-4,
    }

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = ImprovedSkinLesionModel('efficientnet_b0')
    model = model.to(device)

    # Loss function with higher positive weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([150.0]).to(device))

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=CONFIG['learning_rate'],
                                 weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )
    # Training loop
    best_pauc = 0

    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, valid_loader, criterion, device)
        val_pauc = calculate_pauc(val_metrics['targets'], val_metrics['predictions'])

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {val_metrics['loss']:.4f}")
        print(f"Valid pAUC: {val_pauc:.4f}")
        print(f"Valid Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Valid Sensitivity: {val_metrics['sensitivity']:.4f}")
        print(f"Valid Specificity: {val_metrics['specificity']:.4f}")
        # In the training loop, after validation:
        print(f"\nConfusion Matrix:")
        print(f"TN: {val_metrics['tn']}, FP: {val_metrics['fp']}")
        print(f"FN: {val_metrics['fn']}, TP: {val_metrics['tp']}")

        # Update scheduler
        scheduler.step(val_pauc)

        # Save best model
        if val_pauc > best_pauc:
            best_pauc = val_pauc
            torch.save(model.state_dict(), 'best_improved_model.pth')
            print("Saved new best model")

        # Save best model
        if val_pauc > best_pauc:
            best_pauc = val_pauc
            torch.save(model.state_dict(), 'best_improved_model.pth')
            print("Saved new best model")

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
