from train import *
from model import *
from metrics import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import gc


def main():
    # Set paths for your Kaggle environment
    hdf5_path = '/kaggle/input/isic-2024-challenge/train-image.hdf5'
    metadata_path = '/kaggle/input/isic-2024-challenge/train-metadata.csv'

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    print("Preparing datasets...")
    train_loader, valid_loader = prepare_data(hdf5_path, metadata_path)

    # Initialize model and move to device
    print("Initializing model...")
    model = ImprovedSkinLesionModel()
    model = model.to(device)

    # Initialize optimizer and loss
    pos_weight = torch.tensor([100.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=0.01
    )

    # Initialize scheduler and scaler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2
    )
    scaler = GradScaler()

    # Training loop
    print("Starting training...")
    best_val_score = 0
    patience = 3
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # Validate
        val_metrics = validate(model, valid_loader, criterion, device)

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {val_metrics['loss']:.4f}")
        print(f"Valid pAUC: {val_metrics['pAUC']:.4f}")
        print(f"Valid Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Valid Sensitivity: {val_metrics['sensitivity']:.4f}")
        print(f"Valid Specificity: {val_metrics['specificity']:.4f}")

        # Update scheduler
        scheduler.step(val_metrics['pAUC'])

        # Save best model
        if val_metrics['pAUC'] > best_val_score:
            best_val_score = val_metrics['pAUC']
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved new best model")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

        print(f"Epoch completed in {time.time() - epoch_start_time:.2f} seconds")

        # Clear memory
        gc.collect()
        torch.cuda.empty_cache()

    print("Training completed!")
    print(f"Best validation pAUC: {best_val_score:.4f}")

if __name__ == "__main__":
    main()
