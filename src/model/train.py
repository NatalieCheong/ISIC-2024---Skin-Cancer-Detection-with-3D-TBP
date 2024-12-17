import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    start_time = time.time()

    for idx, batch in enumerate(tqdm(loader, desc="Training")):
        # Check time limit (15 minutes)
        if time.time() - start_time > 800:
            print("Time limit reached, stopping epoch early")
            break

        images = batch['image'].to(device)
        targets = batch['target'].float().to(device)

        # Mixed precision training
        with autocast():
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()

        if (idx + 1) % CONFIG['gradient_accumulation_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []

    for batch in tqdm(loader, desc="Validating"):
        images = batch['image'].to(device)
        target = batch['target'].float().to(device)

        outputs = model(images).squeeze()
        loss = criterion(outputs, target)

        total_loss += loss.item()
        predictions.extend(torch.sigmoid(outputs).cpu().numpy())
        targets.extend(target.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)
    preds = (predictions > 0.5).astype(int)

    # Calculate metrics
    accuracy = (preds == targets).mean()
    sensitivity = (preds[targets == 1] == 1).mean() if any(targets == 1) else 0
    specificity = (preds[targets == 0] == 0).mean()
    pauc = calculate_pauc(targets, predictions)

    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'pAUC': pauc,
        'predictions': predictions,
        'targets': targets
    }
