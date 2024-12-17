import torch
import torch.nn as nn
import timm
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc
import gc
import os
from torch.cuda.amp import autocast, GradScaler

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
        # Get features from the backbone
        features = self.model(x)
        # Pass through classifier
        return self.classifier(features)
