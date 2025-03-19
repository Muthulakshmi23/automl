# src/feature_extraction.py
import torch
import torchvision.models as models
import torch.nn as nn
from xgboost import XGBClassifier

# Load pretrained ResNet50
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Identity()

def extract_image_features(images):
    with torch.no_grad():
        images_tensor = torch.tensor(images).permute(0, 3, 1, 2).float()
        features = resnet(images_tensor)
    return features.numpy()

def extract_tabular_features(tabular_features, labels_encoded):
    xgb = XGBClassifier()
    xgb.fit(tabular_features, labels_encoded)
    return xgb.apply(tabular_features)
