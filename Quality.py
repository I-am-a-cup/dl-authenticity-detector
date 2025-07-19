import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import numpy as np

def build_model(meta):
    arch = meta["arch"]
    num_classes = meta["num_classes"]
    dropout = meta.get("dropout", 0.5)
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    else:  # efficientnet_b0
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    return model

def main():
    MODEL_PATH = r"D:\Файлы и всё такое\Институт\Практика 3 год\trial_273_epoch_7_f1_0.9674_resnet50.pt"
    VAL_DIR   = r"D:\Файлы и всё такое\Институт\Практика 3 год\data\val"
    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", DEVICE)

    # Загружаем чекпоинт с параметром weights_only=True для безопасности
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    meta_keys = ["arch", "dropout", "num_classes"]
    meta = {k: checkpoint[k] for k in meta_keys}

    model = build_model(meta)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    img_size = checkpoint.get("img_size", 224)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    val_dataset = ImageFolder(VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    f1 = f1_score(all_targets, all_preds, average='macro')
    print(f"Macro F1-score: {f1:.4f}\n")

    print("Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=val_dataset.classes))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)

if __name__ == "__main__":
    main()
