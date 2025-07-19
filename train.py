import os
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from multiprocessing import freeze_support

BASE_DIR = "D:/Файлы и всё такое/Институт/Практика 3 год/data"
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR   = os.path.join(BASE_DIR, 'val')

if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    raise FileNotFoundError(f"Не найдены директории: {TRAIN_DIR} или {VAL_DIR}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # Гиперпараметры: размер изображения
    img_size = trial.suggest_categorical("img_size", [224, 256])
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # Датасеты
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tfms)

    # Weighted sampler
    counts = {}
    for _, label in train_ds.samples:
        counts[label] = counts.get(label, 0) + 1
    sample_weights = [1.0/counts[label] for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # DataLoaders
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    # Архитектура
    arch = trial.suggest_categorical("arch", ["resnet18", "resnet50", "efficientnet_b0"])
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, len(train_ds.classes))
    elif arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, len(train_ds.classes))
    else:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, len(train_ds.classes))
    model.to(DEVICE)

    # Функция потерь
    use_focal = trial.suggest_categorical("use_focal", [False, True])
    if use_focal:
        gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
        class FocalLoss(nn.Module):
            def __init__(self, gamma):
                super().__init__()
                self.gamma = gamma
            def forward(self, logits, targets):
                ce = nn.functional.cross_entropy(logits, targets, reduction="none")
                pt = torch.exp(-ce)
                return (((1-pt)**self.gamma)*ce).mean()
        criterion = FocalLoss(gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    # Оптимизатор и scheduler
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    opt_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    if opt_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.0, 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    elif opt_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    # Тренировка и валидация
    for epoch in range(1, 11):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        # Валидация
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                out = model(imgs)
                preds.extend(out.argmax(1).cpu().tolist())
                targets.extend(labels.tolist())
        f1 = f1_score(targets, preds, average="macro")
        scheduler.step(f1)
        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return f1

if __name__ == "__main__":
    freeze_support()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, timeout=3600)

    print("Best trial:")
    trial = study.best_trial
    for k, v in trial.params.items():
        print(f"  {k}: {v}")

