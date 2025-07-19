import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score
from multiprocessing import freeze_support
import optuna

BASE_DIR = "D:/Файлы и всё такое/Институт/Практика 3 год/data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
SAVED_MODELS_DIR = "saved_models"
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

def objective(trial):
    img_size = trial.suggest_categorical("img_size", [224, 256, 299])
    crop_min = trial.suggest_float("crop_min", 0.5, 1.0)
    hflip = trial.suggest_float("hflip", 0.0, 0.5)
    bright = trial.suggest_float("bright", 0.0, 0.5)
    contrast = trial.suggest_float("contrast", 0.0, 0.5)
    sat = trial.suggest_float("sat", 0.0, 0.5)
    hue = trial.suggest_float("hue", 0.0, 0.1)
    use_amp = trial.suggest_categorical("use_amp", [True, False])

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(crop_min, 1.0)),
        transforms.RandomHorizontalFlip(p=hflip),
        transforms.ColorJitter(brightness=bright, contrast=contrast, saturation=sat, hue=hue),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tfms)
    val_ds = datasets.ImageFolder(VAL_DIR, transform=val_tfms)

    counts = {}
    for _, lbl in train_ds.samples:
        counts[lbl] = counts.get(lbl, 0) + 1
    weights = [1.0 / counts[lbl] for _, lbl in train_ds.samples]
    sampler = WeightedRandomSampler(weights, len(weights), True)

    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    arch = trial.suggest_categorical("arch", ["resnet18", "resnet50", "efficientnet_b0"])
    dropout_val = trial.suggest_float("dropout", 0.0, 0.5)
    num_classes = len(train_ds.classes)

    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Linear(in_f, num_classes)
        )
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Linear(in_f, num_classes)
        )
    else:
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Sequential(
            nn.Dropout(dropout_val),
            nn.Linear(in_f, num_classes)
        )

    model = m.to(DEVICE)

    use_focal = trial.suggest_categorical("use_focal", [True, False])
    if use_focal:
        gamma = trial.suggest_float("gamma", 1.0, 3.0)
        class FocalLoss(nn.Module):
            def __init__(self, gamma):
                super().__init__()
                self.gamma = gamma
            def forward(self, logits, targets):
                ce = nn.functional.cross_entropy(logits, targets, reduction="none")
                pt = torch.exp(-ce)
                return ((1 - pt) ** self.gamma * ce).mean()
        criterion = FocalLoss(gamma)
    else:
        label_smoothing = trial.suggest_float("ls", 0.0, 0.2)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    wd = trial.suggest_float("wd", 1e-6, 1e-3, log=True)
    l1_weight = trial.suggest_float("l1_weight", 1e-8, 1e-3, log=True)
    opt_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])

    if opt_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.0, 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    else:
        momentum = None
        if opt_name == "AdamW":
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    scheduler_name = trial.suggest_categorical("scheduler", ["none", "steplr", "reduce"])
    if scheduler_name == "steplr":
        step_size = trial.suggest_int("step", 3, 7)
        gamma_step = trial.suggest_float("gamma_step", 0.1, 0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma_step)
        patience = None
    elif scheduler_name == "reduce":
        patience = trial.suggest_int("patience", 1, 3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=patience, factor=0.5)
        step_size = None
        gamma_step = None
    else:
        scheduler = None
        step_size = None
        gamma_step = None
        patience = None

    scaler = amp.GradScaler(enabled=use_amp)

    epochs = trial.suggest_int("epochs", 5, 15)
    patience_es = 7
    best_f1 = 0
    no_improve = 0
    best_model_path = None

    def remove_all_trial_models(trial_num):
        # Удаляем все сохранённые модели этого trial
        for filename in os.listdir(SAVED_MODELS_DIR):
            if filename.startswith(f"trial_{trial_num}_"):
                try:
                    os.remove(os.path.join(SAVED_MODELS_DIR, filename))
                except Exception as e:
                    print(f"Ошибка при удалении {filename}: {e}")

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with amp.autocast(device_type='cuda', enabled=use_amp):
                out = model(xb)
                loss = criterion(out, yb)
                if l1_weight > 0:
                    l1_reg = sum(torch.norm(param, 1) for param in model.parameters())
                    loss += l1_weight * l1_reg
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with amp.autocast(device_type='cuda', enabled=use_amp):
                    out = model(xb)
                preds += out.argmax(1).cpu().tolist()
                trues += yb.cpu().tolist()

        f1 = f1_score(trues, preds, average="macro")

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(f1)
            else:
                scheduler.step()

        trial.report(f1, ep)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0

            # Удаляем все предыдущие модели этого trial
            remove_all_trial_models(trial.number)

            # Сохраняем текущую лучшую модель
            best_model_path = os.path.join(SAVED_MODELS_DIR, f"trial_{trial.number}_epoch_{ep}_f1_{best_f1:.4f}_{arch}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),

                "arch": arch,
                "dropout": dropout_val,
                "num_classes": num_classes,

                "img_size": img_size,
                "crop_min": crop_min,
                "hflip": hflip,
                "bright": bright,
                "contrast": contrast,
                "sat": sat,
                "hue": hue,

                "use_amp": use_amp,
                "use_focal": use_focal,
                "gamma_focal": gamma if use_focal else None,
                "label_smoothing": label_smoothing if not use_focal else None,

                "optimizer": opt_name,
                "lr": lr,
                "weight_decay": wd,
                "momentum": momentum,
                "l1_weight": l1_weight,

                "scheduler": scheduler_name,
                "step_size": step_size,
                "gamma_step": gamma_step,
                "patience": patience,

                "best_f1": best_f1,
                "epoch": ep,

            }, best_model_path)
        else:
            no_improve += 1
            if no_improve >= patience_es:
                break

    return best_f1

if __name__ == "__main__":
    freeze_support()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_jobs=1)

    print("Best trial:", study.best_value)
    print("Best params:", study.best_trial.params)
