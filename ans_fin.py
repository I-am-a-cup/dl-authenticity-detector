import os
import glob
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import f1_score
from collections import defaultdict

MODELS_DIR = "saved_models"
VAL_DIR    = r"D:/Файлы и всё такое/Институт/Практика 3 год/data/val"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

def load_checkpoints(top_k=20):
    paths = glob.glob(os.path.join(MODELS_DIR, "trial_*_f1_*.pt"))
    ckpts = []
    for p in paths:
        data = torch.load(p, map_location="cpu", weights_only=True)
        bf = data.get("best_f1", 0.0)
        ckpts.append((p, data, bf))
    ckpts = sorted(ckpts, key=lambda x: x[2], reverse=True)
    return [(p, meta) for p, meta, _ in ckpts[:top_k]]

checkpoints = load_checkpoints(20)
print(f"Loaded {len(checkpoints)} top checkpoints.")


classes = sorted(os.listdir(VAL_DIR))
filepaths = []
true_labels = []
for idx, cls in enumerate(classes):
    cls_dir = os.path.join(VAL_DIR, cls)
    for fn in os.listdir(cls_dir):
        if fn.lower().endswith((".jpg", ".png", ".jpeg")):
            filepaths.append(os.path.join(cls_dir, fn))
            true_labels.append(idx)
n_samples = len(filepaths)
print(f"Validation samples: {n_samples}, Classes: {classes}")


def build_model(meta):
    arch = meta["arch"]
    num_classes = meta["num_classes"]
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(meta["dropout"]), nn.Linear(in_f, num_classes))
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(meta["dropout"]), nn.Linear(in_f, num_classes))
    else:  # efficientnet_b0
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Sequential(nn.Dropout(meta["dropout"]), nn.Linear(in_f, num_classes))
    return m


all_probs = []
for idx, (path, meta) in enumerate(checkpoints):
    print(f"Processing model {idx+1}: {os.path.basename(path)}, f1={meta['best_f1']:.4f}")
    model = build_model(meta).to(DEVICE).eval()
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    img_size = meta["img_size"]
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    # inference
    probs = torch.zeros((n_samples, meta["num_classes"]), device="cpu")
    with torch.no_grad():
        for i, fp in enumerate(filepaths):
            img = Image.open(fp).convert("RGB")
            x = tfm(img).unsqueeze(0).to(DEVICE)
            out = model(x)
            p = torch.softmax(out, dim=1).cpu().squeeze(0)
            probs[i] = p
    all_probs.append(probs)
print("Finished per-model inference.")


results = {}
# single best
pred1 = all_probs[0].argmax(1).tolist()
results[1] = f1_score(true_labels, pred1, average="macro")
# ensembles
stack = torch.stack(all_probs, dim=0)
for k in range(2, len(all_probs)+1):
    avg_p = stack[:k].mean(dim=0)
    preds = avg_p.argmax(1).tolist()
    results[k] = f1_score(true_labels, preds, average="macro")


print("\nMacro-F1 Comparison:")
print("Models | Macro-F1")
for k in sorted(results.keys()):
    label = f"Ensemble {k}" if k>1 else "Best single"
    print(f"{label:12s}: {results[k]:.4f}")
