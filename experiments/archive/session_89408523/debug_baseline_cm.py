import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

ckpt_path = Path("artifacts/pneumonia_3class_baseline_backup.pt")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

model = models.mobilenet_v2(weights=None)
model.classifier[-1] = nn.Linear(model.last_channel, 3)
model.load_state_dict(ckpt["state_dict"])
model = model.to(device).eval()

all_preds = []
all_targets = []
all_probs = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

acc = accuracy_score(all_targets, all_preds)
cm = confusion_matrix(all_targets, all_preds)

print("=== EXACT BASELINE EVALUATION ON VALIDATION ===")
print(f"Total samples: {len(all_targets)}")
print(f"Correct predictions: {(all_preds == all_targets).sum()} / {len(all_targets)}")
print(f"Calculated accuracy: {acc:.6f} ({acc*100:.4f}%)")
print(f"Diagonals sum: {np.trace(cm)} / {len(all_targets)} = {np.trace(cm)/len(all_targets):.6f}")
print("\nConfusion Matrix (Rows=True, Cols=Pred):")
print(f"{'':12s} {'BACTERIAL':10s} {'NORMAL':10s} {'VIRAL':10s} {'Total':6s}")
class_names = ["BACTERIAL", "NORMAL", "VIRAL"]
for i, c in enumerate(class_names):
    print(f"{c:12s} {cm[i,0]:10d} {cm[i,1]:10d} {cm[i,2]:10d} {cm[i].sum():6d}")

print("\nClassification Report:")
print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))
