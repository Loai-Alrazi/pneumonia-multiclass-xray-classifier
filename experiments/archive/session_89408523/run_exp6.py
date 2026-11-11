import sys
sys.path.append(r"C:\Users\Loay_Alrazi\.gemini\antigravity\brain\89408523-973d-4ee4-8297-b78793515970\scratch")
import copy
import torch
import numpy as np
from experiment_framework import (
    get_base_transform, build_model, evaluate_model,
    DATA_DIR, CLASS_NAMES, set_seed, IMAGE_SIZE, MEAN, STD
)
from torchvision import datasets
from torch.utils.data import DataLoader

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp2_ckpt = torch.load("artifacts/model_exp2_class_weighted_head.pt", map_location=device)

# Load converged model from Exp 2
model = build_model(device, freeze_backbone=True)
model.load_state_dict(exp2_ckpt["state_dict"])

# Unfreeze top 4 blocks (blocks 15-18)
total_blocks = len(model.features)
for idx in range(total_blocks - 4, total_blocks):
    for p in model.features[idx].parameters():
        p.requires_grad = True

transform = get_base_transform()
train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

counts = np.bincount(train_dataset.targets, minlength=len(CLASS_NAMES))
weights = len(train_dataset) / (len(CLASS_NAMES) * counts.astype(float))
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))

optimizer = torch.optim.Adam([
    {"params": [p for idx in range(total_blocks - 4, total_blocks) for p in model.features[idx].parameters() if p.requires_grad], "lr": 3e-5},
    {"params": model.classifier.parameters(), "lr": 1e-4}
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-6)

best_val_loss = float("inf")
best_state = None
best_epoch = 0

print("Starting Stage 2 Fine-Tuning from Exp2 Checkpoint...")
for epoch in range(1, 9):
    model.train()
    model.features.eval()
    t_loss, t_corr, t_tot = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        t_loss += loss.item() * len(y)
        t_corr += (out.argmax(dim=1) == y).sum().item()
        t_tot += len(y)
        
    scheduler.step()
    val_m = evaluate_model(model, val_loader, device, loss_fn)
    print(f"Epoch {epoch:2d}/8 | Train Loss: {t_loss/t_tot:.4f}, Acc: {t_corr/t_tot:.2%} | Val Loss: {val_m['loss']:.4f}, Acc: {val_m['accuracy']:.2%}, Macro F1: {val_m['macro_f1']:.2%}")
    if val_m["loss"] < best_val_loss:
        best_val_loss = val_m["loss"]
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
final_val_metrics = evaluate_model(model, val_loader, device, loss_fn)
print("BEST_STAGE2_VAL_METRICS:", final_val_metrics)
torch.save({
    "state_dict": {k: v.cpu() for k, v in best_state.items()},
    "class_names": CLASS_NAMES,
    "image_size": IMAGE_SIZE,
    "mean": MEAN,
    "std": STD,
    "best_epoch": best_epoch,
    "val_metrics": final_val_metrics,
    "experiment_name": "Exp6_Stage2_FineTune_from_Exp2"
}, "artifacts/model_exp6_stage2_finetune.pt")
print("Saved artifacts/model_exp6_stage2_finetune.pt successfully!")
