import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from scipy import stats

DATA_DIR = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")
CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# True patient ID extraction
def get_true_patient_id(path_str):
    name = Path(path_str).name
    mb = re.match(r"(person\d+_bacteria)_\d+", name)
    if mb:
        return mb.group(1)
    mv = re.match(r"(person\d+_virus)_\d+", name)
    if mv:
        return mv.group(1)
    parts = Path(path_str).stem.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:-1])
    return Path(path_str).stem

val_paths = [s[0] for s in val_dataset.samples]
val_patients = np.array([get_true_patient_id(p) for p in val_paths])
val_targets = np.array(val_dataset.targets)
unique_patients = np.unique(val_patients)

print(f"Validation dataset: {len(val_dataset)} images, {len(unique_patients)} unique patient IDs (true IDs)")

# Class weights
train_dataset = datasets.ImageFolder(DATA_DIR / "train")
train_counts = np.bincount(train_dataset.targets, minlength=3)
train_weights = len(train_dataset) / (3.0 * train_counts.astype(float))
class_weights_t = torch.tensor(train_weights, dtype=torch.float32).to(device)

crit_unweighted = nn.CrossEntropyLoss().to(device)
crit_weighted = nn.CrossEntropyLoss(weight=class_weights_t).to(device)

ckpts = [
    ("Baseline (Orig)", "artifacts/pneumonia_3class_baseline_backup.pt", None),
    ("Baseline (Exp 0)", "artifacts/model_exp0_baseline_3epochs.pt", 42),
    ("Exp 1 (s42)", "artifacts/model_exp1_extended_head_s42.pt", 42),
    ("Exp 1 (s123)", "artifacts/model_exp1_extended_head_s123.pt", 123),
    ("Exp 1 (s456)", "artifacts/model_exp1_extended_head_s456.pt", 456),
    ("Exp 2 (s42)", "artifacts/model_exp2_class_weighted_head_s42.pt", 42),
    ("Exp 2 (s123)", "artifacts/model_exp2_class_weighted_head_s123.pt", 123),
    ("Exp 2 (s456)", "artifacts/model_exp2_class_weighted_head_s456.pt", 456),
    ("Exp 5 (s42)", "artifacts/model_exp5_finetune_top4blocks_s42.pt", 42),
    ("Exp 5 (s123)", "artifacts/model_exp5_finetune_top4blocks_s123.pt", 123),
    ("Exp 5 (s456)", "artifacts/model_exp5_finetune_top4blocks_s456.pt", 456),
]

all_model_results = []
all_probs_dict = {}
all_preds_dict = {}

for name, rel_path, seed in ckpts:
    path = Path(rel_path)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = models.mobilenet_v2(weights=None)
    model.classifier[-1] = nn.Linear(model.last_channel, 3)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    
    total_loss_u = 0.0
    total_loss_w = 0.0
    preds = []
    probs = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            lu = crit_unweighted(logits, y)
            lw = crit_weighted(logits, y)
            total_loss_u += lu.item() * len(y)
            total_loss_w += lw.item() * len(y)
            p = F.softmax(logits, dim=1).cpu().numpy()
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            probs.extend(p)
            
    preds = np.array(preds)
    probs = np.array(probs)
    all_preds_dict[name] = preds
    all_probs_dict[name] = probs
    
    loss_u = total_loss_u / len(val_dataset)
    loss_w = total_loss_w / len(val_dataset)
    
    acc = accuracy_score(val_targets, preds)
    f1_m = f1_score(val_targets, preds, average="macro", zero_division=0)
    rec_m = recall_score(val_targets, preds, average="macro", zero_division=0)
    recs = recall_score(val_targets, preds, average=None, zero_division=0)
    precs = precision_score(val_targets, preds, average=None, zero_division=0)
    f1s = f1_score(val_targets, preds, average=None, zero_division=0)
    cm = confusion_matrix(val_targets, preds)
    
    # Clinical metrics
    # Binary pneumonia screening: Pneumonia = Bact(0) or Viral(2), Normal = 1
    y_true_binary = (val_targets != 1).astype(int) # 1 = Pneumonia, 0 = Normal
    y_pred_binary = (preds != 1).astype(int)
    
    pneu_tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    pneu_fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0)) # False Normal (missed pneumonia!)
    pneu_tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    pneu_fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
    
    pneu_sens = pneu_tp / (pneu_tp + pneu_fn) # Pneumonia sensitivity
    pneu_spec = pneu_tn / (pneu_tn + pneu_fp) # Pneumonia specificity (= Normal recall)
    
    # Etiology discrimination among true pneumonia cases (Targets 0 and 2)
    pneu_mask = (val_targets != 1)
    targets_pneu = val_targets[pneu_mask]
    preds_pneu = preds[pneu_mask]
    # In etiology, how many bacterial were predicted bacterial? How many viral predicted viral?
    bact_cases = (targets_pneu == 0)
    viral_cases = (targets_pneu == 2)
    bact_to_viral = np.sum((targets_pneu == 0) & (preds_pneu == 2))
    viral_to_bact = np.sum((targets_pneu == 2) & (preds_pneu == 0))
    
    # Patient-level aggregation using TRUE patient IDs
    df_p = pd.DataFrame({
        "patient": val_patients,
        "target": val_targets,
        "p0": probs[:, 0],
        "p1": probs[:, 1],
        "p2": probs[:, 2]
    })
    grouped = df_p.groupby("patient").agg({"target": "first", "p0": "mean", "p1": "mean", "p2": "mean"})
    pat_y = grouped["target"].values
    pat_pred = grouped[["p0", "p1", "p2"]].values.argmax(axis=1)
    pat_acc = accuracy_score(pat_y, pat_pred)
    pat_f1 = f1_score(pat_y, pat_pred, average="macro", zero_division=0)
    
    all_model_results.append({
        "name": name,
        "seed": seed,
        "loss_u": loss_u,
        "loss_w": loss_w,
        "acc": acc,
        "macro_f1": f1_m,
        "pat_acc": pat_acc,
        "pat_f1": pat_f1,
        "bact_rec": recs[0],
        "norm_rec": recs[1],
        "viral_rec": recs[2],
        "bact_f1": f1s[0],
        "norm_f1": f1s[1],
        "viral_f1": f1s[2],
        "pneu_sens": pneu_sens,
        "pneu_spec": pneu_spec,
        "false_normals": pneu_fn,
        "false_pneu": pneu_fp,
        "bact_to_viral": bact_to_viral,
        "viral_to_bact": viral_to_bact,
        "cm": cm
    })

df_all = pd.DataFrame(all_model_results)
print("\n=== SUMMARY TABLE OF ALL CHECKPOINTS ===")
for r in all_model_results:
    print(f"\nModel: {r['name']}")
    print(f"  Loss (Unweighted): {r['loss_u']:.4f} | Loss (Weighted): {r['loss_w']:.4f}")
    print(f"  Image Acc: {r['acc']:.2%} | Image Macro F1: {r['macro_f1']:.2%}")
    print(f"  Patient Acc: {r['pat_acc']:.2%} | Patient Macro F1: {r['pat_f1']:.2%}")
    print(f"  Bacterial Rec: {r['bact_rec']:.2%}, F1: {r['bact_f1']:.2%}")
    print(f"  Normal Rec:    {r['norm_rec']:.2%}, F1: {r['norm_f1']:.2%}")
    print(f"  Viral Rec:     {r['viral_rec']:.2%}, F1: {r['viral_f1']:.2%}")
    print(f"  Pneumonia Screening -> Sens: {r['pneu_sens']:.2%}, Spec: {r['pneu_spec']:.2%}, False-Normals (Missed): {r['false_normals']}/655")
    print(f"  Etiology Confusion  -> Bact predicted Viral: {r['bact_to_viral']}/404, Viral predicted Bact: {r['viral_to_bact']}/251")
    print(f"  Confusion Matrix:\n{r['cm']}")
