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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

# Set local torch cache
TORCH_HUB = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\artifacts\torch")
torch.hub.set_dir(str(TORCH_HUB))

DATA_DIR = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")
CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Load validation dataset with paths to extract patient IDs
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# Extract patient IDs for validation set
def extract_patient_id(path_str):
    stem = Path(path_str).stem
    m = re.search(r'(person\d+|NORMAL2-IM-\d+|IM-\d+)', stem, re.IGNORECASE)
    return m.group(1) if m else stem

val_paths = [s[0] for s in val_dataset.samples]
val_patients = [extract_patient_id(p) for p in val_paths]
val_targets = np.array(val_dataset.targets)

# Compute standard class weights from training set
train_dataset = datasets.ImageFolder(DATA_DIR / "train")
train_counts = np.bincount(train_dataset.targets, minlength=len(CLASS_NAMES))
train_total = len(train_dataset)
weights = train_total / (len(CLASS_NAMES) * train_counts.astype(float))
class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

criterion_unweighted = nn.CrossEntropyLoss().to(device)
criterion_weighted = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)

print(f"Validation dataset size: {len(val_dataset)} images, {len(set(val_patients))} unique patients")
print(f"Class weights (derived from train): {dict(zip(CLASS_NAMES, weights.round(3).tolist()))}")

checkpoints_to_eval = [
    ("Baseline (Orig)", "artifacts/pneumonia_3class_baseline_backup.pt", None),
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

results = []
all_preds_dict = {}

for name, rel_path, seed in checkpoints_to_eval:
    ckpt_path = Path(rel_path)
    if not ckpt_path.exists():
        print(f"Warning: {ckpt_path} missing")
        continue
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Instantiate MobileNetV2
    model = models.mobilenet_v2(weights=None)
    model.classifier[-1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    
    total_unweighted_loss = 0.0
    total_weighted_loss = 0.0
    preds_list = []
    probs_list = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_u = criterion_unweighted(logits, y)
            loss_w = criterion_weighted(logits, y)
            total_unweighted_loss += loss_u.item() * len(y)
            total_weighted_loss += loss_w.item() * len(y)
            
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            preds_list.extend(preds)
            probs_list.extend(probs)
            
    preds_arr = np.array(preds_list)
    probs_arr = np.array(probs_list)
    all_preds_dict[name] = preds_arr
    
    mean_unweighted_loss = total_unweighted_loss / len(val_dataset)
    mean_weighted_loss = total_weighted_loss / len(val_dataset)
    
    acc = accuracy_score(val_targets, preds_arr)
    macro_f1 = f1_score(val_targets, preds_arr, average="macro", zero_division=0)
    macro_recall = recall_score(val_targets, preds_arr, average="macro", zero_division=0)
    per_class_rec = recall_score(val_targets, preds_arr, average=None, zero_division=0)
    per_class_f1 = f1_score(val_targets, preds_arr, average=None, zero_division=0)
    per_class_prec = precision_score(val_targets, preds_arr, average=None, zero_division=0)
    cm = confusion_matrix(val_targets, preds_arr)
    
    # Patient-level aggregation: average predicted probabilities per patient
    df_patient = pd.DataFrame({
        "patient": val_patients,
        "target": val_targets,
        "p0": probs_arr[:, 0],
        "p1": probs_arr[:, 1],
        "p2": probs_arr[:, 2]
    })
    grouped = df_patient.groupby("patient").agg({
        "target": "first",
        "p0": "mean",
        "p1": "mean",
        "p2": "mean"
    })
    pat_targets = grouped["target"].values
    pat_probs = grouped[["p0", "p1", "p2"]].values
    pat_preds = pat_probs.argmax(axis=1)
    pat_acc = accuracy_score(pat_targets, pat_preds)
    pat_macro_f1 = f1_score(pat_targets, pat_preds, average="macro", zero_division=0)
    
    results.append({
        "name": name,
        "seed": seed,
        "val_loss_unweighted": mean_unweighted_loss,
        "val_loss_weighted": mean_weighted_loss,
        "img_acc": acc,
        "img_macro_f1": macro_f1,
        "img_macro_recall": macro_recall,
        "pat_acc": pat_acc,
        "pat_macro_f1": pat_macro_f1,
        "bact_recall": per_class_rec[0],
        "norm_recall": per_class_rec[1],
        "viral_recall": per_class_rec[2],
        "bact_f1": per_class_f1[0],
        "norm_f1": per_class_f1[1],
        "viral_f1": per_class_f1[2],
        "confusion_matrix": cm.tolist()
    })

df_res = pd.DataFrame(results)
print("\n================ DETAILED AUDITED METRICS ================")
cols_to_show = ["name", "val_loss_unweighted", "val_loss_weighted", "img_acc", "img_macro_f1", "pat_acc", "pat_macro_f1", "viral_recall", "norm_recall"]
print(df_res[cols_to_show].to_string(index=False))

# Grouped summaries for Exp 1, Exp 2, Exp 5
print("\n================ GROUPED MEANS +/- STD ================")
for prefix in ["Exp 1", "Exp 2", "Exp 5"]:
    sub = df_res[df_res["name"].str.startswith(prefix)]
    print(f"\n{prefix}:")
    print(f"  Unweighted Val Loss: {sub['val_loss_unweighted'].mean():.4f} +/- {sub['val_loss_unweighted'].std():.4f}")
    print(f"  Weighted Val Loss:   {sub['val_loss_weighted'].mean():.4f} +/- {sub['val_loss_weighted'].std():.4f}")
    print(f"  Image Macro F1:      {sub['img_macro_f1'].mean()*100:.2f}% +/- {sub['img_macro_f1'].std()*100:.2f}%")
    print(f"  Image Accuracy:      {sub['img_acc'].mean()*100:.2f}% +/- {sub['img_acc'].std()*100:.2f}%")
    print(f"  Patient Macro F1:    {sub['pat_macro_f1'].mean()*100:.2f}% +/- {sub['pat_macro_f1'].std()*100:.2f}%")
    print(f"  Patient Accuracy:    {sub['pat_acc'].mean()*100:.2f}% +/- {sub['pat_acc'].std()*100:.2f}%")
    print(f"  Viral Recall:        {sub['viral_recall'].mean()*100:.2f}% +/- {sub['viral_recall'].std()*100:.2f}%")
    print(f"  Normal Recall:       {sub['norm_recall'].mean()*100:.2f}% +/- {sub['norm_recall'].std()*100:.2f}%")
