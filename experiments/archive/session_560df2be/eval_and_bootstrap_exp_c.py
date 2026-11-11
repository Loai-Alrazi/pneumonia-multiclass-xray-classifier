"""
Evaluation and Paired Patient-Clustered Bootstrap for Experiment C:
Frozen DenseNet-121 vs. Frozen MobileNetV2 (Control)
Loads saved checkpoints and computes full metrics and statistical bootstrap.
"""

import os
import json
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

# Paths
WORKSPACE = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\pneumonia-multiclass-xray-classifier")
DATA_DIR = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")
TORCH_HUB = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\artifacts\torch")
torch.hub.set_dir(str(TORCH_HUB))
ARTIFACTS_DIR = WORKSPACE / "artifacts"

CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
SEEDS = [42, 123, 456]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Evaluation device: {device}")

def get_patient_id(path_str):
    stem = Path(path_str).stem
    m_bact = re.search(r"(person\d+_bacteria)", stem)
    if m_bact:
        return m_bact.group(1)
    m_vir = re.search(r"(person\d+_virus)", stem)
    if m_vir:
        return m_vir.group(1)
    parts = stem.split("-")
    if len(parts) >= 2:
        return "-".join(parts[:-1])
    return stem

standard_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Load Validation Dataset
val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=standard_transform)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
val_targets = np.array(val_ds.targets)
val_paths = [s[0] for s in val_ds.samples]
val_patients = [get_patient_id(p) for p in val_paths]
unique_patients = np.unique(val_patients)

pat_to_indices = {}
for idx, p in enumerate(val_patients):
    pat_to_indices.setdefault(p, []).append(idx)

# Loss functions with identical train class weights
train_ds = datasets.ImageFolder(DATA_DIR / "train")
counts = np.bincount(train_ds.targets, minlength=len(CLASS_NAMES))
weights = len(train_ds) / (len(CLASS_NAMES) * counts.astype(float))
class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

weighted_loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
unweighted_loss_fn = nn.CrossEntropyLoss()

def build_mobilenet_model():
    model = models.mobilenet_v2(weights=None)
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, len(CLASS_NAMES))
    return model.to(device)

def build_densenet_model():
    model = models.densenet121(weights=None)
    in_feat = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_feat, len(CLASS_NAMES))
    )
    return model.to(device)

def eval_checkpoint(ckpt_path, arch="mobilenet"):
    ckpt = torch.load(ckpt_path, map_location=device)
    if arch == "mobilenet":
        model = build_mobilenet_model()
    else:
        model = build_densenet_model()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    
    total_w_loss, total_u_loss, total_samples = 0.0, 0.0, 0
    all_logits = []
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            wl = weighted_loss_fn(logits, y)
            ul = unweighted_loss_fn(logits, y)
            total_w_loss += wl.item() * len(y)
            total_u_loss += ul.item() * len(y)
            all_logits.append(logits.cpu().numpy())
            total_samples += len(y)
            
    all_logits = np.concatenate(all_logits, axis=0)
    probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    
    acc = accuracy_score(val_targets, preds)
    macro_f1 = f1_score(val_targets, preds, average="macro", zero_division=0)
    macro_rec = recall_score(val_targets, preds, average="macro", zero_division=0)
    recs = recall_score(val_targets, preds, average=None, zero_division=0)
    precs = precision_score(val_targets, preds, average=None, zero_division=0)
    f1s = f1_score(val_targets, preds, average=None, zero_division=0)
    cm = confusion_matrix(val_targets, preds)
    
    return {
        "best_epoch": ckpt.get("best_epoch", "N/A"),
        "training_time_sec": ckpt.get("training_time_sec", None),
        "weighted_loss": float(total_w_loss / total_samples),
        "unweighted_loss": float(total_u_loss / total_samples),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_recall": float(macro_rec),
        "bacterial_recall": float(recs[0]),
        "normal_recall": float(recs[1]),
        "viral_recall": float(recs[2]),
        "bacterial_precision": float(precs[0]),
        "normal_precision": float(precs[1]),
        "viral_precision": float(precs[2]),
        "bacterial_f1": float(f1s[0]),
        "normal_f1": float(f1s[1]),
        "viral_f1": float(f1s[2]),
        "confusion_matrix": cm.tolist(),
        "probs": probs,
        "preds": preds
    }

print("\n--- Evaluating Models on Validation Set ---")
control_results = {}
densenet_results = {}

for s in SEEDS:
    c_path = ARTIFACTS_DIR / f"exp_a_control_s{s}.pt"
    d_path = ARTIFACTS_DIR / f"exp_c_densenet_s{s}.pt"
    control_results[s] = eval_checkpoint(c_path, arch="mobilenet")
    densenet_results[s] = eval_checkpoint(d_path, arch="densenet")
    print(f"Seed {s}:")
    print(f"  Control:  Best Ep={control_results[s]['best_epoch']}, Macro F1={control_results[s]['macro_f1']:.2%}, Acc={control_results[s]['accuracy']:.2%}, Normal Rec={control_results[s]['normal_recall']:.2%}, Viral Rec={control_results[s]['viral_recall']:.2%}, U-Loss={control_results[s]['unweighted_loss']:.4f}")
    print(f"  DenseNet: Best Ep={densenet_results[s]['best_epoch']}, Macro F1={densenet_results[s]['macro_f1']:.2%}, Acc={densenet_results[s]['accuracy']:.2%}, Normal Rec={densenet_results[s]['normal_recall']:.2%}, Viral Rec={densenet_results[s]['viral_recall']:.2%}, U-Loss={densenet_results[s]['unweighted_loss']:.4f}")

# Compute Ensembles
control_ens_probs = np.mean([control_results[s]["probs"] for s in SEEDS], axis=0)
control_ens_preds = np.argmax(control_ens_probs, axis=1)

dense_ens_probs = np.mean([densenet_results[s]["probs"] for s in SEEDS], axis=0)
dense_ens_preds = np.argmax(dense_ens_probs, axis=1)

def get_ensemble_metrics(preds, probs):
    acc = accuracy_score(val_targets, preds)
    macro_f1 = f1_score(val_targets, preds, average="macro", zero_division=0)
    macro_rec = recall_score(val_targets, preds, average="macro", zero_division=0)
    recs = recall_score(val_targets, preds, average=None, zero_division=0)
    precs = precision_score(val_targets, preds, average=None, zero_division=0)
    f1s = f1_score(val_targets, preds, average=None, zero_division=0)
    cm = confusion_matrix(val_targets, preds)
    
    log_probs = torch.tensor(np.log(np.clip(probs, 1e-12, 1.0)), dtype=torch.float32, device=device)
    targets_t = torch.tensor(val_targets, dtype=torch.long, device=device)
    w_l = weighted_loss_fn(log_probs, targets_t).item()
    u_l = unweighted_loss_fn(log_probs, targets_t).item()
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_recall": float(macro_rec),
        "bacterial_recall": float(recs[0]),
        "normal_recall": float(recs[1]),
        "viral_recall": float(recs[2]),
        "bacterial_precision": float(precs[0]),
        "normal_precision": float(precs[1]),
        "viral_precision": float(precs[2]),
        "bacterial_f1": float(f1s[0]),
        "normal_f1": float(f1s[1]),
        "viral_f1": float(f1s[2]),
        "weighted_loss": float(w_l),
        "unweighted_loss": float(u_l),
        "confusion_matrix": cm.tolist()
    }

control_ens_metrics = get_ensemble_metrics(control_ens_preds, control_ens_probs)
dense_ens_metrics = get_ensemble_metrics(dense_ens_preds, dense_ens_probs)

print("\n--- Ensemble Metrics ---")
print(f"Control Ensemble:  Macro F1={control_ens_metrics['macro_f1']:.2%}, Acc={control_ens_metrics['accuracy']:.2%}, Normal Rec={control_ens_metrics['normal_recall']:.2%}, Viral Rec={control_ens_metrics['viral_recall']:.2%}, U-Loss={control_ens_metrics['unweighted_loss']:.4f}")
print(f"DenseNet Ensemble: Macro F1={dense_ens_metrics['macro_f1']:.2%}, Acc={dense_ens_metrics['accuracy']:.2%}, Normal Rec={dense_ens_metrics['normal_recall']:.2%}, Viral Rec={dense_ens_metrics['viral_recall']:.2%}, U-Loss={dense_ens_metrics['unweighted_loss']:.4f}")

# Paired Patient-Clustered Bootstrap (B = 2,000)
print("\n--- Running Paired Patient-Clustered Bootstrap (B=2,000) ---")
np.random.seed(42)
B = 2000

boot_diffs = {
    "Ensemble": {"macro_f1": [], "acc": [], "rec_bact": [], "rec_norm": [], "rec_viral": []},
    "Seed42": {"macro_f1": [], "acc": [], "rec_bact": [], "rec_norm": [], "rec_viral": []},
    "Seed123": {"macro_f1": [], "acc": [], "rec_bact": [], "rec_norm": [], "rec_viral": []},
    "Seed456": {"macro_f1": [], "acc": [], "rec_bact": [], "rec_norm": [], "rec_viral": []},
}

for b_idx in range(B):
    boot_pats = np.random.choice(unique_patients, size=len(unique_patients), replace=True)
    boot_idx = []
    for p in boot_pats:
        boot_idx.extend(pat_to_indices[p])
    boot_idx = np.array(boot_idx)
    
    y_b = val_targets[boot_idx]
    
    # Ensemble diff (DenseNet - Control)
    p_c_ens = control_ens_preds[boot_idx]
    p_d_ens = dense_ens_preds[boot_idx]
    
    f1_c = f1_score(y_b, p_c_ens, average="macro", zero_division=0)
    f1_d = f1_score(y_b, p_d_ens, average="macro", zero_division=0)
    acc_c = accuracy_score(y_b, p_c_ens)
    acc_d = accuracy_score(y_b, p_d_ens)
    rec_c = recall_score(y_b, p_c_ens, average=None, zero_division=0)
    rec_d = recall_score(y_b, p_d_ens, average=None, zero_division=0)
    
    boot_diffs["Ensemble"]["macro_f1"].append(f1_d - f1_c)
    boot_diffs["Ensemble"]["acc"].append(acc_d - acc_c)
    boot_diffs["Ensemble"]["rec_bact"].append(rec_d[0] - rec_c[0])
    boot_diffs["Ensemble"]["rec_norm"].append(rec_d[1] - rec_c[1])
    boot_diffs["Ensemble"]["rec_viral"].append(rec_d[2] - rec_c[2])
    
    for s in SEEDS:
        k = f"Seed{s}"
        p_c = control_results[s]["preds"][boot_idx]
        p_d = densenet_results[s]["preds"][boot_idx]
        
        f1_cs = f1_score(y_b, p_c, average="macro", zero_division=0)
        f1_ds = f1_score(y_b, p_d, average="macro", zero_division=0)
        acc_cs = accuracy_score(y_b, p_c)
        acc_ds = accuracy_score(y_b, p_d)
        rec_cs = recall_score(y_b, p_c, average=None, zero_division=0)
        rec_ds = recall_score(y_b, p_d, average=None, zero_division=0)
        
        boot_diffs[k]["macro_f1"].append(f1_ds - f1_cs)
        boot_diffs[k]["acc"].append(acc_ds - acc_cs)
        boot_diffs[k]["rec_bact"].append(rec_ds[0] - rec_cs[0])
        boot_diffs[k]["rec_norm"].append(rec_ds[1] - rec_cs[1])
        boot_diffs[k]["rec_viral"].append(rec_ds[2] - rec_cs[2])

def compute_stats(arr):
    diff = float(np.mean(arr))
    ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
    p_val = 2.0 * min(np.mean(np.array(arr) <= 0), np.mean(np.array(arr) >= 0))
    p_val = min(float(p_val), 1.0)
    return {
        "mean_diff": diff,
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "p_value": p_val
    }

bootstrap_results = {}
for group_k in ["Ensemble", "Seed42", "Seed123", "Seed456"]:
    bootstrap_results[group_k] = {
        m: compute_stats(boot_diffs[group_k][m]) for m in ["macro_f1", "acc", "rec_bact", "rec_norm", "rec_viral"]
    }

print("\n" + "=" * 75)
print("EXPERIMENT C STATISTICAL SUMMARY (DenseNet-121 - MobileNetV2)")
print("=" * 75)
for grp in ["Ensemble", "Seed42", "Seed123", "Seed456"]:
    f1_st = bootstrap_results[grp]["macro_f1"]
    acc_st = bootstrap_results[grp]["acc"]
    vir_st = bootstrap_results[grp]["rec_viral"]
    norm_st = bootstrap_results[grp]["rec_norm"]
    bact_st = bootstrap_results[grp]["rec_bact"]
    print(f"\n[{grp}]")
    print(f"  Delta Macro F1:    {f1_st['mean_diff']:+.2%} [95% CI: {f1_st['ci_95_low']:+.2%}, {f1_st['ci_95_high']:+.2%}], p = {f1_st['p_value']:.4f}")
    print(f"  Delta Accuracy:    {acc_st['mean_diff']:+.2%} [95% CI: {acc_st['ci_95_low']:+.2%}, {acc_st['ci_95_high']:+.2%}], p = {acc_st['p_value']:.4f}")
    print(f"  Delta Viral Rec:   {vir_st['mean_diff']:+.2%} [95% CI: {vir_st['ci_95_low']:+.2%}, {vir_st['ci_95_high']:+.2%}], p = {vir_st['p_value']:.4f}")
    print(f"  Delta Normal Rec:  {norm_st['mean_diff']:+.2%} [95% CI: {norm_st['ci_95_low']:+.2%}, {norm_st['ci_95_high']:+.2%}], p = {norm_st['p_value']:.4f}")
    print(f"  Delta Bact Rec:    {bact_st['mean_diff']:+.2%} [95% CI: {bact_st['ci_95_low']:+.2%}, {bact_st['ci_95_high']:+.2%}], p = {bact_st['p_value']:.4f}")

# Save full results JSON
def serialize_dict(d):
    out = {}
    for k, v in d.items():
        if k in ["probs", "preds"]:
            continue
        if isinstance(v, dict):
            out[k] = serialize_dict(v)
        else:
            out[k] = v
    return out

full_output = {
    "control_mobilenetv2": {
        "per_seed": {s: serialize_dict(control_results[s]) for s in SEEDS},
        "ensemble": control_ens_metrics
    },
    "densenet121": {
        "per_seed": {s: serialize_dict(densenet_results[s]) for s in SEEDS},
        "ensemble": dense_ens_metrics
    },
    "paired_patient_bootstrap": bootstrap_results
}

out_file = ARTIFACTS_DIR / "experiment_c_densenet_results.json"
with open(out_file, "w") as f:
    json.dump(full_output, f, indent=2)
print(f"\nFull results saved to: {out_file}")
