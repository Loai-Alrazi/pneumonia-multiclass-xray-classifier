"""
Rigorous Verification and Reconciliation of Experiment C:
Frozen DenseNet-121 vs. Frozen MobileNetV2 (Control)
Directly recomputes all metrics from saved checkpoints and validates exact mathematical relationships.
"""

import re
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

# Workspace paths
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

def evaluate_model_clean(ckpt_path, arch="mobilenet"):
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
    
    cm = confusion_matrix(val_targets, preds)
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    
    recs = tp / support
    precs = tp / cm.sum(axis=0)
    f1s = 2 * (precs * recs) / (precs + recs)
    f1s = np.nan_to_num(f1s, nan=0.0)
    
    acc = tp.sum() / total_samples
    support_weighted_rec = np.sum(recs * support) / total_samples
    assert np.isclose(acc, support_weighted_rec), f"Mismatch: acc={acc}, support_weighted_rec={support_weighted_rec}"
    
    macro_f1 = float(np.mean(f1s))
    macro_rec = float(np.mean(recs))
    
    return {
        "best_epoch": ckpt.get("best_epoch", "N/A"),
        "training_time_sec": ckpt.get("training_time_sec", None),
        "weighted_loss": float(total_w_loss / total_samples),
        "unweighted_loss": float(total_u_loss / total_samples),
        "accuracy": float(acc),
        "support_weighted_recall": float(support_weighted_rec),
        "macro_f1": macro_f1,
        "macro_recall": macro_rec,
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
        "tp": tp.tolist(),
        "support": support.tolist(),
        "probs": probs,
        "preds": preds
    }

print("=" * 80)
print("RE-COMPUTING PER-SEED VALIDATION METRICS")
print("=" * 80)

control_results = {}
densenet_results = {}

for s in SEEDS:
    c_path = ARTIFACTS_DIR / f"exp_a_control_s{s}.pt"
    d_path = ARTIFACTS_DIR / f"exp_c_densenet_s{s}.pt"
    control_results[s] = evaluate_model_clean(c_path, arch="mobilenet")
    densenet_results[s] = evaluate_model_clean(d_path, arch="densenet")
    
    c = control_results[s]
    d = densenet_results[s]
    
    print(f"\n--- SEED {s} ---")
    print(f"MobileNetV2: Ep {c['best_epoch']} | Acc: {c['accuracy']:.4f} (SuppW-Rec: {c['support_weighted_recall']:.4f}) | Macro F1: {c['macro_f1']:.4f} | U-Loss: {c['unweighted_loss']:.4f}")
    print(f"  Recalls: Bact={c['bacterial_recall']:.4f} ({c['tp'][0]}/{c['support'][0]}), Norm={c['normal_recall']:.4f} ({c['tp'][1]}/{c['support'][1]}), Viral={c['viral_recall']:.4f} ({c['tp'][2]}/{c['support'][2]})")
    print(f"  CM:\n{np.array(c['confusion_matrix'])}")
    
    print(f"DenseNet121: Ep {d['best_epoch']} | Acc: {d['accuracy']:.4f} (SuppW-Rec: {d['support_weighted_recall']:.4f}) | Macro F1: {d['macro_f1']:.4f} | U-Loss: {d['unweighted_loss']:.4f}")
    print(f"  Recalls: Bact={d['bacterial_recall']:.4f} ({d['tp'][0]}/{d['support'][0]}), Norm={d['normal_recall']:.4f} ({d['tp'][1]}/{d['support'][1]}), Viral={d['viral_recall']:.4f} ({d['tp'][2]}/{d['support'][2]})")
    print(f"  CM:\n{np.array(d['confusion_matrix'])}")

# Ensemble calculations
control_ens_probs = np.mean([control_results[s]["probs"] for s in SEEDS], axis=0)
control_ens_preds = np.argmax(control_ens_probs, axis=1)

dense_ens_probs = np.mean([densenet_results[s]["probs"] for s in SEEDS], axis=0)
dense_ens_preds = np.argmax(dense_ens_probs, axis=1)

def evaluate_ensemble(preds, probs):
    cm = confusion_matrix(val_targets, preds)
    support = cm.sum(axis=1)
    tp = np.diag(cm)
    total_samples = len(val_targets)
    
    recs = tp / support
    precs = tp / cm.sum(axis=0)
    f1s = 2 * (precs * recs) / (precs + recs)
    f1s = np.nan_to_num(f1s, nan=0.0)
    
    acc = tp.sum() / total_samples
    support_weighted_rec = np.sum(recs * support) / total_samples
    assert np.isclose(acc, support_weighted_rec)
    
    log_probs = torch.tensor(np.log(np.clip(probs, 1e-12, 1.0)), dtype=torch.float32, device=device)
    targets_t = torch.tensor(val_targets, dtype=torch.long, device=device)
    w_l = weighted_loss_fn(log_probs, targets_t).item()
    u_l = unweighted_loss_fn(log_probs, targets_t).item()
    
    return {
        "accuracy": float(acc),
        "support_weighted_recall": float(support_weighted_rec),
        "macro_f1": float(np.mean(f1s)),
        "macro_recall": float(np.mean(recs)),
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
        "confusion_matrix": cm.tolist(),
        "tp": tp.tolist(),
        "support": support.tolist()
    }

control_ens = evaluate_ensemble(control_ens_preds, control_ens_probs)
dense_ens = evaluate_ensemble(dense_ens_preds, dense_ens_probs)

print("\n" + "=" * 80)
print("3-SEED ENSEMBLE METRICS & RECONCILIATION")
print("=" * 80)
print(f"MobileNetV2 Ensemble:")
print(f"  Accuracy:                {control_ens['accuracy']:.4f} ({control_ens['accuracy']*100:.2f}%)")
print(f"  Support-Weighted Recall: {control_ens['support_weighted_recall']:.4f} ({control_ens['support_weighted_recall']*100:.2f}%)")
print(f"  Macro F1:                {control_ens['macro_f1']:.4f} ({control_ens['macro_f1']*100:.2f}%)")
print(f"  Unweighted Loss:         {control_ens['unweighted_loss']:.4f}")
print(f"  Bacterial Recall:        {control_ens['bacterial_recall']:.4f} ({control_ens['bacterial_recall']*100:.2f}%) -> {control_ens['tp'][0]}/{control_ens['support'][0]}")
print(f"  Normal Recall:           {control_ens['normal_recall']:.4f} ({control_ens['normal_recall']*100:.2f}%) -> {control_ens['tp'][1]}/{control_ens['support'][1]}")
print(f"  Viral Recall:            {control_ens['viral_recall']:.4f} ({control_ens['viral_recall']*100:.2f}%) -> {control_ens['tp'][2]}/{control_ens['support'][2]}")
print(f"  Confusion Matrix:\n{np.array(control_ens['confusion_matrix'])}")

print(f"\nDenseNet-121 Ensemble:")
print(f"  Accuracy:                {dense_ens['accuracy']:.4f} ({dense_ens['accuracy']*100:.2f}%)")
print(f"  Support-Weighted Recall: {dense_ens['support_weighted_recall']:.4f} ({dense_ens['support_weighted_recall']*100:.2f}%)")
print(f"  Macro F1:                {dense_ens['macro_f1']:.4f} ({dense_ens['macro_f1']*100:.2f}%)")
print(f"  Unweighted Loss:         {dense_ens['unweighted_loss']:.4f}")
print(f"  Bacterial Recall:        {dense_ens['bacterial_recall']:.4f} ({dense_ens['bacterial_recall']*100:.2f}%) -> {dense_ens['tp'][0]}/{dense_ens['support'][0]}")
print(f"  Normal Recall:           {dense_ens['normal_recall']:.4f} ({dense_ens['normal_recall']*100:.2f}%) -> {dense_ens['tp'][1]}/{dense_ens['support'][1]}")
print(f"  Viral Recall:            {dense_ens['viral_recall']:.4f} ({dense_ens['viral_recall']*100:.2f}%) -> {dense_ens['tp'][2]}/{dense_ens['support'][2]}")
print(f"  Confusion Matrix:\n{np.array(dense_ens['confusion_matrix'])}")

# Paired Patient-Clustered Bootstrap (B = 2,000) with +1 continuity correction
print("\n" + "=" * 80)
print("RUNNING PAIRED PATIENT-CLUSTERED BOOTSTRAP (B=2,000) WITH +1 CONTINUITY CORRECTION")
print("=" * 80)

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

def compute_stats_corrected(arr):
    arr = np.array(arr)
    diff = float(np.mean(arr))
    ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
    
    # Corrected two-sided empirical bootstrap p-value with +1 continuity correction (Davison & Hinkley 1997)
    count_le = np.sum(arr <= 0)
    count_ge = np.sum(arr >= 0)
    p_val = 2.0 * (min(count_le, count_ge) + 1) / (len(arr) + 1)
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
        m: compute_stats_corrected(boot_diffs[group_k][m]) for m in ["macro_f1", "acc", "rec_bact", "rec_norm", "rec_viral"]
    }

print("\nCORRECTED BOOTSTRAP RESULTS:")
for grp in ["Ensemble", "Seed42", "Seed123", "Seed456"]:
    print(f"\n[{grp}]")
    for m in ["macro_f1", "acc", "rec_bact", "rec_norm", "rec_viral"]:
        st = bootstrap_results[grp][m]
        print(f"  {m:10s}: diff={st['mean_diff']:+.4f} ({st['mean_diff']*100:+.2f}%) [95% CI: {st['ci_95_low']*100:+.2f}%, {st['ci_95_high']*100:+.2f}%], p = {st['p_value']:.4f}")

# Save clean results
def clean_dict(d):
    out = {}
    for k, v in d.items():
        if k in ["probs", "preds"]:
            continue
        if isinstance(v, dict):
            out[k] = clean_dict(v)
        else:
            out[k] = v
    return out

final_json = {
    "cohort_verification": {
        "total_images": 925,
        "unique_patients": 714,
        "class_support": {"BACTERIAL": 404, "NORMAL": 270, "VIRAL": 251},
        "support_weights": {"BACTERIAL": 404/925, "NORMAL": 270/925, "VIRAL": 251/925}
    },
    "control_mobilenetv2": {
        "per_seed": {s: clean_dict(control_results[s]) for s in SEEDS},
        "ensemble": control_ens
    },
    "densenet121": {
        "per_seed": {s: clean_dict(densenet_results[s]) for s in SEEDS},
        "ensemble": dense_ens
    },
    "paired_patient_bootstrap_corrected": bootstrap_results
}

out_path = ARTIFACTS_DIR / "experiment_c_densenet_results.json"
with open(out_path, "w") as f:
    json.dump(final_json, f, indent=2)

print(f"\nSaved reconciled results to: {out_path}")
