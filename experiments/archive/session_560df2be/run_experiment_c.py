"""
Experiment C: Frozen DenseNet-121 vs. Frozen MobileNetV2 (Control)
Evaluates DenseNet-121 across Seeds 42, 123, 456 on internal validation cohort (N=925, 714 patients).
Compares with existing MobileNetV2 Control checkpoints using Paired Patient-Clustered Bootstrap (B=2,000).
"""

import os
import sys
import copy
import json
import time
import random
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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def build_densenet_model(device):
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    for p in model.features.parameters():
        p.requires_grad = False
    in_feat = model.classifier.in_features
    # Align head configuration with MobileNetV2: Dropout(0.2) + Linear(in_feat, 3)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_feat, len(CLASS_NAMES))
    )
    return model.to(device)

def build_mobilenet_model(device):
    model = models.mobilenet_v2(weights=None)
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, len(CLASS_NAMES))
    return model.to(device)

def evaluate_model_full(model, loader, device, weighted_loss_fn, unweighted_loss_fn, targets_arr):
    model.eval()
    total_w_loss, total_u_loss, total_samples = 0.0, 0.0, 0
    all_logits = []
    
    with torch.no_grad():
        for x, y in loader:
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
    
    acc = accuracy_score(targets_arr, preds)
    macro_f1 = f1_score(targets_arr, preds, average="macro", zero_division=0)
    macro_rec = recall_score(targets_arr, preds, average="macro", zero_division=0)
    recs = recall_score(targets_arr, preds, average=None, zero_division=0)
    precs = precision_score(targets_arr, preds, average=None, zero_division=0)
    f1s = f1_score(targets_arr, preds, average=None, zero_division=0)
    cm = confusion_matrix(targets_arr, preds)
    
    return {
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

def train_densenet_seed(seed, train_loader, val_loader, device, weighted_loss_fn, unweighted_loss_fn, targets_arr):
    print(f"\n>>> Starting DenseNet-121 Training for Seed {seed}...")
    set_seed(seed)
    
    model = build_densenet_model(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=0.0)
    
    epochs = 12
    patience = 3
    best_val_w_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    
    start_t = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        model.features.eval()  # Keep frozen backbone feature statistics consistent
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = weighted_loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += len(y)
            
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation evaluation
        val_eval = evaluate_model_full(model, val_loader, device, weighted_loss_fn, unweighted_loss_fn, targets_arr)
        val_w_loss = val_eval["weighted_loss"]
        val_u_loss = val_eval["unweighted_loss"]
        val_acc = val_eval["accuracy"]
        val_f1 = val_eval["macro_f1"]
        
        print(f"[DenseNet121|s{seed}] Ep {ep:2d}/{epochs:2d} -> Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | Val W-Loss: {val_w_loss:.4f}, U-Loss: {val_u_loss:.4f}, Acc: {val_acc:.2%}, Macro F1: {val_f1:.2%}")
        
        if val_w_loss < best_val_w_loss:
            best_val_w_loss = val_w_loss
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[DenseNet121|s{seed}] Early stopping at epoch {ep} (patience={patience}).")
                break
                
    elapsed = time.time() - start_t
    print(f"[DenseNet121|s{seed}] Finished in {elapsed:.1f}s. Best Epoch: {best_epoch} (Val W-Loss: {best_val_w_loss:.4f})")
    
    # Reload best model and evaluate
    model.load_state_dict(best_state)
    final_val = evaluate_model_full(model, val_loader, device, weighted_loss_fn, unweighted_loss_fn, targets_arr)
    final_val["best_epoch"] = best_epoch
    final_val["training_time_sec"] = elapsed
    
    # Save checkpoint
    ckpt_name = f"exp_c_densenet_s{seed}.pt"
    ckpt_path = ARTIFACTS_DIR / ckpt_name
    torch.save({
        "architecture": "densenet121",
        "pretrained_weights": "IMAGENET1K_V1",
        "seed": seed,
        "best_epoch": best_epoch,
        "training_time_sec": elapsed,
        "state_dict": best_state,
        "final_val_metrics": {k: v for k, v in final_val.items() if k not in ["probs", "preds"]}
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")
    
    return final_val

def main():
    print("=" * 75)
    print("EXPERIMENT C: FROZEN DENSENET-121 VS. FROZEN MOBILENETV2 (CONTROL)")
    print("=" * 75)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Datasets
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=standard_transform)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=standard_transform)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    val_targets = np.array(val_ds.targets)
    val_paths = [s[0] for s in val_ds.samples]
    val_patients = [get_patient_id(p) for p in val_paths]
    unique_patients = np.unique(val_patients)
    
    pat_to_indices = {}
    for idx, p in enumerate(val_patients):
        pat_to_indices.setdefault(p, []).append(idx)
        
    print(f"Validation samples: {len(val_ds)}, Unique patients: {len(unique_patients)}")
    
    # Class weights (identical to Exp A/B)
    counts = np.bincount(train_ds.targets, minlength=len(CLASS_NAMES))
    weights = len(train_ds) / (len(CLASS_NAMES) * counts.astype(float))
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Train class counts: {counts}, Weights: {weights.round(4)}")
    
    weighted_loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    unweighted_loss_fn = nn.CrossEntropyLoss()
    
    # 1. Load MobileNetV2 Control Results
    print("\n--- Loading Existing Validated MobileNetV2 Control Checkpoints ---")
    control_results = {}
    for s in SEEDS:
        ckpt_path = ARTIFACTS_DIR / f"exp_a_control_s{s}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Control checkpoint missing: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        m = build_mobilenet_model(device)
        m.load_state_dict(ckpt["state_dict"])
        m_eval = evaluate_model_full(m, val_loader, device, weighted_loss_fn, unweighted_loss_fn, val_targets)
        m_eval["best_epoch"] = ckpt.get("best_epoch", "N/A")
        control_results[s] = m_eval
        print(f"Control s{s}: Best Ep={m_eval['best_epoch']}, Val W-Loss={m_eval['weighted_loss']:.4f}, U-Loss={m_eval['unweighted_loss']:.4f}, Acc={m_eval['accuracy']:.2%}, Macro F1={m_eval['macro_f1']:.2%}, Viral Rec={m_eval['viral_recall']:.2%}")
        
    # 2. Train DenseNet-121 Models
    print("\n--- Training DenseNet-121 Models Across 3 Seeds ---")
    densenet_results = {}
    for s in SEEDS:
        densenet_results[s] = train_densenet_seed(s, train_loader, val_loader, device, weighted_loss_fn, unweighted_loss_fn, val_targets)
        
    # 3. Compute Ensembles
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
        
        # Loss for ensemble
        w_l = weighted_loss_fn(torch.tensor(np.log(np.clip(probs, 1e-12, 1.0))), torch.tensor(val_targets).to(device)).item()
        u_l = unweighted_loss_fn(torch.tensor(np.log(np.clip(probs, 1e-12, 1.0))), torch.tensor(val_targets).to(device)).item()
        
        return {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "macro_recall": float(macro_rec),
            "bacterial_recall": float(recs[0]),
            "normal_recall": float(recs[1]),
            "viral_recall": float(recs[2]),
            "bacterial_f1": float(f1s[0]),
            "normal_f1": float(f1s[1]),
            "viral_f1": float(f1s[2]),
            "weighted_loss": float(w_l),
            "unweighted_loss": float(u_l),
            "confusion_matrix": cm.tolist()
        }
        
    control_ens_metrics = get_ensemble_metrics(control_ens_preds, control_ens_probs)
    dense_ens_metrics = get_ensemble_metrics(dense_ens_preds, dense_ens_probs)
    
    # 4. Paired Patient-Clustered Bootstrap (B = 2,000)
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
        diff = np.mean(arr)
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        p_val = 2.0 * min(np.mean(np.array(arr) <= 0), np.mean(np.array(arr) >= 0))
        p_val = min(p_val, 1.0)
        return {
            "mean_diff": float(diff),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "p_value": float(p_val)
        }
        
    bootstrap_results = {}
    for group_k in ["Ensemble", "Seed42", "Seed123", "Seed456"]:
        bootstrap_results[group_k] = {
            m: compute_stats(boot_diffs[group_k][m]) for m in ["macro_f1", "acc", "rec_bact", "rec_norm", "rec_viral"]
        }
        
    print("\n" + "=" * 75)
    print("EXPERIMENT C VALIDATION RESULTS (DenseNet-121 vs. MobileNetV2)")
    print("=" * 75)
    print(f"Control Ensemble Macro F1: {control_ens_metrics['macro_f1']:.2%}")
    print(f"DenseNet Ensemble Macro F1: {dense_ens_metrics['macro_f1']:.2%}")
    ens_f1_stat = bootstrap_results['Ensemble']['macro_f1']
    print(f"Ensemble Delta Macro F1: {ens_f1_stat['mean_diff']:+.2%} [95% CI: {ens_f1_stat['ci_95_low']:+.2%}, {ens_f1_stat['ci_95_high']:+.2%}], p = {ens_f1_stat['p_value']:.4f}")
    
    ens_vir_stat = bootstrap_results['Ensemble']['rec_viral']
    print(f"Control Ensemble Viral Recall: {control_ens_metrics['viral_recall']:.2%}")
    print(f"DenseNet Ensemble Viral Recall: {dense_ens_metrics['viral_recall']:.2%}")
    print(f"Ensemble Delta Viral Recall: {ens_vir_stat['mean_diff']:+.2%} [95% CI: {ens_vir_stat['ci_95_low']:+.2%}, {ens_vir_stat['ci_95_high']:+.2%}], p = {ens_vir_stat['p_value']:.4f}")
    
    ens_norm_stat = bootstrap_results['Ensemble']['rec_norm']
    print(f"Control Ensemble Normal Recall: {control_ens_metrics['normal_recall']:.2%}")
    print(f"DenseNet Ensemble Normal Recall: {dense_ens_metrics['normal_recall']:.2%}")
    print(f"Ensemble Delta Normal Recall: {ens_norm_stat['mean_diff']:+.2%} [95% CI: {ens_norm_stat['ci_95_low']:+.2%}, {ens_norm_stat['ci_95_high']:+.2%}], p = {ens_norm_stat['p_value']:.4f}")
    
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
        "control": {
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

if __name__ == '__main__':
    main()
