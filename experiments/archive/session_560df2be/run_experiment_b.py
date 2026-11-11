"""Experiment B: Multi-Class Focal Loss (gamma=2) vs. Weighted Cross-Entropy Baseline."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

# Directories
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

def get_base_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def get_true_patient_id(path_str):
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

class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # 1D tensor of class weights
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        
        log_pt = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        pt = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        focal_term = (1.0 - pt).clamp(min=0.0) ** self.gamma
        loss = -focal_term * log_pt
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(dim=-1, index=targets)
            loss = loss * alpha_t
            
        if self.reduction == "mean":
            if self.alpha is not None:
                return loss.sum() / alpha_t.sum()
            else:
                return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def build_model(device):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    for p in model.features.parameters():
        p.requires_grad = False
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, len(CLASS_NAMES))
    return model.to(device)

def evaluate_model_full(model, loader, device, train_loss_fn, unweighted_loss_fn):
    model.eval()
    total_tr_loss, total_u_loss, total_samples = 0.0, 0.0, 0
    all_logits, all_targets = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            tl = train_loss_fn(logits, y)
            ul = unweighted_loss_fn(logits, y)
            total_tr_loss += tl.item() * len(y)
            total_u_loss += ul.item() * len(y)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            total_samples += len(y)
            
    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    
    acc = accuracy_score(all_targets, preds)
    macro_f1 = f1_score(all_targets, preds, average="macro", zero_division=0)
    macro_rec = recall_score(all_targets, preds, average="macro", zero_division=0)
    recs = recall_score(all_targets, preds, average=None, zero_division=0)
    precs = precision_score(all_targets, preds, average=None, zero_division=0)
    f1s = f1_score(all_targets, preds, average=None, zero_division=0)
    cm = confusion_matrix(all_targets, preds)
    
    return {
        "train_type_loss": total_tr_loss / total_samples,
        "unweighted_loss": total_u_loss / total_samples,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_recall": float(macro_rec),
        "per_class": {
            cls: {
                "precision": float(precs[i]),
                "recall": float(recs[i]),
                "f1": float(f1s[i])
            } for i, cls in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
        "probs": probs,
        "preds": preds
    }

def train_focal_single_run(seed, device, focal_loss_fn, unweighted_loss_fn, train_loader, val_loader):
    set_seed(seed)
    model = build_model(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=0.0)
    
    epochs = 12
    patience = 3
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0
    
    start_t = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        model.features.eval()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = focal_loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += len(y)
            
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation evaluation
        val_eval = evaluate_model_full(model, val_loader, device, focal_loss_fn, unweighted_loss_fn)
        val_fl_loss = val_eval["train_type_loss"]
        val_u_loss = val_eval["unweighted_loss"]
        val_acc = val_eval["accuracy"]
        val_f1 = val_eval["macro_f1"]
        
        print(f"[Focal-gamma2|s{seed}] Ep {ep:2d}/{epochs:2d} -> Train FL: {train_loss:.4f}, Acc: {train_acc:.2%} | Val FL: {val_fl_loss:.4f}, U-Loss: {val_u_loss:.4f}, Acc: {val_acc:.2%}, Macro F1: {val_f1:.2%}")
        
        if val_fl_loss < best_val_loss:
            best_val_loss = val_fl_loss
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[Focal-gamma2|s{seed}] Early stopping at epoch {ep} (patience={patience}).")
                break
                
    elapsed = time.time() - start_t
    print(f"[Focal-gamma2|s{seed}] Finished in {elapsed:.1f}s. Best Epoch: {best_epoch} (Val FL: {best_val_loss:.4f})")
    
    # Reload best model and evaluate final metrics
    model.load_state_dict(best_state)
    final_val = evaluate_model_full(model, val_loader, device, focal_loss_fn, unweighted_loss_fn)
    final_val["best_epoch"] = best_epoch
    
    # Save checkpoint separately
    ckpt_name = f"exp_b_focal_s{seed}.pt"
    ckpt_path = ARTIFACTS_DIR / ckpt_name
    torch.save({
        "condition": "Focal_gamma2",
        "seed": seed,
        "best_epoch": best_epoch,
        "state_dict": best_state,
        "final_val_metrics": {k: v for k, v in final_val.items() if k not in ["probs", "preds"]}
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")
    
    return final_val

def run_experiment_b():
    print("=" * 70)
    print("EXPERIMENT B: MULTI-CLASS FOCAL LOSS (GAMMA=2) VS. WEIGHTED CROSS-ENTROPY")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    transform = get_base_transform()
    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    
    # Class weights strictly from training set (identical to Exp A)
    counts = np.bincount(train_dataset.targets, minlength=len(CLASS_NAMES))
    total_samples = len(train_dataset)
    weights = total_samples / (len(CLASS_NAMES) * counts.astype(float))
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class weights derived from Train set: {dict(zip(CLASS_NAMES, weights.round(3).tolist()))}")
    
    focal_loss_fn = MultiClassFocalLoss(alpha=class_weights_tensor, gamma=2.0, reduction="mean")
    unweighted_ce_loss_fn = nn.CrossEntropyLoss()
    
    # Validation patient clusters for clustered bootstrap
    val_paths = [s[0] for s in val_dataset.samples]
    val_patients = [get_true_patient_id(p) for p in val_paths]
    unique_patients = np.unique(val_patients)
    val_targets = np.array(val_dataset.targets)
    print(f"Validation cohort: {len(val_dataset)} images across {len(unique_patients)} unique patient clusters.\n")
    
    pat_to_indices = {}
    for idx, p in enumerate(val_patients):
        pat_to_indices.setdefault(p, []).append(idx)
        
    focal_results = {}
    
    # Train 3 seeds of Focal Loss
    for s in SEEDS:
        print(f"\n--- Training Focal Loss gamma=2 (Seed {s}) ---")
        val_res = train_focal_single_run(s, device, focal_loss_fn, unweighted_ce_loss_fn, train_loader, val_loader)
        focal_results[s] = val_res
        
    # Load established Control baseline from Experiment A (exp_a_control_s*.pt)
    print("\n--- Loading Established Experiment A Control Checkpoints ---")
    control_results = {}
    weighted_ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    for s in SEEDS:
        ckpt_path = ARTIFACTS_DIR / f"exp_a_control_s{s}.pt"
        ckpt = torch.load(ckpt_path, map_location=device)
        m = build_model(device)
        m.load_state_dict(ckpt["state_dict"])
        eval_ctrl = evaluate_model_full(m, val_loader, device, weighted_ce_loss_fn, unweighted_ce_loss_fn)
        eval_ctrl["best_epoch"] = ckpt["best_epoch"]
        control_results[s] = eval_ctrl
        print(f"Loaded Control s{s} (Best Ep {ckpt['best_epoch']}): Macro F1 = {eval_ctrl['macro_f1']:.2%}, Acc = {eval_ctrl['accuracy']:.2%}, U-Loss = {eval_ctrl['unweighted_loss']:.4f}")
        
    # Ensembles
    focal_ens_probs = np.mean([focal_results[s]["probs"] for s in SEEDS], axis=0)
    control_ens_probs = np.mean([control_results[s]["probs"] for s in SEEDS], axis=0)
    
    focal_ens_preds = np.argmax(focal_ens_probs, axis=1)
    control_ens_preds = np.argmax(control_ens_probs, axis=1)
    
    # Paired Patient-Clustered Bootstrap (B = 2,000)
    print("\n" + "=" * 70)
    print("RUNNING PAIRED PATIENT-CLUSTERED BOOTSTRAP (B=2,000 RESAMPLES)...")
    print("=" * 70)
    
    np.random.seed(42)
    B = 2000
    
    diff_records = {
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
        
        # Ensemble diff
        p_c_ens = control_ens_preds[boot_idx]
        p_f_ens = focal_ens_preds[boot_idx]
        
        f1_c = f1_score(y_b, p_c_ens, average="macro", zero_division=0)
        f1_f = f1_score(y_b, p_f_ens, average="macro", zero_division=0)
        acc_c = accuracy_score(y_b, p_c_ens)
        acc_f = accuracy_score(y_b, p_f_ens)
        rec_c = recall_score(y_b, p_c_ens, average=None, zero_division=0)
        rec_f = recall_score(y_b, p_f_ens, average=None, zero_division=0)
        
        diff_records["Ensemble"]["macro_f1"].append(f1_f - f1_c)
        diff_records["Ensemble"]["acc"].append(acc_f - acc_c)
        diff_records["Ensemble"]["rec_bact"].append(rec_f[0] - rec_c[0])
        diff_records["Ensemble"]["rec_norm"].append(rec_f[1] - rec_c[1])
        diff_records["Ensemble"]["rec_viral"].append(rec_f[2] - rec_c[2])
        
        # Per-seed diffs
        for s in SEEDS:
            k = f"Seed{s}"
            p_c = control_results[s]["preds"][boot_idx]
            p_f = focal_results[s]["preds"][boot_idx]
            
            f1_cs = f1_score(y_b, p_c, average="macro", zero_division=0)
            f1_fs = f1_score(y_b, p_f, average="macro", zero_division=0)
            acc_cs = accuracy_score(y_b, p_c)
            acc_fs = accuracy_score(y_b, p_f)
            rec_cs = recall_score(y_b, p_c, average=None, zero_division=0)
            rec_fs = recall_score(y_b, p_f, average=None, zero_division=0)
            
            diff_records[k]["macro_f1"].append(f1_fs - f1_cs)
            diff_records[k]["acc"].append(acc_fs - acc_cs)
            diff_records[k]["rec_bact"].append(rec_fs[0] - rec_cs[0])
            diff_records[k]["rec_norm"].append(rec_fs[1] - rec_cs[1])
            diff_records[k]["rec_viral"].append(rec_fs[2] - rec_cs[2])
            
    print("\n" + "=" * 70)
    print("EXPERIMENT B: FOCAL LOSS (GAMMA=2) VS. CONTROL ON VALIDATION SET")
    print("=" * 70)
    
    # Compile summary
    summary = {}
    for cond_name, res_dict, ens_preds in [("Control_WeightedCE", control_results, control_ens_preds),
                                            ("FocalLoss_gamma2", focal_results, focal_ens_preds)]:
        f1s = [res_dict[s]["macro_f1"] for s in SEEDS]
        accs = [res_dict[s]["accuracy"] for s in SEEDS]
        u_losses = [res_dict[s]["unweighted_loss"] for s in SEEDS]
        b_recs = [res_dict[s]["per_class"]["BACTERIAL"]["recall"] for s in SEEDS]
        n_recs = [res_dict[s]["per_class"]["NORMAL"]["recall"] for s in SEEDS]
        v_recs = [res_dict[s]["per_class"]["VIRAL"]["recall"] for s in SEEDS]
        
        ens_f1 = f1_score(val_targets, ens_preds, average="macro")
        ens_acc = accuracy_score(val_targets, ens_preds)
        ens_recs = recall_score(val_targets, ens_preds, average=None)
        
        summary[cond_name] = {
            "seeds": {
                s: {
                    "best_epoch": res_dict[s]["best_epoch"],
                    "macro_f1": float(res_dict[s]["macro_f1"]),
                    "accuracy": float(res_dict[s]["accuracy"]),
                    "unweighted_loss": float(res_dict[s]["unweighted_loss"]),
                    "bacterial_recall": float(res_dict[s]["per_class"]["BACTERIAL"]["recall"]),
                    "normal_recall": float(res_dict[s]["per_class"]["NORMAL"]["recall"]),
                    "viral_recall": float(res_dict[s]["per_class"]["VIRAL"]["recall"]),
                } for s in SEEDS
            },
            "mean_std": {
                "macro_f1": f"{np.mean(f1s):.4f} +/- {np.std(f1s):.4f}",
                "accuracy": f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}",
                "unweighted_loss": f"{np.mean(u_losses):.4f} +/- {np.std(u_losses):.4f}",
                "bacterial_recall": f"{np.mean(b_recs):.4f} +/- {np.std(b_recs):.4f}",
                "normal_recall": f"{np.mean(n_recs):.4f} +/- {np.std(n_recs):.4f}",
                "viral_recall": f"{np.mean(v_recs):.4f} +/- {np.std(v_recs):.4f}",
            },
            "ensemble": {
                "macro_f1": float(ens_f1),
                "accuracy": float(ens_acc),
                "bacterial_recall": float(ens_recs[0]),
                "normal_recall": float(ens_recs[1]),
                "viral_recall": float(ens_recs[2]),
                "confusion_matrix": confusion_matrix(val_targets, ens_preds).tolist()
            }
        }
        print(f"\n>>> Condition: {cond_name}")
        for s in SEEDS:
            print(f"  Seed {s:3d} (Best Ep {res_dict[s]['best_epoch']}): Macro F1 = {res_dict[s]['macro_f1']:.2%}, Acc = {res_dict[s]['accuracy']:.2%}, U-Loss = {res_dict[s]['unweighted_loss']:.4f} | Bact={res_dict[s]['per_class']['BACTERIAL']['recall']:.2%}, Norm={res_dict[s]['per_class']['NORMAL']['recall']:.2%}, Viral={res_dict[s]['per_class']['VIRAL']['recall']:.2%}")
        print(f"  SINGLE SEEDS MEAN +/- STD:")
        print(f"    Macro F1:        {np.mean(f1s):.2%} +/- {np.std(f1s):.2%}")
        print(f"    Accuracy:        {np.mean(accs):.2%} +/- {np.std(accs):.2%}")
        print(f"    Unweighted Loss: {np.mean(u_losses):.4f} +/- {np.std(u_losses):.4f}")
        print(f"    Bacterial Rec:   {np.mean(b_recs):.2%} +/- {np.std(b_recs):.2%}")
        print(f"    Normal Recall:   {np.mean(n_recs):.2%} +/- {np.std(n_recs):.2%}")
        print(f"    Viral Recall:    {np.mean(v_recs):.2%} +/- {np.std(v_recs):.2%}")
        print(f"  3-SEED ENSEMBLE:")
        print(f"    Macro F1:        {ens_f1:.2%}")
        print(f"    Accuracy:        {ens_acc:.2%}")
        print(f"    Bacterial Rec:   {ens_recs[0]:.2%}")
        print(f"    Normal Recall:   {ens_recs[1]:.2%}")
        print(f"    Viral Recall:    {ens_recs[2]:.2%}")
        print(f"    Confusion Matrix [Bact, Norm, Viral]:")
        for r in confusion_matrix(val_targets, ens_preds):
            print(f"      {r.tolist()}")

    print("\n" + "=" * 70)
    print("PAIRED PATIENT-CLUSTERED BOOTSTRAP (B=2,000 RESAMPLES): FOCAL LOSS - CONTROL")
    print("=" * 70)
    
    bootstrap_results = {}
    for comp_key in ["Ensemble", "Seed42", "Seed123", "Seed456"]:
        print(f"\n--- {comp_key} Paired Difference (Focal Loss - Control) ---")
        bootstrap_results[comp_key] = {}
        for metric in ["macro_f1", "acc", "rec_bact", "rec_norm", "rec_viral"]:
            arr = np.array(diff_records[comp_key][metric])
            mean_d = float(np.mean(arr))
            ci = np.percentile(arr, [2.5, 97.5]).tolist()
            le_0 = int(np.sum(arr <= 0))
            ge_0 = int(np.sum(arr >= 0))
            p_val = float(2.0 * min((le_0 + 1) / (B + 1), (ge_0 + 1) / (B + 1)))
            p_val = min(p_val, 1.0)
            
            bootstrap_results[comp_key][metric] = {
                "mean_difference": mean_d,
                "ci_95": ci,
                "p_value": p_val
            }
            print(f"  Delta {metric:12s}: {mean_d:+.2%}  [95% CI: {ci[0]:+.2%}, {ci[1]:+.2%}] | Two-sided p = {p_val:.4f}")
            
    out_json = ARTIFACTS_DIR / "experiment_b_focal_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "bootstrap_results": bootstrap_results
        }, f, indent=2)
    print(f"\nExperiment B complete. Results saved to: {out_json}")

if __name__ == "__main__":
    run_experiment_b()
