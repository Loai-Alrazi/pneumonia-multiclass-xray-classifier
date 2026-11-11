"""Experiment A: Isolated Control vs. CLAHE Evaluation on Pediatric Chest X-Ray."""

import os
import sys
import copy
import json
import time
import random
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
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

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
    def __call__(self, img_pil):
        gray = np.array(img_pil.convert("L"))
        enhanced = self.clahe.apply(gray)
        return Image.fromarray(np.stack([enhanced, enhanced, enhanced], axis=-1))

def get_control_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def get_clahe_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8)),
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

def build_model(device):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    for p in model.features.parameters():
        p.requires_grad = False
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, len(CLASS_NAMES))
    return model.to(device)

def evaluate_model_full(model, loader, device, weighted_loss_fn, unweighted_loss_fn):
    model.eval()
    total_w_loss, total_u_loss, total_samples = 0.0, 0.0, 0
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            w_loss = weighted_loss_fn(logits, y)
            u_loss = unweighted_loss_fn(logits, y)
            total_w_loss += w_loss.item() * len(y)
            total_u_loss += u_loss.item() * len(y)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            total_samples += len(y)
            
    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_probs = torch.softmax(torch.tensor(all_logits), dim=1).numpy()
    all_preds = np.argmax(all_probs, axis=1)
    
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    per_class_rec = recall_score(all_targets, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    per_class_prec = precision_score(all_targets, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        "weighted_loss": total_w_loss / total_samples,
        "unweighted_loss": total_u_loss / total_samples,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_recall": float(macro_rec),
        "per_class": {
            cls: {
                "precision": float(per_class_prec[i]),
                "recall": float(per_class_rec[i]),
                "f1": float(per_class_f1[i])
            } for i, cls in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),
        "logits": all_logits,
        "probs": all_probs,
        "preds": all_preds,
        "targets": all_targets
    }

def train_single_run(condition_name, seed, transform_fn, device, class_weights_tensor):
    set_seed(seed)
    tf = transform_fn()
    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=tf)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=tf)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    
    weighted_loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
    unweighted_loss_fn = nn.CrossEntropyLoss()
    
    model = build_model(device)
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
        model.features.eval()
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
        val_eval = evaluate_model_full(model, val_loader, device, weighted_loss_fn, unweighted_loss_fn)
        val_w_loss = val_eval["weighted_loss"]
        val_u_loss = val_eval["unweighted_loss"]
        val_acc = val_eval["accuracy"]
        val_f1 = val_eval["macro_f1"]
        
        print(f"[{condition_name}|s{seed}] Ep {ep:2d}/{epochs:2d} -> Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | Val W-Loss: {val_w_loss:.4f}, U-Loss: {val_u_loss:.4f}, Acc: {val_acc:.2%}, Macro F1: {val_f1:.2%}")
        
        if val_w_loss < best_val_w_loss:
            best_val_w_loss = val_w_loss
            best_epoch = ep
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{condition_name}|s{seed}] Early stopping at epoch {ep} (patience={patience}).")
                break
                
    elapsed = time.time() - start_t
    print(f"[{condition_name}|s{seed}] Finished in {elapsed:.1f}s. Best Epoch: {best_epoch} (Val W-Loss: {best_val_w_loss:.4f})")
    
    # Reload best model and evaluate final metrics
    model.load_state_dict(best_state)
    final_val = evaluate_model_full(model, val_loader, device, weighted_loss_fn, unweighted_loss_fn)
    
    # Save checkpoint separately
    ckpt_name = f"exp_a_{condition_name.lower()}_s{seed}.pt"
    ckpt_path = ARTIFACTS_DIR / ckpt_name
    torch.save({
        "condition": condition_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "state_dict": best_state,
        "final_val_metrics": {k: v for k, v in final_val.items() if k not in ["logits", "probs", "preds", "targets"]}
    }, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")
    
    return final_val

def run_experiment_a():
    print("=" * 70)
    print("EXPERIMENT A: CONTROL VS. CLAHE (ISOLATED DEVELOPMENT RUN)")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # Compute inverse class weights strictly from training set
    raw_train = datasets.ImageFolder(DATA_DIR / "train")
    counts = np.bincount(raw_train.targets, minlength=len(CLASS_NAMES))
    total_samples = len(raw_train)
    weights = total_samples / (len(CLASS_NAMES) * counts.astype(float))
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    print(f"Class weights derived from Train set: {dict(zip(CLASS_NAMES, weights.round(3).tolist()))}")
    
    # Extract validation patient IDs for clustered bootstrap
    val_raw = datasets.ImageFolder(DATA_DIR / "val")
    val_paths = [s[0] for s in val_raw.samples]
    val_patients = [get_true_patient_id(p) for p in val_paths]
    unique_patients = np.unique(val_patients)
    val_targets = np.array(val_raw.targets)
    print(f"Validation cohort: {len(val_raw)} images across {len(unique_patients)} unique patient clusters.\n")
    
    pat_to_indices = {}
    for idx, p in enumerate(val_patients):
        pat_to_indices.setdefault(p, []).append(idx)
        
    results = {"Control": {}, "CLAHE": {}}
    
    # 1. Run Control (Seeds 42, 123, 456)
    for s in SEEDS:
        print(f"\n--- Running Control (Seed {s}) ---")
        val_res = train_single_run("Control", s, get_control_transform, device, class_weights_tensor)
        results["Control"][s] = val_res
        
    # 2. Run CLAHE (Seeds 42, 123, 456)
    for s in SEEDS:
        print(f"\n--- Running CLAHE (Seed {s}) ---")
        val_res = train_single_run("CLAHE", s, get_clahe_transform, device, class_weights_tensor)
        results["CLAHE"][s] = val_res
        
    print("\n" + "=" * 70)
    print("ALL RUNS COMPLETE. PERFORMING PAIRED PATIENT-CLUSTERED BOOTSTRAP (B=2,000)...")
    print("=" * 70)
    
    # Compute 3-seed ensemble probabilities
    control_ens_probs = np.mean([results["Control"][s]["probs"] for s in SEEDS], axis=0)
    clahe_ens_probs = np.mean([results["CLAHE"][s]["probs"] for s in SEEDS], axis=0)
    
    control_ens_preds = np.argmax(control_ens_probs, axis=1)
    clahe_ens_preds = np.argmax(clahe_ens_probs, axis=1)
    
    # Clustered Bootstrap
    np.random.seed(42)
    B = 2000
    
    diff_records = {
        "Ensemble": {"macro_f1": [], "u_loss": [], "acc": [], "rec_bact": [], "rec_norm": [], "rec_viral": []},
        "Seed42": {"macro_f1": [], "u_loss": [], "acc": [], "rec_bact": [], "rec_norm": [], "rec_viral": []},
        "Seed123": {"macro_f1": [], "u_loss": [], "acc": [], "rec_bact": [], "rec_norm": [], "rec_viral": []},
        "Seed456": {"macro_f1": [], "u_loss": [], "acc": [], "rec_bact": [], "rec_norm": [], "rec_viral": []},
    }
    
    for b_idx in range(B):
        boot_pats = np.random.choice(unique_patients, size=len(unique_patients), replace=True)
        boot_idx = []
        for p in boot_pats:
            boot_idx.extend(pat_to_indices[p])
        boot_idx = np.array(boot_idx)
        
        y_b = val_targets[boot_idx]
        
        # Ensemble diff
        p_ctrl_ens = control_ens_preds[boot_idx]
        p_clahe_ens = clahe_ens_preds[boot_idx]
        
        f1_c = f1_score(y_b, p_ctrl_ens, average="macro", zero_division=0)
        f1_l = f1_score(y_b, p_clahe_ens, average="macro", zero_division=0)
        acc_c = accuracy_score(y_b, p_ctrl_ens)
        acc_l = accuracy_score(y_b, p_clahe_ens)
        rec_c = recall_score(y_b, p_ctrl_ens, average=None, zero_division=0)
        rec_l = recall_score(y_b, p_clahe_ens, average=None, zero_division=0)
        
        diff_records["Ensemble"]["macro_f1"].append(f1_l - f1_c)
        diff_records["Ensemble"]["acc"].append(acc_l - acc_c)
        diff_records["Ensemble"]["rec_bact"].append(rec_l[0] - rec_c[0])
        diff_records["Ensemble"]["rec_norm"].append(rec_l[1] - rec_c[1])
        diff_records["Ensemble"]["rec_viral"].append(rec_l[2] - rec_c[2])
        
        # Per-seed diffs
        for s in SEEDS:
            k = f"Seed{s}"
            p_c = results["Control"][s]["preds"][boot_idx]
            p_l = results["CLAHE"][s]["preds"][boot_idx]
            
            f1_cs = f1_score(y_b, p_c, average="macro", zero_division=0)
            f1_ls = f1_score(y_b, p_l, average="macro", zero_division=0)
            acc_cs = accuracy_score(y_b, p_c)
            acc_ls = accuracy_score(y_b, p_l)
            rec_cs = recall_score(y_b, p_c, average=None, zero_division=0)
            rec_ls = recall_score(y_b, p_l, average=None, zero_division=0)
            
            diff_records[k]["macro_f1"].append(f1_ls - f1_cs)
            diff_records[k]["acc"].append(acc_ls - acc_cs)
            diff_records[k]["rec_bact"].append(rec_ls[0] - rec_cs[0])
            diff_records[k]["rec_norm"].append(rec_ls[1] - rec_cs[1])
            diff_records[k]["rec_viral"].append(rec_ls[2] - rec_cs[2])
            
    # Compile Summary
    print("\n" + "=" * 70)
    print("EXPERIMENTAL RESULTS SUMMARY: CONTROL VS. CLAHE ON VALIDATION SET")
    print("=" * 70)
    
    summary = {}
    for cond in ["Control", "CLAHE"]:
        f1s = [results[cond][s]["macro_f1"] for s in SEEDS]
        accs = [results[cond][s]["accuracy"] for s in SEEDS]
        u_losses = [results[cond][s]["unweighted_loss"] for s in SEEDS]
        w_losses = [results[cond][s]["weighted_loss"] for s in SEEDS]
        b_recs = [results[cond][s]["per_class"]["BACTERIAL"]["recall"] for s in SEEDS]
        n_recs = [results[cond][s]["per_class"]["NORMAL"]["recall"] for s in SEEDS]
        v_recs = [results[cond][s]["per_class"]["VIRAL"]["recall"] for s in SEEDS]
        
        # Ensemble metrics
        ens_preds = control_ens_preds if cond == "Control" else clahe_ens_preds
        ens_f1 = f1_score(val_targets, ens_preds, average="macro")
        ens_acc = accuracy_score(val_targets, ens_preds)
        ens_recs = recall_score(val_targets, ens_preds, average=None)
        
        summary[cond] = {
            "seeds": {
                s: {
                    "best_epoch": results[cond][s]["best_epoch"],
                    "macro_f1": float(results[cond][s]["macro_f1"]),
                    "accuracy": float(results[cond][s]["accuracy"]),
                    "unweighted_loss": float(results[cond][s]["unweighted_loss"]),
                    "weighted_loss": float(results[cond][s]["weighted_loss"]),
                    "bacterial_recall": float(results[cond][s]["per_class"]["BACTERIAL"]["recall"]),
                    "normal_recall": float(results[cond][s]["per_class"]["NORMAL"]["recall"]),
                    "viral_recall": float(results[cond][s]["per_class"]["VIRAL"]["recall"]),
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
            }
        }
        print(f"\n>>> {cond} Configuration:")
        print(f"  Single Seeds (Mean +/- Std): Macro F1 = {np.mean(f1s):.2%} +/- {np.std(f1s):.2%}, Acc = {np.mean(accs):.2%} +/- {np.std(accs):.2%}, U-Loss = {np.mean(u_losses):.4f}")
        print(f"    Bacterial Recall: {np.mean(b_recs):.2%} +/- {np.std(b_recs):.2%}")
        print(f"    Normal Recall:    {np.mean(n_recs):.2%} +/- {np.std(n_recs):.2%}")
        print(f"    Viral Recall:     {np.mean(v_recs):.2%} +/- {np.std(v_recs):.2%}")
        print(f"  3-Seed Ensemble: Macro F1 = {ens_f1:.2%}, Acc = {ens_acc:.2%}")
        print(f"    Bacterial Recall: {ens_recs[0]:.2%}, Normal Recall: {ens_recs[1]:.2%}, Viral Recall: {ens_recs[2]:.2%}")

    # Bootstrap Paired Comparisons
    print("\n" + "=" * 70)
    print("PAIRED PATIENT-CLUSTERED BOOTSTRAP INFERENCE (CLAHE vs. CONTROL, B=2,000)")
    print("=" * 70)
    
    bootstrap_results = {}
    for comp_key in ["Ensemble", "Seed42", "Seed123", "Seed456"]:
        print(f"\n--- Comparison: {comp_key} ---")
        bootstrap_results[comp_key] = {}
        for metric, vals in diff_records[comp_key].items():
            arr = np.array(vals)
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
            print(f"  Delta {metric:12s}: {mean_d:+.2%} [95% CI: {ci[0]:+.2%}, {ci[1]:+.2%}] | Two-sided p = {p_val:.4f}")
            
    out_json = ARTIFACTS_DIR / "experiment_a_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "bootstrap_results": bootstrap_results
        }, f, indent=2)
    print(f"\nFull results saved to: {out_json}")
    return summary, bootstrap_results

if __name__ == "__main__":
    run_experiment_a()
