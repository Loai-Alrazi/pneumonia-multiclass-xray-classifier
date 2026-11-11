import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import re

# Paths
WORKSPACE = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\pneumonia-multiclass-xray-classifier")
DATA_DIR = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")
ARTIFACTS_DIR = WORKSPACE / "artifacts"

CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
SEEDS = [42, 123, 456]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
    def __call__(self, img_pil):
        gray = np.array(img_pil.convert("L"))
        enhanced = self.clahe.apply(gray)
        return Image.fromarray(np.stack([enhanced, enhanced, enhanced], axis=-1))

control_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

clahe_transform = transforms.Compose([
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

# DataLoaders for Validation
val_control_ds = datasets.ImageFolder(DATA_DIR / "val", transform=control_transform)
val_clahe_ds = datasets.ImageFolder(DATA_DIR / "val", transform=clahe_transform)

val_control_loader = DataLoader(val_control_ds, batch_size=32, shuffle=False, num_workers=0)
val_clahe_loader = DataLoader(val_clahe_ds, batch_size=32, shuffle=False, num_workers=0)

val_paths = [s[0] for s in val_control_ds.samples]
val_patients = [get_true_patient_id(p) for p in val_paths]
unique_patients = np.unique(val_patients)
val_targets = np.array(val_control_ds.targets)

pat_to_indices = {}
for idx, p in enumerate(val_patients):
    pat_to_indices.setdefault(p, []).append(idx)

# Class weights
raw_train = datasets.ImageFolder(DATA_DIR / "train")
counts = np.bincount(raw_train.targets, minlength=len(CLASS_NAMES))
weights = len(raw_train) / (len(CLASS_NAMES) * counts.astype(float))
class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

w_loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
u_loss_fn = nn.CrossEntropyLoss()

def eval_ckpt(ckpt_path, loader):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = models.mobilenet_v2(weights=None)
    model.classifier[-1] = nn.Linear(model.last_channel, 3)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    
    total_w_loss, total_u_loss, total_samples = 0.0, 0.0, 0
    all_logits = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            wl = w_loss_fn(logits, y)
            ul = u_loss_fn(logits, y)
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
        "weighted_loss": total_w_loss / total_samples,
        "unweighted_loss": total_u_loss / total_samples,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_recall": float(macro_rec),
        "bacterial_recall": float(recs[0]),
        "normal_recall": float(recs[1]),
        "viral_recall": float(recs[2]),
        "bacterial_f1": float(f1s[0]),
        "normal_f1": float(f1s[1]),
        "viral_f1": float(f1s[2]),
        "confusion_matrix": cm.tolist(),
        "probs": probs,
        "preds": preds
    }

results = {"Control": {}, "CLAHE": {}}

for s in SEEDS:
    results["Control"][s] = eval_ckpt(ARTIFACTS_DIR / f"exp_a_control_s{s}.pt", val_control_loader)
    results["CLAHE"][s] = eval_ckpt(ARTIFACTS_DIR / f"exp_a_clahe_s{s}.pt", val_clahe_loader)

# Ensemble probabilities
control_ens_probs = np.mean([results["Control"][s]["probs"] for s in SEEDS], axis=0)
clahe_ens_probs = np.mean([results["CLAHE"][s]["probs"] for s in SEEDS], axis=0)

control_ens_preds = np.argmax(control_ens_probs, axis=1)
clahe_ens_preds = np.argmax(clahe_ens_probs, axis=1)

# Paired Patient-Clustered Bootstrap (B = 2,000)
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
    
    # Ensemble
    p_c_ens = control_ens_preds[boot_idx]
    p_l_ens = clahe_ens_preds[boot_idx]
    
    f1_c = f1_score(y_b, p_c_ens, average="macro", zero_division=0)
    f1_l = f1_score(y_b, p_l_ens, average="macro", zero_division=0)
    acc_c = accuracy_score(y_b, p_c_ens)
    acc_l = accuracy_score(y_b, p_l_ens)
    rec_c = recall_score(y_b, p_c_ens, average=None, zero_division=0)
    rec_l = recall_score(y_b, p_l_ens, average=None, zero_division=0)
    
    boot_diffs["Ensemble"]["macro_f1"].append(f1_l - f1_c)
    boot_diffs["Ensemble"]["acc"].append(acc_l - acc_c)
    boot_diffs["Ensemble"]["rec_bact"].append(rec_l[0] - rec_c[0])
    boot_diffs["Ensemble"]["rec_norm"].append(rec_l[1] - rec_c[1])
    boot_diffs["Ensemble"]["rec_viral"].append(rec_l[2] - rec_c[2])
    
    # Per-seed
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
        
        boot_diffs[k]["macro_f1"].append(f1_ls - f1_cs)
        boot_diffs[k]["acc"].append(acc_ls - acc_cs)
        boot_diffs[k]["rec_bact"].append(rec_ls[0] - rec_cs[0])
        boot_diffs[k]["rec_norm"].append(rec_ls[1] - rec_cs[1])
        boot_diffs[k]["rec_viral"].append(rec_ls[2] - rec_cs[2])

print("\n" + "=" * 70)
print("EXPERIMENT A: CONTROL VS. CLAHE ON VALIDATION COHORT (N=925, 714 PATIENTS)")
print("=" * 70)

for cond in ["Control", "CLAHE"]:
    f1s = [results[cond][s]["macro_f1"] for s in SEEDS]
    accs = [results[cond][s]["accuracy"] for s in SEEDS]
    u_losses = [results[cond][s]["unweighted_loss"] for s in SEEDS]
    w_losses = [results[cond][s]["weighted_loss"] for s in SEEDS]
    b_recs = [results[cond][s]["bacterial_recall"] for s in SEEDS]
    n_recs = [results[cond][s]["normal_recall"] for s in SEEDS]
    v_recs = [results[cond][s]["viral_recall"] for s in SEEDS]
    
    ens_preds = control_ens_preds if cond == "Control" else clahe_ens_preds
    ens_f1 = f1_score(val_targets, ens_preds, average="macro")
    ens_acc = accuracy_score(val_targets, ens_preds)
    ens_recs = recall_score(val_targets, ens_preds, average=None)
    
    print(f"\n>>> Condition: {cond.upper()}")
    for s in SEEDS:
        print(f"  Seed {s:3d} (Best Ep {results[cond][s]['best_epoch']}): Macro F1 = {results[cond][s]['macro_f1']:.2%}, Acc = {results[cond][s]['accuracy']:.2%}, U-Loss = {results[cond][s]['unweighted_loss']:.4f} | Bact={results[cond][s]['bacterial_recall']:.2%}, Norm={results[cond][s]['normal_recall']:.2%}, Viral={results[cond][s]['viral_recall']:.2%}")
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
    cm_ens = confusion_matrix(val_targets, ens_preds)
    for r in cm_ens:
        print(f"      {r.tolist()}")

print("\n" + "=" * 70)
print("PAIRED PATIENT-CLUSTERED BOOTSTRAP (B=2,000 RESAMPLES): CLAHE - CONTROL")
print("=" * 70)

for comp_key in ["Ensemble", "Seed42", "Seed123", "Seed456"]:
    print(f"\n--- {comp_key} Paired Difference (CLAHE vs. Control) ---")
    for metric in ["macro_f1", "acc", "rec_bact", "rec_norm", "rec_viral"]:
        vals = np.array(boot_diffs[comp_key][metric])
        mean_d = np.mean(vals)
        ci = np.percentile(vals, [2.5, 97.5])
        le_0 = np.sum(vals <= 0)
        ge_0 = np.sum(vals >= 0)
        p_val = 2.0 * min((le_0 + 1) / (B + 1), (ge_0 + 1) / (B + 1))
        p_val = min(p_val, 1.0)
        print(f"  Delta {metric:12s}: {mean_d:+.2%}  [95% CI: {ci[0]:+.2%}, {ci[1]:+.2%}] | Two-sided p = {p_val:.4f}")

# Save clean final JSON artifact
final_artifact = {
    "experiment": "Experiment A (Control vs. CLAHE)",
    "dataset": "Kermany 2018 Pediatric CXR (Validation Set, N=925, 714 patients)",
    "seeds": SEEDS,
    "control": {
        "single_seeds": {s: {k: v for k, v in results["Control"][s].items() if k not in ["probs", "preds"]} for s in SEEDS},
        "mean_f1": float(np.mean([results["Control"][s]["macro_f1"] for s in SEEDS])),
        "std_f1": float(np.std([results["Control"][s]["macro_f1"] for s in SEEDS])),
        "mean_u_loss": float(np.mean([results["Control"][s]["unweighted_loss"] for s in SEEDS])),
    },
    "clahe": {
        "single_seeds": {s: {k: v for k, v in results["CLAHE"][s].items() if k not in ["probs", "preds"]} for s in SEEDS},
        "mean_f1": float(np.mean([results["CLAHE"][s]["macro_f1"] for s in SEEDS])),
        "std_f1": float(np.std([results["CLAHE"][s]["macro_f1"] for s in SEEDS])),
        "mean_u_loss": float(np.mean([results["CLAHE"][s]["unweighted_loss"] for s in SEEDS])),
    },
    "paired_bootstrap_ensemble": {
        metric: {
            "mean_delta": float(np.mean(boot_diffs["Ensemble"][metric])),
            "ci_95": [float(np.percentile(boot_diffs["Ensemble"][metric], 2.5)), float(np.percentile(boot_diffs["Ensemble"][metric], 97.5))],
            "p_value": float(2.0 * min((np.sum(np.array(boot_diffs["Ensemble"][metric]) <= 0) + 1) / (B + 1), (np.sum(np.array(boot_diffs["Ensemble"][metric]) >= 0) + 1) / (B + 1)))
        } for metric in ["macro_f1", "acc", "rec_bact", "rec_norm", "rec_viral"]
    }
}

out_path = ARTIFACTS_DIR / "experiment_a_validated_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(final_artifact, f, indent=2)
print(f"\nFinal validated artifact saved to: {out_path}")
