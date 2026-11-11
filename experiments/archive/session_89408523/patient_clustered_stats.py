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
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from scipy import stats

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

val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

def extract_patient_id(path_str):
    stem = Path(path_str).stem
    m = re.search(r'(person\d+|NORMAL2-IM-\d+|IM-\d+)', stem, re.IGNORECASE)
    return m.group(1) if m else stem

val_paths = [s[0] for s in val_dataset.samples]
val_patients = np.array([extract_patient_id(p) for p in val_paths])
val_targets = np.array(val_dataset.targets)
unique_patients = np.unique(val_patients)

# Get predictions and probabilities for s42 of Exp 1, Exp 2, Exp 5, and Baseline
ckpts = {
    "Baseline": "artifacts/pneumonia_3class_baseline_backup.pt",
    "Exp 1": "artifacts/model_exp1_extended_head_s42.pt",
    "Exp 2": "artifacts/model_exp2_class_weighted_head_s42.pt",
    "Exp 5": "artifacts/model_exp5_finetune_top4blocks_s42.pt",
}

model_preds = {}
model_correct = {}

for name, path in ckpts.items():
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = models.mobilenet_v2(weights=None)
    model.classifier[-1] = nn.Linear(model.last_channel, 3)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    
    preds_list = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            preds_list.extend(preds)
    preds_arr = np.array(preds_list)
    model_preds[name] = preds_arr
    model_correct[name] = (preds_arr == val_targets)

# Patient-clustered Bootstrap for Macro F1 and Accuracy (1000 resamples of patients)
np.random.seed(42)
n_boot = 1000
boot_results = {name: {"acc": [], "macro_f1": [], "viral_rec": [], "norm_rec": []} for name in ckpts}
boot_diffs = {
    "Exp5_vs_Exp1": {"macro_f1": [], "acc": []},
    "Exp2_vs_Exp1": {"macro_f1": [], "acc": [], "viral_rec": []},
    "Exp5_vs_Exp2": {"macro_f1": [], "acc": []}
}

# Create mapping from patient to image indices
pat_to_indices = {}
for idx, p in enumerate(val_patients):
    pat_to_indices.setdefault(p, []).append(idx)

for _ in range(n_boot):
    sampled_patients = np.random.choice(unique_patients, size=len(unique_patients), replace=True)
    sampled_indices = []
    for p in sampled_patients:
        sampled_indices.extend(pat_to_indices[p])
    sampled_indices = np.array(sampled_indices)
    
    y_true_b = val_targets[sampled_indices]
    
    metrics_b = {}
    for name in ckpts:
        y_pred_b = model_preds[name][sampled_indices]
        acc_b = accuracy_score(y_true_b, y_pred_b)
        f1_b = f1_score(y_true_b, y_pred_b, average="macro", zero_division=0)
        recs = recall_score(y_true_b, y_pred_b, average=None, zero_division=0)
        boot_results[name]["acc"].append(acc_b)
        boot_results[name]["macro_f1"].append(f1_b)
        boot_results[name]["viral_rec"].append(recs[2])
        boot_results[name]["norm_rec"].append(recs[1])
        metrics_b[name] = {"acc": acc_b, "f1": f1_b, "viral_rec": recs[2]}
        
    boot_diffs["Exp5_vs_Exp1"]["macro_f1"].append(metrics_b["Exp 5"]["f1"] - metrics_b["Exp 1"]["f1"])
    boot_diffs["Exp5_vs_Exp1"]["acc"].append(metrics_b["Exp 5"]["acc"] - metrics_b["Exp 1"]["acc"])
    
    boot_diffs["Exp2_vs_Exp1"]["macro_f1"].append(metrics_b["Exp 2"]["f1"] - metrics_b["Exp 1"]["f1"])
    boot_diffs["Exp2_vs_Exp1"]["acc"].append(metrics_b["Exp 2"]["acc"] - metrics_b["Exp 1"]["acc"])
    boot_diffs["Exp2_vs_Exp1"]["viral_rec"].append(metrics_b["Exp 2"]["viral_rec"] - metrics_b["Exp 1"]["viral_rec"])

    boot_diffs["Exp5_vs_Exp2"]["macro_f1"].append(metrics_b["Exp 5"]["f1"] - metrics_b["Exp 2"]["f1"])
    boot_diffs["Exp5_vs_Exp2"]["acc"].append(metrics_b["Exp 5"]["acc"] - metrics_b["Exp 2"]["acc"])

print("================ PATIENT-CLUSTERED 95% BOOTSTRAP CONFIDENCE INTERVALS ================")
for name in ckpts:
    acc_ci = np.percentile(boot_results[name]["acc"], [2.5, 97.5])
    f1_ci = np.percentile(boot_results[name]["macro_f1"], [2.5, 97.5])
    vrec_ci = np.percentile(boot_results[name]["viral_rec"], [2.5, 97.5])
    nrec_ci = np.percentile(boot_results[name]["norm_rec"], [2.5, 97.5])
    print(f"\n{name}:")
    print(f"  Accuracy:  {np.mean(boot_results[name]['acc']):.2%}  (95% CI: [{acc_ci[0]:.2%}, {acc_ci[1]:.2%}])")
    print(f"  Macro F1:  {np.mean(boot_results[name]['macro_f1']):.2%}  (95% CI: [{f1_ci[0]:.2%}, {f1_ci[1]:.2%}])")
    print(f"  Viral Rec: {np.mean(boot_results[name]['viral_rec']):.2%}  (95% CI: [{vrec_ci[0]:.2%}, {vrec_ci[1]:.2%}])")
    print(f"  Norm Rec:  {np.mean(boot_results[name]['norm_rec']):.2%}  (95% CI: [{nrec_ci[0]:.2%}, {nrec_ci[1]:.2%}])")

print("\n================ PAIRED DIFFERENCE BOOTSTRAP 95% CI & P-VALUES ================")
for comp, diffs in boot_diffs.items():
    print(f"\nComparison: {comp}")
    for metric, d_vals in diffs.items():
        ci = np.percentile(d_vals, [2.5, 97.5])
        mean_d = np.mean(d_vals)
        p_val = 2 * min(np.mean(np.array(d_vals) <= 0), np.mean(np.array(d_vals) >= 0))
        print(f"  Delta {metric:9s}: {mean_d:+.2%}  (95% CI: [{ci[0]:+.2%}, {ci[1]:+.2%}], p={p_val:.4f})")

# McNemar test on image level
print("\n================ PAIRED MCNEMAR TESTS ================")
for pair in [("Exp 5", "Exp 1"), ("Exp 2", "Exp 1"), ("Exp 5", "Exp 2")]:
    m1, m2 = pair
    c1 = model_correct[m1]
    c2 = model_correct[m2]
    # Contingency table
    # n10: m1 correct, m2 wrong
    # n01: m1 wrong, m2 correct
    n10 = np.sum(c1 & ~c2)
    n01 = np.sum(~c1 & c2)
    # McNemar with continuity correction: (|b - c| - 1)^2 / (b + c)
    if (n10 + n01) > 0:
        stat = (abs(n10 - n01) - 1)**2 / (n10 + n01)
        pval = 1 - stats.chi2.cdf(stat, df=1)
    else:
        stat, pval = 0.0, 1.0
    print(f"{m1} vs {m2}: n10 (better)={n10}, n01 (worse)={n01}, Chi2={stat:.3f}, p={pval:.4e}")
