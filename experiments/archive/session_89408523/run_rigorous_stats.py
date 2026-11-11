import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, recall_score
from scipy import stats

# Load all predictions and targets from comprehensive_clinical_audit
from comprehensive_clinical_audit import (
    val_targets, val_patients, unique_patients, all_preds_dict, all_probs_dict
)

print("=== RIGOROUS STATISTICAL EVALUATION WITH TRUE PATIENT CLUSTERS ===")

# Create mapping from patient to image indices
pat_to_indices = {}
for idx, p in enumerate(val_patients):
    pat_to_indices.setdefault(p, []).append(idx)

# Models to compare: we will evaluate the multi-seed average predictions (soft voting across seeds)
# as well as individual seeds
models_to_test = ["Baseline (Orig)", "Exp 1", "Exp 2", "Exp 5"]

# Compute ensemble/average probabilities across seeds for Exp 1, Exp 2, Exp 5
ensemble_probs = {}
ensemble_preds = {}

for m in ["Exp 1", "Exp 2", "Exp 5"]:
    p_seeds = [all_probs_dict[f"{m} (s{s})"] for s in [42, 123, 456]]
    avg_p = np.mean(p_seeds, axis=0)
    ensemble_probs[m] = avg_p
    ensemble_preds[m] = avg_p.argmax(axis=1)

ensemble_probs["Baseline (Orig)"] = all_probs_dict["Baseline (Orig)"]
ensemble_preds["Baseline (Orig)"] = all_preds_dict["Baseline (Orig)"]

# Patient-clustered Bootstrap (B = 2,000 resamples)
np.random.seed(42)
B = 2000
boot_metrics = {m: {"acc": [], "macro_f1": [], "viral_rec": [], "norm_rec": []} for m in models_to_test}
boot_diffs = {
    "Exp5_vs_Exp1": {"macro_f1": [], "acc": []},
    "Exp5_vs_Exp2": {"macro_f1": [], "acc": []},
    "Exp2_vs_Exp1": {"macro_f1": [], "acc": [], "viral_rec": []}
}

for b_idx in range(B):
    sampled_pats = np.random.choice(unique_patients, size=len(unique_patients), replace=True)
    idx_list = []
    for p in sampled_pats:
        idx_list.extend(pat_to_indices[p])
    idx_arr = np.array(idx_list)
    
    y_true_b = val_targets[idx_arr]
    
    cur_b = {}
    for m in models_to_test:
        y_pred_b = ensemble_preds[m][idx_arr]
        acc_b = accuracy_score(y_true_b, y_pred_b)
        f1_b = f1_score(y_true_b, y_pred_b, average="macro", zero_division=0)
        recs = recall_score(y_true_b, y_pred_b, average=None, zero_division=0)
        boot_metrics[m]["acc"].append(acc_b)
        boot_metrics[m]["macro_f1"].append(f1_b)
        boot_metrics[m]["viral_rec"].append(recs[2])
        boot_metrics[m]["norm_rec"].append(recs[1])
        cur_b[m] = {"acc": acc_b, "f1": f1_b, "viral_rec": recs[2]}
        
    boot_diffs["Exp5_vs_Exp1"]["macro_f1"].append(cur_b["Exp 5"]["f1"] - cur_b["Exp 1"]["f1"])
    boot_diffs["Exp5_vs_Exp1"]["acc"].append(cur_b["Exp 5"]["acc"] - cur_b["Exp 1"]["acc"])
    
    boot_diffs["Exp5_vs_Exp2"]["macro_f1"].append(cur_b["Exp 5"]["f1"] - cur_b["Exp 2"]["f1"])
    boot_diffs["Exp5_vs_Exp2"]["acc"].append(cur_b["Exp 5"]["acc"] - cur_b["Exp 2"]["acc"])
    
    boot_diffs["Exp2_vs_Exp1"]["macro_f1"].append(cur_b["Exp 2"]["f1"] - cur_b["Exp 1"]["f1"])
    boot_diffs["Exp2_vs_Exp1"]["acc"].append(cur_b["Exp 2"]["acc"] - cur_b["Exp 1"]["acc"])
    boot_diffs["Exp2_vs_Exp1"]["viral_rec"].append(cur_b["Exp 2"]["viral_rec"] - cur_b["Exp 1"]["viral_rec"])

print("\n=== PATIENT-CLUSTERED 95% BOOTSTRAP CONFIDENCE INTERVALS (B = 2,000) ===")
for m in models_to_test:
    acc_ci = np.percentile(boot_metrics[m]["acc"], [2.5, 97.5])
    f1_ci = np.percentile(boot_metrics[m]["macro_f1"], [2.5, 97.5])
    v_ci = np.percentile(boot_metrics[m]["viral_rec"], [2.5, 97.5])
    n_ci = np.percentile(boot_metrics[m]["norm_rec"], [2.5, 97.5])
    print(f"\n{m}:")
    print(f"  Accuracy:  {np.mean(boot_metrics[m]['acc']):.2%}  [95% CI: {acc_ci[0]:.2%}, {acc_ci[1]:.2%}]")
    print(f"  Macro F1:  {np.mean(boot_metrics[m]['macro_f1']):.2%}  [95% CI: {f1_ci[0]:.2%}, {f1_ci[1]:.2%}]")
    print(f"  Viral Rec: {np.mean(boot_metrics[m]['viral_rec']):.2%}  [95% CI: {v_ci[0]:.2%}, {v_ci[1]:.2%}]")
    print(f"  Norm Rec:  {np.mean(boot_metrics[m]['norm_rec']):.2%}  [95% CI: {n_ci[0]:.2%}, {n_ci[1]:.2%}]")

print("\n=== PAIRED DIFFERENCE BOOTSTRAP WITH EXACT EMPIRICAL RESOLUTION ===")
# Note: With B=2000, if count is 0, p-value is strictly bounded as p < 1/B = 0.0005 (or smoothed (1)/(B+1) = 0.0005)
for comp, diff_dict in boot_diffs.items():
    print(f"\nComparison: {comp}")
    for metric, vals in diff_dict.items():
        v_arr = np.array(vals)
        mean_d = np.mean(v_arr)
        ci = np.percentile(v_arr, [2.5, 97.5])
        # Two-sided empirical p-value with continuity correction
        count_le_zero = np.sum(v_arr <= 0)
        count_ge_zero = np.sum(v_arr >= 0)
        p_raw = 2.0 * min((count_le_zero + 1) / (B + 1), (count_ge_zero + 1) / (B + 1))
        p_raw = min(p_raw, 1.0)
        print(f"  Delta {metric:9s}: {mean_d:+.2%}  [95% CI: {ci[0]:+.2%}, {ci[1]:+.2%}] | Empirical p={p_raw:.4f} (le_0 count={count_le_zero}/{B})")

# Multiple Comparison Correction (Holm-Bonferroni across the 3 primary Macro F1 tests)
print("\n=== MULTIPLE TESTING CORRECTION (HOLM-BONFERRONI) ON MACRO F1 ===")
p_f1_comp = [
    ("Exp5 vs Exp1", 2.0 * (np.sum(np.array(boot_diffs["Exp5_vs_Exp1"]["macro_f1"]) <= 0) + 1) / (B + 1)),
    ("Exp5 vs Exp2", 2.0 * (np.sum(np.array(boot_diffs["Exp5_vs_Exp2"]["macro_f1"]) <= 0) + 1) / (B + 1)),
    ("Exp2 vs Exp1", 2.0 * (np.sum(np.array(boot_diffs["Exp2_vs_Exp1"]["macro_f1"]) <= 0) + 1) / (B + 1)),
]
# Sort by p-value ascending
p_f1_sorted = sorted(p_f1_comp, key=lambda x: x[1])
m_tests = len(p_f1_sorted)
print(f"Total hypothesis tests (m = {m_tests}):")
for rank, (name, p_val) in enumerate(p_f1_sorted, start=1):
    alpha_holm = 0.05 / (m_tests - rank + 1)
    is_sig = p_val < alpha_holm
    print(f"  Rank {rank}: {name:15s} | Raw p = {p_val:.4f} | Holm Alpha = {alpha_holm:.4f} | Significant? {is_sig}")
