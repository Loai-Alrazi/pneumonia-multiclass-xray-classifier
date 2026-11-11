import torch
from pathlib import Path
import pandas as pd
import numpy as np

artifacts = Path("artifacts")

configs = [
    ("Exp 1 (Extended Head)", "model_exp1_extended_head_s{seed}.pt"),
    ("Exp 2 (Class-Weighted Head)", "model_exp2_class_weighted_head_s{seed}.pt"),
    ("Exp 5 (Fine-Tune Top 4 Blocks)", "model_exp5_finetune_top4blocks_s{seed}.pt")
]

seeds = [42, 123, 456]
rows = []

for cfg_label, pattern in configs:
    for seed in seeds:
        ckpt_path = artifacts / pattern.format(seed=seed)
        if not ckpt_path.exists():
            print(f"Warning: {ckpt_path} not found")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        vm = ckpt["val_metrics"]
        best_epoch = ckpt.get("best_epoch", "N/A")
        
        row = {
            "Config": cfg_label,
            "Seed": seed,
            "Best_Epoch": best_epoch,
            "Val_Loss": vm["loss"],
            "Val_Acc": vm["accuracy"],
            "Val_Macro_F1": vm["macro_f1"],
            "Val_Macro_Recall": vm["macro_recall"],
            "Bact_Recall": vm["per_class"]["BACTERIAL"]["recall"],
            "Norm_Recall": vm["per_class"]["NORMAL"]["recall"],
            "Viral_Recall": vm["per_class"]["VIRAL"]["recall"],
            "Bact_F1": vm["per_class"]["BACTERIAL"]["f1"],
            "Norm_F1": vm["per_class"]["NORMAL"]["f1"],
            "Viral_F1": vm["per_class"]["VIRAL"]["f1"],
        }
        rows.append(row)

df = pd.DataFrame(rows)
print("\n========== MULTI-SEED RUN DETAILS ==========")
print(df.to_string(index=False))

print("\n========== SUMMARY (MEAN +/- STD) ==========")
summary_rows = []
for cfg_label, grp in df.groupby("Config", sort=False):
    s = {
        "Config": cfg_label,
        "Val Loss": f"{grp['Val_Loss'].mean():.4f} +/- {grp['Val_Loss'].std():.4f}",
        "Val Acc (%)": f"{grp['Val_Acc'].mean()*100:.2f} +/- {grp['Val_Acc'].std()*100:.2f}",
        "Val Macro F1 (%)": f"{grp['Val_Macro_F1'].mean()*100:.2f} +/- {grp['Val_Macro_F1'].std()*100:.2f}",
        "Val Macro Rec (%)": f"{grp['Val_Macro_Recall'].mean()*100:.2f} +/- {grp['Val_Macro_Recall'].std()*100:.2f}",
        "Bact F1 (%)": f"{grp['Bact_F1'].mean()*100:.2f} +/- {grp['Bact_F1'].std()*100:.2f}",
        "Norm F1 (%)": f"{grp['Norm_F1'].mean()*100:.2f} +/- {grp['Norm_F1'].std()*100:.2f}",
        "Viral F1 (%)": f"{grp['Viral_F1'].mean()*100:.2f} +/- {grp['Viral_F1'].std()*100:.2f}",
        "Viral Rec (%)": f"{grp['Viral_Recall'].mean()*100:.2f} +/- {grp['Viral_Recall'].std()*100:.2f}",
        "Norm Rec (%)": f"{grp['Norm_Recall'].mean()*100:.2f} +/- {grp['Norm_Recall'].std()*100:.2f}",
    }
    summary_rows.append(s)

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))
