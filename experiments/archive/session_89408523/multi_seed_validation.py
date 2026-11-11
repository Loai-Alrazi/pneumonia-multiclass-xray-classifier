import sys
sys.path.append(r"C:\Users\Loay_Alrazi\.gemini\antigravity\brain\89408523-973d-4ee4-8297-b78793515970\scratch")
import numpy as np
import pandas as pd
from experiment_framework import run_experiment, get_base_transform

SEEDS = [42, 123, 456]
transform = get_base_transform()

configs = [
    {
        "name": "Exp1_Extended_Head",
        "epochs": 12,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "use_class_weights": False,
        "use_scheduler": False,
        "finetune_blocks": 0,
        "patience": 3
    },
    {
        "name": "Exp2_Class_Weighted_Head",
        "epochs": 12,
        "lr": 1e-3,
        "weight_decay": 0.0,
        "use_class_weights": True,
        "use_scheduler": False,
        "finetune_blocks": 0,
        "patience": 3
    },
    {
        "name": "Exp5_FineTune_Top4Blocks",
        "epochs": 8,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "use_class_weights": True,
        "use_scheduler": True,
        "finetune_blocks": 4,
        "finetune_lr_backbone": 5e-5,
        "patience": 3
    }
]

all_results = []
for cfg in configs:
    for seed in SEEDS:
        print(f"\n>>> Running {cfg['name']} with SEED={seed}")
        res, _ = run_experiment(
            exp_name=f"{cfg['name']}_s{seed}",
            train_transform=transform,
            epochs=cfg["epochs"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            use_class_weights=cfg["use_class_weights"],
            use_scheduler=cfg["use_scheduler"],
            finetune_blocks=cfg["finetune_blocks"],
            finetune_lr_backbone=cfg.get("finetune_lr_backbone", 5e-5),
            patience=cfg["patience"],
            seed=seed
        )
        all_results.append({
            "config": cfg["name"],
            "seed": seed,
            "best_epoch": res["best_epoch"],
            "val_loss": res["val_loss"],
            "val_accuracy": res["val_accuracy"],
            "val_macro_f1": res["val_macro_f1"],
            "val_macro_recall": res["val_macro_recall"],
            "bact_recall": res["val_per_class"]["BACTERIAL"]["recall"],
            "norm_recall": res["val_per_class"]["NORMAL"]["recall"],
            "viral_recall": res["val_per_class"]["VIRAL"]["recall"],
            "bact_f1": res["val_per_class"]["BACTERIAL"]["f1"],
            "norm_f1": res["val_per_class"]["NORMAL"]["f1"],
            "viral_f1": res["val_per_class"]["VIRAL"]["f1"],
        })

df_res = pd.DataFrame(all_results)
df_res.to_csv("scratch/multi_seed_results.csv", index=False)
print("\n================ MULTI-SEED VALIDATION SUMMARY ================")
for cfg_name, grp in df_res.groupby("config"):
    print(f"\nConfiguration: {cfg_name}")
    print(f"  Val Loss:         {grp['val_loss'].mean():.4f} +/- {grp['val_loss'].std():.4f}")
    print(f"  Val Accuracy:     {grp['val_accuracy'].mean():.2%} +/- {grp['val_accuracy'].std():.2%}")
    print(f"  Val Macro F1:     {grp['val_macro_f1'].mean():.2%} +/- {grp['val_macro_f1'].std():.2%}")
    print(f"  Val Macro Recall: {grp['val_macro_recall'].mean():.2%} +/- {grp['val_macro_recall'].std():.2%}")
    print(f"  Bacterial F1:     {grp['bact_f1'].mean():.2%} +/- {grp['bact_f1'].std():.2%}")
    print(f"  Normal F1:        {grp['norm_f1'].mean():.2%} +/- {grp['norm_f1'].std():.2%}")
    print(f"  Viral F1:         {grp['viral_f1'].mean():.2%} +/- {grp['viral_f1'].std():.2%}")
    print(f"  Viral Recall:     {grp['viral_recall'].mean():.2%} +/- {grp['viral_recall'].std():.2%}")
