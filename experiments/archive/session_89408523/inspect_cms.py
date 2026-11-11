import torch
from pathlib import Path
import numpy as np

artifacts = Path("artifacts")
class_names = ["BACTERIAL", "NORMAL", "VIRAL"]

for name, fname in [
    ("Exp 1 (Extended Head, Seed 42)", "model_exp1_extended_head_s42.pt"),
    ("Exp 2 (Class-Weighted Head, Seed 42)", "model_exp2_class_weighted_head_s42.pt"),
    ("Exp 5 (Fine-Tune Top 4 Blocks, Seed 42)", "model_exp5_finetune_top4blocks_s42.pt")
]:
    ckpt = torch.load(artifacts / fname, map_location="cpu", weights_only=False)
    cm = np.array(ckpt["val_metrics"]["confusion_matrix"])
    print(f"\n--- {name} Validation Confusion Matrix ---")
    print(f"Rows = True, Cols = Pred")
    print(f"{'':12s} {'BACTERIAL':10s} {'NORMAL':10s} {'VIRAL':10s}")
    for i, cls in enumerate(class_names):
        print(f"{cls:12s} {cm[i,0]:10d} {cm[i,1]:10d} {cm[i,2]:10d}")
