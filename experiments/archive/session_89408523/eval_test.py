import sys
sys.path.append(r"C:\Users\Loay_Alrazi\.gemini\antigravity\brain\89408523-973d-4ee4-8297-b78793515970\scratch")
import torch
from experiment_framework import get_base_transform, evaluate_model, DATA_DIR, CLASS_NAMES
from torchvision import datasets, models
from torch.utils.data import DataLoader
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_base_transform()
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

models_to_test = [
    ("Original Baseline (3 Ep, Frozen)", "artifacts/pneumonia_3class_baseline_backup.pt"),
    ("Exp 1: Extended Head (8 Ep, Frozen)", "artifacts/model_exp1_extended_head_12epochs.pt"),
    ("Exp 2: Class-Weighted Head (Frozen)", "artifacts/model_exp2_class_weighted_head.pt"),
    ("Exp 5: Fine-Tuned Top 4 Blocks", "artifacts/model_exp5_finetune_top4blocks.pt"),
]

for name, path in models_to_test:
    ckpt = torch.load(path, map_location=device)
    model = models.mobilenet_v2(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.last_channel, 3)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    
    m = evaluate_model(model, test_loader, device)
    acc = m["accuracy"]
    f1 = m["macro_f1"]
    rec = m["macro_recall"]
    print(f"=== {name} ===")
    print(f"Test Accuracy: {acc:.2%}")
    print(f"Test Macro F1: {f1:.2%}")
    print(f"Test Macro Recall: {rec:.2%}")
    for cls in CLASS_NAMES:
        p = m["per_class"][cls]["precision"]
        r = m["per_class"][cls]["recall"]
        f = m["per_class"][cls]["f1"]
        print(f"  {cls:9s} -> Recall: {r:.2%}, Precision: {p:.2%}, F1: {f:.2%}")
    print("Confusion Matrix (rows: [Bact, Norm, Viral], cols: [Bact, Norm, Viral]):")
    for row in m["confusion_matrix"]:
        print(" ", row)
    print()
