import sys
from pathlib import Path
import torch
import numpy as np
from torchvision import datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

sys.path.append(r"C:\Users\Loay_Alrazi\.gemini\antigravity\brain\89408523-973d-4ee4-8297-b78793515970\scratch")
from experiment_framework import get_base_transform, DATA_DIR, CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_base_transform()
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
targets = np.array(test_dataset.targets)

def get_probs(model_path):
    ckpt = torch.load(model_path, map_location=device)
    model = models.mobilenet_v2(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.last_channel, 3)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    all_probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            out = model(x.to(device))
            p = torch.softmax(out, dim=1).cpu().numpy()
            all_probs.append(p)
    return np.concatenate(all_probs, axis=0)

seeds = [42, 123, 456]
experiments = [
    ("Exp 1 (Extended Head)", "artifacts/model_exp1_extended_head"),
    ("Exp 2 (Class-Weighted Head)", "artifacts/model_exp2_class_weighted_head"),
    ("Exp 5 (Fine-Tune Top 4 Blocks)", "artifacts/model_exp5_finetune_top4blocks"),
]

all_exp_ensemble_probs = []

for label, prefix in experiments:
    accs, f1s, norm_recs, vir_recs, bac_recs = [], [], [], [], []
    exp_probs = []
    for s in seeds:
        path = f"{prefix}_s{s}.pt"
        probs = get_probs(path)
        exp_probs.append(probs)
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="macro")
        rep = classification_report(targets, preds, target_names=CLASS_NAMES, output_dict=True)
        accs.append(acc)
        f1s.append(f1)
        norm_recs.append(rep["NORMAL"]["recall"])
        vir_recs.append(rep["VIRAL"]["recall"])
        bac_recs.append(rep["BACTERIAL"]["recall"])
    
    # 3-Seed Ensemble
    ens_probs = np.mean(exp_probs, axis=0)
    all_exp_ensemble_probs.append(ens_probs)
    ens_preds = np.argmax(ens_probs, axis=1)
    ens_acc = accuracy_score(targets, ens_preds)
    ens_f1 = f1_score(targets, ens_preds, average="macro")
    ens_rep = classification_report(targets, ens_preds, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(targets, ens_preds)
    
    print(f"=== {label} on TEST SET ===")
    print(f"  Individual Seeds (mean +/- std):")
    print(f"    Accuracy:      {np.mean(accs):.2%} +/- {np.std(accs):.2%}")
    print(f"    Macro F1:      {np.mean(f1s):.2%} +/- {np.std(f1s):.2%}")
    print(f"    Bacterial Rec: {np.mean(bac_recs):.2%} +/- {np.std(bac_recs):.2%}")
    print(f"    Normal Recall: {np.mean(norm_recs):.2%} +/- {np.std(norm_recs):.2%}")
    print(f"    Viral Recall:  {np.mean(vir_recs):.2%} +/- {np.std(vir_recs):.2%}")
    print(f"  3-Seed Ensemble:")
    print(f"    Accuracy:      {ens_acc:.2%}")
    print(f"    Macro F1:      {ens_f1:.2%}")
    print(f"    Bacterial Rec: {ens_rep['BACTERIAL']['recall']:.2%}")
    print(f"    Normal Recall: {ens_rep['NORMAL']['recall']:.2%}")
    print(f"    Viral Recall:  {ens_rep['VIRAL']['recall']:.2%}")
    print(f"    Confusion Matrix [Bact, Norm, Viral]:")
    for row in cm:
        print(f"      {row.tolist()}")
    print()

# Cross-experiment ensemble (Exp 1 + Exp 2)
mix_probs = (all_exp_ensemble_probs[0] + all_exp_ensemble_probs[1]) / 2.0
mix_preds = np.argmax(mix_probs, axis=1)
mix_acc = accuracy_score(targets, mix_preds)
mix_f1 = f1_score(targets, mix_preds, average="macro")
mix_rep = classification_report(targets, mix_preds, target_names=CLASS_NAMES, output_dict=True)
print("=== Cross-Model Ensemble (Exp 1 + Exp 2 across all 6 models) ===")
print(f"  Accuracy:      {mix_acc:.2%}")
print(f"  Macro F1:      {mix_f1:.2%}")
print(f"  Normal Recall: {mix_rep['NORMAL']['recall']:.2%}")
print(f"  Viral Recall:  {mix_rep['VIRAL']['recall']:.2%}")
print(f"  Bacterial Rec: {mix_rep['BACTERIAL']['recall']:.2%}")
