import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
test_paths = [s[0] for s in test_dataset.samples]
test_targets = np.array(test_dataset.targets)

# Normal images are class index 1
normal_indices = np.where(test_targets == 1)[0]
normal_paths = [test_paths[i] for i in normal_indices]

# Load Exp 5 and Exp 1 models
def get_model(ckpt_rel_path):
    ckpt = torch.load(ckpt_rel_path, map_location=device, weights_only=False)
    m = models.mobilenet_v2(weights=None)
    m.classifier[-1] = nn.Linear(m.last_channel, 3)
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device).eval()

m_exp5 = get_model("artifacts/model_exp5_finetune_top4blocks_s42.pt")
m_exp1 = get_model("artifacts/model_exp1_extended_head_s42.pt")
m_base = get_model("artifacts/pneumonia_3class_baseline_backup.pt")

def predict_logits(model, loader):
    all_logits = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            l = model(x)
            all_logits.append(l.cpu().numpy())
    return np.concatenate(all_logits, axis=0)

logits_exp5 = predict_logits(m_exp5, test_loader)
logits_exp1 = predict_logits(m_exp1, test_loader)
logits_base = predict_logits(m_base, test_loader)

probs_exp5 = torch.softmax(torch.tensor(logits_exp5), dim=1).numpy()
probs_exp1 = torch.softmax(torch.tensor(logits_exp1), dim=1).numpy()
probs_base = torch.softmax(torch.tensor(logits_base), dim=1).numpy()

# Look at Normal test samples
p_norm_exp5 = probs_exp5[normal_indices]
p_norm_exp1 = probs_exp1[normal_indices]
p_norm_base = probs_base[normal_indices]

preds_norm_exp5 = p_norm_exp5.argmax(axis=1)
preds_norm_exp1 = p_norm_exp1.argmax(axis=1)
preds_norm_base = p_norm_base.argmax(axis=1)

print("=== DIAGNOSTIC INSPECTION OF 234 TEST NORMAL IMAGES ===")
print(f"Exp 5 Predictions on Normal: Bact={np.sum(preds_norm_exp5==0)}, Norm={np.sum(preds_norm_exp5==1)}, Viral={np.sum(preds_norm_exp5==2)}")
print(f"Exp 1 Predictions on Normal: Bact={np.sum(preds_norm_exp1==0)}, Norm={np.sum(preds_norm_exp1==1)}, Viral={np.sum(preds_norm_exp1==2)}")
print(f"Base Predictions on Normal:  Bact={np.sum(preds_norm_base==0)}, Norm={np.sum(preds_norm_base==1)}, Viral={np.sum(preds_norm_base==2)}")

# Mean predicted probabilities for Normal test images
print("\nMean Predicted Probabilities on Test Normal Images [Bact, Norm, Viral]:")
print(f"Exp 5: {p_norm_exp5.mean(axis=0).round(4)}")
print(f"Exp 1: {p_norm_exp1.mean(axis=0).round(4)}")
print(f"Base:  {p_norm_base.mean(axis=0).round(4)}")

# Analyze margin between Normal and Viral on Normal test samples
margin_exp5 = p_norm_exp5[:, 1] - p_norm_exp5[:, 2] # prob(Normal) - prob(Viral)
margin_exp1 = p_norm_exp1[:, 1] - p_norm_exp1[:, 2]
print(f"\nMean Margin (P(Normal) - P(Viral)) on Test Normal:")
print(f"Exp 5: {margin_exp5.mean():.4f} (median: {np.median(margin_exp5):.4f})")
print(f"Exp 1: {margin_exp1.mean():.4f} (median: {np.median(margin_exp1):.4f})")

# Look at images where Exp 1 was correct (predicted Normal) but Exp 5 predicted Viral
diff_mask = (preds_norm_exp1 == 1) & (preds_norm_exp5 == 2)
print(f"\nCount of Normal images correct in Exp 1 but predicted Viral in Exp 5: {np.sum(diff_mask)}")

sample_diff_paths = [normal_paths[i] for i in np.where(diff_mask)[0][:10]]
print("Sample files:", [Path(p).name for p in sample_diff_paths])
