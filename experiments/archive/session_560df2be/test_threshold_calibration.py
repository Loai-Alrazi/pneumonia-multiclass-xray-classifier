import sys
from pathlib import Path
import torch
import numpy as np
from torchvision import datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from scipy.optimize import minimize

sys.path.append(r"C:\Users\Loay_Alrazi\.gemini\antigravity\brain\89408523-973d-4ee4-8297-b78793515970\scratch")
from experiment_framework import get_base_transform, DATA_DIR, CLASS_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_base_transform()

val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
val_targets = np.array(val_dataset.targets)

test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
test_targets = np.array(test_dataset.targets)

def get_logits(model_path, loader):
    ckpt = torch.load(model_path, map_location=device)
    model = models.mobilenet_v2(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.last_channel, 3)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    all_logits = []
    with torch.no_grad():
        for x, _ in loader:
            l = model(x.to(device))
            all_logits.append(l.cpu().numpy())
    return np.concatenate(all_logits, axis=0)

# Load logits for Exp 1 and Exp 2 multi-seed ensemble
val_logits_exp1 = np.mean([get_logits(f"artifacts/model_exp1_extended_head_s{s}.pt", val_loader) for s in [42, 123, 456]], axis=0)
test_logits_exp1 = np.mean([get_logits(f"artifacts/model_exp1_extended_head_s{s}.pt", test_loader) for s in [42, 123, 456]], axis=0)

val_logits_exp2 = np.mean([get_logits(f"artifacts/model_exp2_class_weighted_head_s{s}.pt", val_loader) for s in [42, 123, 456]], axis=0)
test_logits_exp2 = np.mean([get_logits(f"artifacts/model_exp2_class_weighted_head_s{s}.pt", test_loader) for s in [42, 123, 456]], axis=0)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Test simple threshold adjustment: multiplier on class probabilities w = [w_bact, w_norm, w_viral]
# Optimize w on validation to maximize macro F1
for name, v_logits, t_logits in [("Exp 1 Ensemble", val_logits_exp1, test_logits_exp1),
                                  ("Exp 2 Ensemble", val_logits_exp2, test_logits_exp2)]:
    v_probs = softmax(v_logits)
    t_probs = softmax(t_logits)
    
    # Baseline (argmax)
    b_v_pred = np.argmax(v_probs, axis=1)
    b_t_pred = np.argmax(t_probs, axis=1)
    b_t_rep = classification_report(test_targets, b_t_pred, target_names=CLASS_NAMES, output_dict=True)
    
    # Search for optimal threshold multipliers on validation set
    best_weights = [1.0, 1.0, 1.0]
    best_v_f1 = f1_score(val_targets, b_v_pred, average="macro")
    
    for w_norm in np.linspace(0.8, 2.5, 35):
        for w_viral in np.linspace(0.5, 1.5, 21):
            w = np.array([1.0, w_norm, w_viral])
            adj_v = v_probs * w
            pred_v = np.argmax(adj_v, axis=1)
            score = f1_score(val_targets, pred_v, average="macro")
            if score > best_v_f1:
                best_v_f1 = score
                best_weights = [1.0, w_norm, w_viral]
                
    # Apply optimal validation weights to test set
    adj_t = t_probs * np.array(best_weights)
    cal_t_pred = np.argmax(adj_t, axis=1)
    cal_t_rep = classification_report(test_targets, cal_t_pred, target_names=CLASS_NAMES, output_dict=True)
    
    print(f"=== {name} Calibration via Validation Optimization ===")
    print(f"  Best Weights [Bact, Norm, Viral] from Val: {np.round(best_weights, 3)}")
    print(f"  Validation Macro F1: Baseline={f1_score(val_targets, b_v_pred, average='macro'):.2%} -> Tuned={best_v_f1:.2%}")
    print(f"  Test Accuracy:       Baseline={accuracy_score(test_targets, b_t_pred):.2%} -> Tuned={accuracy_score(test_targets, cal_t_pred):.2%}")
    print(f"  Test Macro F1:       Baseline={f1_score(test_targets, b_t_pred, average='macro'):.2%} -> Tuned={f1_score(test_targets, cal_t_pred, average='macro'):.2%}")
    print(f"  Test Normal Recall:  Baseline={b_t_rep['NORMAL']['recall']:.2%} -> Tuned={cal_t_rep['NORMAL']['recall']:.2%}")
    print(f"  Test Viral Recall:   Baseline={b_t_rep['VIRAL']['recall']:.2%} -> Tuned={cal_t_rep['VIRAL']['recall']:.2%}")
    print(f"  Test Bacterial Rec:  Baseline={b_t_rep['BACTERIAL']['recall']:.2%} -> Tuned={cal_t_rep['BACTERIAL']['recall']:.2%}")
    print("  Tuned Test Confusion Matrix:")
    print(confusion_matrix(test_targets, cal_t_pred))
    print()
