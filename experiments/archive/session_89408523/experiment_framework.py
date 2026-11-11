"""Reproducible Experiment Framework for Chest X-Ray Multiclass Optimization."""

import os
import sys
import copy
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score

# Set local torch cache to avoid network calls
TORCH_HUB = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\artifacts\torch")
torch.hub.set_dir(str(TORCH_HUB))

DATA_DIR = Path(r"D:\PAEP_AI_Program\05_Deep_Learning\Projects_and_Homeworks\Pneumonia_Chest_XRay_Kit\data\chest_xray_3class")
CLASS_NAMES = ["BACTERIAL", "NORMAL", "VIRAL"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_base_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

def evaluate_model(model, loader, device, loss_fn=None):
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if loss_fn is not None:
                loss = loss_fn(logits, y)
                total_loss += loss.item() * len(y)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())
            total_samples += len(y)
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    per_class_rec = recall_score(all_targets, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_targets, all_preds, average=None, zero_division=0)
    per_class_prec = precision_score(all_targets, all_preds, average=None, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)
    
    metrics = {
        "loss": (total_loss / total_samples) if loss_fn is not None else None,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "macro_recall": float(macro_recall),
        "per_class": {
            cls: {
                "precision": float(per_class_prec[i]),
                "recall": float(per_class_rec[i]),
                "f1": float(per_class_f1[i])
            } for i, cls in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist()
    }
    return metrics

def build_model(device, freeze_backbone=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    in_feat = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feat, len(CLASS_NAMES))
    return model.to(device)

def run_experiment(
    exp_name: str,
    train_transform,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    use_class_weights: bool = False,
    use_scheduler: bool = False,
    finetune_blocks: int = 0, # 0 = frozen backbone, >0 = unfreeze top N blocks
    finetune_lr_backbone: float = 5e-5,
    patience: int = 3,
    seed: int = 42
):
    print(f"\n==================================================")
    print(f"STARTING EXPERIMENT: {exp_name}")
    print(f"==================================================")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    val_transform = get_base_transform()
    train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    
    # Calculate class weights if requested
    if use_class_weights:
        counts = np.bincount(train_dataset.targets, minlength=len(CLASS_NAMES))
        total = len(train_dataset)
        weights = total / (len(CLASS_NAMES) * counts.astype(float))
        class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print(f"Using class weights: {dict(zip(CLASS_NAMES, weights.round(3).tolist()))}")
    else:
        loss_fn = nn.CrossEntropyLoss()
        
    model = build_model(device, freeze_backbone=True)
    
    # Configure optimizer
    if finetune_blocks == 0:
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # Unfreeze top feature layers
        total_feature_blocks = len(model.features)
        unfreeze_start = total_feature_blocks - finetune_blocks
        for idx in range(unfreeze_start, total_feature_blocks):
            for p in model.features[idx].parameters():
                p.requires_grad = True
        trainable_params = [
            {"params": [p for idx in range(unfreeze_start, total_feature_blocks) for p in model.features[idx].parameters() if p.requires_grad], "lr": finetune_lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr}
        ]
        optimizer = torch.optim.Adam(trainable_params, weight_decay=weight_decay)
        print(f"Unfrozen top {finetune_blocks} feature blocks (from block {unfreeze_start} to {total_feature_blocks-1})")
        
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        
    history = []
    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        if finetune_blocks == 0:
            model.features.eval() # Keep backbone deterministic
        else:
            # When fine-tuning, keep running stats in batchnorm stable
            model.features.eval()
            
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(y)
            train_correct += (logits.argmax(dim=1) == y).sum().item()
            train_total += len(y)
            
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        if scheduler is not None:
            scheduler.step()
            
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device, loss_fn)
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_f1": val_f1
        })
        
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}, Macro F1: {val_f1:.2%}")
        
        # Checkpoint selection strictly on Validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement in {patience} epochs).")
                break
                
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.1f}s. Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}, Val Macro F1: {best_val_f1:.2%}")
    
    # Reload best model state and evaluate final validation performance
    model.load_state_dict(best_model_state)
    final_val_metrics = evaluate_model(model, val_loader, device, loss_fn)
    
    result = {
        "experiment_name": exp_name,
        "best_epoch": best_epoch,
        "elapsed_time": elapsed_time,
        "val_loss": final_val_metrics["loss"],
        "val_accuracy": final_val_metrics["accuracy"],
        "val_macro_f1": final_val_metrics["macro_f1"],
        "val_macro_recall": final_val_metrics["macro_recall"],
        "val_per_class": final_val_metrics["per_class"],
        "val_confusion_matrix": final_val_metrics["confusion_matrix"],
        "history": history
    }
    
    # Save checkpoint for this experiment
    ckpt_out = Path("artifacts") / f"model_{exp_name.replace(' ', '_').lower()}.pt"
    torch.save({
        "state_dict": {k: v.cpu() for k, v in best_model_state.items()},
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "mean": MEAN,
        "std": STD,
        "best_epoch": best_epoch,
        "val_metrics": final_val_metrics,
        "experiment_name": exp_name
    }, ckpt_out)
    print(f"Saved checkpoint: {ckpt_out}")
    
    return result, model
