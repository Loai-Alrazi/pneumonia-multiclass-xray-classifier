#!/usr/bin/env python3
"""CLI Inference tool for Chest X-Ray Multiclass Pneumonia Classification.

Supports the Dual-Model Framework:
- Exp 2 (Default): Controlled Experimental Baseline (Frozen MobileNetV2, Seed 42).
- Exp 5 (Optional): Validation-Optimized Variant (Top-4 blocks fine-tuned, Seed 42).
"""

import argparse
import json
from pathlib import Path
import sys
import time

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.model import PneumoniaClassifier
from app.gradcam import GradCAMExplainer


def format_table(scores: dict, predicted_class: str) -> str:
    """Format scores into an aligned terminal table."""
    lines = [
        "┌────────────┬─────────────┬────────────────────────────────────┐",
        "│ Class      │ Probability │ Distribution                       │",
        "├────────────┼─────────────┼────────────────────────────────────┤",
    ]
    for cls, prob in scores.items():
        is_pred = " (★)" if cls == predicted_class else "    "
        bar = "█" * int(prob * 30)
        lines.append(f"│ {cls:<10} │ {prob * 100:6.2f}%{is_pred} │ {bar:<34} │")
    lines.append("└────────────┴─────────────┴────────────────────────────────────┘")
    return "\n".join(lines)


def main():
    # The result table uses box-drawing characters, which legacy Windows code pages
    # (e.g. cp1252 when output is piped or redirected) cannot encode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Chest X-Ray Pneumonia Multiclass Diagnosis CLI (MobileNetV2 + Grad-CAM)"
    )
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input chest X-ray image (JPEG/PNG)")
    parser.add_argument(
        "--model", "-m", type=str, choices=["exp2", "exp5"], default="exp2",
        help="Model variant: 'exp2' (default baseline, frozen MobileNetV2) or 'exp5' (validation-tuned, top-4 blocks fine-tuned)"
    )
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Custom checkpoint path (.pt), overrides --model")
    parser.add_argument("--device", "-d", type=str, default=None, help="Inference device ('cpu' or 'cuda')")
    parser.add_argument("--gradcam", "-g", action="store_true", help="Generate Grad-CAM interpretability overlay")
    parser.add_argument("--output", "-o", type=str, default="gradcam_result.png", help="Path to save Grad-CAM image")
    parser.add_argument("--json", action="store_true", help="Output pure JSON format")

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}", file=sys.stderr)
        sys.exit(1)

    # Initialize classifier
    try:
        classifier = PneumoniaClassifier(
            checkpoint_path=args.checkpoint,
            model_variant=args.model,
            device=args.device
        )
    except Exception as e:
        print(f"Error initializing model: {e}", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    predicted_class, confidence, scores = classifier.predict(image_path)
    elapsed_ms = (time.time() - t0) * 1000.0

    gradcam_saved = None
    if args.gradcam:
        explainer = GradCAMExplainer(classifier)
        _, overlay_pil, _ = explainer.explain(image_path)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_pil.save(out_path)
        gradcam_saved = str(out_path)

    model_desc = f"{classifier.experiment_name} (Loaded from: {classifier.checkpoint_path.name})"

    if args.json:
        result = {
            "image": str(image_path),
            "model_variant": classifier.model_variant,
            "checkpoint": str(classifier.checkpoint_path),
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in scores.items()},
            "inference_time_ms": round(elapsed_ms, 2),
            "gradcam_saved": gradcam_saved
        }
        print(json.dumps(result, indent=2))
    else:
        print("\n========================================================")
        print("   Chest X-Ray Pneumonia Multiclass AI Diagnostic Tool   ")
        print("========================================================")
        print(f"Input Image:     {image_path.name}")
        print(f"Model Variant:   {model_desc}")
        print(f"Diagnosis:       {predicted_class} ({confidence * 100:.2f}%)")
        print(f"Inference Time:  {elapsed_ms:.1f} ms on {classifier.device.type.upper()}\n")
        print(format_table(scores, predicted_class))
        if gradcam_saved:
            print(f"\n[+] Grad-CAM explanation saved to: {gradcam_saved}")
        print("--------------------------------------------------------")
        print("Notice: Educational & research exploration tool only.")
        print("Not approved for autonomous clinical triage or diagnosis.")
        print("========================================================\n")


if __name__ == "__main__":
    main()
