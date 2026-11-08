#!/usr/bin/env python3
"""CLI Inference tool for Chest X-Ray Multiclass Pneumonia Classification."""

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
    parser = argparse.ArgumentParser(
        description="Chest X-Ray Pneumonia Multiclass Diagnosis CLI (MobileNetV2 + Grad-CAM)"
    )
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input chest X-ray image (JPEG/PNG)")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Custom checkpoint path (.pt)")
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
        classifier = PneumoniaClassifier(checkpoint_path=args.checkpoint, device=args.device)
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

    if args.json:
        result = {
            "image": str(image_path),
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
        print(f"Diagnosis:       {predicted_class} ({confidence * 100:.2f}%)")
        print(f"Inference Time:  {elapsed_ms:.1f} ms on {classifier.device.type.upper()}\n")
        print(format_table(scores, predicted_class))
        if gradcam_saved:
            print(f"\n[+] Grad-CAM explanation saved to: {gradcam_saved}")
        print("--------------------------------------------------------")
        print("Notice: AI assistive tool for research purposes only.")
        print("========================================================\n")


if __name__ == "__main__":
    main()

