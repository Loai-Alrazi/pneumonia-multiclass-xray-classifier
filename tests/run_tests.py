"""Unit test suite for the Pneumonia Multiclass AI Inference Engine.
Tests both Exp 2 (Default Baseline) and Exp 5 (Validation Winner).
"""

import sys
import unittest
from unittest import mock
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.model import PneumoniaClassifier
from app.gradcam import GradCAMExplainer


class TestInferenceEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine_exp2 = PneumoniaClassifier(model_variant="exp2")
        cls.engine_exp5 = PneumoniaClassifier(model_variant="exp5")
        cls.explainer_exp2 = GradCAMExplainer(cls.engine_exp2)
        cls.explainer_exp5 = GradCAMExplainer(cls.engine_exp5)
        
        # Synthetic 224x224 RGB image
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cls.dummy_img = Image.fromarray(arr)

    def test_exp2_loading(self):
        """Verify Exp 2 model loads correctly with 3 classes."""
        self.assertIsNotNone(self.engine_exp2.model)
        self.assertEqual(len(self.engine_exp2.class_names), 3)
        self.assertEqual(self.engine_exp2.class_names, ["BACTERIAL", "NORMAL", "VIRAL"])
        self.assertEqual(self.engine_exp2.image_size, 224)
        self.assertEqual(self.engine_exp2.model_variant, "exp2")
        self.assertEqual(self.engine_exp2.checkpoint_path.name, "pneumonia_3class.pt")
        self.assertEqual(self.engine_exp2.experiment_name, "Exp2_Class_Weighted_Head_s42")

    def test_exp5_loading(self):
        """Verify Exp 5 model loads correctly with 3 classes."""
        self.assertIsNotNone(self.engine_exp5.model)
        self.assertEqual(len(self.engine_exp5.class_names), 3)
        self.assertEqual(self.engine_exp5.class_names, ["BACTERIAL", "NORMAL", "VIRAL"])
        self.assertEqual(self.engine_exp5.image_size, 224)
        self.assertEqual(self.engine_exp5.model_variant, "exp5")
        self.assertEqual(self.engine_exp5.checkpoint_path.name, "model_exp5_finetune_top4blocks_s42.pt")
        self.assertEqual(self.engine_exp5.experiment_name, "Exp5_FineTune_Top4Blocks_s42")

    def test_missing_exp5_checkpoint_raises(self):
        """A missing Exp 5 file must fail loudly instead of serving other weights."""
        missing = Path(__file__).resolve().parent / "does_not_exist.pt"
        with mock.patch.object(PneumoniaClassifier, "EXP5_CHECKPOINT", missing):
            with self.assertRaises(FileNotFoundError):
                PneumoniaClassifier(model_variant="exp5", device="cpu")

    def test_mismatched_checkpoint_raises(self):
        """Exp 2 weights placed at the Exp 5 path must be rejected, not mislabelled."""
        with mock.patch.object(PneumoniaClassifier, "EXP5_CHECKPOINT", PneumoniaClassifier.DEFAULT_CHECKPOINT):
            with self.assertRaises(ValueError):
                PneumoniaClassifier(model_variant="exp5", device="cpu")

    def test_unknown_variant_raises(self):
        """Unknown variant names must not silently fall back to Exp 2."""
        with self.assertRaises(ValueError):
            PneumoniaClassifier(model_variant="exp3", device="cpu")

    def test_exp2_predict(self):
        """Verify Exp 2 inference output types, ranges, and probability sum."""
        pred_class, conf, scores = self.engine_exp2.predict(self.dummy_img)
        self.assertIn(pred_class, self.engine_exp2.class_names)
        self.assertTrue(0.0 <= conf <= 1.0)
        self.assertEqual(len(scores), 3)
        total_prob = sum(scores.values())
        self.assertAlmostEqual(total_prob, 1.0, places=4)

    def test_exp5_predict(self):
        """Verify Exp 5 inference output types, ranges, and probability sum."""
        pred_class, conf, scores = self.engine_exp5.predict(self.dummy_img)
        self.assertIn(pred_class, self.engine_exp5.class_names)
        self.assertTrue(0.0 <= conf <= 1.0)
        self.assertEqual(len(scores), 3)
        total_prob = sum(scores.values())
        self.assertAlmostEqual(total_prob, 1.0, places=4)

    def test_gradcam_generation(self):
        """Verify Grad-CAM output shapes, range, and overlay image creation."""
        for explainer in [self.explainer_exp2, self.explainer_exp5]:
            cam, overlay, target = explainer.explain(self.dummy_img)
            self.assertIsInstance(cam, np.ndarray)
            self.assertEqual(cam.shape, (224, 224))
            self.assertTrue(0.0 <= cam.min() <= cam.max() <= 1.0)
            self.assertIsInstance(overlay, Image.Image)
            self.assertEqual(overlay.size, (224, 224))
            self.assertIn(target, (0, 1, 2))


if __name__ == "__main__":
    unittest.main()

