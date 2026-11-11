"""Automated Unit Tests for the Pneumonia Multiclass AI Inference Engine."""

try:
    import pytest
except ImportError:
    # Graceful fallback when pytest is not installed in the Python environment
    class _MockPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func
            if len(args) == 1 and callable(args[0]):
                return args[0]
            return decorator
    pytest = _MockPytest()

import numpy as np
from PIL import Image
import torch
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.model import PneumoniaClassifier
from app.gradcam import GradCAMExplainer


@pytest.fixture(scope="session")
def engine():
    return PneumoniaClassifier()


@pytest.fixture(scope="session")
def explainer(engine):
    return GradCAMExplainer(engine)


@pytest.fixture
def dummy_image():
    """Create a synthetic 224x224 RGB image for testing."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_model_loading(engine):
    """Verify that model loads with expected 3-class configuration."""
    assert engine.model is not None
    assert len(engine.class_names) == 3
    assert set(engine.class_names) == {"BACTERIAL", "NORMAL", "VIRAL"}
    assert engine.image_size == 224


def test_predict_probabilities(engine, dummy_image):
    """Verify prediction outputs, types, and probability calibration."""
    pred_class, conf, scores = engine.predict(dummy_image)

    assert pred_class in engine.class_names
    assert 0.0 <= conf <= 1.0
    assert len(scores) == 3

    # Probabilities must sum to approximately 1.0
    total_prob = sum(scores.values())
    assert abs(total_prob - 1.0) < 1e-4


def test_gradcam_generation(explainer, dummy_image):
    """Verify Grad-CAM output shapes, range, and overlay image creation."""
    cam, overlay, target = explainer.explain(dummy_image)

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (224, 224)
    assert 0.0 <= cam.min() <= cam.max() <= 1.0
    assert isinstance(overlay, Image.Image)
    assert overlay.size == (224, 224)
    assert target in (0, 1, 2)


if __name__ == "__main__":
    print("Executing tests in test_inference.py directly...")
    eng = PneumoniaClassifier()
    exp = GradCAMExplainer(eng)
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr)

    test_model_loading(eng)
    print("  [OK] test_model_loading passed")

    test_predict_probabilities(eng, img)
    print("  [OK] test_predict_probabilities passed")

    test_gradcam_generation(exp, img)
    print("  [OK] test_gradcam_generation passed")

    print("\nAll tests passed successfully.")


