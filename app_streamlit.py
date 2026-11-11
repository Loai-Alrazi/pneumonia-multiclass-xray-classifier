"""Streamlit Web Application: Chest X-Ray Pneumonia Multiclass Diagnosis with Grad-CAM.

Supports Dual-Model Academic Architecture:
- Exp 2 (Default): Controlled Baseline (Frozen MobileNetV2, Seed 42)
- Exp 5 (Optional): Validation Winner (Top 4 Feature Blocks Fine-Tuned, Seed 42)
"""

from pathlib import Path
import time
from PIL import Image
import numpy as np
import streamlit as st

# Import backend modules
from app.model import PneumoniaClassifier
from app.gradcam import GradCAMExplainer

# Page configuration
st.set_page_config(
    page_title="Pneumonia AI Diagnostic Suite",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .badge-bacterial {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-viral {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-normal {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .disclaimer-box {
        font-size: 0.85rem;
        color: #64748B;
        border-left: 3px solid #CBD5E1;
        padding-left: 10px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ai_engine(variant: str = "exp2"):
    """Cache and load the selected PyTorch MobileNetV2 classifier variant."""
    classifier = PneumoniaClassifier(model_variant=variant)
    explainer = GradCAMExplainer(classifier)
    return classifier, explainer


# Sidebar: Controls & Information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lungs.png", width=70)
    st.title("Diagnostic Suite")
    st.markdown("Academic research system for 3-class pediatric Chest X-Ray triage.")

    st.markdown("---")
    st.subheader("🔬 Model Architecture Variant")
    model_choice = st.radio(
        "Select Active Checkpoint:",
        options=[
            "Exp 2: Controlled Baseline (Frozen MobileNetV2) [Default]",
            "Exp 5: Validation Winner (Top-4 Fine-Tuned)"
        ],
        index=0,
        help="Exp 2 is the controlled experimental baseline (frozen features + class-weighted head). Exp 5 fine-tunes the top 4 feature blocks and scored higher on internal validation, but lower on the historical Test split."
    )
    
    variant_key = "exp2" if model_choice.startswith("Exp 2") else "exp5"
    classifier, explainer = load_ai_engine(variant_key)

    st.markdown("---")
    st.subheader("📁 Demo Images")
    st.caption("Drawn from the historical Test split; for demonstration only.")
    samples_dir = Path(__file__).parent / "sample_images"
    sample_files = list(samples_dir.glob("*.jpeg")) + list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))

    sample_dict = {f.name.replace("_", " ").replace(".jpeg", "").title(): f for f in sample_files}
    sample_choice = st.selectbox(
        "Choose pre-loaded sample:",
        options=["-- Upload your own --"] + list(sample_dict.keys()),
        index=0,
    )

    st.markdown("---")
    st.subheader("⚙️ Active Model Profile")
    if variant_key == "exp2":
        st.markdown("""
        - **Model:** Exp 2 Baseline (Seed 42)
        - **Architecture:** MobileNetV2 (Frozen)
        - **Classifier:** 3-Class Weighted Head
        - **Val Macro F1 (Single):** `77.55%`
        - **Val Normal Recall:** `95.19%`
        - **Val Viral Recall:** `61.75%`
        - **Val Bacterial Recall:** `76.49%`
        - **Role:** Controlled Baseline Anchor
        """)
    else:
        st.markdown("""
        - **Model:** Exp 5 Fine-Tuned (Seed 42)
        - **Architecture:** MobileNetV2 (Top 4 Feature Blocks Unfrozen)
        - **Classifier:** 3-Class Weighted Head
        - **Val Macro F1 (Single):** `79.56%`
        - **Val Normal Recall:** `96.67%`
        - **Val Viral Recall:** `65.74%`
        - **Val Bacterial Recall:** `76.49%`
        - **Role:** Peak Validation Explorer
        """)
    st.caption("Validation figures come from the same set used for model selection and are therefore optimistic.")

    st.markdown("---")
    st.caption("Developed by Loay Alrazi • Applied AI Engineer")

# Main Page Header
st.markdown('<div class="main-header">🫁 Interpretable Chest X-Ray Diagnosis</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Automated multiclass screening (Normal vs. Bacterial vs. Viral) with Grad-CAM visual explanations. Active Model: <b>{classifier.experiment_name}</b>.</div>', unsafe_allow_html=True)

# Image Input Section
uploaded_file = st.file_uploader(
    "Drop a patient chest X-ray image (DICOM export, JPEG, or PNG):",
    type=["jpeg", "jpg", "png"],
)

input_image = None
image_title = ""

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("RGB")
    image_title = uploaded_file.name
elif sample_choice != "-- Upload your own --":
    sample_path = sample_dict[sample_choice]
    input_image = Image.open(sample_path).convert("RGB")
    image_title = sample_choice

if input_image is None:
    # Default initial view
    st.info("👆 Please select a sample from the sidebar or upload a chest X-ray to run diagnosis.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Original Release Model (Exp 0): Historical Test Results")
        conf_img = Path(__file__).parent / "docs" / "assets" / "confusion_matrix.png"
        if conf_img.exists():
            st.image(str(conf_img), caption="Exp 0 (original 3-epoch model, not the active model): confusion matrix on the historical Test split (N=624)", use_container_width=True)
    with col2:
        st.subheader("Explainability Overview (Grad-CAM)")
        cam_img = Path(__file__).parent / "docs" / "assets" / "gradcam_sample.png"
        if cam_img.exists():
            st.image(str(cam_img), caption="Grad-CAM example generated with the original Exp 0 model", use_container_width=True)
else:
    # Run Inference
    t0 = time.time()
    predicted_class, confidence, scores = classifier.predict(input_image)
    inference_time = (time.time() - t0) * 1000.0

    tabs = st.tabs(["🎯 Diagnostic Result", "🔬 Grad-CAM Explainability", "📊 Benchmark Curves"])

    with tabs[0]:
        col_img, col_pred = st.columns([1, 1.2], gap="large")

        with col_img:
            st.image(input_image, caption=f"Input X-Ray: {image_title}", use_container_width=True)

        with col_pred:
            st.markdown("### Model Assessment")
            
            badge_class = f"badge-{predicted_class.lower()}"
            st.markdown(f'<span class="{badge_class}">{predicted_class} PNEUMONIA</span>' if predicted_class != "NORMAL" else f'<span class="{badge_class}">NORMAL (NO PNEUMONIA)</span>', unsafe_allow_html=True)
            st.markdown(f"**Confidence Level:** `{confidence * 100:.2f}%` &nbsp;|&nbsp; **Latency:** `{inference_time:.1f} ms` on `{classifier.device.type.upper()}`")

            st.markdown("---")
            st.markdown("#### Differential Probability Distribution")
            
            for cls_name, prob in scores.items():
                col_lbl, col_bar, col_pct = st.columns([1.2, 3, 0.8])
                with col_lbl:
                    st.write(f"**{cls_name}**")
                with col_bar:
                    st.progress(prob)
                with col_pct:
                    st.write(f"{prob * 100:.1f}%")

            st.markdown("""
            <div class="disclaimer-box">
                <b>Academic Research Disclaimer:</b><br>
                This software is intended for research and educational demonstration only.
                It is not approved by medical regulatory authorities for primary diagnosis or autonomous triage.
                Radiographic interpretation must always be performed by licensed healthcare professionals.
            </div>
            """, unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("### Grad-CAM Anatomic Localization")
        st.markdown("Visualizing which pulmonary regions contributed most to the model's classification.")

        ctrl_col1, ctrl_col2 = st.columns([1, 1])
        with ctrl_col1:
            explain_class = st.selectbox(
                "Class to explain:",
                options=classifier.class_names,
                index=classifier.class_names.index(predicted_class)
            )
        with ctrl_col2:
            alpha_val = st.slider("Heatmap Transparency (Alpha):", min_value=0.1, max_value=0.9, value=0.45, step=0.05)

        target_idx = classifier.class_names.index(explain_class)
        heatmap, overlay_img, _ = explainer.explain(input_image, target_class=target_idx, alpha=alpha_val)

        cam_c1, cam_c2, cam_c3 = st.columns([1, 1, 1])
        with cam_c1:
            st.image(input_image.resize((224, 224)), caption="1. Original Preprocessed X-Ray", use_container_width=True)
        with cam_c2:
            st.image(heatmap, caption=f"2. Grad-CAM Activation ({explain_class})", clamp=True, use_container_width=True)
        with cam_c3:
            st.image(overlay_img, caption="3. Anatomical Overlay", use_container_width=True)

        st.caption("Warm regions (Red/Yellow) denote high feature activation; cool regions (Blue/Cyan) denote low activation.")

    with tabs[2]:
        st.markdown("### Historical Analytics (Original Exp 0 Release Model)")
        st.caption("These figures were produced by the original 3-epoch Exp 0 model, not by the active Exp 2 / Exp 5 checkpoint.")
        b1, b2 = st.columns([1, 1])
        with b1:
            curves_path = Path(__file__).parent / "docs" / "assets" / "learning_curves.png"
            if curves_path.exists():
                st.image(str(curves_path), caption="Exp 0: training & validation loss / accuracy across 3 epochs", use_container_width=True)
        with b2:
            matrix_path = Path(__file__).parent / "docs" / "assets" / "confusion_matrix.png"
            if matrix_path.exists():
                st.image(str(matrix_path), caption="Exp 0: confusion matrix on the historical Test split (N=624)", use_container_width=True)
