"""Streamlit Web Application: Chest X-Ray Pneumonia Multiclass Diagnosis with Grad-CAM."""

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
def load_ai_engine():
    """Cache and load the PyTorch MobileNetV2 classifier."""
    checkpoint_path = Path(__file__).parent / "artifacts" / "pneumonia_3class.pt"
    classifier = PneumoniaClassifier(checkpoint_path=checkpoint_path)
    explainer = GradCAMExplainer(classifier)
    return classifier, explainer


classifier, explainer = load_ai_engine()

# Sidebar: Controls & Information
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lungs.png", width=70)
    st.title("Diagnostic Suite")
    st.markdown("Deep Learning system for 3-class pediatric Chest X-Ray triage.")

    st.markdown("---")
    st.subheader("📁 Test Samples")
    samples_dir = Path(__file__).parent / "sample_images"
    sample_files = list(samples_dir.glob("*.jpeg")) + list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))

    sample_dict = {f.name.replace("_", " ").replace(".jpeg", "").title(): f for f in sample_files}
    sample_choice = st.selectbox(
        "Choose pre-loaded sample:",
        options=["-- Upload your own --"] + list(sample_dict.keys()),
        index=0,
    )

    st.markdown("---")
    st.subheader("⚙️ Model Specifications")
    st.markdown("""
    - **Architecture:** MobileNetV2
    - **Classes:** Normal / Bacterial / Viral
    - **Pre-trained:** ImageNet-1K V2
    - **Target Resolution:** 224 × 224 px
    - **Bacterial Sensitivity:** `93.8%`
    - **Viral Sensitivity:** `74.3%`
    """)

    st.markdown("---")
    st.caption("Developed by Loay Alrazi • Applied AI Engineer")

# Main Page Header
st.markdown('<div class="main-header">🫁 Interpretable Chest X-Ray Diagnosis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated multiclass screening (Normal vs. Bacterial vs. Viral Pneumonia) with Grad-CAM visual explanations.</div>', unsafe_allow_html=True)

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
        st.subheader("Model Performance on Test Callset")
        conf_img = Path(__file__).parent / "docs" / "assets" / "confusion_matrix.png"
        if conf_img.exists():
            st.image(str(conf_img), caption="Test Set Confusion Matrix", use_container_width=True)
    with col2:
        st.subheader("Explainability Overview (Grad-CAM)")
        cam_img = Path(__file__).parent / "docs" / "assets" / "gradcam_sample.png"
        if cam_img.exists():
            st.image(str(cam_img), caption="Grad-CAM Anatomic Activation Example", use_container_width=True)
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
            st.markdown("### Clinical Assessment")
            
            badge_class = f"badge-{predicted_class.lower()}"
            st.markdown(f'<span class="{badge_class}">{predicted_class} PNEUMONIA</span>' if predicted_class != "NORMAL" else f'<span class="{badge_class}">NORMAL (NO PNEUMONIA)</span>', unsafe_allow_html=True)
            st.markdown(f"**Confidence Level:** `{confidence * 100:.2f}%` &nbsp;|&nbsp; **Latency:** `{inference_time:.1f} ms`")

            st.markdown("---")
            st.markdown("#### Differential Probability Distribution")
            
            colors = {
                "BACTERIAL": "#EF4444",
                "NORMAL": "#10B981",
                "VIRAL": "#F59E0B"
            }

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
                <b>Clinical Context Note:</b><br>
                Bacterial pneumonia typically shows focal lobar consolidation and requires antibiotic treatment.
                Viral pneumonia presents with diffuse interstitial patterns and requires supportive/antiviral care.
                This software is intended as an investigational decision-support tool.
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
        st.markdown("### Training & Validation Analytics")
        b1, b2 = st.columns([1, 1])
        with b1:
            curves_path = Path(__file__).parent / "docs" / "assets" / "learning_curves.png"
            if curves_path.exists():
                st.image(str(curves_path), caption="Training & Validation Loss / Accuracy Across Epochs", use_container_width=True)
        with b2:
            matrix_path = Path(__file__).parent / "docs" / "assets" / "confusion_matrix.png"
            if matrix_path.exists():
                st.image(str(matrix_path), caption="Multiclass Confusion Matrix on Independent Test Set (624 images)", use_container_width=True)
