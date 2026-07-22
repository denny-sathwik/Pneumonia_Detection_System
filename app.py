from pathlib import Path

import streamlit as st
from PIL import Image

from predict import CLASS_NAMES_PATH, MODEL_PATH, load_class_names, load_model, predict_image


st.set_page_config(
    page_title="Pneumonia-detection-system Chest X-ray Analysis",
    page_icon="🫁",
    layout="centered",
)

st.markdown(
    """
    <style>
      /* ── global ── */
      html, body, [class*="css"] { font-family: -apple-system, "Segoe UI", system-ui, sans-serif; }

      /* ── hero ── */
      .hero { text-align: center; padding: 2.8rem 0 1.6rem; }
      .hero-badge {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.28rem 0.8rem;
        border-radius: 99px;
        border: 1px solid #bfdbfe;
        margin-bottom: 1rem;
      }
      .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: inherit;
        line-height: 1.1;
        margin-bottom: 0.6rem;
      }
      .hero-sub {
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 520px;
        margin: 0 auto 0;
        line-height: 1.6;
      }

      /* ── upload card ── */
      .upload-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: #64748b;
        margin-bottom: 0.4rem;
      }

      /* ── result card ── */
      .result-card {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 1.4rem 1.6rem;
        margin-top: 1.2rem;
        background: #ffffff;
      }
      .result-label-normal   { color: #15803d; font-size: 1.7rem; font-weight: 800; }
      .result-label-pneumonia { color: #b91c1c; font-size: 1.7rem; font-weight: 800; }
      .result-confidence {
        font-size: 0.9rem;
        color: #64748b;
        margin-top: 0.15rem;
        margin-bottom: 1rem;
      }
      .conf-bar-wrap {
        background: #f1f5f9;
        border-radius: 99px;
        height: 8px;
        overflow: hidden;
        margin-bottom: 1.2rem;
      }
      .conf-bar {
        height: 8px;
        border-radius: 99px;
        transition: width 0.4s ease;
      }
      .conf-normal   { background: #22c55e; }
      .conf-pneumonia { background: #ef4444; }

      /* ── footer ── */
      .footer {
        text-align: center;
        font-size: 0.78rem;
        color: #94a3b8;
        border-top: 1px solid #e2e8f0;
        padding-top: 1.2rem;
        margin-top: 2.8rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_class_names():
    return load_class_names()


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <div class="hero-badge">AI-Powered · Chest X-ray Analysis</div>
      <div class="hero-title">Pneumonia-detection-system</div>
      <div class="hero-sub">
        Upload a chest X-ray and get an instant AI-driven assessment for
        signs of pneumonia — powered by MobileNetV2 transfer learning.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Classifier ────────────────────────────────────────────────────────────────
model_exists = Path(MODEL_PATH).exists()
classes_exist = Path(CLASS_NAMES_PATH).exists()

if not model_exists or not classes_exist:
    st.warning(
        "No trained model found. Run `python train_model.py` first to generate "
        "`model/pneumonia_detector.keras` and `model/class_names.txt`."
    )
else:
    model = get_model()
    class_names = get_class_names()

    st.markdown('<div class="upload-label">Upload X-ray image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("Drag and drop a chest X-ray image (JPG or PNG) to begin analysis.")
    else:
        image = Image.open(uploaded_file)
        result = predict_image(image, model=model, class_names=class_names)

        img_col, res_col = st.columns([1, 1], gap="large")

        with img_col:
            st.image(image, use_container_width=True)

        with res_col:
            label = result["label"]
            confidence = result["confidence"]
            is_pneumonia = label.upper() == "PNEUMONIA"
            label_class = "result-label-pneumonia" if is_pneumonia else "result-label-normal"
            bar_class = "conf-pneumonia" if is_pneumonia else "conf-normal"
            bar_pct = round(confidence * 100, 1)

            st.markdown(
                f"""
                <div class="result-card">
                  <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;
                              text-transform:uppercase;color:#94a3b8;margin-bottom:0.4rem;">
                    Prediction
                  </div>
                  <div class="{label_class}">{label}</div>
                  <div class="result-confidence">{bar_pct}% confidence</div>
                  <div class="conf-bar-wrap">
                    <div class="conf-bar {bar_class}" style="width:{bar_pct}%"></div>
                  </div>
                  <div style="font-size:0.78rem;font-weight:700;letter-spacing:0.07em;
                              text-transform:uppercase;color:#64748b;margin-bottom:0.6rem;">
                    Class probabilities
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            for cls, prob in result["probabilities"].items():
                st.progress(prob, text=f"{cls}  {prob:.1%}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">PneumoScan · Built with Streamlit &amp; TensorFlow · Educational use only</div>',
    unsafe_allow_html=True,
)
