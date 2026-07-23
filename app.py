from pathlib import Path

import numpy as np
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from PIL import Image, ImageOps, UnidentifiedImageError

from predict import CLASS_NAMES_PATH, MODEL_PATH, load_class_names, load_model, predict_image
from xray_gate import XRAY_GATE_MODEL_PATH, XRAY_GATE_THRESHOLD, is_chest_xray, load_xray_gate_model


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

      .login-panel {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        background: #ffffff;
        margin: 0.5rem 0 1.5rem;
      }
      .login-title {
        font-size: 1rem;
        font-weight: 750;
        margin-bottom: 0.3rem;
      }
      .login-copy {
        color: #64748b;
        font-size: 0.9rem;
        line-height: 1.5;
        margin-bottom: 0;
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


@st.cache_resource
def get_xray_gate_model():
    return load_xray_gate_model()


@st.cache_data
def get_class_names():
    return load_class_names()


def get_auth_config():
    try:
        return st.secrets.get("auth", {})
    except StreamlitSecretNotFoundError:
        return {}


def auth_is_configured():
    auth_config = get_auth_config()
    required_keys = (
        "redirect_uri",
        "cookie_secret",
        "client_id",
        "client_secret",
        "server_metadata_url",
    )

    return (
        bool(auth_config)
        and all(auth_config.get(key) for key in required_keys)
        and auth_config.get("server_metadata_url")
        == "https://accounts.google.com/.well-known/openid-configuration"
    )


def get_allowed_emails():
    try:
        app_config = st.secrets.get("app", {})
    except StreamlitSecretNotFoundError:
        return set()

    emails = app_config.get("allowed_emails", [])
    if isinstance(emails, str):
        emails = [emails]

    return {str(email).strip().lower() for email in emails if str(email).strip()}


def require_auth():
    if not auth_is_configured():
        st.error(
            "Authentication is not configured yet. Copy "
            "`.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` "
            "and fill in your Google OAuth client values."
        )
        st.stop()

    if not st.user.is_logged_in:
        st.markdown(
            """
            <div class="login-panel">
              <div class="login-title">Sign in required</div>
              <p class="login-copy">
                This app analyzes chest X-ray images, so access is limited to
                authenticated users before any uploads or predictions are shown.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button("Log in with Google", on_click=st.login, use_container_width=True)
        st.stop()

    allowed_emails = get_allowed_emails()
    user_email = str(st.user.get("email", "")).lower()

    if allowed_emails and user_email not in allowed_emails:
        st.error("Your account is authenticated, but it is not authorized to use this app.")
        st.button("Log out", on_click=st.logout, use_container_width=True)
        st.stop()

    with st.sidebar:
        st.caption(f"Signed in as {st.user.get('email') or st.user.get('name') or 'user'}")
        st.button("Log out", on_click=st.logout, use_container_width=True)


def load_uploaded_image(uploaded_file):
    try:
        image = Image.open(uploaded_file)
        image.verify()
        uploaded_file.seek(0)
        image = Image.open(uploaded_file)
        return ImageOps.exif_transpose(image)
    except (OSError, UnidentifiedImageError):
        return None


def passes_color_gate(image):
    rgb_image = image.convert("RGB").resize((224, 224))
    image_array = np.asarray(rgb_image, dtype=np.float32)

    max_channel = np.max(image_array, axis=2)
    min_channel = np.min(image_array, axis=2)
    channel_spread = max_channel - min_channel
    saturation = np.divide(
        channel_spread,
        np.maximum(max_channel, 1),
        out=np.zeros_like(channel_spread),
        where=max_channel > 0,
    )

    mean_saturation = float(np.mean(saturation))
    colored_pixel_ratio = float(np.mean((saturation > 0.35) & (max_channel > 45)))

    passed = mean_saturation <= 0.22 or colored_pixel_ratio <= 0.18
    return passed, {
        "mean_saturation": mean_saturation,
        "colored_pixel_ratio": colored_pixel_ratio,
    }


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
require_auth()

model_exists = Path(MODEL_PATH).exists()
classes_exist = Path(CLASS_NAMES_PATH).exists()
xray_gate_exists = Path(XRAY_GATE_MODEL_PATH).exists()

if not model_exists or not classes_exist or not xray_gate_exists:
    st.warning(
        "Required model files are missing. Run `python train_model.py` for pneumonia "
        "classification and `python train_xray_gate.py` for chest X-ray validation."
    )
else:
    st.markdown('<div class="upload-label">Upload chest X-ray image only</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="upload",
        type=["jpg", "jpeg", "png"],
        help="Upload only a clear chest X-ray.",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("Drag and drop a clear chest X-ray image (JPG or PNG) to begin analysis.")
    else:
        image = load_uploaded_image(uploaded_file)
        if image is None:
            st.error("The uploaded file could not be opened as an image. Please upload a chest X-ray JPG or PNG.")
            st.stop()

        passed_color_gate, color_gate_metrics = passes_color_gate(image)
        if not passed_color_gate:
            st.image(image, use_container_width=True)
            st.error("Image appears to be a color image.")
            st.warning("Please upload a clear grayscale chest X-ray.")
            st.caption(
                "Color score: "
                f"{color_gate_metrics['mean_saturation']:.1%} mean saturation, "
                f"{color_gate_metrics['colored_pixel_ratio']:.1%} colored pixels."
            )
            st.stop()

        xray_gate_model = get_xray_gate_model()
        is_valid_xray, xray_probability = is_chest_xray(image, model=xray_gate_model)
        if not is_valid_xray:
            st.image(image, use_container_width=True)
            st.error("Image does not appear to be a chest X-ray.")
            st.warning(
                "Please upload only clear chest X-ray images."
            )
            st.caption(
                f"Chest X-ray gate score: {xray_probability:.1%} "
                f"(required: {XRAY_GATE_THRESHOLD:.0%})"
            )
            st.stop()

        model = get_model()
        class_names = get_class_names()
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
