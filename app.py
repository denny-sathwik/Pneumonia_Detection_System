from pathlib import Path

import streamlit as st
from PIL import Image

from predict import CLASS_NAMES_PATH, MODEL_PATH, load_class_names, load_model, predict_image


st.set_page_config(
    page_title="My Pneumonia Detector",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main-title {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.5rem;
      }
      .subtitle {
        font-size: 1.08rem;
        color: #4b5563;
        max-width: 760px;
        margin-bottom: 1.25rem;
      }
      .notice {
        border-left: 4px solid #0f766e;
        background: #ecfdf5;
        padding: 0.9rem 1rem;
        border-radius: 6px;
        color: #134e4a;
      }
      .section-label {
        color: #0f766e;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
      }
      .metric-box {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
        min-height: 118px;
      }
      .metric-box strong {
        display: block;
        font-size: 1.4rem;
        color: #111827;
      }
      .metric-box span {
        color: #6b7280;
        font-size: 0.92rem;
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


left, right = st.columns([1.35, 1])

with left:
    st.markdown('<div class="section-label">Chest X-ray AI demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">My Pneumonia Detector</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="subtitle">
        A learning project that trains a TensorFlow image classifier and serves it through a
        simple Streamlit web app. Upload a chest X-ray image to see the model prediction.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="notice">
        Educational demo only. This app is not a medical diagnosis tool and should not be
        used for real medical decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

with right:
    st.markdown("#### Project Flow")
    st.write("Dataset folders -> training script -> saved model -> prediction helper -> web app")
    st.write("Current model: `MobileNetV2` transfer learning")

model_exists = Path(MODEL_PATH).exists()
classes_exist = Path(CLASS_NAMES_PATH).exists()

st.divider()

status_cols = st.columns(3)
with status_cols[0]:
    st.markdown(
        '<div class="metric-box"><strong>2 classes</strong><span>NORMAL and PNEUMONIA</span></div>',
        unsafe_allow_html=True,
    )
with status_cols[1]:
    model_status = "Ready" if model_exists and classes_exist else "Not trained"
    st.markdown(
        f'<div class="metric-box"><strong>{model_status}</strong><span>Saved model status</span></div>',
        unsafe_allow_html=True,
    )
with status_cols[2]:
    st.markdown(
        '<div class="metric-box"><strong>Streamlit</strong><span>Interactive web interface</span></div>',
        unsafe_allow_html=True,
    )

st.divider()

st.markdown("### Try The Classifier")

if not model_exists or not classes_exist:
    st.warning(
        "Train the model first by running `python train_model.py`. "
        "The app needs `model/pneumonia_detector.keras` and `model/class_names.txt`."
    )
else:
    model = get_model()
    class_names = get_class_names()

    upload_col, result_col = st.columns([1, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload a chest X-ray image",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is None:
            st.info("Choose an image to see a prediction.")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with upload_col:
            st.image(image, caption="Uploaded image", use_container_width=True)

        result = predict_image(image, model=model, class_names=class_names)

        with result_col:
            st.markdown("#### Prediction")
            st.subheader(result["label"])
            st.metric("Confidence", f"{result['confidence']:.2%}")
            st.write("Class probabilities")
            st.bar_chart(result["probabilities"])

st.divider()

with st.expander("About this project"):
    st.write(
        "This app separates the project into three parts: model training in "
        "`train_model.py`, reusable prediction logic in `predict.py`, and the web "
        "interface in `app.py`."
    )
    st.write(
        "For accurate results, train with a proper dataset split where training and "
        "validation images are different."
    )
