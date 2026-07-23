from pathlib import Path

import numpy as np
import tensorflow as tf


IMAGE_SIZE = (224, 224)
XRAY_GATE_THRESHOLD = 0.6

BASE_DIR = Path(__file__).parent
XRAY_GATE_MODEL_PATH = BASE_DIR / "model" / "xray_gate.keras"


def load_xray_gate_model():
    return tf.keras.models.load_model(XRAY_GATE_MODEL_PATH)


def prepare_gate_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.array(image)
    return np.expand_dims(image_array, axis=0)


def predict_chest_xray_probability(image, model=None):
    model = model or load_xray_gate_model()
    prepared_image = prepare_gate_image(image)
    prediction = model.predict(prepared_image, verbose=0)[0]
    return float(np.ravel(prediction)[0])


def is_chest_xray(image, model=None, threshold=XRAY_GATE_THRESHOLD):
    probability = predict_chest_xray_probability(image, model=model)
    return probability >= threshold, probability
