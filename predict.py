from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


IMAGE_SIZE = (224, 224)

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "pneumonia_detector.keras"
CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.txt"


def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def load_class_names():
    return [line.strip() for line in CLASS_NAMES_PATH.read_text().splitlines() if line.strip()]


def prepare_image(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = np.array(image)
    return np.expand_dims(image_array, axis=0)


def predict_image(image, model=None, class_names=None):
    model = model or load_model()
    class_names = class_names or load_class_names()

    prepared_image = prepare_image(image)
    probabilities = model.predict(prepared_image, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    return {
        "label": class_names[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            class_names[index]: float(probability)
            for index, probability in enumerate(probabilities)
        },
    }


if __name__ == "__main__":
    image_path = input("Enter image path: ").strip().strip('"')
    result = predict_image(Image.open(image_path))
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
