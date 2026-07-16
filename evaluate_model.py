from pathlib import Path

import numpy as np
import tensorflow as tf


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model" / "best_pneumonia_detector.keras"
TEST_DIR = Path.home() / "Downloads" / "chest_xray" / "chest_xray" / "test"


def main():
    model = tf.keras.models.load_model(MODEL_PATH)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = test_ds.class_names

    true_labels = []
    predicted_labels = []
    confidences = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)

        true_labels.extend(np.argmax(labels.numpy(), axis=1))
        predicted_labels.extend(np.argmax(predictions, axis=1))
        confidences.extend(np.max(predictions, axis=1))

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    confidences = np.array(confidences)

    print("Classes:", class_names)
    print("Overall accuracy:", np.mean(true_labels == predicted_labels))

    print("\nConfusion matrix")
    print("Rows = actual, columns = predicted")

    matrix = np.zeros((len(class_names), len(class_names)), dtype=int)

    for actual, predicted in zip(true_labels, predicted_labels):
        matrix[actual][predicted] += 1

    print(matrix)

    print("\nPer-class accuracy")
    for index, class_name in enumerate(class_names):
        class_mask = true_labels == index
        class_accuracy = np.mean(predicted_labels[class_mask] == true_labels[class_mask])
        print(f"{class_name}: {class_accuracy:.2%}")

    correct_mask = true_labels == predicted_labels
    print("\nAverage confidence")
    print(f"Correct predictions: {np.mean(confidences[correct_mask]):.2%}")
    print(f"Wrong predictions: {np.mean(confidences[~correct_mask]):.2%}")


if __name__ == "__main__":
    main()