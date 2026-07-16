from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "chest_xray"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "pneumonia_detector.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"


def load_dataset(split_name):
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / split_name,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=split_name == "train",
    )


def build_model(class_count):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_plot(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="train accuracy")
    plt.plot(history.history["val_accuracy"], label="validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "training_accuracy.png")


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_DIR}. "
            "Create data/chest_xray/train, data/chest_xray/val, and data/chest_xray/test first."
        )

    MODEL_DIR.mkdir(exist_ok=True)

    train_ds = load_dataset("train")
    val_ds = load_dataset("val")

    class_names = train_ds.class_names
    CLASS_NAMES_PATH.write_text("\n".join(class_names))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(class_count=len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    model.save(MODEL_PATH)
    save_training_plot(history)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved class names to {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    main()
