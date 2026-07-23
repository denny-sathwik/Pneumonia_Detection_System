from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
FEATURE_EXTRACTION_EPOCHS = 5
FINE_TUNE_EPOCHS = 3
FINE_TUNE_AT = 100
VALIDATION_SPLIT = 0.2
SEED = 42

BASE_DIR = Path(__file__).parent
LOCAL_DATA_DIR = BASE_DIR / "data" / "chest_xray"
KAGGLE_DOWNLOAD_DATA_DIR = Path.home() / "Downloads" / "chest_xray" / "chest_xray"
DATA_DIR = KAGGLE_DOWNLOAD_DATA_DIR if KAGGLE_DOWNLOAD_DATA_DIR.exists() else LOCAL_DATA_DIR
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "pneumonia_detector.keras"
BEST_MODEL_PATH = MODEL_DIR / "best_pneumonia_detector.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.txt"


def get_image_paths(class_dir):
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(class_dir.glob(extension))
    return sorted(image_paths)


def load_and_resize_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    return image, label


def make_dataset(image_paths, labels, shuffle):
    path_ds = tf.data.Dataset.from_tensor_slices([str(path) for path in image_paths])
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((path_ds, label_ds))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=SEED)

    return (
        dataset
        .map(load_and_resize_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
    )


def load_balanced_training_datasets():
    class_names = sorted(
        path.name for path in (DATA_DIR / "train").iterdir() if path.is_dir()
    )

    normal_paths = get_image_paths(DATA_DIR / "train" / class_names[0])
    pneumonia_paths = get_image_paths(DATA_DIR / "train" / class_names[1])

    sample_count = min(len(normal_paths), len(pneumonia_paths))
    rng = np.random.default_rng(SEED)

    normal_paths = rng.choice(normal_paths, size=sample_count, replace=False)
    pneumonia_paths = rng.choice(pneumonia_paths, size=sample_count, replace=False)

    image_paths = np.concatenate([normal_paths, pneumonia_paths])
    labels = np.array(
        [[1, 0]] * sample_count + [[0, 1]] * sample_count,
        dtype=np.float32,
    )

    indices = rng.permutation(len(image_paths))
    image_paths = image_paths[indices]
    labels = labels[indices]

    validation_count = int(len(image_paths) * VALIDATION_SPLIT)
    val_paths = image_paths[:validation_count]
    val_labels = labels[:validation_count]
    train_paths = image_paths[validation_count:]
    train_labels = labels[validation_count:]

    print(f"Balanced training images per class: {sample_count}")
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")

    train_ds = make_dataset(train_paths, train_labels, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels, shuffle=False)
    return train_ds, val_ds, class_names


def load_test_dataset():
    return tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
    )


def count_images_by_class(split_name, class_names):
    return {
        index: len(list((DATA_DIR / split_name / class_name).glob("*")))
        for index, class_name in enumerate(class_names)
    }


def build_class_weights(class_counts):
    total_images = sum(class_counts.values())
    class_count = len(class_counts)
    return {
        class_index: total_images / (class_count * image_count)
        for class_index, image_count in class_counts.items()
        if image_count > 0
    }


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
    return model, base_model


def fine_tune_model(model, base_model):
    base_model.trainable = True

    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def merge_histories(*histories):
    combined = {}
    for history in histories:
        for key, values in history.history.items():
            combined.setdefault(key, []).extend(values)
    return combined


def save_training_plot(history_data):
    plt.figure(figsize=(8, 4))
    plt.plot(history_data["accuracy"], label="train accuracy")
    plt.plot(history_data["val_accuracy"], label="validation accuracy")
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

    print(f"Using dataset from {DATA_DIR}")

    train_ds, val_ds, class_names = load_balanced_training_datasets()
    CLASS_NAMES_PATH.write_text("\n".join(class_names))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = load_test_dataset().prefetch(tf.data.AUTOTUNE)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
        ),
    ]

    model, base_model = build_model(class_count=len(class_names))
    feature_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FEATURE_EXTRACTION_EPOCHS,
        callbacks=callbacks,
    )

    fine_tune_model(model, base_model)
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_PATH)
    save_training_plot(merge_histories(feature_history, fine_tune_history))
    test_loss, test_accuracy = model.evaluate(test_ds)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved best model to {BEST_MODEL_PATH}")
    print(f"Saved class names to {CLASS_NAMES_PATH}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
