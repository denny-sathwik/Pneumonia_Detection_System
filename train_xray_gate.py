from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
FEATURE_EXTRACTION_EPOCHS = 6
FINE_TUNE_EPOCHS = 4
FINE_TUNE_AT = 100
VALIDATION_SPLIT = 0.2
SEED = 42
MIN_IMAGES_PER_CLASS = 10
RECOMMENDED_IMAGES_PER_CLASS = 200

BASE_DIR = Path(__file__).parent
CHEST_XRAY_DIR = BASE_DIR / "data" / "chest_xray"
NOT_CHEST_XRAY_DIR = BASE_DIR / "data" / "not_chest_xray"
MODEL_DIR = BASE_DIR / "model"
XRAY_GATE_MODEL_PATH = MODEL_DIR / "xray_gate.keras"
XRAY_GATE_PLOT_PATH = MODEL_DIR / "xray_gate_accuracy.png"


def get_image_paths(root_dir):
    image_extensions = ("*.jpg", "*.jpeg", "*.png")
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(root_dir.rglob(extension))
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


def load_gate_datasets():
    if not CHEST_XRAY_DIR.exists():
        raise FileNotFoundError(f"Chest X-ray data not found at {CHEST_XRAY_DIR}")

    if not NOT_CHEST_XRAY_DIR.exists():
        NOT_CHEST_XRAY_DIR.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(
            f"Created negative examples folder at {NOT_CHEST_XRAY_DIR}. "
            "Add at least 20 non-chest-X-ray JPG or PNG images there, then run this script again."
        )

    chest_paths = get_image_paths(CHEST_XRAY_DIR)
    not_chest_paths = get_image_paths(NOT_CHEST_XRAY_DIR)
    sample_count = min(len(chest_paths), len(not_chest_paths))

    if sample_count < MIN_IMAGES_PER_CLASS:
        missing_dir = CHEST_XRAY_DIR if len(chest_paths) < len(not_chest_paths) else NOT_CHEST_XRAY_DIR
        raise ValueError(
            f"The X-ray gate needs at least {MIN_IMAGES_PER_CLASS} chest X-ray images "
            f"and {MIN_IMAGES_PER_CLASS} non-chest-X-ray images. "
            f"Found {len(chest_paths)} chest X-rays and {len(not_chest_paths)} negatives. "
            f"Add more JPG or PNG files under {missing_dir}."
        )

    if sample_count < RECOMMENDED_IMAGES_PER_CLASS:
        print(
            "WARNING: Training with a very small gate dataset. "
            f"Found {len(chest_paths)} chest X-rays and {len(not_chest_paths)} negatives; "
            f"using {sample_count} per class. For a stronger barrier, use at least "
            f"{RECOMMENDED_IMAGES_PER_CLASS} varied images per class."
        )

    rng = np.random.default_rng(SEED)
    chest_paths = rng.choice(chest_paths, size=sample_count, replace=False)
    not_chest_paths = rng.choice(not_chest_paths, size=sample_count, replace=False)

    chest_paths = chest_paths[rng.permutation(sample_count)]
    not_chest_paths = not_chest_paths[rng.permutation(sample_count)]

    validation_count = max(1, int(sample_count * VALIDATION_SPLIT))

    val_paths = np.concatenate([
        chest_paths[:validation_count],
        not_chest_paths[:validation_count],
    ])
    val_labels = np.array([1] * validation_count + [0] * validation_count, dtype=np.float32)

    train_paths = np.concatenate([
        chest_paths[validation_count:],
        not_chest_paths[validation_count:],
    ])
    train_labels = np.array(
        [1] * (sample_count - validation_count)
        + [0] * (sample_count - validation_count),
        dtype=np.float32,
    )

    train_indices = rng.permutation(len(train_paths))
    val_indices = rng.permutation(len(val_paths))
    train_paths = train_paths[train_indices]
    train_labels = train_labels[train_indices]
    val_paths = val_paths[val_indices]
    val_labels = val_labels[val_indices]

    print(f"Chest X-ray examples used: {sample_count}")
    print(f"Non-chest-X-ray examples used: {sample_count}")
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Validation images per class: {validation_count}")

    train_ds = make_dataset(train_paths, train_labels, shuffle=True)
    val_ds = make_dataset(val_paths, val_labels, shuffle=False)
    return train_ds, val_ds


def build_model():
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
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
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    compile_model(model, learning_rate=0.0001)
    return model, base_model


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def fine_tune_model(model, base_model):
    base_model.trainable = True

    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    compile_model(model, learning_rate=0.00001)


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
    plt.savefig(XRAY_GATE_PLOT_PATH)


def main():
    MODEL_DIR.mkdir(exist_ok=True)

    train_ds, val_ds = load_gate_datasets()
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            XRAY_GATE_MODEL_PATH,
            monitor="val_auc",
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

    model, base_model = build_model()
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

    model.save(XRAY_GATE_MODEL_PATH)
    save_training_plot(merge_histories(feature_history, fine_tune_history))
    validation_metrics = model.evaluate(val_ds, verbose=0, return_dict=True)
    print(f"Saved X-ray gate model to {XRAY_GATE_MODEL_PATH}")
    print(f"Saved training plot to {XRAY_GATE_PLOT_PATH}")
    print("Final validation metrics:")
    for metric_name, metric_value in validation_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
