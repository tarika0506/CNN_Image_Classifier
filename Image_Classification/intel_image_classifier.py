import os
import tensorflow as tf

# Load datasets
def load_datasets(data_dir, img_size=(150, 150), batch_size=32):
    train_path = os.path.join(data_dir, "seg_train")
    test_path = os.path.join(data_dir, "seg_test")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=123
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=123
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Normalize images
    normalization = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization(x), y))

    return train_ds, val_ds, test_ds

# Build a basic CNN
def build_basic_cnn(input_shape=(150, 150, 3), num_classes=6):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

# Train and evaluate
def train_and_evaluate(model, train_ds, val_ds, test_ds, epochs=5):
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}\n")

    return history, test_acc

# Main
if __name__ == "__main__":
    dataset_path = r"C:\Users\dummy\Downloads\archive"
    train_ds, val_ds, test_ds = load_datasets(dataset_path)

    model = build_basic_cnn()
    train_and_evaluate(model, train_ds, val_ds, test_ds, epochs=5)






















