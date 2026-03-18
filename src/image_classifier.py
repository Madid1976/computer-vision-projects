import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def build_cnn_model(input_shape=(32, 32, 3), num_classes=10):
    """
    Builds a simple Convolutional Neural Network (CNN) model.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.

    Returns:
        tf.keras.Model: A compiled Keras CNN model.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation=\'relu\', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=\'relu\'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation=\'relu\'),
        layers.Flatten(),
        layers.Dense(64, activation=\'relu\'),
        layers.Dense(num_classes, activation=\'softmax\')
    ])

    model.compile(optimizer=\'adam\', 
                  loss=\'categorical_crossentropy\', 
                  metrics=[\'accuracy\'])
    return model

def load_and_preprocess_cifar10():
    """
    Loads and preprocesses the CIFAR-10 dataset.

    Returns:
        tuple: (x_train, y_train, x_test, y_test) preprocessed data.
    """
    print("Loading CIFAR-10 dataset...")
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # One-hot encode labels
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels

def train_and_evaluate_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=64):
    """
    Trains and evaluates the given CNN model.

    Args:
        model (tf.keras.Model): The compiled Keras model.
        x_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        x_test (numpy.ndarray): Test images.
        y_test (numpy.ndarray): Test labels.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
    """
    print("Training model...")
    history = model.fit(x_train, y_train, epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(x_test, y_test))

    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    # Load and preprocess data
    x_train, y_train, x_test, y_test = load_and_preprocess_cifar10()

    # Build the model
    model = build_cnn_model()
    model.summary()

    # Train and evaluate
    train_and_evaluate_model(model, x_train, y_train, x_test, y_test, epochs=1)
    print("Image classification pipeline finished successfully!")
