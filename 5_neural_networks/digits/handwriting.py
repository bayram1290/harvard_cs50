import argparse, numpy as np
from keras.datasets.mnist import load_data
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense, Dropout
from keras.saving import save_model

def loadData() -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:

    (x_train, y_train), (x_test, y_test) = load_data()

    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


def loadModel() -> Sequential:
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(rate=0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    pr = argparse.ArgumentParser(description='Train MNIST CNN model')
    pr.add_argument('--save', type=str, help='Filename to save the model')
    args = pr.parse_args()

    # Preprocess data
    (x_train, y_train), (x_test, y_test) = loadData()

    # Create & train model
    model = loadModel()

    # Display model
    model.summary()

    # Train the model
    print("\nTraining the model...")
    history = model.fit(
        x_train, y_train,
        epochs=10,
        validation_data=(x_test, y_test),
        verbose='1'
    )

    # Evaluate the model
    print("\nEvaluating the model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose='0')
    print(f'Test accuracy: {test_accuracy: .4f}')
    print(f'Test loss: {test_loss: .4f}')

    # Save model if filename provided
    if args.save:
        filename = args.save
        save_model(model, filename)
        print(f'Model saved to {filename}')


if __name__ == '__main__':
    main()