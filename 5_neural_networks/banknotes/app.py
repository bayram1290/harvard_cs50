import sys, csv, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.callbacks import History
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def loadData(filePath: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the data from the specified file path and return as numpy arrays.

    Parameters:
        filePath (str): The path to the file containing the data.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the evidence and labels as numpy arrays.
    """
    evidence = []
    labels = []

    # Open the specified file and read its contents
    with open(file=filePath, encoding='utf-8') as file:
        # Create a CSV reader object from the file
        reader = csv.reader(file)

        # Skip the header row
        next(reader)

        # Initialize two empty lists to store the evidence and labels
        evidence = []
        labels = []

        # Iterate over each row in the CSV file
        for row in reader:
            # Extract the evidence values from the row
            evidence_values = [float(cell) for cell in row[:4]]

            # Append the evidence values to the evidence list
            evidence.append(evidence_values)

            # Extract the label value from the row
            label_value = float(row[4])

            # Append the label value to the labels list
            labels.append(label_value)

    # Convert the evidence and labels lists to numpy arrays
    evidence_array = np.array(evidence)
    labels_array = np.array(labels)

    # Return the evidence and labels as numpy arrays
    return (evidence_array, labels_array)


def plotTrainingHistory(history: History) -> None:
    """
    Plot the training history of a neural network model.

    Parameters:
        history (History): The training history of the neural network model.

    Returns:
        None
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 1. Plot the training loss and validation loss (if available)
    ax1.plot(history.history['loss'], label='Training Loss')
    # If the model was trained with validation data, plot the validation loss
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # 2. Plot the training accuracy and validation accuracy (if available)
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    # If the model was trained with validation data, plot the validation accuracy
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # Ensure the plot is laid out nicely
    plt.tight_layout()
    # Display the plot
    plt.show()


def evaluateModel(model: Sequential, X_test: np.ndarray, y_test: np.ndarray, X_training: np.ndarray, y_training: np.ndarray):
    """
    Evaluate the model using the test and training data, and provide the model evaluation metrics.

    Parameters:
        model (Sequential): The neural network model to evaluate.
        X_test (np.ndarray): The input test data.
        y_test (np.ndarray): The target test labels.
        X_training (np.ndarray): The input training data.
        y_training (np.ndarray): The target training labels.

    Returns:
        dict: A dictionary containing the test accuracy, train accuracy, predicted labels, and probabilities.
    """
    print('-' * 30 + '\nMODEL EVALUATION\n' + '-' * 30)

    # 1. Test set predictions and evaluation
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    test_accuracy = accuracy_score(y_test, y_pred) # Calculate accuracy
    print(f'Test accuracy: {test_accuracy:.4f}')

    # 2. Training set evaluation (if provided)
    if X_training is not None and y_training is not None:
        y_training_pred_prob = model.predict(X_training)
        y_training_pred = (y_training_pred_prob > 0.5).astype(int).flatten()
        train_accuracy = accuracy_score(y_training, y_training_pred)
        print(f'Training accuracy: {train_accuracy:.4f}')
        print(f'Accuracy difference (Train - Test): {train_accuracy - test_accuracy}')

    # 3. Detailed classification report
    print('\n' + '-' * 30 + '\nCLASSIFICATION REPORT\n' + '-' * 30)
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

    # 4. Confusion Matrix
    print('\n' + '-' * 30 + '\nCONFUSION MATRIX\n' + '-' * 30)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion matrix')
    plt.show()

    # 5. Model loss and accuracy (if history is available)
    return {
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'predictions': y_pred,
        'probabilites': y_pred_prob
    }


def main() -> tuple[Sequential, History, dict]:
    """
    Runs the main function to train a neural network model using the banknotes dataset.
    Loads the data, preprocesses it, defines the neural network model architecture,
    compiles the model, trains the model, evaluates the model performance, and returns
    the trained model, training history, and evaluation results.

    Returns:
        tuple[Sequential, History, dict]: A tuple containing the trained model, training history, and evaluation results.
    """
    if len(sys.argv) != 2:
        sys.exit('App usage: python app.py path_to_banknotes_data_file.csv')

    filePath = sys.argv[2]

    evidence, labels = loadData(filePath)
    print(f'Evidence shape: {evidence.shape}')
    print(f'Labels shape: {labels.shape}')

    X_training, X_test, y_training, y_test = train_test_split(evidence, labels, test_size=0.4, random_state=42)
    input_shape = (X_training.shape[1],)
    print(f'Input shape: {input_shape}')

    model = Sequential([
        Input(shape=input_shape),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print('\n' + '-' * 30 + '\nMODEL ARCHITECTURE\n' + '-' * 30)
    model.summary()

    print('\n' + '-' * 30 + '\nMODEL TRAINING\n' + '-' * 30)
    history = model.fit(
        X_training,
        y_training,
        epochs=20,
        validation_data=(X_test, y_test),
        verbose='1'
    )
    plotTrainingHistory(history)

    evaluation_results = evaluateModel(
        model,
        X_test,
        y_test,
        X_training,
        y_training
    )

    print('\n' + '-' * 30 + '\nMODEL PERFORMANCE SUMMARY\n' + '-' * 30)
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validataion accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

    return model, history, evaluation_results


if __name__ == '__main__':
    main()