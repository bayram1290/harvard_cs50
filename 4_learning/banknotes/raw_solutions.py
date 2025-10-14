import csv
import numpy as np
import random

with open('./data/banknotes.csv') as file:
    reader = csv.reader(file)
    next(reader)

    data = []
    for row in reader:
        data.append({
            'evidence': [float(cell) for cell in row[:4]],
            'label': 'Authentic' if row[4] == '0' else 'Counterfeit'
        })

from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

types = ['perceptron', 'svc', 'kneighbors', 'gaussian']
for type in types:
    match type:
        case 'perceptron':
            model = Perceptron()
        case 'svc':
            model = svm.SVC()
        case 'kneighbors':
            model = KNeighborsClassifier(n_neighbors=5)
        case 'gaussian':
            model = GaussianNB()
        case _:
            exit('Invalid model type. Exiting...')

    holdout = int(.50 * len(data))
    random.shuffle(data)
    training = data[holdout:]
    testing = data[:holdout]

    X_training = np.array([row['evidence'] for row in training])
    y_training = [row['label'] for row in training]
    model.fit(X_training, y_training)

    X_testing = np.array([row['evidence'] for row in testing])
    y_testing = [row['label'] for row in testing]
    predictions = model.predict(X_testing)

    correct, incorrect, total = 0, 0, 0
    for actual, predicted in zip(y_testing, predictions):
        total += 1
        if actual == predicted:
            correct += 1
        else:
            incorrect += 1

    print(f"Results for model {model.__class__.__name__}")
    print(f'Correct: {correct}\nIncorrect: {incorrect}\nAccuracy: {100 * correct / total: .2f}%')
    print('*'*40)


