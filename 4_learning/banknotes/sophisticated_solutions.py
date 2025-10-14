import sys
import csv

with open('./data/banknotes.csv') as file:
    reader = csv.reader(file)
    next(reader)

    data = []
    for row in reader:
        data.append({
            'evidence': [float(cell) for cell in row[:4]],
            'label': 'Authentic' if row[4] == '0' else 'Counterfeit'
        })

from sklearn.model_selection import train_test_split
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
            sys.exit('Wrong model type. Exiting ...')


    evidence = [row['evidence'] for row in data]
    labels = [row['label'] for row in data]

    X_training, X_testing, y_training, y_testing = train_test_split(
        evidence, labels, test_size=0.4
    )

    model.fit(X_training, y_training)
    predictions = model.predict(X_testing)

    correct = (y_testing == predictions).sum()
    incorrect = (y_testing != predictions).sum()
    total = len(predictions)

    print(f"Results for model {model.__class__.__name__}")
    print(f'Correct: {correct}\nIncorrect: {incorrect}\nAccuracy: {100 * correct / total: .2f}%')
    print('*'*40)