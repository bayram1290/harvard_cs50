import sys
import csv
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

Test_size = 0.4
Month_enum = dict(Jan=0, Feb=1, Mar=2, Apr=3, May=4, June=5, Jul=6, Aug=7, Sep=8, Oct=9, Nov=10, Dec=11)


def loadData(file_path: str) -> tuple[list, list]:
    evidence = []
    labels = []

    with open(file=file_path, encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                Month_enum[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'] == 'TRUE' else 0
            ])

            labels.append(1 if row['Revenue'] == 'TRUE' else 0)

    return (evidence, labels)

def trainModel(evidence:np.ndarray, label: list) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, label)

    return model


def evaluate(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    pos_cnt=0
    neg_cnt=0
    pos_iden=0
    neg_iden=0

    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            pos_cnt += 1
            if actual == predicted:
                pos_iden += 1
        else:
            neg_cnt += 1
            if actual == predicted:
                neg_iden += 1

    sensitivity = float((pos_iden/pos_cnt) if pos_cnt > 0 else 0)
    specificity = float( (neg_iden/neg_cnt) if neg_cnt > 0 else 0)

    return (sensitivity, specificity)


def main():
    if len(sys.argv) != 2:
        sys.exit('App usage: python app.py path_to_data_file.csv')
    file_path = sys.argv[1]

    evidence, labels = loadData(file_path)

    x_train, x_test, y_train, y_test = train_test_split(evidence, labels, test_size=Test_size)


    model = trainModel(x_train, y_train)
    predictions = model.predict(x_test)

    sensitivity, specificity = 0, 0
    if y_test is not None and predictions is not None:
        sensitivity, specificity = evaluate(y_test, predictions)

    print(f'Correct: {(y_test == predictions).sum()}')
    print(f'Incorrect: {(y_test != predictions).sum()}')
    print(f'True positive rate: {100 * sensitivity:.2f}%')
    print(f'True negative rate: {100 * specificity:.2f}%')

if __name__ == "__main__":
    main()