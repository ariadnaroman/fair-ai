import numpy as np
import csv

from preprocessing.data_cleaning import DataCleaning
from sklearn.model_selection import train_test_split
from preprocessing.data_evaluation import DataEvaluation
from algorithms.decision_tree_sklearn import DecisionTreeSklearn
from algorithms.decision_tree import DecisionTree
from algorithms.evaluation import evaluate_algorithm

dataCleaning = DataCleaning('german')
x, y = dataCleaning.get_preprocessed_data()
print('x')
print(x)
print('y')
print(y)
x_np = x.to_numpy(copy=True)
y_np = y.to_numpy(copy=True)
x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(x, y, test_size=0.25, random_state=82, shuffle=True)
train = np.append(x_train_np, y_train_np, axis=1)
test = np.append(x_test_np, y_test_np, axis=1)
# dataset = np.concatenate((train, test)).tolist()

columns = [column for column in x.columns] + ['good/bad']

with open('./datasets/processed/german/german_processed_train.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(columns)
    write.writerows(train)

with open('./datasets/processed/german/german_processed_test.csv', 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(columns)
    write.writerows(test)

# dataEvaluation = DataEvaluation(x_train, y_train)
# dataEvaluation.evaluate_dataset()
# decisionTree = DecisionTreeSklearn(x_train, y_train, x_test, y_test)
# decisionTree.train()
# decisionTree.test()
# decisionTree.evaluate()


with open('D:\\Facultate\Disertatie\\experiments\\fair-ai\\datasets\\processed\\german\\german_processed_train.csv', newline='') as f:
    reader = csv.reader(f)
    train = list(reader)
    with open('D:\\Facultate\Disertatie\\experiments\\fair-ai\\datasets\\processed\\german\\german_processed_test.csv',
              newline='') as f:
        reader = csv.reader(f)
        test = list(reader)
        print(evaluate_algorithm(train, test, 10))