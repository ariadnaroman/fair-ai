import numpy as np
import csv

from preprocessing.data_cleaning import DataCleaning
from preprocessing.data_evaluation import DataEvaluation
from algorithms.decision_tree_sklearn import DecisionTreeSklearn
from algorithms.decision_tree import DecisionTree
from algorithms.evaluation import evaluate_algorithm

# dataCleaning = DataCleaning('adult')
# x_train, y_train, x_test, y_test = dataCleaning.get_preprocessed_data()
# print('x_train')
# print(x_train)
# print('y_train')
# print(y_train)
# print('x_test')
# print(x_test)
# print('y_test')
# print(y_test)
# x_train_np = x_train.to_numpy(copy=True)
# y_train_np = y_train.to_numpy(copy=True)
# x_test_np = x_test.to_numpy(copy=True)
# y_test_np = y_test.to_numpy(copy=True)
# train = np.append(x_train_np, y_train_np, axis=1)
# test = np.append(x_test_np, y_test_np, axis=1)
# dataset = np.concatenate((train, test)).tolist()
#
#
# print(len(dataset))
#
# columns = [column for column in x_train.columns] + ['income']
#
# with open('./datasets/processed/adult/adult_processed.csv', 'w', newline='') as f:
#     # using csv.writer method from CSV package
#     write = csv.writer(f)
#
#     write.writerow(columns)
#     write.writerows(dataset)
# dataEvaluation = DataEvaluation(x_train, y_train)
# dataEvaluation.evaluate_dataset()
# decisionTree = DecisionTreeSklearn(x_train, y_train, x_test, y_test)
# decisionTree.train()
# decisionTree.test()
# decisionTree.evaluate()


with open('D:\\Facultate\Disertatie\\experiments\\fair-ai\\datasets\\processed\\adult\\adult_processed.csv', newline='') as f:
    reader = csv.reader(f)
    dataset = list(reader)
    print(evaluate_algorithm(dataset, 10))