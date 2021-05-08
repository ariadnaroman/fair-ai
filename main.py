from preprocessing.data_cleaning import DataCleaning
from algorithms.decision_tree import DecisionTree

dataCleaning = DataCleaning('adult')
x_train, y_train, x_test, y_test = dataCleaning.get_preprocessed_data()
print('x_train')
print(x_train)
print('y_train')
print(y_train)
print('x_test')
print(x_test)
print('y_test')
print(y_test)
decisionTree = DecisionTree(x_train, y_train, x_test, y_test)
decisionTree.train()
decisionTree.test()
decisionTree.evaluate()