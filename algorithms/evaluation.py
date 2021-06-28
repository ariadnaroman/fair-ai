import random
import copy

# Split a dataset into k folds
import numpy as np
import pandas as pd

from algorithms.decision_tree import DecisionTree
from algorithms.fairness_measures import demographic_parity_urm_vs_others, equalized_odds_urm_vs_others

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    random.shuffle(dataset_copy)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# Calculate error rate based on confusion matrix
def error_rate(cm):
    return (cm['false_positive'] + cm['false_negative']) / float(
        cm['true_positive'] + cm['true_negative'] + cm['false_negative'] + cm['false_positive']) * 100.0


# Calculate accuracy percentage based on confusion matrix
def accuracy(cm):
    return (cm['true_positive'] + cm['true_negative']) / float(
        cm['true_positive'] + cm['true_negative'] + cm['false_negative'] + cm['false_positive']) * 100.0


# Calculate precision percentage based on confusion matrix
def precision(cm):
    print(cm)
    return cm['true_positive'] / float(cm['true_positive'] + cm['false_positive'] + 0.1) * 100.0


# Calculate sensitivity (recall) based on confusion matrix
def sensitivity(cm):
    return cm['true_positive'] / float(cm['true_positive'] + cm['false_negative'] + 0.1) * 100.0


# Calculate specificity based on confusion matrix
def specificity(cm):
    return cm['true_negative'] / float(cm['true_negative'] + cm['false_positive'] + 0.1) * 100.0


# Calculate F1 score based on confusion matrix
def f1_score(cm):
    return (2 * cm['true_positive']) / float(
        2 * cm['true_positive'] + cm['false_positive'] + cm['false_negative']) * 100.0


# Calculate confusion matrix
def confusion_matrix(actual, predicted):
    tp = 0  # true positive
    tn = 0  # true negative
    fp = 0  # false positive
    fn = 0  # false negative
    for i in range(len(actual)):
        if (actual[i] == predicted[i]) & (actual[i] == 1):
            tp += 1
        elif (actual[i] == predicted[i]) & (actual[i] == 0):
            tn += 1
        elif (actual[i] != predicted[i]) & (actual[i] == 1):
            fn += 1
        elif (actual[i] != predicted[i]) & (actual[i] == 0):
            fp += 1
    return {'true_positive': tp,
            'true_negative': tn,
            'false_negative': fn,
            'false_positive': fp}


# Evaluate an algorithm using k-fold cross validation
def evaluate_algorithm(train, test, n_folds, *args):
    columns = train.pop(0)
    train_set = []
    for row in train:
        train_set.append([float(x) for x in row])
    test.pop(0)
    test_set = []
    for row in test:
        test_set.append([float(x) for x in row])
    # folds = cross_validation_split(dataset, n_folds)
    mean_err = 0
    mean_acc = 0
    mean_prec = 0
    mean_sens = 0
    mean_spec = 0
    mean_f1 = 0
    decisionTree = DecisionTree(train_set, test_set, columns)
    tree = decisionTree.build_tree()
    decisionTree.visualize_tree()
    predicted = [decisionTree.predict(tree, row) for row in test_set]
    actual = [row[-1] for row in test_set]
    cm = confusion_matrix(actual, predicted)
    mean_err += error_rate(cm)
    mean_acc += accuracy(cm)
    mean_prec += precision(cm)
    mean_sens += sensitivity(cm)
    mean_spec += specificity(cm)
    mean_f1 += f1_score(cm)
    print('Demographic parity test set: ')
    print(demographic_parity_urm_vs_others(test_set, -1))
    print('Demographic parity predicted set: ')
    predicted_test_set = copy.deepcopy(test_set)
    for i in range(0, len(predicted_test_set)):
        predicted_test_set[i][-1] = predicted[i]
    print(demographic_parity_urm_vs_others(predicted_test_set, -1))
    print('Equalized odds: ')
    print(equalized_odds_urm_vs_others(test_set, predicted, -1))
    return {
        'error_rate': mean_err,
        'accuracy': mean_acc,
        'precision': mean_prec,
        'sensitivity': mean_sens,
        'specificity': mean_spec,
        'f1_score': mean_f1
    }
