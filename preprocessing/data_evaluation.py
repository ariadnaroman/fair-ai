import typing as t
import numpy as np
import random
import itertools

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import accuracy_score

from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from scipy.spatial import distance


class DataEvaluation:
    def __init__(self, X, Y):
        self.name = 'DataEvaluation'

        self.X = X.to_numpy(copy=True)
        self.Y = Y['income'].squeeze().to_numpy(copy=True)

        self.precomp_fx = self.precompute_fx()
        self.cls_index = self.precomp_fx['cls_index']
        self.cls_n_ex = self.precomp_fx['cls_n_ex']
        self.ovo_comb = self.precomp_fx['ovo_comb']

    def precompute_fx(self):
        prepcomp_vals = {}

        classes, class_freqs = np.unique(self.Y, return_counts=True)
        cls_index = [np.equal(self.Y, i) for i in range(classes.shape[0])]

        # cls_n_ex = np.array([np.sum(aux) for aux in cls_index])
        cls_n_ex = list(class_freqs)
        ovo_comb = list(itertools.combinations(range(classes.shape[0]), 2))
        prepcomp_vals["ovo_comb"] = ovo_comb
        prepcomp_vals["cls_index"] = cls_index
        prepcomp_vals["cls_n_ex"] = cls_n_ex
        return prepcomp_vals

    def numerator(self, X, cls_index, cls_n_ex, i):
        return np.sum([cls_n_ex[j] * np.power((np.mean(X[cls_index[j], i]) -
                                               np.mean(X[:, i], axis=0)), 2) for j in range(len(cls_index))])

    def denominator(self, X, cls_index, cls_n_ex, i):
        return np.sum([np.sum(np.power(X[cls_index[j], i] - np.mean(X[cls_index[j], i], axis=0), 2))
                       for j in range(0, len(cls_n_ex))])

    def compute_rfi(self, X, cls_index, cls_n_ex):
        result = []
        for i in range(np.shape(X)[1]):
            numerator = self.numerator(X, cls_index, cls_n_ex, i)
            denominator = self.denominator(X, cls_index, cls_n_ex, i)
            # print('numerator: ' + numerator)
            # print ('denominator: ' + denominator)
            if denominator != 0:
                result.append(numerator / denominator)
            return result
        # return [self.numerator(X, cls_index, cls_n_ex, i) / self.denominator(X, cls_index, cls_n_ex, i)
        #         for i in range(np.shape(X)[1])]

    def ft_F1(self, X, cls_index, cls_n_ex):
        return 1 / (1 + np.max(self.compute_rfi(X, cls_index, cls_n_ex)))

    def compute_F1(self):
        print('Maximum Fisher Discriminant Ratio (F1): ')
        F1 = self.ft_F1(self.X, self.cls_index, self.cls_n_ex)
        print(F1)

    def evaluate_dataset(self):
        self.compute_F1()
