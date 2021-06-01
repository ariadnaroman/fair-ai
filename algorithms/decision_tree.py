from random import randrange
import numpy as np
import pydot
from math import log2
import copy
from algorithms.fairness_measures import demographic_parity


class DecisionTree:
    def __init__(self, train, test, columns):
        self.name = 'Decision Tree'
        self.train = train
        self.test = test
        self.classifier = None
        self.y_pred = None
        self.columns = columns
        self.root = None
        self.graph = None

    # Build a decision tree
    def build_tree(self):
        score_measure = 'fig'
        class_values = list(set(row[-1] for row in self.train))
        current_score = 999
        if score_measure == 'ig':
            current_score = self.calculate_entropy(self.train, class_values)
        elif score_measure == 'fig':
            current_score = self.calculate_demographic_parity(self.train)
        variable_importance = z = np.zeros(len(self.train[0]) - 1)
        self.root = self.select_best_feature(self.train, score_measure, current_score, variable_importance, [])
        self.split_node(self.root, 10, 30, 1, score_measure, variable_importance, None, None)
        return self.root

    # Split a dataset based on a feature and its split value
    def split_dataset_by_feature(self, feature_index, feature_split_value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[feature_index] < feature_split_value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    def calculate_entropy(self, dataset, classes):
        entropy = 0
        if len(dataset):
            for class_val in classes:
                count_class = [row[-1] for row in dataset].count(class_val)
                if count_class:
                    p = count_class/float(len(dataset))
                    entropy = entropy - p * log2(p)
        return entropy

    def calculate_entropy_clusters(self, clusters, classes):
        weights = []
        for i in range(0, len(clusters)):
            weights.append(len(clusters[i]))
        total_number_instances = sum(weights)
        for i in range (0, len(weights)):
            weights[i] = weights[i]/float(total_number_instances)
        entropy = 0
        for i in range(0, len(clusters)):
            entropy = entropy + weights[i] * self.calculate_entropy(clusters[i], classes)
        return entropy

    def calculate_demographic_parity(self, dataset):
        sex0 = self.columns.index('sex_0')
        sex1 = self.columns.index('sex_1')
        return demographic_parity(dataset, [sex0, sex1], -1)

    def calculate_demographic_parity_clusters(self, clusters):
        weights = []
        for i in range(0, len(clusters)):
            weights.append(len(clusters[i]))
        total_number_instances = sum(weights)
        for i in range (0, len(weights)):
            weights[i] = weights[i]/float(total_number_instances)
        demographic_parity = 0
        for i in range(0, len(clusters)):
            demographic_parity = demographic_parity + weights[i] * self.calculate_demographic_parity(clusters[i])
        return demographic_parity

    # Calculate the Gini index for a split dataset
    def gini_index(self, clusters, classes):
        # count all instances at split point
        n_instances = float(sum([len(cluster) for cluster in clusters]))
        # sum weighted Gini index for each cluster
        gini = 0.0
        for cluster in clusters:
            size = float(len(cluster))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the cluster based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in cluster].count(class_val) / size
                score += p * p
            # weight the cluster score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # Select the best split point for a dataset
    def select_best_feature(self, dataset, score_measure, current_score, variable_importance, features_already_used):
        class_values = list(set(row[-1] for row in dataset))
        feature_index, feature_split_value, feature_score, feature_clusters = 999, 999, 999, None  # best feature
        features = range(len(dataset[0]) - 1)

        # find the (feature, feature_split_value) pair that result in the best score

        for index in features:  # for each feature
            if index in features_already_used:
                continue
            possible_values = list(set(row[index] for row in dataset))
            if len(possible_values) == 2 and ((possible_values[0] == 0 and possible_values[1] == 1) or (possible_values[0] == 1 and possible_values[1] == 0)):
                value = 0.5
                clusters = self.split_dataset_by_feature(index, value, dataset)
                score = 999
                if score_measure == 'ig':
                    score = self.calculate_entropy_clusters(clusters, class_values)
                elif score_measure == 'fig':
                    demographic_parity = self.calculate_demographic_parity_clusters(clusters)
                    if demographic_parity == 0:
                        score = self.calculate_entropy_clusters(clusters, class_values)
                    else:
                        score = self.calculate_entropy_clusters(clusters, class_values) * demographic_parity
                variable_importance[index] += current_score - score
                if score < feature_score:
                    feature_index, feature_split_value, feature_score, feature_clusters = index, value, score, clusters
            else:
                for value in possible_values:  # for each value of that feature
                    # compute the clusters that result from the split
                    clusters = self.split_dataset_by_feature(index, value, dataset)
                    score = 999
                    if score_measure == 'ig':
                        score = self.calculate_entropy_clusters(clusters, class_values)
                    elif score_measure == 'fig':
                        demographic_parity = self.calculate_demographic_parity_clusters(clusters)
                        if demographic_parity == 0:
                            score = self.calculate_entropy_clusters(clusters, class_values)
                        else:
                            score = self.calculate_entropy_clusters(clusters, class_values) * demographic_parity
                    variable_importance[index] += current_score - score
                    if score < feature_score:
                        feature_index, feature_split_value, feature_score, feature_clusters = index, value, score, clusters
        if feature_index == 999:
            return {'index': None, 'feature': None, 'value': None, 'score': score,
                    'clusters': None, 'features_already_used': features_already_used}
        features_already_used.append(feature_index)
        return {'index': feature_index, 'feature': self.columns[feature_index], 'value': feature_split_value,
                'score': feature_score,
                'clusters': feature_clusters, 'features_already_used': features_already_used,
                'unique_name': self.columns[feature_index] + randrange(9999).__str__()}

    # Create a terminal node
    def create_terminal_node(self, cluster):
        outcomes = [row[-1] for row in cluster]
        return max(set(outcomes), key=outcomes.count)

    # Recursive function that splits the dataset while building the tree, until getting to terminal nodes
    def split_node(self, node, max_depth, min_size, depth, score_measure, variable_importance, parent, childKey):
        left, right = node['clusters']
        del (node['clusters'])
        # check if any of the clusters is empty
        if not left or not right:
            if parent:
                parent[childKey] = self.create_terminal_node(left + right)
            return
        # check if the max_depth of the tree has been reached
        if depth >= max_depth:
            terminal_left = self.create_terminal_node(left)
            terminal_right = self.create_terminal_node(right)
            if parent and terminal_left == terminal_right:
                parent[childKey] = self.create_terminal_node(left + right)
            else:
                node['left'], node['right'] = terminal_left, terminal_right
            return
        # check if the min_size of a node has been reached (for left child)
        if len(left) <= min_size:
            node['left'] = self.create_terminal_node(left)
        else:  # compute the next split
            node['left'] = self.select_best_feature(left, score_measure, node['score'], variable_importance,
                                                    copy.deepcopy(node['features_already_used']))
            if node['left']['index'] == None:
                node['left'] = self.create_terminal_node(left)
            else:
                self.split_node(node['left'], max_depth, min_size, depth + 1, score_measure, variable_importance, node, 'left')
        # check if the min_size of a node has been reached (for right child)
        if len(right) <= min_size:
            node['right'] = self.create_terminal_node(right)
        else:  # compute the next split
            node['right'] = self.select_best_feature(right, score_measure, node['score'], variable_importance,
                                                     copy.deepcopy(node['features_already_used']))
            if node['right']['index'] == None:
                node['right'] = self.create_terminal_node(right)
            else:
                self.split_node(node['right'], max_depth, min_size, depth + 1, score_measure, variable_importance, node, 'right')
        if node['left'] == node['right']:
            parent[childKey] = self.create_terminal_node(left + right)

    # Make a prediction with a decision tree
    def predict(self, node, row):
        if row[node['index']] >= node['value']:
            if 'right' in node and isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            elif 'right' in node:
                return node['right']
        else:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']

    def draw_edge(self, parent_name, child_name):
        edge = pydot.Edge(parent_name, child_name)
        self.graph.add_edge(edge)

    def visit(self, node, parent=None):
        if parent:
            self.draw_edge(parent['unique_name'], node['unique_name'])
        if isinstance(node['left'], dict):
            self.visit(node['left'], node)
        else:
            self.draw_edge(node['unique_name'], node['left'].__str__() + ' ' + node['unique_name'])
        if 'right' in node and isinstance(node['right'], dict):
            self.visit(node['right'], node)
        elif 'right' in node:
            self.draw_edge(node['unique_name'], node['right'].__str__() + ' ' + node['unique_name'])

    def visualize_tree(self):
        self.graph = pydot.Dot(graph_type='graph')
        self.visit(self.root)
        self.graph.write_png('D:\\Facultate\Disertatie\\experiments\\fair-ai\\example2_graph.png')
