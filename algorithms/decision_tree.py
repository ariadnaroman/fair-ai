from random import randrange
import numpy as np
import pydot
import graphviz


class DecisionTree:
    # def __init__(self, x_train, y_train, x_test, y_test):
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
        class_values = list(set(row[-1] for row in self.train))
        print('Class values')
        print(class_values)
        current_gini = self.gini_index([self.train], class_values)
        print('Gini index')
        print(current_gini)
        variable_importance = z = np.zeros(len(self.train[0]) - 1)
        self.root = self.select_best_feature(self.train, current_gini, variable_importance, [])
        self.split_node(self.root, 20, 50, 1, variable_importance)
        return self.root

    # Split a dataset based on a feature and its split value
    def split_dataset_by_feature(self, feature_index, feature_split_value, dataset):
        left, right = list(), list()
        for row in dataset:
            if feature_split_value == 0 or feature_split_value == 1:
                if row[feature_index] == feature_split_value:
                    left.append(row)
                else:
                    right.append(row)
            else:
                if row[feature_index] < feature_split_value:
                    left.append(row)
                else:
                    right.append(row)
        return left, right

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
    def select_best_feature(self, dataset, current_gini, variable_importance, features_already_used):
        # print('Select best feature')
        class_values = list(set(row[-1] for row in dataset))
        feature_index, feature_split_value, feature_gini_score, feature_clusters = 999, 999, 999, None  # best feature
        features = range(len(dataset[0]) - 1)

        # find the (feature, feature_split_value) pair that result in the lowest gini index
        print('Features')
        print(features)
        print('Features already used')
        print(features_already_used)
        for index in features:  # for each feature
            if index in features_already_used:
                continue
            # print('Feature')
            # print(index)
            # for row in dataset:  # for each value of that feature
            #     # compute the clusters that result from the split
            #     clusters = self.split_dataset_by_feature(index, row[index], dataset)
            #     # compute the gini index for the clusters
            #     gini = self.gini_index(clusters, class_values)
            #     variable_importance[index] += current_gini - gini
            #     # replace the selected feature if a better one was found
            #     if gini < feature_gini_score:
            #         feature_index, feature_split_value, feature_gini_score, feature_clusters = index, row[
            #             index], gini, clusters
            # compute the clusters that result from the split
            row = dataset[randrange(len(dataset))]
            # print('Split value')
            # print(row[index])
            clusters = self.split_dataset_by_feature(index, row[index], dataset)
            # compute the gini index for the clusters
            gini = self.gini_index(clusters, class_values)
            # print('Current gini')
            # print(current_gini)
            # print('Gini')
            # print(gini)
            # print('Variable importance: ')
            variable_importance[index] += current_gini - gini
            # print(variable_importance[index])
            # replace the selected feature if a better one was found
            if gini < feature_gini_score:
                feature_index, feature_split_value, feature_gini_score, feature_clusters = index, row[
                    index], gini, clusters
        # print('Feature chosen:')
        # print(feature_index)
        # print('Feature split value:')
        # print(feature_split_value)
        # if feature_clusters:
        #     print('Left: ')
        #     print(len(feature_clusters[0]))
        #     print('Right: ')
        #     print(len(feature_clusters[1]))
        if feature_index == 999:
            return {'index': None, 'feature': None, 'value': None, 'gini': current_gini,
                    'clusters': None, 'features_already_used': features_already_used}
        features_already_used.append(feature_index)
        return {'index': feature_index, 'feature': self.columns[feature_index], 'value': feature_split_value,
                'gini': feature_gini_score,
                'clusters': feature_clusters, 'features_already_used': features_already_used,
                'unique_name': self.columns[feature_index] + randrange(9999).__str__()}

    # Create a terminal node
    def create_terminal_node(self, cluster):
        outcomes = [row[-1] for row in cluster]
        return max(set(outcomes), key=outcomes.count)

    # Recursive function that splits the dataset while building the tree, until getting to terminal nodes
    def split_node(self, node, max_depth, min_size, depth, variable_importance):
        left, right = node['clusters']
        del (node['clusters'])
        # check if any of the clusters is empty
        if not left or not right:
            node['left'] = node['right'] = self.create_terminal_node(left + right)
            return
        # check if the max_depth of the tree has been reached
        if depth >= max_depth:
            node['left'], node['right'] = self.create_terminal_node(left), self.create_terminal_node(right)
            return
        # check if the min_size of a node has been reached (for left child)
        if len(left) <= min_size:
            node['left'] = self.create_terminal_node(left)
        else:  # compute the next split
            node['left'] = self.select_best_feature(left, node['gini'], variable_importance,
                                                    node['features_already_used'])
            if node['left']['index'] == None:
                node['left'] = self.create_terminal_node(left)
            else:
                self.split_node(node['left'], max_depth, min_size, depth + 1, variable_importance)
        # check if the min_size of a node has been reached (for right child)
        if len(right) <= min_size:
            node['right'] = self.create_terminal_node(right)
        else:  # compute the next split
            node['right'] = self.select_best_feature(right, node['gini'], variable_importance,
                                                     node['features_already_used'])
            if node['right']['index'] == None:
                node['right'] = self.create_terminal_node(right)
            else:
                self.split_node(node['right'], max_depth, min_size, depth + 1, variable_importance)

    # Make a prediction with a decision tree
    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    # menu = {'dinner':
    #             {'chicken':'good',
    #              'beef':'average',
    #              'vegetarian':{
    #                    'tofu':'good',
    #                    'salad':{
    #                             'caeser':'bad',
    #                             'italian':'average'}
    #                    },
    #              'pork':'bad'}
    #         }

    def draw(self, parent_name, child_name):
        edge = pydot.Edge(parent_name, child_name)
        self.graph.add_edge(edge)

    def visit(self, node, parent=None):
        if isinstance(node['left'], dict):
            if parent:
                self.draw(parent['unique_name'], node['unique_name'])
            self.visit(node['left'], node)
        else:
            if parent:
                self.draw(parent['unique_name'], node['unique_name'])
            self.draw(node['unique_name'], node['left'].__str__() + ' ' + node['unique_name'])
        if isinstance(node['right'], dict):
            if parent:
                self.draw(parent['unique_name'], node['unique_name'])
            self.visit(node['right'], node)
        else:
            if parent:
                self.draw(parent['unique_name'], node['unique_name'])
            self.draw(node['unique_name'], node['right'].__str__() + ' ' + node['unique_name'])

    def visualize_tree(self):
        self.graph = pydot.Dot(graph_type='graph')
        self.visit(self.root)
        self.graph.write_png('D:\\Facultate\Disertatie\\experiments\\fair-ai\\example1_graph.png')
