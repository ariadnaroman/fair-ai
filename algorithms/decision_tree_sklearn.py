from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree

class DecisionTreeSklearn:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.name = 'Decision Tree'
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.classifier = None
        self.y_pred = None

    def train(self):
        self.classifier = DecisionTreeClassifier()
        self.classifier.fit(self.x_train, self.y_train)

    def test(self):
        self.y_pred = self.classifier.predict(self.x_test)

    def evaluate(self):
        print(confusion_matrix(self.y_test, self.y_pred))
        print(classification_report(self.y_test, self.y_pred))
        text_representation = tree.export_text(self.classifier, feature_names=list(self.x_train.columns))
        print(text_representation)
