from sklearn.neighbors import KNeighborsClassifier as SklearnKNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTreeClassifier

from .base import BaseClassificationMethod


class DecisionTreeClassifier(BaseClassificationMethod):
    def __init__(self, *args, **kwargs):
        self.model = SklearnDecisionTreeClassifier(*args, **kwargs)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        return self.model.predict(X)


class KNeighborsClassifier(BaseClassificationMethod):
    def __init__(self, *args, **kwargs):
        self.model = SklearnKNeighborsClassifier(*args, **kwargs)

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y)

    def predict(self, X, **kwargs):
        return self.model.predict(X)
