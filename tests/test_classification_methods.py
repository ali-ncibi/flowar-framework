import numpy as np
import pytest

from backend.classification.methods import (
    BaseClassificationMethod,
    DecisionTreeClassifier,
    KNeighborsClassifier,
)


class TestBaseABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseClassificationMethod()

    def test_incomplete_subclass_not_instantiable(self):
        class OnlyFit(BaseClassificationMethod):
            def fit(self, X, y, **kwargs):
                return self

        with pytest.raises(TypeError):
            OnlyFit()

    def test_super_calls_raise_not_implemented(self):
        class CallsSuper(BaseClassificationMethod):
            def fit(self, X, y, **kwargs):
                return super().fit(X, y, **kwargs)

            def predict(self, X, **kwargs):
                return super().predict(X, **kwargs)

        clf = CallsSuper()
        with pytest.raises(NotImplementedError, match="Subclasses should implement"):
            clf.fit(None, None)
        with pytest.raises(NotImplementedError, match="Subclasses should implement"):
            clf.predict(None)


@pytest.fixture
def xy():
    X = np.array([[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 4.9]])
    y = np.array([0, 0, 1, 1])
    return X, y


class TestDecisionTree:
    def test_fit_predict(self, xy):
        X, y = xy
        clf = DecisionTreeClassifier(max_depth=2, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert list(preds) == [0, 0, 1, 1]

    def test_kwargs_forwarded(self, xy):
        X, y = xy
        clf = DecisionTreeClassifier(criterion="entropy")
        clf.fit(X, y)
        assert clf.model.criterion == "entropy"


class TestKNeighbors:
    def test_fit_predict(self, xy):
        X, y = xy
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert list(preds) == [0, 0, 1, 1]

    def test_kwargs_forwarded(self, xy):
        X, y = xy
        clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
        clf.fit(X, y)
        assert clf.model.weights == "distance"
