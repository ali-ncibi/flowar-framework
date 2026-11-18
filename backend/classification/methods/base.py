from abc import ABC, abstractmethod


class BaseClassificationMethod(ABC):
    @abstractmethod
    def fit(self, X, y, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def predict(self, X, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")
