from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from ...models import EventStream, SegmentationResult


class BaseSegmentationMethod(ABC):
    def __init__(self, method_name: str):
        self.method_name = method_name
        self._is_calibrated = False

    @abstractmethod
    def fit(self, events: EventStream, **kwargs: Any) -> "BaseSegmentationMethod":
        pass

    @abstractmethod
    def transform(self, events: EventStream) -> SegmentationResult:
        pass

    def __deepcopy__(self, memo: dict) -> "BaseSegmentationMethod":
        """
        Create a deep copy of the segmentation method.

        This resets the fitted state (_is_calibrated) to prevent data leakage
        in cross-validation. Subclasses should override this if they have
        additional state that needs special handling.
        """
        # Create a new instance with the same method_name
        cls = self.__class__
        new_instance = cls.__new__(cls)

        # Copy all attributes from the original instance
        for key, value in self.__dict__.items():
            if key == "_is_calibrated":
                # Reset fitted state - this is crucial for avoiding data leakage
                setattr(new_instance, key, False)
            else:
                # Deep copy other attributes
                setattr(new_instance, key, deepcopy(value, memo))

        memo[id(self)] = new_instance
        return new_instance
