from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from backend.models import Segment, SensorActivations


@dataclass
class BaseRepresentationMethod(ABC):
    @abstractmethod
    def fit(
        self,
        sensor_activations: SensorActivations,
        **kwargs,
    ):
        pass

    @abstractmethod
    def transform(
        self,
        segments: list[Segment],
        sensor_activations: SensorActivations,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        pass

    def __deepcopy__(self, memo: dict) -> "BaseRepresentationMethod":
        """
        Create a deep copy of the representation method.

        This resets any fitted state to prevent data leakage in cross-validation.
        Subclasses with additional mutable state should override this method
        to properly reset their state during cloning.
        """
        cls = self.__class__
        new_instance = cls.__new__(cls)

        # For dataclass instances, we need to handle fields differently
        for key, value in self.__dict__.items():
            # Deep copy all attributes; subclasses can override for custom logic
            setattr(new_instance, key, deepcopy(value, memo))

        memo[id(self)] = new_instance
        return new_instance
