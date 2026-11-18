"""
Cloning utilities for avoiding data leakage in cross-validation.

Inspired by sklearn's clone function, this module provides utilities to safely
clone segmentation and representation methods while resetting their fitted state.
"""

from copy import deepcopy
from typing import TypeVar

T = TypeVar("T")


def clone_method[T](method: T, reset_state: bool = True) -> T:
    """
    Clone a segmentation or representation method.

    This creates a deep copy of the method that prevents data leakage during
    cross-validation by resetting fitted state flags.

    Parameters
    ----------
    method : BaseSegmentationMethod or BaseRepresentationMethod
        The method instance to clone.
    reset_state : bool, default=True
        If True, reset the `_is_calibrated` flag on segmentation methods
        to False. This prevents using fitted state from training data.

    Returns
    -------
    cloned_method : type of method
        A deep copy of the input method with state reset.

    Examples
    --------
    >>> segmenter = FixedTime(length_in_seconds=60.0)
    >>> segmenter.fit(training_events)
    >>> cloned_seg = clone_method(segmenter)
    >>> # cloned_seg has the same parameters but _is_calibrated=False
    >>> cloned_seg.fit(test_events)  # Fresh fit on new data

    Notes
    -----
    The method's `__deepcopy__` implementation handles the cloning logic.
    Subclasses can override `__deepcopy__` for custom cloning behavior.
    """
    cloned = deepcopy(method)

    # Additional reset of _is_calibrated flag if it exists
    # (provides defensive reset in case __deepcopy__ doesn't handle it).
    # setattr is required here: `cloned` is an unbounded TypeVar, so direct
    # attribute assignment is a type error even after the hasattr check.
    if reset_state and hasattr(cloned, "_is_calibrated"):
        setattr(cloned, "_is_calibrated", False)  # noqa: B010

    return cloned


def clone_segmentation_method[T](segmenter: T) -> T:
    """
    Clone a segmentation method for cross-validation.

    Convenience wrapper around clone_method for segmentation methods.

    Parameters
    ----------
    segmenter : BaseSegmentationMethod
        The segmentation method to clone.

    Returns
    -------
    cloned_segmenter : BaseSegmentationMethod
        Cloned segmentation method with fitted state reset.
    """
    return clone_method(segmenter, reset_state=True)


def clone_representation_method[T](extractor: T) -> T:
    """
    Clone a representation extraction method for cross-validation.

    Convenience wrapper around clone_method for representation methods.

    Parameters
    ----------
    extractor : BaseRepresentationMethod
        The representation method to clone.

    Returns
    -------
    cloned_extractor : BaseRepresentationMethod
        Cloned representation method with state reset.
    """
    return clone_method(extractor, reset_state=True)
