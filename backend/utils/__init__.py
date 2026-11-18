"""Backend utilities for segmentation, representation, and other ML operations."""

from .cloning import (
    clone_method,
    clone_representation_method,
    clone_segmentation_method,
)

__all__ = [
    "clone_method",
    "clone_segmentation_method",
    "clone_representation_method",
]
