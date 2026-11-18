from backend.segmentation.methods import FixedCount
from backend.utils import (
    clone_method,
    clone_representation_method,
    clone_segmentation_method,
)


class _Plain:
    def __init__(self):
        self.value = 1


class TestCloneMethod:
    def test_clone_resets_calibrated(self):
        seg = FixedCount(nb_events=4)
        seg._is_calibrated = True
        cloned = clone_method(seg)
        assert cloned is not seg
        assert cloned._is_calibrated is False
        assert cloned.nb_events == 4

    def test_clone_no_reset(self):
        obj = _Plain()
        obj._is_calibrated = True
        cloned = clone_method(obj, reset_state=False)
        assert cloned._is_calibrated is True

    def test_clone_reset_plain_object(self):
        obj = _Plain()
        obj._is_calibrated = True
        cloned = clone_method(obj, reset_state=True)
        assert cloned._is_calibrated is False

    def test_clone_object_without_flag(self):
        obj = _Plain()
        cloned = clone_method(obj)
        assert cloned is not obj
        assert cloned.value == 1
        assert not hasattr(cloned, "_is_calibrated")

    def test_wrappers(self):
        seg = FixedCount(nb_events=2)
        assert isinstance(clone_segmentation_method(seg), FixedCount)
        assert isinstance(clone_representation_method(seg), FixedCount)
