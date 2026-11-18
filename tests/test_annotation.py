from datetime import datetime, timedelta

import numpy as np
import pytest

from backend.annotation.methods import (
    AnyOverlappingAnnotator,
    AtSegmentEndAnnotator,
    BaseAnnotationMethod,
    MostOverlappingAnnotator,
)
from backend.annotation.methods.base import convert_to_array, convert_to_df
from backend.models import ActivityOccurrence, Interval, IntervalType, Segment

T0 = datetime(2024, 1, 1, 8, 0, 0)


def _seg(start_min, end_min, j=(0, 1)):
    return Segment(
        interval=Interval(
            T0 + timedelta(minutes=start_min),
            T0 + timedelta(minutes=end_min),
            IntervalType.LEFT_OPEN,
        ),
        event_indices=j,
    )


def _occ(name, start_min, end_min):
    return ActivityOccurrence(
        interval=Interval(
            T0 + timedelta(minutes=start_min),
            T0 + timedelta(minutes=end_min),
            IntervalType.LEFT_OPEN,
        ),
        name=name,
    )


@pytest.fixture
def occurrences():
    return [_occ("COOKING", 0, 10), _occ("SHOWERING", 20, 30)]


@pytest.fixture
def segments():
    return [_seg(0, 10), _seg(10, 20), _seg(25, 35)]


class TestBase:
    def test_abstract(self):
        with pytest.raises(TypeError):
            BaseAnnotationMethod()

    def test_super_call_covers_pass(self, segments, occurrences):
        class Dummy(BaseAnnotationMethod):
            def transform(self, segments, occurrences):
                return super().transform(segments, occurrences)

        assert Dummy().transform(segments, occurrences) is None


class TestConvertHelpers:
    def test_convert_to_df(self):
        labels = [{"A": True, "B": False}, {"A": False, "B": True}]
        df = convert_to_df(labels)
        assert list(df.columns) == ["A", "B"]
        assert df["A"].tolist() == [1, 0]
        assert df["B"].tolist() == [0, 1]

    def test_convert_to_array(self):
        df = convert_to_df([{"A": False, "B": True}, {"A": True, "B": False}])
        arr = convert_to_array(df)
        assert isinstance(arr, np.ndarray)
        assert arr.tolist() == [1, 0]


class TestAnyOverlapping:
    def test_transform(self, segments, occurrences):
        labels, idx = AnyOverlappingAnnotator().transform(segments, occurrences)
        assert idx == {0: "COOKING", 1: "SHOWERING"}
        # seg0 [0,10] overlaps COOKING; seg1 [10,20] touches COOKING end at 10 (overlaps) but not SHOWERING
        # seg2 [25,35] overlaps SHOWERING
        assert labels.tolist() == [0, 0, 1]

    def test_no_overlap_gives_idle(self, occurrences):
        segs = [_seg(100, 110)]
        labels, idx = AnyOverlappingAnnotator().transform(segs, occurrences)
        # "IDLE" string: membership test against activity names
        assert labels.tolist() == [0]
        assert idx == {0: "COOKING", 1: "SHOWERING"}


class TestAtSegmentEnd:
    def test_transform(self, segments, occurrences):
        labels, idx = AtSegmentEndAnnotator().transform(segments, occurrences)
        assert idx == {0: "COOKING", 1: "SHOWERING"}
        # seg ends: 10 -> COOKING contains 10; 20 -> SHOWERING contains 20; 35 -> none
        assert labels.tolist() == [0, 1, 0]

    def test_end_inside_activity(self):
        segs = [_seg(0, 5), _seg(20, 30)]
        occs = [_occ("COOKING", 0, 10), _occ("SHOWERING", 20, 30)]
        labels, _ = AtSegmentEndAnnotator().transform(segs, occs)
        assert labels.tolist() == [0, 1]


class TestMostOverlapping:
    def test_transform(self, occurrences):
        # segment overlapping both: [5, 25] -> COOKING overlap 5min, SHOWERING 5min; first max wins
        segs = [_seg(5, 25), _seg(100, 110)]
        labels, idx = MostOverlappingAnnotator().transform(segs, occurrences)
        assert idx == {0: "COOKING", 1: "SHOWERING"}
        assert labels.tolist() == [
            0,
            0,
        ]  # tie -> index of max -> first (COOKING); idle -> all False -> argmax 0

    def test_clear_winner(self):
        segs = [_seg(0, 25)]  # COOKING overlap 10, SHOWERING overlap 5
        occs = [_occ("COOKING", 0, 10), _occ("SHOWERING", 20, 30)]
        labels, _ = MostOverlappingAnnotator().transform(segs, occs)
        assert labels.tolist() == [0]

    def test_winner_second(self):
        segs = [_seg(5, 30)]  # COOKING overlap 5, SHOWERING overlap 10
        occs = [_occ("COOKING", 0, 10), _occ("SHOWERING", 20, 30)]
        labels, _ = MostOverlappingAnnotator().transform(segs, occs)
        assert labels.tolist() == [1]
