from datetime import datetime, timedelta

import pytest

from backend.models import (
    ActivityOccurrence,
    DatasetStatistics,
    Event,
    EventStream,
    EventWindow,
    FunctionalArea,
    Interval,
    IntervalType,
    Segment,
    SegmentationResult,
    Sensor,
    SensorActivation,
    SensorActivations,
    SensorMeta,
    SensorObject,
    SensorType,
)

T0 = datetime(2024, 1, 1, 8, 0, 0)
T1 = T0 + timedelta(minutes=10)
T2 = T0 + timedelta(minutes=20)


def _interval(start, end, itype=IntervalType.LEFT_OPEN):
    return Interval(start=start, end=end, interval_type=itype)


class TestInterval:
    def test_post_init_rejects_end_before_start(self):
        with pytest.raises(ValueError, match="must be >="):
            _interval(T1, T0)

    def test_duration(self):
        assert _interval(T0, T1).duration == 600.0

    @pytest.mark.parametrize(
        ("itype", "t", "expected"),
        [
            (IntervalType.LEFT_OPEN, T0, True),
            (IntervalType.LEFT_OPEN, T1, True),
            (IntervalType.LEFT_OPEN, T0 - timedelta(seconds=1), False),
            (IntervalType.OPEN, T0, False),
            (IntervalType.OPEN, T1, False),
            (IntervalType.OPEN, T0 + timedelta(seconds=1), True),
            (IntervalType.CLOSED, T0, True),
            (IntervalType.CLOSED, T1, False),
            (IntervalType.RIGHT_OPEN, T0, True),
            (IntervalType.RIGHT_OPEN, T1, False),
        ],
    )
    def test_contains(self, itype, t, expected):
        assert _interval(T0, T1, itype).contains(t) is expected

    def test_overlaps_no_overlap(self):
        a = _interval(T0, T1)
        b = _interval(T2, T2 + timedelta(minutes=5))
        assert a.overlaps(b) is False
        assert b.overlaps(a) is False

    @pytest.mark.parametrize(
        ("self_type", "other_type", "expected"),
        [
            (IntervalType.CLOSED, IntervalType.CLOSED, False),
            (IntervalType.CLOSED, IntervalType.LEFT_OPEN, False),
            (IntervalType.CLOSED, IntervalType.RIGHT_OPEN, False),
            (IntervalType.CLOSED, IntervalType.OPEN, False),
            (IntervalType.LEFT_OPEN, IntervalType.CLOSED, False),
            (IntervalType.LEFT_OPEN, IntervalType.LEFT_OPEN, True),
            (IntervalType.LEFT_OPEN, IntervalType.RIGHT_OPEN, True),
            (IntervalType.LEFT_OPEN, IntervalType.OPEN, False),
            (IntervalType.RIGHT_OPEN, IntervalType.CLOSED, False),
            (IntervalType.RIGHT_OPEN, IntervalType.LEFT_OPEN, False),
            (IntervalType.RIGHT_OPEN, IntervalType.RIGHT_OPEN, False),
            (IntervalType.OPEN, IntervalType.OPEN, False),
        ],
    )
    def test_overlaps_touching_endpoints(self, self_type, other_type, expected):
        # a ends exactly when b starts
        a = _interval(T0, T1, self_type)
        b = _interval(T1, T2, other_type)
        assert a.overlaps(b) is expected

    @pytest.mark.parametrize(
        ("self_type", "other_type", "expected"),
        [
            (IntervalType.CLOSED, IntervalType.CLOSED, False),
            (IntervalType.LEFT_OPEN, IntervalType.CLOSED, False),
            (IntervalType.LEFT_OPEN, IntervalType.LEFT_OPEN, True),
            (IntervalType.LEFT_OPEN, IntervalType.RIGHT_OPEN, False),
            (IntervalType.RIGHT_OPEN, IntervalType.CLOSED, False),
            (IntervalType.RIGHT_OPEN, IntervalType.LEFT_OPEN, True),
            (IntervalType.OPEN, IntervalType.OPEN, False),
        ],
    )
    def test_overlaps_touching_startpoints(self, self_type, other_type, expected):
        # a starts exactly when b ends
        a = _interval(T1, T2, self_type)
        b = _interval(T0, T1, other_type)
        assert a.overlaps(b) is expected

    def test_overlaps_true(self):
        a = _interval(T0, T2)
        b = _interval(T1, T2 + timedelta(minutes=5))
        assert a.overlaps(b) is True

    @pytest.mark.parametrize(
        ("self_type", "other_type", "expected"),
        [
            (IntervalType.LEFT_OPEN, IntervalType.LEFT_OPEN, True),
            (IntervalType.OPEN, IntervalType.LEFT_OPEN, False),
            (IntervalType.LEFT_OPEN, IntervalType.OPEN, True),
            (IntervalType.CLOSED, IntervalType.LEFT_OPEN, False),
            (IntervalType.LEFT_OPEN, IntervalType.RIGHT_OPEN, True),
            (IntervalType.RIGHT_OPEN, IntervalType.LEFT_OPEN, False),
        ],
    )
    def test_contains_interval_same_bounds(self, self_type, other_type, expected):
        a = _interval(T0, T1, self_type)
        b = _interval(T0, T1, other_type)
        assert a.contains_interval(b) is expected

    def test_contains_interval_strict_inside(self):
        outer = _interval(T0, T2)
        inner = _interval(T1, T1 + timedelta(minutes=5))
        assert outer.contains_interval(inner) is True
        assert inner.contains_interval(outer) is False

    def test_contains_interval_equal_start(self):
        a = _interval(T0, T2, IntervalType.RIGHT_OPEN)
        b = _interval(T0, T1, IntervalType.LEFT_OPEN)
        assert a.contains_interval(b) is True

    def test_contains_interval_equal_end(self):
        a = _interval(T0, T2, IntervalType.LEFT_OPEN)
        b = _interval(T1, T2, IntervalType.RIGHT_OPEN)
        assert a.contains_interval(b) is True

    def test_intersection_no_overlap(self):
        a = _interval(T0, T1)
        b = _interval(T2, T2 + timedelta(minutes=5))
        assert a.intersection(b) is None

    @pytest.mark.parametrize(
        ("a_type", "b_type", "expected_type"),
        [
            (IntervalType.OPEN, IntervalType.OPEN, IntervalType.OPEN),
            (IntervalType.OPEN, IntervalType.CLOSED, IntervalType.OPEN),
            (IntervalType.CLOSED, IntervalType.OPEN, IntervalType.OPEN),
            (IntervalType.CLOSED, IntervalType.CLOSED, IntervalType.LEFT_OPEN),
            (IntervalType.RIGHT_OPEN, IntervalType.RIGHT_OPEN, IntervalType.RIGHT_OPEN),
            (IntervalType.LEFT_OPEN, IntervalType.RIGHT_OPEN, IntervalType.OPEN),
            (IntervalType.RIGHT_OPEN, IntervalType.LEFT_OPEN, IntervalType.OPEN),
        ],
    )
    def test_intersection_same_bounds(self, a_type, b_type, expected_type):
        a = _interval(T0, T1, a_type)
        b = _interval(T0, T1, b_type)
        inter = a.intersection(b)
        assert inter is not None
        assert inter.interval_type == expected_type
        assert inter.start == T0 and inter.end == T1

    def test_intersection_partial_overlap_left_from_self(self):
        a = _interval(T0, T2, IntervalType.OPEN)
        b = _interval(T1, T2 + timedelta(minutes=5), IntervalType.CLOSED)
        inter = a.intersection(b)
        assert inter is not None
        assert inter.start == T1 and inter.end == T2
        assert inter.interval_type == IntervalType.RIGHT_OPEN

    def test_intersection_partial_overlap_left_from_other(self):
        a = _interval(T1, T2 + timedelta(minutes=5), IntervalType.CLOSED)
        b = _interval(T0, T2, IntervalType.OPEN)
        inter = a.intersection(b)
        assert inter is not None
        assert inter.start == T1 and inter.end == T2
        assert inter.interval_type == IntervalType.RIGHT_OPEN

    def test_intersection_left_open_only(self):
        a = _interval(T0, T2, IntervalType.OPEN)
        b = _interval(T0, T1, IntervalType.CLOSED)
        inter = a.intersection(b)
        assert inter is not None
        assert inter.start == T0 and inter.end == T1
        assert inter.interval_type == IntervalType.LEFT_OPEN

    def test_intersection_right_open_from_self_end(self):
        a = _interval(T0, T2, IntervalType.RIGHT_OPEN)
        b = _interval(T1, T2 + timedelta(minutes=5), IntervalType.CLOSED)
        inter = a.intersection(b)
        assert inter is not None
        assert inter.end == T2
        assert inter.interval_type == IntervalType.RIGHT_OPEN

    def test_intersection_right_open_from_other_end(self):
        a = _interval(T1, T2 + timedelta(minutes=5), IntervalType.CLOSED)
        b = _interval(T0, T2, IntervalType.RIGHT_OPEN)
        inter = a.intersection(b)
        assert inter is not None
        assert inter.end == T2
        assert inter.interval_type == IntervalType.RIGHT_OPEN

    def test_overlap_duration(self):
        a = _interval(T0, T2)
        b = _interval(T1, T2 + timedelta(minutes=5))
        assert a.overlap_duration(b) == 600.0
        c = _interval(T2 + timedelta(minutes=10), T2 + timedelta(minutes=20))
        assert a.overlap_duration(c) == 0.0

    @pytest.mark.parametrize(
        ("itype", "left", "right"),
        [
            (IntervalType.OPEN, "(", ")"),
            (IntervalType.LEFT_OPEN, "(", "]"),
            (IntervalType.RIGHT_OPEN, "[", ")"),
            (IntervalType.CLOSED, "[", "]"),
        ],
    )
    def test_repr(self, itype, left, right):
        r = repr(_interval(T0, T1, itype))
        assert r.startswith(left) and r.endswith(right)


class TestEnums:
    def test_values(self):
        assert FunctionalArea.KITCHEN.value == "kitchen"
        assert SensorType.MOTION.value == "motion"
        assert SensorObject.COOKTOP.value == "cooktop"
        assert FunctionalArea.UNDEFINED.value == "undefined"
        assert list(IntervalType) == [
            IntervalType.CLOSED,
            IntervalType.OPEN,
            IntervalType.LEFT_OPEN,
            IntervalType.RIGHT_OPEN,
        ]


class TestSensor:
    def test_hash_by_name(self, sensor_a):
        other = Sensor(
            name="SENSOR_A",
            meta=SensorMeta(
                sensor_type=SensorType.CONTACT,
                area=FunctionalArea.BATHROOM,
                sensor_object=SensorObject.DOOR,
            ),
        )
        assert hash(sensor_a) == hash("SENSOR_A")
        assert hash(sensor_a) == hash(other)  # hash is by name only
        assert sensor_a != other  # dataclass eq compares meta too


class TestEvent:
    def test_invalid_value(self, sensor_a):
        with pytest.raises(ValueError, match="must be 0 or 1"):
            Event(timestamp=T0, sensor=sensor_a, value=2)

    def test_lt_by_timestamp(self, sensor_a, sensor_b):
        e1 = Event(timestamp=T0, sensor=sensor_b, value=1)
        e2 = Event(timestamp=T1, sensor=sensor_a, value=1)
        assert e1 < e2

    def test_lt_same_timestamp_by_sensor_name(self, sensor_a, sensor_b):
        e1 = Event(timestamp=T0, sensor=sensor_a, value=1)
        e2 = Event(timestamp=T0, sensor=sensor_b, value=1)
        assert e1 < e2
        assert not e2 < e1


class TestEventWindow:
    def test_invalid(self):
        with pytest.raises(ValueError, match="j_last"):
            EventWindow(j_first=5, j_last=2)

    def test_size_and_contains(self):
        w = EventWindow(j_first=2, j_last=5)
        assert w.size == 4
        assert w.contains_index(2)
        assert w.contains_index(5)
        assert not w.contains_index(1)
        assert not w.contains_index(6)


class TestEventStream:
    def test_empty(self):
        s = EventStream([])
        assert len(s) == 0
        assert s.sensors == set()
        assert s.interval is None
        assert s.find_events_in_interval(_interval(T0, T1)) == []

    def test_unordered_raises(self, sensor_a, make_event):
        with pytest.raises(ValueError, match="time-ordered"):
            EventStream([make_event(sensor_a, 10), make_event(sensor_a, 5)])

    def test_index_assigned(self, event_stream):
        assert [e.index for e in event_stream.events] == [0, 1, 2, 3, 4]

    def test_sensors(self, event_stream, sensor_a, sensor_b):
        assert event_stream.sensors == {sensor_a, sensor_b}

    def test_interval(self, event_stream):
        iv = event_stream.interval
        assert iv is not None
        assert iv.start == T0
        assert iv.end == T0 + timedelta(minutes=20)

    def test_len_and_getitem(self, event_stream):
        assert len(event_stream) == 5
        assert event_stream[0].value == 1
        assert len(event_stream[0:2]) == 2
        window = EventWindow(j_first=1, j_last=3)
        assert len(event_stream[window]) == 3

    def test_find_events_in_interval(self, event_stream):
        iv = _interval(T0 + timedelta(minutes=5), T0 + timedelta(minutes=15))
        idx = event_stream.find_events_in_interval(iv, include_boundaries=True)
        assert idx == [1, 2, 3]
        idx_all = event_stream.find_events_in_interval(iv, include_boundaries=False)
        assert idx_all == [1, 2, 3]
        # with boundaries excluded by interval semantics (OPEN interval)
        iv_open = _interval(
            T0 + timedelta(minutes=5), T0 + timedelta(minutes=15), IntervalType.OPEN
        )
        assert event_stream.find_events_in_interval(iv_open) == [2]

    def test_filter_by_sensor(self, event_stream, sensor_a):
        filtered = event_stream.filter_by_sensor(sensor_a)
        assert len(filtered) == 3
        assert all(e.sensor == sensor_a for e in filtered.events)

    def test_filter_by_area(self, event_stream):
        filtered = event_stream.filter_by_area(FunctionalArea.BATHROOM)
        assert len(filtered) == 2
        filtered_k = event_stream.filter_by_area(FunctionalArea.GARAGE)
        assert len(filtered_k) == 0


class TestSensorActivations:
    def test_duration(self, sensor_a):
        act = SensorActivation(interval=_interval(T0, T1), sensor=sensor_a)
        assert act.duration == 600.0

    def test_to_event_stream(self, sensor_a, sensor_b):
        acts = SensorActivations(
            data=[
                SensorActivation(interval=_interval(T0, T1), sensor=sensor_a),
                SensorActivation(
                    interval=_interval(T0 + timedelta(minutes=5), T2), sensor=sensor_b
                ),
            ]
        )
        stream = acts.to_event_stream()
        assert len(stream) == 4
        assert [e.value for e in stream.events] == [1, 1, 0, 0]
        assert [e.index for e in stream.events] == [0, 1, 2, 3]


class TestActivityOccurrence:
    def test_duration(self):
        occ = ActivityOccurrence(interval=_interval(T0, T1), name="COOKING")
        assert occ.duration == 600.0
        assert occ.name == "COOKING"


def test_dataset_statistics():
    stats = DatasetStatistics(
        num_sensors=2,
        num_activities=1,
        num_sensor_activations=10,
        num_activity_occurrences=3,
        time_span=(T0, T1),
        total_duration_seconds=600.0,
    )
    assert stats.num_sensors == 2
    assert stats.time_span == (T0, T1)


class TestSegment:
    def test_invalid_confidence(self, sensor_a):
        with pytest.raises(ValueError, match="Confidence"):
            Segment(
                interval=_interval(T0, T1),
                event_indices=(0, 1),
                confidence=1.5,
            )

    def test_invalid_indices(self):
        with pytest.raises(ValueError, match="Invalid event indices"):
            Segment(interval=_interval(T0, T1), event_indices=(3, 1))

    def test_properties(self, event_stream):
        seg = Segment(
            interval=_interval(T0, T1),
            event_indices=(0, 2),
            label="COOKING",
            confidence=0.9,
            metadata={"k": "v"},
        )
        assert seg.num_events == 3
        assert seg.duration == 600.0
        assert seg.label == "COOKING"
        window = seg.to_event_window(event_stream)
        assert window.j_first == 0 and window.j_last == 2
        assert len(window.events) == 3

    def test_defaults(self):
        seg = Segment(interval=_interval(T0, T1), event_indices=(0, 0))
        assert seg.label is None
        assert seg.confidence is None
        assert seg.metadata == {}


class TestSegmentationResult:
    def test_empty(self):
        res = SegmentationResult(segments=[], method_name="m", parameters={})
        assert res.num_segments == 0
        assert res.total_duration == 0.0
        assert res.get_boundaries() == []

    def test_with_segments(self):
        segs = [
            Segment(interval=_interval(T0, T1), event_indices=(0, 1)),
            Segment(
                interval=_interval(T1, T2), event_indices=(2, 3), metadata={"a": 1}
            ),
        ]
        res = SegmentationResult(
            segments=segs, method_name="m", parameters={"p": 1}, metadata={"x": 2}
        )
        assert res.num_segments == 2
        assert res.total_duration == 1200.0
        assert res.get_boundaries() == [T0, T1, T2]
        assert res.metadata == {"x": 2}
