from datetime import datetime, timedelta

import pytest

from backend.models import (
    FunctionalArea,
    Interval,
    Segment,
    Sensor,
    SensorActivation,
    SensorActivations,
    SensorMeta,
    SensorObject,
    SensorType,
)
from backend.representation.methods import (
    BaseRepresentationMethod,
    FuzzyMatrixExtractor,
    StatsVectorExtractor,
    TextualMatrixExtractor,
    TextualVectorExtractor,
)
from backend.representation.methods.manual.base import group_activations_by_sensor
from backend.representation.methods.manual.matrix.fuzzy.base import (
    FuzzyTemporalWindow,
    _get_closest_fib_index,
    _make_ftws,
    fib_sequence_from_f0,
)

T0 = datetime(2024, 1, 1, 8, 0, 0)


def _sensor(
    name, area=FunctionalArea.KITCHEN, stype=SensorType.MOTION, obj=SensorObject.COOKTOP
):
    return Sensor(
        name=name,
        meta=SensorMeta(sensor_type=stype, area=area, sensor_object=obj),
    )


def _acts():
    a, b = (
        _sensor("A"),
        _sensor("B", FunctionalArea.BATHROOM, SensorType.CONTACT, SensorObject.DOOR),
    )
    return SensorActivations(
        data=[
            SensorActivation(
                interval=Interval(T0, T0 + timedelta(minutes=5)), sensor=a
            ),
            SensorActivation(
                interval=Interval(
                    T0 + timedelta(minutes=10), T0 + timedelta(minutes=12)
                ),
                sensor=b,
            ),
            SensorActivation(
                interval=Interval(
                    T0 + timedelta(minutes=30), T0 + timedelta(minutes=45)
                ),
                sensor=a,
            ),
        ]
    )


def _segments(n=2):
    return [
        Segment(
            interval=Interval(
                T0 + timedelta(minutes=10 * i), T0 + timedelta(minutes=10 * i + 8)
            ),
            event_indices=(2 * i, 2 * i + 1),
        )
        for i in range(n)
    ]


class TestBase:
    def test_abstract(self):
        with pytest.raises(TypeError):
            BaseRepresentationMethod()

    def test_super_calls(self):
        class Dummy(BaseRepresentationMethod):
            def fit(self, sensor_activations, **kwargs):
                return super().fit(sensor_activations, **kwargs)

            def transform(self, segments, sensor_activations):
                return super().transform(segments, sensor_activations)

        d = Dummy()
        assert d.fit(SensorActivations(data=[])) is None
        assert d.transform([], SensorActivations(data=[])) is None

    def test_deepcopy(self):
        import copy

        ext = FuzzyMatrixExtractor(minutes=30)
        ext.sensors = {_sensor("A")}
        cloned = copy.deepcopy(ext)
        assert cloned is not ext
        assert cloned.minutes == 30
        assert cloned.sensors == ext.sensors


class TestGroupBySensor:
    def test_groups_and_sorts(self):
        acts = _acts()
        grouped = group_activations_by_sensor(acts)
        assert set(grouped.keys()) == {"A", "B"}
        assert [a.interval.start for a in grouped["A"].data] == sorted(
            a.interval.start for a in grouped["A"].data
        )


class TestFuzzyTemporalWindow:
    def _ftw(self, direction="past"):
        return FuzzyTemporalWindow(
            direction=direction,
            l1=timedelta(minutes=4),
            l2=timedelta(minutes=3),
            l3=timedelta(minutes=2),
            l4=timedelta(minutes=1),
        )

    def test_apply_trapezoidal_all_branches(self):
        ftw = self._ftw()
        t_l1, t_l2, t_l3, t_l4 = (
            T0,
            T0 + timedelta(minutes=2),
            T0 + timedelta(minutes=3),
            T0 + timedelta(minutes=4),
        )
        assert (
            ftw.apply_trapezoidal_fn(
                t_l1 - timedelta(seconds=1), t_l1, t_l2, t_l3, t_l4
            )
            == 0.0
        )  # <= l1
        assert ftw.apply_trapezoidal_fn(t_l1, t_l1, t_l2, t_l3, t_l4) == 0.0
        r = ftw.apply_trapezoidal_fn(
            t_l1 + timedelta(minutes=1), t_l1, t_l2, t_l3, t_l4
        )
        assert r == pytest.approx(0.5)  # between l1 and l2
        assert ftw.apply_trapezoidal_fn(t_l3, t_l1, t_l2, t_l3, t_l4) == 1.0  # <= l3
        r = ftw.apply_trapezoidal_fn(
            t_l3 + timedelta(minutes=0.5), t_l1, t_l2, t_l3, t_l4
        )
        assert 0.0 < r < 1.0  # between l3 and l4
        assert ftw.apply_trapezoidal_fn(t_l4, t_l1, t_l2, t_l3, t_l4) == 0.0  # >= l4

    def test_compute_degree_past_and_future(self):
        acts = _acts()
        for direction in ("past", "future"):
            ftw = self._ftw(direction)
            deg = ftw.compute_degree(T0 + timedelta(minutes=40), acts)
            assert 0.0 <= deg <= 1.0

    def test_compute_degree_single_full_coverage(self):
        a = _sensor("A")
        ftw = FuzzyTemporalWindow(
            direction="past",
            l1=timedelta(minutes=4),
            l2=timedelta(minutes=3),
            l3=timedelta(minutes=2),
            l4=timedelta(minutes=1),
        )
        # window [t-4min, t-1min] fully covered by the single activation
        acts = SensorActivations(
            data=[
                SensorActivation(
                    interval=Interval(T0, T0 + timedelta(minutes=20)), sensor=a
                )
            ]
        )
        deg = ftw.compute_degree(T0 + timedelta(minutes=13), acts)
        assert deg == 1.0

    def test_compute_degree_no_overlap(self):
        acts = _acts()
        ftw = self._ftw("past")
        # far in the past: no activations in window
        deg = ftw.compute_degree(T0 - timedelta(hours=5), acts)
        assert deg == 0.0


class TestFibHelpers:
    def test_fib_sequence(self):
        assert fib_sequence_from_f0(0) == []
        assert fib_sequence_from_f0(-3) == []
        assert fib_sequence_from_f0(1) == [0]
        assert fib_sequence_from_f0(5) == [0, 0, 1, 1, 2, 3]

    def test_closest_fib_index(self):
        assert _get_closest_fib_index(0) == 0
        assert _get_closest_fib_index(-5) == 0
        assert _get_closest_fib_index(1) == 1
        assert _get_closest_fib_index(2) == 3
        assert _get_closest_fib_index(100) == _get_closest_fib_index(
            100
        )  # deterministic

    def test_make_ftws_branches(self):
        with pytest.raises(AssertionError):
            _make_ftws(None, None, None, "past")
        a = _make_ftws(minutes=30, nb_ftws=None, ftw_params=None, direction="past")
        assert len(a) > 0
        b = _make_ftws(minutes=None, nb_ftws=4, ftw_params=None, direction="future")
        assert len(b) == 4
        c = _make_ftws(
            minutes=None, nb_ftws=None, ftw_params=[[1, 2, 3, 4]], direction="past"
        )
        assert len(c) == 1
        assert c[0].l1 == timedelta(minutes=4)
        # past reverses order
        assert (
            _make_ftws(minutes=30, nb_ftws=None, ftw_params=None, direction="past")[
                0
            ].l4
            >= _make_ftws(minutes=30, nb_ftws=None, ftw_params=None, direction="past")[
                -1
            ].l4
        )


class TestFuzzyMatrixExtractor:
    def test_fit_sets_sensors(self):
        ext = FuzzyMatrixExtractor(minutes=30)
        ext.fit(_acts())
        assert ext.sensors == {a.sensor for a in _acts().data}

    def test_set_sensors(self):
        ext = FuzzyMatrixExtractor(minutes=30)
        s = {_sensor("Z")}
        ext.set_sensors(s)
        assert ext.sensors is s

    def test_transform(self):
        ext = FuzzyMatrixExtractor(minutes=30)
        ext.fit(_acts())
        feats, masks = ext.transform(_segments(2), _acts())
        assert feats.ndim == 3  # (segments, ftws, sensors)
        assert feats.shape[0] == 2
        assert masks.size == 0

    def test_transform_nb_ftws(self):
        ext = FuzzyMatrixExtractor(minutes=None, nb_ftws=3)
        ext.fit(_acts())
        feats, _ = ext.transform(_segments(2), _acts())
        assert feats.shape[1] == 3

    def test_transform_ftw_params(self):
        ext = FuzzyMatrixExtractor(
            minutes=None, nb_ftws=None, ftw_params=[[1, 2, 3, 4], [2, 3, 4, 5]]
        )
        ext.fit(_acts())
        feats, _ = ext.transform(_segments(2), _acts())
        assert feats.shape[1] == 2

    def test_compute_requires_sensors(self):
        ext = FuzzyMatrixExtractor(minutes=30)
        with pytest.raises(AssertionError, match="Sensors must be set"):
            ext.compute_fuzzy_features(_segments(1), _acts(), 30, None, None)

    def test_empty_segments(self):
        ext = FuzzyMatrixExtractor(minutes=30)
        ext.fit(_acts())
        feats = ext.compute_fuzzy_features([], _acts(), 30, None, None)
        assert feats.size == 0


class TestStatsVectorExtractor:
    ALL = [
        "sensor_state_changed",
        "sensor_last_fired",
        "sensor_is_active",
        "sensor_shifted_activation",
        "sensor_event_count",
        "sensor_dwell_time",
        "segment_trigonometric_time_encoding",
        "segment_duration",
    ]

    def test_fit_transform_all_features(self):
        ext = StatsVectorExtractor(desired_features=self.ALL)
        ext.fit(_acts())
        feats, masks = ext.transform(_segments(3), _acts())
        assert feats.shape[0] == 3
        assert masks.size == 0
        # 6 sensor features * 2 sensors + 4 trig + 1 duration
        assert feats.shape[1] == 6 * 2 + 4 + 1

    def test_single_feature(self):
        ext = StatsVectorExtractor(desired_features=["segment_duration"])
        ext.fit(_acts())
        feats, _ = ext.transform(_segments(2), _acts())
        assert feats.shape == (2, 1)

    def test_empty_segments(self):
        ext = StatsVectorExtractor(desired_features=["segment_duration"])
        ext.fit(_acts())
        feats, _ = ext.transform([], _acts())
        assert feats.shape == (0, 0)

    def test_empty_desired(self):
        ext = StatsVectorExtractor(desired_features=[])
        ext.fit(_acts())
        feats, _ = ext.transform(_segments(2), _acts())
        assert feats.shape == (2, 0)

    def test_unknown_feature_raises(self):
        ext = StatsVectorExtractor(desired_features=["bogus"])
        ext.fit(_acts())
        with pytest.raises(ValueError, match="Unknown feature"):
            ext.transform(_segments(1), _acts())

    def test_no_sensor_features(self):
        # desired features without sensor_* -> sensor block skipped, zero fallbacks used
        ext = StatsVectorExtractor(
            desired_features=["segment_duration", "segment_trigonometric_time_encoding"]
        )
        ext.fit(_acts())
        feats, _ = ext.transform(_segments(2), _acts())
        assert feats.shape == (2, 5)

    def test_sensor_feature_without_sensor_block(self):
        # sensor feature requested but training_sensors empty and no activations
        ext = StatsVectorExtractor(desired_features=["sensor_is_active"])
        ext.fit(SensorActivations(data=[]))
        feats, _ = ext.transform(_segments(2), SensorActivations(data=[]))
        assert feats.shape == (2, 0)

    def test_segment_before_events(self):
        # segment entirely before all activations -> last_si stays None
        ext = StatsVectorExtractor(
            desired_features=["sensor_last_fired", "sensor_state_changed"]
        )
        ext.fit(_acts())
        segs = [
            Segment(
                interval=Interval(T0 - timedelta(hours=2), T0 - timedelta(hours=1)),
                event_indices=(0, 0),
            )
        ]
        feats, _ = ext.transform(segs, _acts())
        assert feats.shape == (1, 4)
        assert feats[0, :2].sum() == 0  # last_fired all zero

    def test_zero_duration_segment(self):
        ext = StatsVectorExtractor(desired_features=["sensor_dwell_time"])
        ext.fit(_acts())
        segs = [
            Segment(interval=Interval(T0, T0), event_indices=(0, 0)),
            Segment(
                interval=Interval(T0, T0 + timedelta(minutes=50)), event_indices=(0, 1)
            ),
        ]
        feats, _ = ext.transform(segs, _acts())
        assert feats[0].sum() == 0.0  # zero-duration -> dwell 0
        assert feats[1].sum() > 0.0

    def test_single_segment_no_shift(self):
        ext = StatsVectorExtractor(desired_features=["sensor_shifted_activation"])
        ext.fit(_acts())
        feats, _ = ext.transform(_segments(1), _acts())
        assert feats.shape == (1, 2)
        assert feats.sum() == 0  # n_segments == 1 -> no shift


class TestTextualExtractors:
    NAMES_VEC = [
        "tdost_plain_basic_tfidf",
        "tdost_plain_basic_word2vec",
        "tdost_plain_basic_sentence_transformer",
        "tdost_plain_temporal_tfidf",
        "tdost_plain_temporal_word2vec",
        "tdost_plain_temporal_sentence_transformer",
    ]
    NAMES_MAT = [
        "tdost_seq_basic_tfidf",
        "tdost_seq_basic_word2vec",
        "tdost_seq_basic_sentence_transformer",
        "tdost_seq_temporal_tfidf",
        "tdost_seq_temporal_word2vec",
        "tdost_seq_temporal_sentence_transformer",
    ]

    def test_unknown_name_vector(self):
        with pytest.raises(ValueError, match="Unknown extractor name"):
            TextualVectorExtractor(name="nope")

    def test_unknown_name_matrix(self):
        with pytest.raises(ValueError, match="Unknown extractor name"):
            TextualMatrixExtractor(name="nope")

    def test_vector_extractor_tfidf(self):
        ext = TextualVectorExtractor(name="tdost_plain_basic_tfidf", max_features=50)
        segs = _segments(2)
        ext.fit(_acts(), segments=segs)
        feats, masks = ext.transform(segs, _acts())
        assert feats.ndim == 2
        assert feats.shape[0] == 2
        assert masks.size == 0

    def test_matrix_extractor_tfidf_pad_and_truncate(self):
        ext = TextualMatrixExtractor(
            name="tdost_seq_basic_tfidf", max_len=3, max_features=50
        )
        segs = _segments(2)
        ext.fit(_acts(), segments=segs)
        feats, masks = ext.transform(segs, _acts())
        assert feats.shape == (2, 3, feats.shape[2])
        assert masks.shape == (2, 3)
        # second segment has more events than max_len? seq basic: one desc per event
        # segment has events (0,1) -> 2 descs <= 3 -> padded
        assert masks[0].sum() == 2

    def test_matrix_extractor_truncation(self):
        # 4 events in one segment, max_len=2 -> truncate branch
        ext = TextualMatrixExtractor(
            name="tdost_seq_basic_tfidf", max_len=2, max_features=50
        )
        acts = SensorActivations(
            data=[
                SensorActivation(
                    interval=Interval(
                        T0 + timedelta(minutes=i), T0 + timedelta(minutes=i, seconds=30)
                    ),
                    sensor=_sensor("A"),
                )
                for i in range(4)
            ]
        )
        segs = [
            Segment(
                interval=Interval(T0, T0 + timedelta(minutes=4)), event_indices=(0, 7)
            )
        ]
        ext.fit(acts, segments=segs)
        feats, masks = ext.transform(segs, acts)
        assert feats.shape[1] == 2
        assert masks[0].all()

    def test_fit_requires_segments(self):
        ext = TextualVectorExtractor(name="tdost_plain_basic_tfidf")
        with pytest.raises(AssertionError, match="Segments must be provided"):
            ext.fit(_acts())


class TestStatsBranches:
    def test_sensor_missing_from_activations(self):
        # training sensors include B, but transform activations only have A
        acts_full = _acts()
        acts_a_only = SensorActivations(
            data=[a for a in acts_full.data if a.sensor.name == "A"]
        )
        ext = StatsVectorExtractor(
            desired_features=[
                "sensor_is_active",
                "sensor_event_count",
                "sensor_dwell_time",
                "sensor_state_changed",
            ]
        )
        ext.fit(acts_full)
        feats, _ = ext.transform(_segments(2), acts_a_only)
        assert feats.shape[0] == 2
        # B column all zeros (no activations)
        assert feats[:, 1].sum() == 0.0

    def test_segments_outside_event_range(self):
        ext = StatsVectorExtractor(
            desired_features=[
                "sensor_last_fired",
                "sensor_state_changed",
                "sensor_is_active",
                "sensor_event_count",
                "sensor_dwell_time",
            ]
        )
        ext.fit(_acts())
        segs = [
            Segment(
                interval=Interval(T0 - timedelta(hours=2), T0 - timedelta(hours=1)),
                event_indices=(0, 0),
            ),  # before all
            Segment(
                interval=Interval(T0 + timedelta(hours=5), T0 + timedelta(hours=6)),
                event_indices=(0, 0),
            ),  # after all
        ]
        feats, _ = ext.transform(segs, _acts())
        assert feats.shape[0] == 2
        assert feats[0].sum() == 0.0  # before all events: nothing fired yet
        assert feats[1, 0] == 1.0  # after all events: last fired sensor flagged


class TestExtractorNames:
    @pytest.mark.parametrize("name", TestTextualExtractors.NAMES_VEC)
    def test_vector_names(self, name):
        kwargs = {"model_name": "fake-model"} if "sentence_transformer" in name else {}
        ext = TextualVectorExtractor(name=name, **kwargs)
        assert ext.name == name

    @pytest.mark.parametrize("name", TestTextualExtractors.NAMES_MAT)
    def test_matrix_names(self, name):
        kwargs = {"model_name": "fake-model"} if "sentence_transformer" in name else {}
        ext = TextualMatrixExtractor(name=name, **kwargs)
        assert ext.name == name


class TestMoreBranches:
    def test_fit_with_presensors(self):
        ext = FuzzyMatrixExtractor(minutes=30)
        ext.set_sensors({_sensor("A")})
        ext.fit(_acts())  # sensors already set: no overwrite
        assert ext.sensors == {_sensor("A")}

    def test_stats_without_eventcount_and_dwell(self):
        ext = StatsVectorExtractor(
            desired_features=["sensor_is_active", "sensor_state_changed"]
        )
        ext.fit(_acts())
        feats, _ = ext.transform(_segments(2), _acts())
        assert feats.shape == (2, 4)
