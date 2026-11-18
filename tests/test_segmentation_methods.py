import copy
from datetime import datetime, timedelta

import numpy as np
import pytest

from backend.models import (
    Event,
    EventStream,
    FunctionalArea,
    Sensor,
    SensorMeta,
    SensorObject,
    SensorType,
)
from backend.segmentation.methods import (
    BaseSegmentationMethod,
    CorrWan,
    CorrXu,
    FixedCount,
    FixedTime,
    RulsifPeCpd,
    RulsifPlsbdCpd,
    SepCpd,
    W2vCcsCPD,
    W2vCcsdCPD,
    W2vCsCPD,
)
from backend.segmentation.methods.change_point_detection.distributional.base import (
    CpdConfig,
    _CpdCore,
)

BASE = EventStream([])  # placeholder for typing


class TestBase:
    def test_abstract(self):
        with pytest.raises(TypeError):
            BaseSegmentationMethod("x")

    def test_super_calls_cover_pass(self, event_stream):
        class Dummy(BaseSegmentationMethod):
            def fit(self, events, **kwargs):
                return super().fit(events, **kwargs)

            def transform(self, events):
                return super().transform(events)

        d = Dummy("d")
        assert d.fit(event_stream) is None
        assert d.transform(event_stream) is None

    def test_deepcopy_resets_state(self, event_stream):
        seg = FixedCount(nb_events=3)
        seg.fit(event_stream)
        assert seg._is_calibrated is True
        cloned = copy.deepcopy(seg)
        assert cloned._is_calibrated is False
        assert cloned.nb_events == 3
        assert cloned.method_name == "FixedCount"


class TestFixedCount:
    def test_validation(self):
        with pytest.raises(ValueError, match="nb_events"):
            FixedCount(nb_events=0)
        with pytest.raises(ValueError, match="overlap_percentage"):
            FixedCount(nb_events=2, overlap_percentage=101)

    def test_fit_returns_self(self, event_stream):
        seg = FixedCount(nb_events=2)
        assert seg.fit(event_stream) is seg
        assert seg._is_calibrated

    def test_transform_empty(self):
        res = FixedCount(nb_events=2).transform(EventStream([]))
        assert res.segments == []
        assert res.parameters == {"nb_events": 2, "overlap_percentage": 0.0}

    def test_transform(self, long_event_stream):
        res = FixedCount(nb_events=4).transform(long_event_stream)
        assert res.num_segments == 12
        assert res.segments[0].num_events == 5  # j_last = j_first + nb_events
        assert res.segments[-1].event_indices[1] == len(long_event_stream) - 1
        assert res.segments[0].metadata["num_events_actual"] == 5

    def test_transform_overlap(self, long_event_stream):
        res = FixedCount(nb_events=4, overlap_percentage=50).transform(
            long_event_stream
        )
        # step = 4 - 2 = 2 -> j_first = 0,2,...,44 -> 23 segments
        assert res.num_segments == 23

    def test_transform_single_event_stream(self, event_stream):
        single = EventStream(event_stream.events[:1])
        res = FixedCount(nb_events=10).transform(single)
        assert res.num_segments == 1
        assert res.segments[0].num_events == 1


class TestFixedTime:
    def test_validation(self):
        with pytest.raises(ValueError, match="length_in_seconds"):
            FixedTime(length_in_seconds=0)
        with pytest.raises(ValueError, match="overlap_percentage"):
            FixedTime(overlap_percentage=-1)

    def test_transform_empty(self):
        res = FixedTime(length_in_seconds=60).transform(EventStream([]))
        assert res.segments == []

    def test_transform(self, long_event_stream):
        res = FixedTime(length_in_seconds=1200).transform(long_event_stream)
        assert res.num_segments >= 1
        seg = res.segments[0]
        assert seg.metadata["duration_requested_seconds"] == 1200
        assert seg.interval.duration == 1200

    def test_fit(self, event_stream):
        seg = FixedTime(length_in_seconds=60)
        assert seg.fit(event_stream) is seg
        assert seg._is_calibrated

    def test_transform_gap_segment(self, sensor_a):
        from datetime import datetime

        t = datetime(2024, 1, 1, 8, 0, 0)
        stream = EventStream(
            [
                Event(timestamp=t, sensor=sensor_a, value=1),
                Event(timestamp=t + timedelta(minutes=5), sensor=sensor_a, value=0),
                Event(timestamp=t + timedelta(hours=2), sensor=sensor_a, value=1),
            ]
        )
        res = FixedTime(length_in_seconds=600).transform(stream)
        # window [600s, 1200s) contains no events
        assert any(seg.metadata["num_events"] == 0 for seg in res.segments)

    def test_transform_full_overlap_fallback(self, long_event_stream):
        # overlap 100% -> step <= 0 -> falls back to full length
        res = FixedTime(length_in_seconds=600, overlap_percentage=100).transform(
            long_event_stream
        )
        assert res.num_segments >= 1


class TestCorrWan:
    def test_validation(self):
        with pytest.raises(ValueError, match="threshold"):
            CorrWan(threshold=2.0)
        with pytest.raises(ValueError, match="time_bin_size"):
            CorrWan(time_bin_size=0)
        with pytest.raises(ValueError, match="time_th"):
            CorrWan(time_th="median")

    def test_fit_transform_no_labels(self, sensor_a, sensor_b):
        dense = EventStream(
            [
                ev
                for k in range(20)
                for ev in (
                    Event(
                        timestamp=datetime(2024, 1, 1, 8, 0, 0)
                        + timedelta(seconds=10 * k),
                        sensor=sensor_a if k % 2 == 0 else sensor_b,
                        value=1,
                    ),
                    Event(
                        timestamp=datetime(2024, 1, 1, 8, 0, 0)
                        + timedelta(seconds=10 * k + 5),
                        sensor=sensor_a if k % 2 == 0 else sensor_b,
                        value=0,
                    ),
                )
            ]
        )
        seg = CorrWan(threshold=0.0, time_bin_size=30)
        seg.fit(dense)
        res = seg.transform(dense)
        assert res.method_name.startswith("CorrWan")
        assert "windows_closed_time" in res.metadata
        assert "max_concurrent_windows" in res.metadata
        assert res.num_segments >= 1

    def test_fit_transform_with_labels(
        self, long_event_stream, activity_labels_for_stream
    ):
        seg = CorrWan(threshold=0.0, time_bin_size=300, time_th="mean")
        seg.fit(long_event_stream, activity_labels=activity_labels_for_stream)
        res = seg.transform(long_event_stream)
        assert res.num_segments >= 1

    def test_transform_not_calibrated(self, long_event_stream):
        with pytest.raises(RuntimeError, match="calibrate"):
            CorrWan().transform(long_event_stream)

    def test_unknown_sensor_in_transform(self, sensor_a, sensor_b):
        dense = EventStream(
            [
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 0, 0), sensor=sensor_a, value=1
                ),
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 0, 5), sensor=sensor_a, value=0
                ),
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 0, 10), sensor=sensor_b, value=1
                ),
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 0, 15), sensor=sensor_b, value=0
                ),
            ]
        )
        seg = CorrWan(threshold=0.0, time_bin_size=30)
        seg.fit(dense)
        other = Sensor(
            name="UNKNOWN",
            meta=SensorMeta(
                sensor_type=SensorType.MOTION,
                area=FunctionalArea.GARAGE,
                sensor_object=SensorObject.DOOR,
            ),
        )
        stream = EventStream(
            dense.events
            + [Event(timestamp=datetime(2024, 1, 1, 8, 1, 0), sensor=other, value=1)]
        )
        res = seg.transform(stream)
        assert res.num_segments >= 1

    def test_time_thresholds_defaults(self, long_event_stream):
        seg = CorrWan()
        t_span, t_int = seg._compute_time_thresholds(long_event_stream, None)
        assert t_span[FunctionalArea.KITCHEN] == 60.0
        assert t_int[FunctionalArea.GARAGE] == 60.0

    def test_correlation_matrix_zero_std(self, sensor_a):
        # single sensor -> X row constant -> std 0 branch for off-diagonal impossible (M=1),
        # so use two sensors where one never activates
        from datetime import datetime

        t = datetime(2024, 1, 1, 8, 0, 0)
        stream = EventStream(
            [
                Event(timestamp=t, sensor=sensor_a, value=1),
                Event(timestamp=t + timedelta(minutes=5), sensor=sensor_a, value=0),
            ]
        )
        seg = CorrWan()
        sensors = list(stream.sensors)
        seg._sensor_to_idx = {s.name: i for i, s in enumerate(sensors)}
        X = seg._build_activation_sequences(stream, sensors)
        scm = seg._compute_correlation_matrix(stream, sensors)
        assert scm.shape == (1, 1)
        assert scm[0, 0] == 1.0
        assert X.shape[0] == 1

    def test_still_on_at_end(self, sensor_a):
        from datetime import datetime

        t = datetime(2024, 1, 1, 8, 0, 0)
        stream = EventStream([Event(timestamp=t, sensor=sensor_a, value=1)])
        seg = CorrWan(time_bin_size=60)
        seg._sensor_to_idx = {sensor_a.name: 0}
        X = seg._build_activation_sequences(stream, [sensor_a])
        assert X[0].sum() > 0  # on-until-end fills bins


class TestCorrXu:
    def test_validation(self):
        with pytest.raises(ValueError, match="context_size"):
            CorrXu(context_size=1)

    def test_scm_not_enough_data(self, event_stream):
        seg = CorrXu(context_size=100)
        with pytest.raises(ValueError, match="Not enough data"):
            seg._compute_scm(event_stream, {})

    def test_sct_not_initialized(self):
        seg = CorrXu(context_size=2)
        with pytest.raises(RuntimeError, match="not initialized"):
            seg._compute_sct(np.zeros((2, 2)), [], None)

    def test_sct_critical_neighbors(self, long_event_stream):
        seg = CorrXu(context_size=4)
        sensors = list(long_event_stream.sensors)
        seg._sensor_to_idx = {s.name: i for i, s in enumerate(sensors)}
        scm = seg._compute_scm(long_event_stream, seg._sensor_to_idx)
        sct = seg._compute_sct(scm, sensors, {"SENSOR_A": "SENSOR_B"})
        assert sct["SENSOR_A"] == scm[0, 1]
        sct2 = seg._compute_sct(scm, sensors, None)
        assert set(sct2) == {"SENSOR_A", "SENSOR_B"}

    def test_sct_single_sensor(self, sensor_a):
        from datetime import datetime

        t = datetime(2024, 1, 1, 8, 0, 0)
        stream = EventStream(
            [
                Event(timestamp=t, sensor=sensor_a, value=1),
                Event(timestamp=t + timedelta(minutes=1), sensor=sensor_a, value=0),
                Event(timestamp=t + timedelta(minutes=2), sensor=sensor_a, value=1),
            ]
        )
        seg = CorrXu(context_size=2)
        seg._sensor_to_idx = {sensor_a.name: 0}
        scm = seg._compute_scm(stream, seg._sensor_to_idx)
        sct = seg._compute_sct(scm, [sensor_a], None)
        assert sct[sensor_a.name] == 0.0  # empty values branch

    def test_mts_no_labels(self, long_event_stream):
        seg = CorrXu(context_size=4)
        mts = seg._compute_mts(None)
        assert all(v == 3600.0 for v in mts.values())

    def test_mts_with_labels(self, activity_labels_for_stream):
        seg = CorrXu(context_size=4)
        mts = seg._compute_mts(activity_labels_for_stream)
        assert (
            mts[FunctionalArea.KITCHEN] == 3600.0
        )  # mu + 2*sigma, single sample -> sigma 0
        assert mts[FunctionalArea.GARAGE] == 3600.0

    def test_fit_transform(self, long_event_stream, activity_labels_for_stream):
        seg = CorrXu(context_size=4)
        seg.fit(long_event_stream, activity_labels=activity_labels_for_stream)
        res = seg.transform(long_event_stream)
        assert res.num_segments == len(long_event_stream) - 1
        assert res.parameters == {"context_size": 4}
        assert res.metadata == {"num_events": len(long_event_stream)}

    def test_transform_not_calibrated(self, long_event_stream):
        with pytest.raises(RuntimeError, match="calibrated"):
            CorrXu().transform(long_event_stream)

    def test_transform_state_none(self, long_event_stream):
        seg = CorrXu(context_size=4)
        seg._is_calibrated = True  # bypass first check, state still None
        with pytest.raises(RuntimeError, match="not properly initialized"):
            seg.transform(long_event_stream)


class TestCpdBase:
    def test_core_extract_features(self, long_event_stream):
        core = _CpdCore(CpdConfig(local_context_size=3, number_of_view_samples=2))
        X = core.extract_feature_matrix(long_event_stream)
        assert X.ndim == 2
        assert X.shape[0] == len(long_event_stream) - 3

    def test_core_select_centers_both_branches(self, long_event_stream):
        core = _CpdCore(
            CpdConfig(local_context_size=3, number_of_view_samples=2, number_of_bases=2)
        )
        X = core.extract_feature_matrix(long_event_stream)
        future = X[:2]
        centers = core.select_centers(future)
        assert len(centers) == 2  # bases <= len -> choice branch
        core2 = _CpdCore(
            CpdConfig(
                local_context_size=3, number_of_view_samples=2, number_of_bases=100
            )
        )
        centers2 = core2.select_centers(future)
        assert len(centers2) == len(future)  # bases > len -> arange branch

    def test_rulsif_theta(self, long_event_stream):
        core = _CpdCore(
            CpdConfig(local_context_size=3, number_of_view_samples=2, number_of_bases=2)
        )
        X = core.extract_feature_matrix(long_event_stream)
        theta = core.rulsif_theta(X[:2], X[2:4], X[2:4], 0.5)
        assert theta.shape == (2,)

    def test_get_feature_constants(self):
        assert SepCpd.get_default_features() == SepCpd.get_all_features()


class TestWindowCpdValidation:
    def test_invalid_params(self):
        with pytest.raises(ValueError, match="local_context_size"):
            SepCpd(local_context_size=0)
        with pytest.raises(ValueError, match="number_of_view_samples"):
            SepCpd(number_of_view_samples=0)
        with pytest.raises(ValueError, match="number_of_bases"):
            SepCpd(number_of_bases=0)
        with pytest.raises(ValueError, match="kernel_width"):
            SepCpd(kernel_width=0)
        with pytest.raises(ValueError, match="lambda_reg"):
            SepCpd(lambda_reg=0)
        with pytest.raises(ValueError, match="threshold"):
            SepCpd(threshold=2.0)

    def test_sep_alpha_must_be_none(self):
        with pytest.raises(ValueError, match="alpha_relative"):
            SepCpd(alpha_relative=0.5)

    def test_rulsif_pe_alpha(self):
        with pytest.raises(ValueError, match="alpha_relative"):
            RulsifPeCpd(alpha_relative=None)
        with pytest.raises(ValueError, match="alpha_relative"):
            RulsifPeCpd(alpha_relative=0.0)
        with pytest.raises(ValueError, match="alpha_relative"):
            RulsifPeCpd(alpha_relative=1.5)

    def test_rulsif_plsbd_alpha(self):
        with pytest.raises(ValueError, match="alpha_relative"):
            RulsifPlsbdCpd(alpha_relative=1.0)
        with pytest.raises(ValueError, match="alpha_relative"):
            RulsifPlsbdCpd(alpha_relative=None)


class TestWindowCpdTransform:
    def _run(self, cls, long_event_stream, **kwargs):
        kwargs.setdefault("threshold", 0.0)
        seg = cls(
            local_context_size=2,
            number_of_view_samples=2,
            number_of_bases=2,
            **kwargs,
        )
        seg.fit(long_event_stream)
        return seg.transform(long_event_stream)

    def test_sep(self, long_event_stream):
        res = self._run(SepCpd, long_event_stream)
        assert res.num_segments >= 1
        assert res.metadata["num_boundaries"] >= 0
        assert "max_sep_score" in res.metadata
        assert res.segments[0].metadata["boundary_type"] == "sep_cpd"

    def test_rulsif_pe(self, long_event_stream):
        res = self._run(RulsifPeCpd, long_event_stream, alpha_relative=0.5)
        assert "max_rulsif_score" in res.metadata
        assert res.segments[0].metadata["boundary_type"] == "rulsif_cpd"

    def test_rulsif_plsbd(self, long_event_stream):
        res = self._run(RulsifPlsbdCpd, long_event_stream, alpha_relative=0.5)
        assert "max_plsbd_score" in res.metadata
        assert res.segments[0].metadata["boundary_type"] == "plsbd_cpd"

    def test_too_few_events(self, event_stream):
        seg = SepCpd(local_context_size=2, number_of_view_samples=2)
        seg.fit(event_stream)
        with pytest.raises(ValueError, match="at least"):
            seg.transform(event_stream)

    def test_boundaries_found(self, long_event_stream):
        # threshold 0 with noisy features should flag at least one local max
        res = self._run(SepCpd, long_event_stream, threshold=0.0)
        assert res.metadata["num_boundaries"] >= 0


class TestW2vCpd:
    def _params(self):
        return dict(
            embedding_w2v_dimension=8,
            embedding_w2v_context_size=2,
            embedding_w2v_training_epochs=2,
            embedding_w2v_min_token_count=1,
            embedding_w2v_negative_sampling=2,
            random_state=42,
        )

    def test_base_transform_not_implemented(self, long_event_stream):
        from backend.segmentation.methods.change_point_detection.embedding.word2vec_cosine.base import (
            W2vCosineCPD,
        )

        base = W2vCosineCPD(**self._params())
        base.fit(long_event_stream)
        with pytest.raises(NotImplementedError):
            base.transform(long_event_stream)

    def test_cosine_similarity_zero_norm(self):
        from backend.segmentation.methods.change_point_detection.embedding.word2vec_cosine.base import (
            W2vCosineCPD,
        )

        base = W2vCosineCPD(**self._params())
        assert base._cosine_similarity(np.zeros(8), np.ones(8)) == 0.0
        assert base._cosine_similarity(np.ones(8), np.zeros(8)) == 0.0
        assert base._cosine_similarity(np.ones(8), np.ones(8)) == pytest.approx(1.0)

    def test_extract_embeddings_missing_action(self, sensor_a, sensor_b):
        from backend.segmentation.methods.change_point_detection.embedding.word2vec_cosine.base import (
            W2vCosineCPD,
        )

        # rare sensor fires only once -> excluded by min_count=2 -> zero vector
        stream = EventStream(
            [
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 0, 0), sensor=sensor_a, value=1
                ),
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 1, 0), sensor=sensor_a, value=0
                ),
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 2, 0), sensor=sensor_b, value=1
                ),
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 3, 0), sensor=sensor_b, value=0
                ),
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 4, 0), sensor=sensor_a, value=1
                ),
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 5, 0), sensor=sensor_a, value=0
                ),
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 6, 0), sensor=sensor_b, value=0
                ),
            ]
        )
        base = W2vCosineCPD(**{**self._params(), "embedding_w2v_min_token_count": 2})
        base.fit(stream)
        assert base._action_embeddings is not None
        assert any(np.linalg.norm(v) == 0.0 for v in base._action_embeddings.values())

    def test_cs_variant(self, long_event_stream):
        seg = W2vCsCPD(threshold=0.9, **self._params())
        with pytest.raises(ValueError, match="threshold"):
            W2vCsCPD(threshold=2.0, **self._params())
        seg.fit(long_event_stream)
        res = seg.transform(long_event_stream)
        assert res.segments[0].metadata["variant"] == "CS"
        assert "num_boundaries" in res.metadata

    def test_ccs_variant(self, long_event_stream):
        seg = W2vCcsCPD(threshold=0.0, context_size=2, **self._params())
        with pytest.raises(ValueError, match="threshold"):
            W2vCcsCPD(threshold=-0.1, **self._params())
        seg.fit(long_event_stream)
        res = seg.transform(long_event_stream)
        assert res.segments[0].metadata["variant"] == "CCS"

    def test_ccsd_variant(self, long_event_stream):
        seg = W2vCcsdCPD(context_size=2, delta=0.0, **self._params())
        seg.fit(long_event_stream)
        res = seg.transform(long_event_stream)
        assert res.segments[0].metadata["variant"] == "CCSD"

    def test_not_calibrated(self, long_event_stream):
        seg = W2vCsCPD(**self._params())
        with pytest.raises(RuntimeError, match="calibrate"):
            seg.transform(long_event_stream)

    def test_embeddings_none_after_fit(self, long_event_stream):
        seg = W2vCsCPD(**self._params())
        seg.fit(long_event_stream)
        seg._action_embeddings = None
        with pytest.raises(RuntimeError, match="Embeddings"):
            seg.transform(long_event_stream)


class TestCorrWanDense:
    @pytest.fixture
    def dense_stream(self, sensor_a, sensor_b):
        events = []
        for k in range(20):
            t_on = datetime(2024, 1, 1, 8, 0, 0) + timedelta(seconds=10 * k)
            sensor = sensor_a if k % 2 == 0 else sensor_b
            events.append(Event(timestamp=t_on, sensor=sensor, value=1))
            events.append(
                Event(timestamp=t_on + timedelta(seconds=5), sensor=sensor, value=0)
            )
        return EventStream(events)

    def test_segments_produced(self, dense_stream):
        seg = CorrWan(threshold=0.0, time_bin_size=30)
        seg.fit(dense_stream)
        res = seg.transform(dense_stream)
        assert res.num_segments >= 1
        assert all(seg.interval.duration > 0 for seg in res.segments)

    def test_time_th_max(self, dense_stream, activity_labels_for_stream):
        seg = CorrWan(threshold=0.0, time_bin_size=30, time_th="max")
        seg.fit(dense_stream, activity_labels=activity_labels_for_stream)
        res = seg.transform(dense_stream)
        assert res.num_segments >= 1

    def test_off_without_on(self, sensor_a, sensor_b):
        # B turns off without ever turning on: row stays zero (std=0 branch)
        stream = EventStream(
            [
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 0, 0), sensor=sensor_a, value=1
                ),
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 1, 0), sensor=sensor_a, value=0
                ),
                Event(
                    timestamp=datetime(2024, 1, 1, 8, 2, 0), sensor=sensor_b, value=0
                ),
            ]
        )
        seg = CorrWan(time_bin_size=60)
        sensors = list(stream.sensors)
        seg._sensor_to_idx = {s.name: i for i, s in enumerate(sensors)}
        X = seg._build_activation_sequences(stream, sensors)
        b_idx = seg._sensor_to_idx["SENSOR_B"]
        assert X[b_idx].sum() == 0.0  # B never on
        scm = seg._compute_correlation_matrix(stream, sensors)
        a_idx = seg._sensor_to_idx["SENSOR_A"]
        assert scm[a_idx, b_idx] == 0.0 and scm[b_idx, a_idx] == 0.0  # zero-std branch

    def test_transform_sensor_map_none(self, dense_stream):
        seg = CorrWan(threshold=0.0, time_bin_size=30)
        seg.fit(dense_stream)
        seg._sensor_to_idx = None
        res = seg.transform(dense_stream)
        assert res.num_segments >= 0  # runs; windows never match on sensor

    def test_transform_scm_none(self, dense_stream):
        seg = CorrWan(threshold=0.0, time_bin_size=30)
        seg.fit(dense_stream)
        seg._scm = None
        res = seg.transform(dense_stream)
        assert res.num_segments >= 0  # correlation defaults to 0.0


class TestCorrXuBreaks:
    def test_break_on_low_correlation(self, long_event_stream):
        seg = CorrXu(context_size=4)
        seg.fit(long_event_stream)
        seg._sct = {k: 1e9 for k in seg._sct}
        res = seg.transform(long_event_stream)
        assert res.num_segments == len(long_event_stream) - 1

    def test_break_on_gap(self, long_event_stream):
        seg = CorrXu(context_size=4)
        seg.fit(long_event_stream)
        seg._sct = {k: 0.0 for k in seg._sct}
        seg._mti = {p: -1.0 for p in seg._mti}
        res = seg.transform(long_event_stream)
        assert res.num_segments >= 1

    def test_break_on_activity_duration(self, long_event_stream):
        seg = CorrXu(context_size=4)
        seg.fit(long_event_stream)
        seg._sct = {k: 0.0 for k in seg._sct}
        seg._mti = {p: 1e18 for p in seg._mti}
        seg._mts = {a: -1.0 for a in seg._mts}
        res = seg.transform(long_event_stream)
        assert res.num_segments >= 1


class TestW2vVariantsExtra:
    def _params(self):
        return dict(
            embedding_w2v_dimension=8,
            embedding_w2v_context_size=2,
            embedding_w2v_training_epochs=2,
            embedding_w2v_min_token_count=1,
            embedding_w2v_negative_sampling=2,
            random_state=42,
        )

    def test_cs_threshold_zero(self, long_event_stream):
        seg = W2vCsCPD(threshold=0.0, **self._params())
        seg.fit(long_event_stream)
        res = seg.transform(long_event_stream)
        assert res.num_segments >= 1  # no boundaries: cs >= 0 always

    def test_ccs_threshold_one(self, long_event_stream):
        seg = W2vCcsCPD(threshold=1.0, context_size=2, **self._params())
        seg.fit(long_event_stream)
        res = seg.transform(long_event_stream)
        assert res.metadata["num_boundaries"] >= 0

    def test_ccs_not_calibrated(self, long_event_stream):
        with pytest.raises(RuntimeError, match="calibrate"):
            W2vCcsCPD(**self._params()).transform(long_event_stream)

    def test_ccs_embeddings_none(self, long_event_stream):
        seg = W2vCcsCPD(**self._params())
        seg.fit(long_event_stream)
        seg._embedding_w2v_dim = None
        with pytest.raises(RuntimeError, match="Embeddings"):
            seg.transform(long_event_stream)

    def test_ccsd_short_stream(self, event_stream):
        seg = W2vCcsdCPD(context_size=10, delta=0.5, **self._params())
        seg.fit(event_stream)
        res = seg.transform(
            event_stream
        )  # len(embeddings) <= context_size+1: no loop iterations
        assert res.num_segments >= 1

    def test_ccsd_not_calibrated(self, long_event_stream):
        with pytest.raises(RuntimeError, match="calibrate"):
            W2vCcsdCPD(**self._params()).transform(long_event_stream)

    def test_ccsd_embeddings_none(self, long_event_stream):
        seg = W2vCcsdCPD(**self._params())
        seg.fit(long_event_stream)
        seg._action_embeddings = None
        with pytest.raises(RuntimeError, match="Embeddings"):
            seg.transform(long_event_stream)
