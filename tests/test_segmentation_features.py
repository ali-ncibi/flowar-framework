import numpy as np
import pytest

from backend.models import EventStream
from backend.segmentation.features.base import (
    ALL_FEATURES,
    DEFAULT_FEATURES,
    FEATURE_DURATION,
    FEATURE_EVENT_COUNTS,
    FEATURE_HOUR_OF_DAY,
    FEATURE_LAST_SENSOR_AREA,
    FEATURE_LAST_SENSOR_ID,
    FEATURE_PER_SENSOR_COUNTS,
    FEATURE_TIMESTAMPS,
    extract_window_features,
)


class TestExtractWindowFeatures:
    def test_default_features(self, long_event_stream):
        out = extract_window_features(long_event_stream, j=10, local_context_size=5)
        assert isinstance(out, np.ndarray)
        assert out.ndim == 1
        assert len(out) > 0
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_all_features_explicit(self, long_event_stream):
        out = extract_window_features(
            long_event_stream, j=20, local_context_size=4, features=ALL_FEATURES
        )
        assert len(out) > 0

    @pytest.mark.parametrize(
        "features",
        [
            [FEATURE_DURATION],
            [FEATURE_EVENT_COUNTS],
            [FEATURE_TIMESTAMPS],
            [FEATURE_HOUR_OF_DAY],
            [FEATURE_LAST_SENSOR_ID],
            [FEATURE_LAST_SENSOR_AREA],
            [FEATURE_PER_SENSOR_COUNTS],
            [FEATURE_DURATION, FEATURE_HOUR_OF_DAY],
        ],
    )
    def test_individual_features(self, long_event_stream, features):
        out = extract_window_features(
            long_event_stream, j=8, local_context_size=3, features=features
        )
        assert len(out) >= 1

    def test_empty_features_raises(self, long_event_stream):
        with pytest.raises(ValueError, match="cannot be empty"):
            extract_window_features(
                long_event_stream, j=5, local_context_size=3, features=[]
            )

    def test_invalid_feature_raises(self, long_event_stream):
        with pytest.raises(ValueError, match="Invalid feature names"):
            extract_window_features(
                long_event_stream, j=5, local_context_size=3, features=["nope"]
            )

    def test_window_beyond_stream(self, long_event_stream):
        out = extract_window_features(long_event_stream, j=1000, local_context_size=3)
        assert np.array_equal(out, np.zeros(1))

    def test_context_clipped_at_start(self, event_stream):
        out = extract_window_features(event_stream, j=1, local_context_size=100)
        assert len(out) > 0

    def test_default_features_constant(self):
        assert (
            DEFAULT_FEATURES == ALL_FEATURES.copy() or DEFAULT_FEATURES == ALL_FEATURES
        )

    def test_empty_stream(self):
        out = extract_window_features(EventStream([]), j=0, local_context_size=3)
        assert np.array_equal(out, np.zeros(1))


class TestAreaNone:
    def test_sensor_with_none_area(self):
        from backend.models import Event, Sensor, SensorMeta, SensorType

        s_none = Sensor(
            name="S_NONE",
            meta=SensorMeta(
                sensor_type=SensorType.MOTION, area=None, sensor_object=None
            ),
        )
        from datetime import datetime

        t = datetime(2024, 1, 1, 8, 0, 0)
        stream = EventStream(
            [
                Event(timestamp=t, sensor=s_none, value=1),
                Event(timestamp=t, sensor=s_none, value=0),
            ]
        )
        out = extract_window_features(
            stream,
            j=1,
            local_context_size=2,
            features=[
                FEATURE_LAST_SENSOR_AREA,
                FEATURE_LAST_SENSOR_ID,
                FEATURE_DURATION,
            ],
        )
        assert len(out) > 0
