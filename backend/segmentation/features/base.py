import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

from ...models import EventStream, Interval

FEATURE_DURATION = "duration"
FEATURE_EVENT_COUNTS = "event_counts"
FEATURE_TIMESTAMPS = "timestamps"
FEATURE_HOUR_OF_DAY = "hour_of_day"
FEATURE_LAST_SENSOR_ID = "last_sensor_id"
FEATURE_LAST_SENSOR_AREA = "last_sensor_area"
FEATURE_PER_SENSOR_COUNTS = "per_sensor_counts"

ALL_FEATURES = [
    FEATURE_DURATION,
    FEATURE_EVENT_COUNTS,
    FEATURE_TIMESTAMPS,
    FEATURE_HOUR_OF_DAY,
    FEATURE_LAST_SENSOR_ID,
    FEATURE_LAST_SENSOR_AREA,
    FEATURE_PER_SENSOR_COUNTS,
]

DEFAULT_FEATURES = ALL_FEATURES.copy()


def extract_window_features(
    events: EventStream,
    j: int,
    local_context_size: int,
    features: list[str] | None = None,
) -> NDArray[np.float64]:
    if features is None:
        features = DEFAULT_FEATURES
    if not features:
        raise ValueError("features list cannot be empty")
    invalid_features = [f for f in features if f not in ALL_FEATURES]
    if invalid_features:
        raise ValueError(
            f"Invalid feature names: {invalid_features}. Valid features: {ALL_FEATURES}"
        )
    start_idx = max(0, j - local_context_size + 1)
    window_events = events.events[start_idx : j + 1]
    all_sensors = sorted(events.sensors, key=lambda s: s.name)
    sensor_id_to_idx = {s.name: idx for idx, s in enumerate(all_sensors)}
    unique_areas: set[tuple[str, ...]] = set()
    for sensor in all_sensors:
        if sensor.meta.area is not None:
            unique_areas.add(tuple(sensor.meta.area.name))
    sorted_areas = sorted(unique_areas)
    area_to_idx = {loc: idx for idx, loc in enumerate(sorted_areas)}
    M = len(all_sensors)
    D = len(sorted_areas)
    if not window_events:
        return np.zeros(1, dtype=np.float64)
    feature_parts = []
    if FEATURE_DURATION in features:
        # Use Interval model for duration calculation
        window_interval = Interval(
            window_events[0].timestamp, window_events[-1].timestamp
        )
        duration = window_interval.duration
        feature_parts.append(np.array([duration], dtype=np.float64))
    if FEATURE_EVENT_COUNTS in features:
        num_on = sum(1 for e in window_events if e.value == 1)
        num_off = sum(1 for e in window_events if e.value == 0)
        feature_parts.append(np.array([num_on, num_off], dtype=np.float64))
    if FEATURE_TIMESTAMPS in features:
        t_start = window_events[0].timestamp.timestamp()
        t_end = window_events[-1].timestamp.timestamp()
        feature_parts.append(np.array([t_start, t_end], dtype=np.float64))
    if FEATURE_HOUR_OF_DAY in features:
        hour_of_day = float(window_events[0].timestamp.hour)
        feature_parts.append(np.array([hour_of_day], dtype=np.float64))
    if FEATURE_LAST_SENSOR_ID in features:
        last_sensor_id = window_events[-1].sensor.name
        last_sensor_onehot = np.zeros(M, dtype=np.float64)
        if (
            last_sensor_id in sensor_id_to_idx
        ):  # pragma: no cover  # always true: ids built from the same stream
            last_sensor_onehot[sensor_id_to_idx[last_sensor_id]] = 1.0
        feature_parts.append(last_sensor_onehot)
    if FEATURE_LAST_SENSOR_AREA in features:
        last_sensor_area_onehot = np.zeros(D, dtype=np.float64)
        if window_events[-1].sensor.meta.area is not None:
            last_loc = tuple(window_events[-1].sensor.meta.area.name)
            if (
                last_loc in area_to_idx
            ):  # pragma: no cover  # always true: areas built from the same stream
                last_sensor_area_onehot[area_to_idx[last_loc]] = 1.0
        feature_parts.append(last_sensor_area_onehot)
    if FEATURE_PER_SENSOR_COUNTS in features:
        per_sensor_features = []
        for sensor in all_sensors:
            sensor_on = sum(
                1
                for e in window_events
                if e.sensor.name == sensor.name and e.value == 1
            )
            sensor_off = sum(
                1
                for e in window_events
                if e.sensor.name == sensor.name and e.value == 0
            )
            per_sensor_features.extend([sensor_on, sensor_off])
        feature_parts.append(np.array(per_sensor_features, dtype=np.float64))
    if (
        not feature_parts
    ):  # pragma: no cover  # unreachable: features validated non-empty above
        output = np.array([0.0], dtype=np.float64)
    else:
        output = np.concatenate(feature_parts)
    if len(output) == 0:  # pragma: no cover  # unreachable: feature_parts is non-empty
        return np.array([0.0], dtype=np.float64)
    scaler = MinMaxScaler()
    output = scaler.fit_transform(output.reshape(-1, 1)).flatten()
    return output
