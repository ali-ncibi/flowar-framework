from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
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

BASE_TIME = datetime(2024, 1, 1, 8, 0, 0)


@pytest.fixture
def sensor_a() -> Sensor:
    return Sensor(
        name="SENSOR_A",
        meta=SensorMeta(
            sensor_type=SensorType.MOTION,
            area=FunctionalArea.KITCHEN,
            sensor_object=SensorObject.COOKTOP,
        ),
    )


@pytest.fixture
def sensor_b() -> Sensor:
    return Sensor(
        name="SENSOR_B",
        meta=SensorMeta(
            sensor_type=SensorType.CONTACT,
            area=FunctionalArea.BATHROOM,
            sensor_object=SensorObject.DOOR,
        ),
    )


@pytest.fixture
def make_event():
    def _make(sensor: Sensor, minutes: float, value: int = 1) -> Event:
        return Event(
            timestamp=BASE_TIME + timedelta(minutes=minutes), sensor=sensor, value=value
        )

    return _make


@pytest.fixture
def event_stream(sensor_a, sensor_b, make_event) -> EventStream:
    events = [
        make_event(sensor_a, 0, 1),
        make_event(sensor_b, 5, 1),
        make_event(sensor_a, 10, 0),
        make_event(sensor_b, 15, 0),
        make_event(sensor_a, 20, 1),
    ]
    return EventStream(events)


@pytest.fixture
def sensors_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": pd.to_datetime(
                [
                    "2024-01-01 08:00:00",
                    "2024-01-01 08:05:00",
                    "2024-01-01 09:00:00",
                    "2024-01-02 08:00:00",
                    "2024-01-02 08:10:00",
                ]
            ),
            "end": pd.to_datetime(
                [
                    "2024-01-01 08:02:00",
                    "2024-01-01 08:07:00",
                    "2024-01-01 09:30:00",
                    "2024-01-02 08:05:00",
                    "2024-01-02 08:12:00",
                ]
            ),
            "name": ["SENSOR_A", "SENSOR_B", "SENSOR_A", "SENSOR_A", "SENSOR_B"],
        }
    )


@pytest.fixture
def activities_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": pd.to_datetime(
                [
                    "2024-01-01 08:00:00",
                    "2024-01-01 09:00:00",
                    "2024-01-02 08:00:00",
                ]
            ),
            "end": pd.to_datetime(
                [
                    "2024-01-01 08:10:00",
                    "2024-01-01 09:30:00",
                    "2024-01-02 08:15:00",
                ]
            ),
            "name": ["COOKING", "SHOWERING", "COOKING"],
        }
    )


@pytest.fixture
def metadata() -> dict:
    return {
        "sensors": {
            "sorted_names": ["SENSOR_A", "SENSOR_B"],
            "metadata": {
                "SENSOR_A": {
                    "type": "motion",
                    "area": "kitchen",
                    "object": "cooktop",
                },
                "SENSOR_B": {
                    "type": "contact",
                    "area": "bathroom",
                    "object": "door",
                },
            },
        },
        "activities": {"sorted_names": ["COOKING", "SHOWERING"]},
    }


@pytest.fixture
def dataset_dir(tmp_path: Path, sensors_df, activities_df, metadata) -> Path:
    """Write a minimal smart-home dataset (sensors.csv, activities.csv, meta.toml)."""
    d = tmp_path / "toy-home"
    d.mkdir()
    sensors_df.to_csv(d / "sensors.csv", index=False)
    activities_df.to_csv(d / "activities.csv", index=False)
    lines = [
        "[sensors]",
        'sorted_names = ["SENSOR_A", "SENSOR_B"]',
        "",
        "[activities]",
        'sorted_names = ["COOKING", "SHOWERING"]',
        "",
    ]
    for name, meta in metadata["sensors"]["metadata"].items():
        lines.append(f"[sensors.metadata.{name}]")
        lines.append(f'type = "{meta["type"]}"')
        lines.append(f'area = "{meta["area"]}"')
        lines.append(f'object = "{meta["object"]}"')
        lines.append("")
    (d / "meta.toml").write_text("\n".join(lines))
    return d


@pytest.fixture
def long_event_stream(sensor_a, sensor_b):
    """48 events over ~4 hours alternating two sensors on/off every 5 minutes."""
    events = []
    for k in range(24):
        t_on = BASE_TIME + timedelta(minutes=10 * k)
        t_off = t_on + timedelta(minutes=5)
        sensor = sensor_a if k % 2 == 0 else sensor_b
        events.append(Event(timestamp=t_on, sensor=sensor, value=1))
        events.append(Event(timestamp=t_off, sensor=sensor, value=0))
    return EventStream(events)


@pytest.fixture
def activity_labels_for_stream():
    from backend.models import FunctionalArea

    return [
        (BASE_TIME, BASE_TIME + timedelta(hours=1), FunctionalArea.KITCHEN),
        (
            BASE_TIME + timedelta(hours=1),
            BASE_TIME + timedelta(hours=2),
            FunctionalArea.BATHROOM,
        ),
    ]
