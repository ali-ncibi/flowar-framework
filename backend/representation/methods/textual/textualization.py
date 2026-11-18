from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from enum import Enum

from num2words import num2words

from backend.models import Event, EventWindow, SensorMeta, SensorName


class TextualizerType(Enum):
    DTS = "dhekane_single"
    CTS = "civitare_single"
    TPB = "thukralb_single"
    TPT = "thukralt_single"
    CSS = "ncibi_single"
    TSB = "thukralb_sequence"
    TST = "thukralt_sequence"
    CSS_NO_AREA = "ncibi_single_no_area"
    CSS_NO_TYPE = "ncibi_single_no_type"
    CSS_NO_TIME = "ncibi_single_no_time"
    CSS_NO_DURATION = "ncibi_single_no_duration"
    CSS_NO_TOKENS = "ncibi_single_no_tokens"


def delta_to_words(delta_seconds):
    if delta_seconds < 60:
        return f"{num2words(delta_seconds)} seconds"
    elif delta_seconds < 3600:
        minutes = delta_seconds // 60
        seconds = delta_seconds % 60
        if seconds == 0:
            return f"{num2words(minutes)} minutes"
        else:
            return f"{num2words(minutes)} minutes and {num2words(seconds)} seconds"
    else:
        hours = delta_seconds // 3600
        minutes = (delta_seconds % 3600) // 60
        seconds = delta_seconds % 60
        parts = []
        if (
            hours > 0
        ):  # pragma: no cover  # always true: delta >= 3600 implies hours >= 1
            parts.append(f"{num2words(hours)} {'hour' if hours == 1 else 'hours'}")
        if minutes > 0:
            parts.append(
                f"{num2words(minutes)} {'minute' if minutes == 1 else 'minutes'}"
            )
        if seconds > 0:
            parts.append(
                f"{num2words(seconds)} {'second' if seconds == 1 else 'seconds'}"
            )
        return " and ".join(parts)


class WindowTextualizer(ABC):
    def describe(self, windows: Iterable[EventWindow]) -> list[list[str]]:
        return [self.describe_window(win) for win in windows]

    @abstractmethod
    def describe_window(self, window: EventWindow) -> list[str]: ...


class TDOSTSeqBasic(WindowTextualizer):
    def describe_window(self, window: EventWindow) -> list[str]:
        assert window.events is not None, "window has no events."
        outputs = []
        for event in window.events:
            meta = event.sensor.meta
            if meta is None:
                raise ValueError(f"Unknown sensor ID: {event.sensor.name}")
            sensor_type = meta.sensor_type.name.lower()
            sensor_area = meta.area.name.lower()
            sensor_object = meta.sensor_object.name.lower()
            sensor_value = "ON" if event.value else "OFF"
            outputs.append(
                f"A {sensor_type} sensor placed on the {sensor_object} in the {sensor_area} fired with value {sensor_value}."
            )
        if not outputs:
            outputs.append("No events in the window.")
        return outputs


class TDOSTSeqTemporal(WindowTextualizer):
    def describe_window(self, window: EventWindow) -> list[str]:
        assert window.events, "window has no events."
        outputs = []
        for i, event in enumerate(window.events):
            meta = event.sensor.meta
            if meta is None:
                raise ValueError(f"Unknown sensor ID: {event.sensor.name}")
            sensor_type = meta.sensor_type.name.lower()
            sensor_area = meta.area.name.lower()
            sensor_object = meta.sensor_object.name.lower()
            sensor_value = "ON" if event.value else "OFF"
            sensor_timestamp = event.timestamp
            if i == 0:
                info_first = self.time_to_words(sensor_timestamp.strftime("%H:%M:%S"))
                s = f"A {sensor_type} sensor placed on the {sensor_object} in the {sensor_area} fired with value {sensor_value} {info_first}."
            else:
                prev_timestamp = window.events[i - 1].timestamp
                delta_seconds = int((sensor_timestamp - prev_timestamp).total_seconds())
                info_next = f"After {delta_to_words(delta_seconds)}"
                s = f"{info_next} a {sensor_type} sensor placed on the {sensor_object} in the {sensor_area} fired with value {sensor_value}."
            outputs.append(s)
        return outputs

    def time_to_words(self, time_str):
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
        hour = time_obj.hour
        minute = time_obj.minute
        second = time_obj.second
        if hour == 0:
            hour_12 = 12
            period = "AM"
        elif hour < 12:
            hour_12 = hour
            period = "AM"
        elif hour == 12:
            hour_12 = 12
            period = "PM"
        else:
            hour_12 = hour - 12
            period = "PM"
        parts = [f"at {num2words(hour_12)} {'hour' if hour_12 == 1 else 'hours'}"]
        if minute > 0:
            parts.append(
                f"{num2words(minute)} {'minute' if minute == 1 else 'minutes'}"
            )
        if second > 0:
            parts.append(
                f"{num2words(second)} {'second' if second == 1 else 'seconds'}"
            )
        if len(parts) == 1:
            result = parts[0]
        elif len(parts) == 2:
            result = f"{parts[0]} and {parts[1]}"
        else:
            result = f"{parts[0]} {parts[1]} and {parts[2]}"
        return f"{result} {period}"


class TDOSTPlainBasic(WindowTextualizer):
    def describe_window(self, window: EventWindow) -> list[str]:
        seqel_outputs = TDOSTSeqBasic().describe_window(window)
        concatenated = " ".join(seqel_outputs)
        outputs = [concatenated]
        return outputs


class TDOSTPlainTemporal(WindowTextualizer):
    def describe_window(self, window: EventWindow) -> list[str]:
        seqel_outputs = TDOSTSeqTemporal().describe_window(window)
        concatenated = " ".join(seqel_outputs)
        outputs = [concatenated]
        return outputs


class Civitarese(WindowTextualizer):
    def build_states(self, events: list[Event]) -> dict[str, dict]:
        """Build ON/OFF state intervals for each sensor.

        Returns a dict mapping sensor names to dicts containing:
        - 'states': List of ON intervals with start/end timestamps
        - 'sensor': The actual Sensor object (for accessing metadata)
        """
        events_by_sensor: dict[SensorName, list[Event]] = {}
        sensor_objects = {}  # Track sensor objects for metadata access
        for event in events:
            if event.sensor.name not in events_by_sensor:
                events_by_sensor[event.sensor.name] = []
                sensor_objects[event.sensor.name] = event.sensor
            events_by_sensor[event.sensor.name].append(event)

        states_by_sensor = {}
        for sensor_id, events in events_by_sensor.items():
            states = []
            current_state = None
            current_start = None
            for event in events:
                if event.value:
                    if current_state is None:
                        current_state = "ON"
                        current_start = event.timestamp
                else:
                    if current_state == "ON":
                        states.append(
                            {
                                "state": "ON",
                                "start": current_start,
                                "end": event.timestamp,
                            }
                        )
                        current_state = None
                        current_start = None
            if current_state == "ON" and current_start is not None:
                states.append(
                    {
                        "state": "ON",
                        "start": current_start,
                        "end": events[-1].timestamp,
                    }
                )
            states_by_sensor[sensor_id] = {
                "states": states,
                "sensor": sensor_objects[sensor_id],
            }
        return states_by_sensor

    def build_temporal(self, window: EventWindow) -> str:
        assert window.events, "window has no events."
        start_time = window.events[0].timestamp
        end_time = window.events[-1].timestamp
        duration = int((end_time - start_time).total_seconds())
        time_str = start_time.strftime("%-I:%M %p").lower()
        output = f"During a {duration}-second time window (around {time_str}), the system observed the following. "
        return output

    def describe_action(self, meta: SensorMeta) -> str:
        sensor_type = meta.sensor_type.name.lower()
        sensor_object = meta.sensor_object.name.lower()
        if sensor_type == "motion":
            return f"get around the {sensor_object}"
        elif sensor_type == "contact":
            return f"open the {sensor_object}"
        elif sensor_type == "flush":
            return f"flushe the {sensor_object}"
        elif sensor_type == "pressure":
            return f"stay on the {sensor_object}"
        elif sensor_type == "electric":
            return f"turn on the {sensor_object}"
        else:
            return f"interact with the {sensor_type} sensor"

    def describe_window(self, window: EventWindow) -> list[str]:
        if not window.events:
            return ["No activity detected"]
        output = self.build_temporal(window)
        seg_by_area = self.group_by_area(window)
        for j, area_seg in enumerate(seg_by_area):
            if j == 0 and len(seg_by_area) > 1:
                outer_link = "First, the subject is in the"
            elif j == 0 and len(seg_by_area) == 1:
                outer_link = "The subject is in the"
            else:
                outer_link = "Then, they go to the"
            area_action_desc = f"{outer_link} {area_seg['area']}."
            area_actions = [
                (action, sensor_data["sensor"])
                for sensor, sensor_data in self.build_states(area_seg["events"]).items()
                for action in sensor_data["states"]
            ]
            area_actions = sorted(area_actions, key=lambda x: x[0]["start"])
            for i, f in enumerate(area_actions):
                action = f[0]
                sensor = f[1]
                if i == 0 and len(area_actions) > 1:
                    innder_link = "Here, they"
                elif i == 0 and len(area_actions) == 1:
                    innder_link = "They"
                else:
                    innder_link = "Then, they"
                action_duration = int((action["end"] - action["start"]).total_seconds())
                action_description = self.describe_action(sensor.meta)
                area_action_desc += (
                    " "
                    + innder_link
                    + " "
                    + f" {action_description} for {action_duration} seconds. "
                )
            output += area_action_desc + " "
        return [output.strip()]

    def group_by_area(self, window: EventWindow) -> list[dict]:
        """Group events by physical area/location in the smart home.

        Returns a list of segments, each containing:
        - 'area': The room/location name
        - 'duration': How long the person stayed in that area (seconds)
        - 'events': All events that occurred in that area
        """
        assert window.events, "window has no events."
        segments = []
        current_area = None
        current_start_time = None
        current_events: list[Event] = []

        for event in window.events:
            meta = event.sensor.meta
            if not meta:
                continue
            area = meta.area.name.lower()

            # If we're entering a new area, save the previous segment
            if area != current_area:
                if current_area is not None:
                    assert current_start_time is not None
                    duration = int(
                        (event.timestamp - current_start_time).total_seconds()
                    )
                    segments.append(
                        {
                            "area": current_area,
                            "duration": duration,
                            "events": current_events,
                        }
                    )
                current_area = area
                current_start_time = event.timestamp
                current_events = []

            current_events.append(event)

        # Don't forget the last segment
        if current_area is not None:
            assert current_start_time is not None
            duration = int(
                (window.events[-1].timestamp - current_start_time).total_seconds()
            )
            segments.append(
                {
                    "area": current_area,
                    "duration": duration,
                    "events": current_events,
                }
            )

        return segments


class Dhekane(WindowTextualizer):
    def __init__(self, attributes: list[str] | None = None):
        super().__init__()
        if attributes is None:
            attributes = [
                "time_of_occurrence",
                "duration",
                "top_k_areas",
                "top_k_sensors",
            ]
        valid_attributes = {
            "time_of_occurrence",
            "duration",
            "top_k_areas",
            "top_k_sensors",
        }
        for attr in attributes:
            if attr not in valid_attributes:
                raise ValueError(
                    f"Invalid attribute '{attr}'. Choose from: {valid_attributes}"
                )
        self.attributes = attributes

    def describe_window(self, window: EventWindow) -> list[str]:
        descriptions = []
        for attr in self.attributes:
            method = getattr(self, attr)
            descriptions.append(method(window))
        outputs = [". ".join(descriptions) + "."]
        return outputs

    def time_of_occurrence(self, window: EventWindow) -> str:
        def period_of_day(hour: int) -> str:
            if 5 <= hour < 12:
                return "in the morning"
            elif 12 <= hour < 17:
                return "in the afternoon"
            elif 17 <= hour < 21:
                return "in the evening"
            else:
                return "at night"

        assert window.events, "window has no events."
        start_time = window.events[0].timestamp
        end_time = window.events[-1].timestamp
        start_hour_part = start_time.strftime("%I:%M %p").lstrip("0")
        end_hour_part = end_time.strftime("%I:%M %p").lstrip("0")
        start_period_part = period_of_day(start_time.hour)
        end_period_part = period_of_day(end_time.hour)
        return f"The activity started at {start_hour_part} {start_period_part} and ended at {end_hour_part} {end_period_part}"

    def duration(self, window: EventWindow) -> str:
        assert window.events, "window has no events."
        duration_seconds = (
            window.events[-1].timestamp - window.events[0].timestamp
        ).total_seconds()
        if duration_seconds < 60:
            return f"The activity was performed for {int(duration_seconds)} seconds"
        elif duration_seconds < 3600:
            minutes = duration_seconds // 60
            return f"The activity was performed for {int(minutes)} minutes"
        else:
            hours = duration_seconds // 3600
            return f"The activity was performed for {int(hours)} hours"

    def top_k_areas(self, window: EventWindow, k: int = 3) -> str:
        assert window.events, "window has no events."
        area_counter: Counter[str] = Counter()
        for event in window.events:
            meta = event.sensor.meta
            if meta:
                area_counter[meta.area.name.lower()] += 1
        most_common = area_counter.most_common(k)
        if not most_common:
            return "No areas detected"
        areas = [area for area, _ in most_common]
        if len(areas) == 1:
            return f"The activity is taking place mainly in the {areas[0]}"
        else:
            return f"The activity is taking place mainly in the {areas[0]} and parts of it in {', '.join(areas[1:])}"

    def top_k_sensors(self, window: EventWindow, k: int = 3) -> str:
        assert window.events, "window has no events."
        sensor_counter: Counter[tuple[str, str, str]] = Counter()
        for event in window.events:
            meta = event.sensor.meta
            if meta:
                sensor_counter[
                    (
                        meta.sensor_type.name.lower(),
                        meta.area.name.lower(),
                        meta.sensor_object.name.lower(),
                    )
                ] += 1
        most_common = sensor_counter.most_common(k)
        if not most_common:
            return "No sensors detected"
        sensors = [
            f"the {stype} sensor placed on the {sobj} in the {sarea}"
            for (stype, sarea, sobj), _ in most_common
        ]
        if len(sensors) == 1:
            return f"The most commonly fired sensor in this activity is {sensors[0]}"
        else:
            return f"The two most commonly fired sensors in this activity are {sensors[0]} and {sensors[1]}"


class CompoundSensorSummary(WindowTextualizer):
    def __init__(self, attributes: list[str] | None = None):
        super().__init__()
        if attributes is None:
            attributes = [
                "unique_types",
                "unique_areas",
                "token_sequence",
                "duration",
                "time_of_occurrence",
            ]
        valid_attributes = {
            "unique_types",
            "unique_areas",
            "token_sequence",
            "duration",
            "time_of_occurrence",
        }
        for attr in attributes:
            if attr not in valid_attributes:
                raise ValueError(
                    f"Invalid attribute '{attr}'. Choose from: {valid_attributes}"
                )
        self.attributes = attributes

    def setup(self, meta: Mapping[SensorName, SensorMeta]) -> "CompoundSensorSummary":
        self.meta = meta
        return self

    def describe_window(self, window: EventWindow) -> list[str]:
        descriptions = []
        for attr in self.attributes:
            method = getattr(self, attr)
            descriptions.append(method(window))
        outputs = [" :: ".join(descriptions) + "."]
        return outputs

    def time_of_occurrence(self, window: EventWindow) -> str:
        def period_of_day(hour: int) -> str:
            if 5 <= hour < 12:
                return "in the morning"
            elif 12 <= hour < 17:
                return "in the afternoon"
            elif 17 <= hour < 21:
                return "in the evening"
            else:
                return "at night"

        assert window.events, "window has no events."
        start_time = window.events[0].timestamp
        end_time = window.events[-1].timestamp
        start_hour_part = start_time.strftime("%I:%M %p").lstrip("0")
        end_hour_part = end_time.strftime("%I:%M %p").lstrip("0")
        start_period_part = period_of_day(start_time.hour)
        end_period_part = period_of_day(end_time.hour)
        return f"OCCURRENCE: from {start_hour_part} {start_period_part} to {end_hour_part} {end_period_part}"

    def duration(self, window: EventWindow) -> str:
        assert window.events, "window has no events."
        duration_seconds = window.events[-1].timestamp - window.events[0].timestamp
        return f"DURATION: {delta_to_words(int(duration_seconds.total_seconds()))}"

    def token_sequence(self, window: EventWindow) -> str:
        assert self.meta is not None, "not set up with metadata."
        assert window.events, "window has no events."
        tokens = []
        for event in window.events:
            meta = self.meta.get(event.sensor.name)
            if meta:
                sensor_object = meta.sensor_object.name.lower()
                sensor_area = meta.area.name.lower()
                sensor_type = meta.sensor_type.name.lower()
                sensor_value = "ON" if event.value else "OFF"
                token = f"{sensor_object}@{sensor_area}({sensor_type})={sensor_value}"
                tokens.append(token)
        if not tokens:
            return "EVENTS: None"
        else:
            return f"EVENTS: {', '.join(tokens)}"

    def unique_types(self, window: EventWindow) -> str:
        assert self.meta is not None, "not set up with metadata."
        assert window.events, "window has no events."
        types = set()
        for event in window.events:
            meta = self.meta.get(event.sensor.name)
            if meta:
                types.add(meta.sensor_type.name.lower())
        if not types:
            return "TYPES: None"
        else:
            return f"TYPES: {'|'.join(sorted(types))}"

    def unique_areas(self, window: EventWindow) -> str:
        assert self.meta is not None, "not set up with metadata."
        assert window.events, "window has no events."
        areas = set()
        for event in window.events:
            meta = self.meta.get(event.sensor.name)
            if meta:
                areas.add(meta.area.name.lower())
        if not areas:
            return "AREAS: None"
        else:
            return f"AREAS: {'|'.join(sorted(areas))}"


class CompoundSensorSummaryNoArea(CompoundSensorSummary):
    def __init__(self):
        super().__init__(
            attributes=[
                "unique_types",
                "token_sequence",
                "duration",
                "time_of_occurrence",
            ]
        )


class CompoundSensorSummaryNoType(CompoundSensorSummary):
    def __init__(self):
        super().__init__(
            attributes=[
                "unique_areas",
                "token_sequence",
                "duration",
                "time_of_occurrence",
            ]
        )


class CompoundSensorSummaryNoTime(CompoundSensorSummary):
    def __init__(self):
        super().__init__(
            attributes=[
                "unique_types",
                "unique_areas",
                "token_sequence",
                "duration",
            ]
        )


class CompoundSensorSummaryNoDuration(CompoundSensorSummary):
    def __init__(self):
        super().__init__(
            attributes=[
                "unique_types",
                "unique_areas",
                "token_sequence",
                "time_of_occurrence",
            ]
        )


class CompoundSensorSummaryNoTokens(CompoundSensorSummary):
    def __init__(self):
        super().__init__(
            attributes=[
                "unique_types",
                "unique_areas",
                "duration",
                "time_of_occurrence",
            ]
        )


class WindowTextualizerFactory:
    _registry: dict[str, Callable[..., WindowTextualizer]] = {
        "tdost_seq_basic": TDOSTSeqBasic,
        "tdost_seq_temporal": TDOSTSeqTemporal,
        "tdost_plain_basic": TDOSTPlainBasic,
        "tdost_plain_temporal": TDOSTPlainTemporal,
        "civitare": Civitarese,
        "dhekane": Dhekane,
        "ncibi_single": CompoundSensorSummary,
        "ncibi_single_no_area": CompoundSensorSummaryNoArea,
        "ncibi_single_no_type": CompoundSensorSummaryNoType,
        "ncibi_single_no_time": CompoundSensorSummaryNoTime,
        "ncibi_single_no_duration": CompoundSensorSummaryNoDuration,
        "ncibi_single_no_tokens": CompoundSensorSummaryNoTokens,
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> WindowTextualizer:
        key = name.strip().lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown  '{name}'. Choose from: {', '.join(cls._registry)}"
            )
        return cls._registry[key](**kwargs)
