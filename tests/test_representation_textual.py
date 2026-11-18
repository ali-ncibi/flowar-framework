from datetime import datetime, timedelta

import pytest

from backend.models import (
    Event,
    EventWindow,
    FunctionalArea,
    Sensor,
    SensorMeta,
    SensorObject,
    SensorType,
)
from backend.representation.methods.textual.textualization import (
    Civitarese,
    CompoundSensorSummary,
    CompoundSensorSummaryNoArea,
    CompoundSensorSummaryNoDuration,
    CompoundSensorSummaryNoTime,
    CompoundSensorSummaryNoTokens,
    CompoundSensorSummaryNoType,
    Dhekane,
    TDOSTPlainBasic,
    TDOSTPlainTemporal,
    TDOSTSeqBasic,
    TDOSTSeqTemporal,
    WindowTextualizer,
    WindowTextualizerFactory,
    delta_to_words,
)

T0 = datetime(2024, 1, 1, 8, 0, 0)


def _sensor(
    name, area=FunctionalArea.KITCHEN, stype=SensorType.MOTION, obj=SensorObject.COOKTOP
):
    return Sensor(
        name=name, meta=SensorMeta(sensor_type=stype, area=area, sensor_object=obj)
    )


def _window(events, j_last=None):
    return EventWindow(
        j_first=0,
        j_last=j_last if j_last is not None else len(events) - 1,
        events=events,
    )


def _ev(sensor, minutes, value=1):
    return Event(timestamp=T0 + timedelta(minutes=minutes), sensor=sensor, value=value)


A = _sensor("A")
B = _sensor("B", FunctionalArea.BATHROOM, SensorType.CONTACT, SensorObject.DOOR)
C = _sensor("C", FunctionalArea.BEDROOM, SensorType.PRESSURE, SensorObject.BED)


class TestDeltaToWords:
    def test_seconds(self):
        assert delta_to_words(30) == "thirty seconds"
        assert delta_to_words(1) == "one seconds"

    def test_minutes(self):
        assert delta_to_words(60) == "one minutes"
        assert delta_to_words(90) == "one minutes and thirty seconds"
        assert delta_to_words(120) == "two minutes"

    def test_hours(self):
        assert delta_to_words(3600) == "one hour"
        assert delta_to_words(7200) == "two hours"
        assert delta_to_words(3660) == "one hour and one minute"
        assert delta_to_words(3661) == "one hour and one minute and one second"
        assert delta_to_words(3601) == "one hour and one second"


class TestWindowTextualizerBase:
    def test_abstract(self):
        with pytest.raises(TypeError):
            WindowTextualizer()

    def test_describe_and_super(self):
        class Dummy(WindowTextualizer):
            def describe_window(self, window):
                return super().describe_window(window)

        d = Dummy()
        w = _window([_ev(A, 0)])
        assert d.describe([w]) == [None]  # super() returns None (covers ellipsis body)


class TestTDOSTSeq:
    def test_basic(self):
        w = _window([_ev(A, 0, 1), _ev(A, 5, 0)])
        out = TDOSTSeqBasic().describe_window(w)
        assert len(out) == 2
        assert "motion" in out[0] and "ON" in out[0]
        assert "OFF" in out[1]

    def test_meta_none_raises(self):
        s = Sensor(name="X", meta=None)
        w = _window([Event(timestamp=T0, sensor=s, value=1)])
        with pytest.raises(ValueError, match="Unknown sensor ID"):
            TDOSTSeqBasic().describe_window(w)

    def test_empty_window(self):
        w = EventWindow(j_first=0, j_last=0, events=[])
        out = TDOSTSeqBasic().describe_window(w)
        assert out == ["No events in the window."]

    def test_temporal(self):
        w = _window([_ev(A, 0, 1), _ev(B, 5, 0), _ev(A, 10, 1)])
        out = TDOSTSeqTemporal().describe_window(w)
        assert len(out) == 3
        assert "fired with value ON at" in out[0]
        assert "After five minutes" in out[1]
        assert "After" in out[2]

    def test_time_to_words_branches(self):
        tz = TDOSTSeqTemporal()
        assert "twelve" in tz.time_to_words("00:00:00") and "AM" in tz.time_to_words(
            "00:00:00"
        )
        assert "AM" in tz.time_to_words("05:00:00")
        assert "twelve" in tz.time_to_words("12:00:00") and "PM" in tz.time_to_words(
            "12:00:00"
        )
        assert "PM" in tz.time_to_words("15:30:05")
        r = tz.time_to_words("08:00:00")
        assert r == "at eight hours AM"
        r = tz.time_to_words("08:05:00")
        assert "and" in r
        r = tz.time_to_words("08:05:07")
        assert r.count("and") == 1 and "seven seconds" in r

    def test_plain_wrappers(self):
        w = _window([_ev(A, 0, 1), _ev(B, 5, 0)])
        assert len(TDOSTPlainBasic().describe_window(w)) == 1
        assert len(TDOSTPlainTemporal().describe_window(w)) == 1
        assert "After" in TDOSTPlainTemporal().describe_window(w)[0]


class TestCivitarese:
    def test_empty_window(self):
        w = EventWindow(j_first=0, j_last=0, events=[])
        assert Civitarese().describe_window(w) == ["No activity detected"]

    def test_build_states(self):
        events = [_ev(A, 0, 1), _ev(A, 5, 0), _ev(A, 10, 1)]  # last ON unterminated
        states = Civitarese().build_states(events)
        assert states["A"]["states"] == [
            {"state": "ON", "start": T0, "end": T0 + timedelta(minutes=5)},
            {
                "state": "ON",
                "start": T0 + timedelta(minutes=10),
                "end": T0 + timedelta(minutes=10),
            },
        ]
        # off without on -> no state
        states2 = Civitarese().build_states([_ev(A, 0, 0)])
        assert states2["A"]["states"] == []

    def test_describe_action(self):
        c = Civitarese()
        assert "get around" in c.describe_action(A.meta)
        assert "open the" in c.describe_action(B.meta)
        assert "flushe" in c.describe_action(
            _sensor("F", stype=SensorType.FLUSH, obj=SensorObject.TOILET).meta
        )
        assert "stay on" in c.describe_action(C.meta)
        assert "turn on" in c.describe_action(
            _sensor("E", stype=SensorType.ELECTRIC, obj=SensorObject.MICROWAVE).meta
        )
        assert "interact with" in c.describe_action(
            _sensor("S", stype=SensorType.SONAR, obj=SensorObject.WINDOW).meta
        )

    def test_describe_window_single_area(self):
        w = _window([_ev(A, 0, 1), _ev(A, 5, 0)])
        out = Civitarese().describe_window(w)
        assert len(out) == 1
        assert "The subject is in the" in out[0]
        assert "get around the cooktop" in out[0]

    def test_describe_window_multiple_areas(self):
        w = _window([_ev(A, 0, 1), _ev(A, 5, 0), _ev(B, 10, 1), _ev(B, 12, 0)])
        out = Civitarese().describe_window(w)
        assert "First, the subject is in the" in out[0]
        assert "Then, they go to the" in out[0]

    def test_group_by_area_meta_none(self):
        s = Sensor(name="X", meta=None)
        w = _window([Event(timestamp=T0, sensor=s, value=1), _ev(A, 1)])
        segs = Civitarese().group_by_area(w)
        assert len(segs) == 1  # meta-None event skipped
        assert segs[0]["area"] == "kitchen"


class TestDhekane:
    def test_invalid_attribute(self):
        with pytest.raises(ValueError, match="Invalid attribute"):
            Dhekane(attributes=["bogus"])

    def test_describe_window(self):
        w = _window([_ev(A, 0, 1), _ev(A, 30, 0)])
        out = Dhekane().describe_window(w)
        assert len(out) == 1
        assert "morning" in out[0]  # 08:00
        assert "minutes" in out[0]

    def test_period_of_day(self):
        d = Dhekane()
        w_morning = _window(
            [Event(timestamp=datetime(2024, 1, 1, 6, 0), sensor=A, value=1)]
        )
        w_afternoon = _window(
            [Event(timestamp=datetime(2024, 1, 1, 13, 0), sensor=A, value=1)]
        )
        w_evening = _window(
            [Event(timestamp=datetime(2024, 1, 1, 18, 0), sensor=A, value=1)]
        )
        w_night = _window(
            [Event(timestamp=datetime(2024, 1, 1, 23, 0), sensor=A, value=1)]
        )
        assert "morning" in d.time_of_occurrence(w_morning)
        assert "afternoon" in d.time_of_occurrence(w_afternoon)
        assert "evening" in d.time_of_occurrence(w_evening)
        assert "night" in d.time_of_occurrence(w_night)

    def test_duration_branches(self):
        d = Dhekane()
        w_sec = _window(
            [
                Event(timestamp=T0, sensor=A, value=1),
                Event(timestamp=T0 + timedelta(seconds=30), sensor=A, value=0),
            ]
        )
        w_min = _window(
            [
                Event(timestamp=T0, sensor=A, value=1),
                Event(timestamp=T0 + timedelta(minutes=30), sensor=A, value=0),
            ]
        )
        w_hour = _window(
            [
                Event(timestamp=T0, sensor=A, value=1),
                Event(timestamp=T0 + timedelta(hours=2), sensor=A, value=0),
            ]
        )
        assert "seconds" in d.duration(w_sec)
        assert "minutes" in d.duration(w_min)
        assert "hours" in d.duration(w_hour)

    def test_top_k_areas(self):
        d = Dhekane()
        w = _window([_ev(A, 0), _ev(A, 1), _ev(B, 2)])
        out = d.top_k_areas(w)
        assert "kitchen" in out and "bathroom" in out
        w1 = _window([_ev(A, 0)])
        assert "mainly in the kitchen" in d.top_k_areas(w1)
        w_none = _window(
            [Event(timestamp=T0, sensor=Sensor(name="X", meta=None), value=1)]
        )
        assert d.top_k_areas(w_none) == "No areas detected"

    def test_top_k_sensors(self):
        d = Dhekane()
        w2 = _window([_ev(A, 0), _ev(A, 1), _ev(B, 2)])
        out = d.top_k_sensors(w2)
        assert "two most commonly" in out or "and" in out
        w1 = _window([_ev(A, 0)])
        assert "most commonly fired sensor" in d.top_k_sensors(w1)
        w_none = _window(
            [Event(timestamp=T0, sensor=Sensor(name="X", meta=None), value=1)]
        )
        assert d.top_k_sensors(w_none) == "No sensors detected"


class TestCompoundSensorSummary:
    def _meta_map(self):
        return {"A": A.meta, "B": B.meta}

    def test_invalid_attribute(self):
        with pytest.raises(ValueError, match="Invalid attribute"):
            CompoundSensorSummary(attributes=["nope"])

    def test_describe_window(self):
        c = CompoundSensorSummary().setup(self._meta_map())
        w = _window([_ev(A, 0, 1), _ev(B, 5, 0)])
        out = c.describe_window(w)
        assert len(out) == 1
        assert "TYPES:" in out[0] and "EVENTS:" in out[0] and "DURATION:" in out[0]

    def test_token_sequence_unknown_sensor(self):
        c = CompoundSensorSummary(attributes=["token_sequence"]).setup(self._meta_map())
        w = _window(
            [
                _ev(A, 0),
                Event(
                    timestamp=T0 + timedelta(minutes=1), sensor=_sensor("ZZZ"), value=1
                ),
            ]
        )
        out = c.describe_window(w)
        assert "cooktop@kitchen" in out[0]
        assert "zzz" not in out[0]

    def test_none_branches(self):
        c = CompoundSensorSummary(
            attributes=["unique_types", "unique_areas", "token_sequence"]
        ).setup({})
        w = _window([_ev(A, 0)])
        out = c.describe_window(w)
        assert "TYPES: None" in out[0]
        assert "AREAS: None" in out[0]
        assert "EVENTS: None" in out[0]

    def test_no_meta_assert(self):
        c = CompoundSensorSummary(attributes=["unique_types"])
        w = _window([_ev(A, 0)])
        with pytest.raises((AssertionError, AttributeError)):
            c.describe_window(w)

    def test_variants(self):
        for cls in (
            CompoundSensorSummaryNoArea,
            CompoundSensorSummaryNoType,
            CompoundSensorSummaryNoTime,
            CompoundSensorSummaryNoDuration,
            CompoundSensorSummaryNoTokens,
        ):
            c = cls().setup(self._meta_map())
            w = _window([_ev(A, 0, 1), _ev(A, 5, 0)])
            out = c.describe_window(w)
            assert len(out) == 1


class TestFactory:
    def test_create_all(self):
        for name in [
            "tdost_seq_basic",
            "tdost_seq_temporal",
            "tdost_plain_basic",
            "tdost_plain_temporal",
            "civitare",
            "dhekane",
            "ncibi_single",
            "ncibi_single_no_area",
            "ncibi_single_no_type",
            "ncibi_single_no_time",
            "ncibi_single_no_duration",
            "ncibi_single_no_tokens",
        ]:
            assert WindowTextualizerFactory.create(name) is not None

    def test_create_case_insensitive(self):
        assert WindowTextualizerFactory.create("  Dhekane ") is not None

    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown"):
            WindowTextualizerFactory.create("nope")


class TestExtraBranches:
    def test_temporal_meta_none_raises(self):
        s = Sensor(name="X", meta=None)
        w = _window([Event(timestamp=T0, sensor=s, value=1)])
        with pytest.raises(ValueError, match="Unknown sensor ID"):
            TDOSTSeqTemporal().describe_window(w)

    def test_build_states_double_on(self):
        events = [_ev(A, 0, 1), _ev(A, 1, 1), _ev(A, 5, 0)]
        states = Civitarese().build_states(events)
        assert states["A"]["states"] == [
            {"state": "ON", "start": T0, "end": T0 + timedelta(minutes=5)}
        ]

    def test_describe_window_multi_action_area(self):
        w = _window([_ev(A, 0, 1), _ev(A, 5, 0), _ev(A, 10, 1), _ev(A, 15, 0)])
        out = Civitarese().describe_window(w)
        assert "Here, they" in out[0]
        assert "Then, they" in out[0]

    def test_group_by_area_all_meta_none(self):
        s = Sensor(name="X", meta=None)
        w = _window([Event(timestamp=T0, sensor=s, value=1)])
        assert Civitarese().group_by_area(w) == []

    @pytest.mark.parametrize(
        ("hour", "period"),
        [(6, "morning"), (13, "afternoon"), (18, "evening"), (23, "night")],
    )
    def test_compound_period_of_day(self, hour, period):
        c = CompoundSensorSummary(attributes=["time_of_occurrence"]).setup(
            {"A": A.meta}
        )
        w = _window([Event(timestamp=datetime(2024, 1, 1, hour, 0), sensor=A, value=1)])
        assert period in c.describe_window(w)[0]
