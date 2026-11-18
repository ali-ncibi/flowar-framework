from collections import defaultdict
from collections.abc import Mapping

from backend.models import SensorActivations, SensorName


def group_activations_by_sensor(
    sensor_activations: SensorActivations,
) -> Mapping[SensorName, SensorActivations]:
    activations_by_name: dict[SensorName, SensorActivations] = defaultdict(
        lambda: SensorActivations(data=[])
    )
    for r in sensor_activations.data:
        activations_by_name[r.sensor.name].data.append(r)
    for lst in activations_by_name.values():
        lst.data.sort(key=lambda x: x.interval.start)
    return activations_by_name
