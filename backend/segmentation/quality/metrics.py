from datetime import datetime

from ...models import ActivityOccurrence


def get_true_boundaries(
    activity_occurrences: list["ActivityOccurrence"],
) -> list[datetime]:
    if not activity_occurrences:
        return []
    sorted_activities = sorted(activity_occurrences, key=lambda a: a.interval.start)
    boundaries = [sorted_activities[0].interval.start]
    for i in range(1, len(sorted_activities)):
        current = sorted_activities[i]
        previous = sorted_activities[i - 1]
        if current.name != previous.name:
            boundaries.append(current.interval.start)
    return sorted(boundaries)


def compute_f1_score(
    predicted_boundaries: list[datetime],
    true_boundaries: list[datetime],
    tolerance_seconds: float = 30.0,
) -> float:
    if not true_boundaries:
        if not predicted_boundaries:
            return 1.0
        return 0.0
    if not predicted_boundaries:
        return 0.0
    matched_predicted = set()
    for pred_boundary in predicted_boundaries:
        for true_boundary in true_boundaries:
            time_diff = abs((pred_boundary - true_boundary).total_seconds())
            if time_diff <= tolerance_seconds:
                matched_predicted.add(id(pred_boundary))
                break
    matched_true = set()
    for true_boundary in true_boundaries:
        for pred_boundary in predicted_boundaries:
            time_diff = abs((pred_boundary - true_boundary).total_seconds())
            if time_diff <= tolerance_seconds:
                matched_true.add(id(true_boundary))
                break
    tp = len(matched_predicted)
    fp = len(predicted_boundaries) - tp
    fn = len(true_boundaries) - len(matched_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
