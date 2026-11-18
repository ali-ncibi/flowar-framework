from typing import Any

from ...dataset.methods import SmartHomeDataset
from ..methods.base import BaseSegmentationMethod
from .metrics import compute_f1_score, get_true_boundaries


class CrossValidationEvaluator:
    def __init__(
        self,
        dataset: SmartHomeDataset,
        tolerance_seconds: float = 30.0,
        n_folds: int | None = None,
    ):
        self.dataset = dataset
        self.tolerance_seconds = tolerance_seconds
        self.splits = dataset.leave_one_day_out_split()
        if n_folds and n_folds < len(self.splits):
            self.splits = self.splits[:n_folds]
        if not self.splits:
            raise ValueError("Dataset must have at least one day of data for CV splits")

    def evaluate_hyperparams(
        self,
        segmenter_class: type[BaseSegmentationMethod],
        params: dict[str, Any],
    ) -> tuple[float, list[float]]:
        fold_scores = []
        for fold_idx, (train_ds, test_ds) in enumerate(self.splits):
            try:
                train_stream = train_ds.to_event_stream()
                test_stream = test_ds.to_event_stream()
                if not train_stream.events or not test_stream.events:
                    continue
                train_activity_labels = train_ds.extract_activity_labels(train_stream)
                segmenter = segmenter_class(**params)
                segmenter.fit(train_stream, activity_labels=train_activity_labels)
                result = segmenter.transform(test_stream)
                true_boundaries = get_true_boundaries(test_ds.activity_occurrences)
                predicted_boundaries = result.get_boundaries()
                score = compute_f1_score(
                    predicted_boundaries,
                    true_boundaries,
                    self.tolerance_seconds,
                )
                fold_scores.append(score)
            except Exception as e:
                print(f"Error in fold {fold_idx}: {str(e)}")
                continue
        if not fold_scores:
            return 0.0, []
        mean_score = sum(fold_scores) / len(fold_scores)
        return mean_score, fold_scores
