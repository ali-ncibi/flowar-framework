from datetime import datetime, timedelta

import pytest

from backend.models import ActivityOccurrence, Interval, IntervalType
from backend.segmentation.methods import FixedCount
from backend.segmentation.optimization import (
    PARAM_RANGES,
    HyperparameterOptimizer,
    OptimizationResult,
    get_default_ranges,
)
from backend.segmentation.quality import (
    CrossValidationEvaluator,
    compute_f1_score,
    get_true_boundaries,
)

T0 = datetime(2024, 1, 1, 8, 0, 0)


def _occ(name, start_min, end_min):
    return ActivityOccurrence(
        interval=Interval(
            T0 + timedelta(minutes=start_min),
            T0 + timedelta(minutes=end_min),
            IntervalType.LEFT_OPEN,
        ),
        name=name,
    )


class TestGetTrueBoundaries:
    def test_empty(self):
        assert get_true_boundaries([]) == []

    def test_single(self):
        assert get_true_boundaries([_occ("A", 0, 10)]) == [T0]

    def test_name_changes(self):
        occs = [_occ("A", 0, 10), _occ("A", 10, 20), _occ("B", 20, 30)]
        assert get_true_boundaries(occs) == [T0, T0 + timedelta(minutes=20)]

    def test_unsorted_input(self):
        occs = [_occ("B", 20, 30), _occ("A", 0, 10)]
        assert get_true_boundaries(occs) == [T0, T0 + timedelta(minutes=20)]


class TestComputeF1:
    def test_both_empty(self):
        assert compute_f1_score([], [], 30.0) == 1.0

    def test_true_empty_pred_not(self):
        assert compute_f1_score([T0], [], 30.0) == 0.0

    def test_pred_empty_true_not(self):
        assert compute_f1_score([], [T0], 30.0) == 0.0

    def test_perfect_match(self):
        assert (
            compute_f1_score(
                [T0, T0 + timedelta(minutes=10)], [T0, T0 + timedelta(minutes=10)], 30.0
            )
            == 1.0
        )

    def test_within_tolerance(self):
        pred = [T0 + timedelta(seconds=20)]
        true = [T0]
        assert compute_f1_score(pred, true, 30.0) == 1.0

    def test_outside_tolerance(self):
        pred = [T0 + timedelta(seconds=60)]
        true = [T0]
        assert compute_f1_score(pred, true, 30.0) == 0.0

    def test_partial(self):
        pred = [T0, T0 + timedelta(minutes=100)]
        true = [T0, T0 + timedelta(minutes=10)]
        # pred[0] matches true[0]; pred[1] matches nothing; true[1] unmatched
        f1 = compute_f1_score(pred, true, 30.0)
        assert f1 == pytest.approx(2 * (0.5 * 0.5) / (0.5 + 0.5))  # = 0.5

    def test_zero_precision_recall(self):
        pred = [T0 + timedelta(hours=5)]
        true = [T0]
        assert compute_f1_score(pred, true, 30.0) == 0.0


class TestSegCrossValidationEvaluator:
    def test_requires_splits(self):
        from backend.dataset import SmartHomeDataset

        with pytest.raises(ValueError, match="at least one day"):
            CrossValidationEvaluator(SmartHomeDataset())

    def test_n_folds_truncation(self, dataset_dir):
        from backend.dataset import SmartHomeDataset

        ds = SmartHomeDataset(dataset_dir)
        ev = CrossValidationEvaluator(ds, tolerance_seconds=30.0, n_folds=1)
        assert len(ev.splits) == 1

    def test_evaluate_hyperparams(self, dataset_dir):
        from backend.dataset import SmartHomeDataset

        ds = SmartHomeDataset(dataset_dir)
        ev = CrossValidationEvaluator(ds, tolerance_seconds=30.0)
        mean_f1, folds = ev.evaluate_hyperparams(FixedCount, {"nb_events": 4})
        assert 0.0 <= mean_f1 <= 1.0
        assert len(folds) == len(ev.splits)
        assert all(0.0 <= f <= 1.0 for f in folds)

    def test_evaluate_hyperparams_empty_streams_skipped(self, dataset_dir, monkeypatch):
        from backend.dataset import SmartHomeDataset

        ds = SmartHomeDataset(dataset_dir)
        ev = CrossValidationEvaluator(ds)
        # force empty streams -> folds skipped -> no scores
        monkeypatch.setattr(
            SmartHomeDataset,
            "to_event_stream",
            lambda self, max_events=None: __import__(
                "backend.models", fromlist=["EventStream"]
            ).EventStream([]),
        )
        mean_f1, folds = ev.evaluate_hyperparams(FixedCount, {"nb_events": 4})
        assert mean_f1 == 0.0
        assert folds == []

    def test_evaluate_hyperparams_exception_in_fold(self, dataset_dir, monkeypatch):
        from backend.dataset import SmartHomeDataset

        ds = SmartHomeDataset(dataset_dir)
        ev = CrossValidationEvaluator(ds)

        def boom(self, events, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(FixedCount, "fit", boom)
        mean_f1, folds = ev.evaluate_hyperparams(FixedCount, {"nb_events": 4})
        assert mean_f1 == 0.0
        assert folds == []


class TestParamRanges:
    def test_get_default_ranges(self):
        assert get_default_ranges("FixedCount") == PARAM_RANGES["FixedCount"]
        assert get_default_ranges("Unknown") == {}

    def test_all_methods_have_ranges(self):
        for name in [
            "SepCpd",
            "RulsifPeCpd",
            "RulsifPlsbdCpd",
            "CorrWan",
            "CorrXu",
            "W2vCsCPD",
            "W2vCcsCPD",
            "W2vCcsdCPD",
            "FixedCount",
            "FixedTime",
        ]:
            assert name in PARAM_RANGES


class TestHyperparameterOptimizer:
    def test_optimize_fixed_count(self, dataset_dir):
        from backend.dataset import SmartHomeDataset

        ds = SmartHomeDataset(dataset_dir)
        ev = CrossValidationEvaluator(ds, tolerance_seconds=30.0, n_folds=1)
        opt = HyperparameterOptimizer(ev)
        result = opt.optimize(
            FixedCount,
            {"nb_events": {"min": 2, "max": 6, "type": "int"}},
            n_calls=3,
            n_initial_points=2,
        )
        assert isinstance(result, OptimizationResult)
        assert result.segmenter_class is FixedCount
        assert 2 <= result.best_params["nb_events"] <= 6
        assert isinstance(result.best_params["nb_events"], int)
        assert 0.0 <= result.best_f1 <= 1.0
        assert len(result.iteration_results) == 3

    def test_optimize_float_param(self, dataset_dir):
        from backend.dataset import SmartHomeDataset

        ds = SmartHomeDataset(dataset_dir)
        ev = CrossValidationEvaluator(ds, n_folds=1)
        opt = HyperparameterOptimizer(ev)
        result = opt.optimize(
            FixedCount,
            {"overlap_percentage": {"min": 0.0, "max": 50.0, "type": "float"}},
            n_calls=2,
            n_initial_points=1,
        )
        assert 0.0 <= result.best_params["overlap_percentage"] <= 50.0

    def test_get_best_params_with_defaults(self):
        res = OptimizationResult(
            best_params={"a": 2},
            best_f1=0.5,
            iteration_results=[],
            segmenter_class=FixedCount,
        )
        assert res.get_best_params_with_defaults({"a": 1, "b": 3}) == {"a": 2, "b": 3}
