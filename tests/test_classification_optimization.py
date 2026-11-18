from backend.annotation.methods import MostOverlappingAnnotator
from backend.classification.methods import KNeighborsClassifier
from backend.classification.optimization import (
    PARAM_RANGES,
    HyperparameterOptimizer,
    OptimizationResult,
    get_all_classifiers,
    get_default_ranges,
    get_matrix_feature_classifiers,
    get_vector_feature_classifiers,
)
from backend.representation.methods.manual.vector.stats import StatsVectorExtractor
from backend.segmentation.methods.fixed_length import FixedCount


class TestParamRanges:
    def test_get_default_ranges_known(self):
        ranges = get_default_ranges("KNeighborsClassifier")
        assert "n_neighbors" in ranges
        assert ranges["n_neighbors"]["type"] == "int"

    def test_get_default_ranges_unknown(self):
        assert get_default_ranges("NoSuchClassifier") == {}

    def test_classifier_lists(self):
        vector = get_vector_feature_classifiers()
        matrix = get_matrix_feature_classifiers()
        assert "KNeighborsClassifier" in vector
        assert "DeepSequenceGridClassifier" in matrix
        all_clf = get_all_classifiers()
        assert set(all_clf) == set(vector) | set(matrix)

    def test_param_ranges_structure(self):
        for name, params in PARAM_RANGES.items():
            assert isinstance(name, str)
            for info in params.values():
                assert "type" in info
                assert "default" in info
                if info["type"] == "categorical":
                    assert "values" in info
                else:
                    assert "min" in info and "max" in info


class StubEvaluator:
    def __init__(self, score=0.75):
        self.score = score
        self.calls = []

    def evaluate_hyperparams(
        self, classifier_class, params, segmenter, extractor, annotator
    ):
        self.calls.append(dict(params))
        return self.score, [self.score, self.score]


def make_optimizer(evaluator):
    return HyperparameterOptimizer(
        evaluator=evaluator,
        segmenter_class=FixedCount,
        segmenter_params={"nb_events": 2},
        extractor_class=StatsVectorExtractor,
        extractor_params={"desired_features": ["segment_duration"]},
        annotator_class=MostOverlappingAnnotator,
    )


class TestOptimizer:
    def test_init_constructs_methods(self):
        opt = make_optimizer(StubEvaluator())
        assert isinstance(opt.segmenter, FixedCount)
        assert isinstance(opt.extractor, StatsVectorExtractor)
        assert isinstance(opt.annotator, MostOverlappingAnnotator)
        assert opt.iteration_results == []

    def test_optimize_mixed_param_types(self):
        evaluator = StubEvaluator(score=0.6)
        opt = make_optimizer(evaluator)
        ranges = {
            "n_neighbors": {"min": 1, "max": 5, "type": "int"},
            "weights": {"values": ["uniform", "distance"], "type": "categorical"},
            "p": {"min": 1.0, "max": 3.0, "type": "float"},
        }
        result = opt.optimize(
            KNeighborsClassifier,
            ranges,
            n_calls=4,
            n_initial_points=2,
            random_state=0,
        )
        assert isinstance(result, OptimizationResult)
        assert result.classifier_class is KNeighborsClassifier
        assert result.best_score == 0.6
        assert isinstance(result.best_params["n_neighbors"], int)
        assert result.best_params["weights"] in ("uniform", "distance")
        assert isinstance(result.best_params["p"], float)
        assert len(result.iteration_results) == 4
        assert len(evaluator.calls) == 4
        for call, iteration in zip(
            evaluator.calls, result.iteration_results, strict=True
        ):
            assert call == iteration["params"]
            assert iteration["mean_accuracy"] == 0.6
            assert iteration["fold_scores"] == [0.6, 0.6]
        assert [r["iteration"] for r in result.iteration_results] == [0, 1, 2, 3]

    def test_optimize_default_param_type_is_float(self):
        evaluator = StubEvaluator()
        opt = make_optimizer(evaluator)
        # no "type" key -> treated as float
        result = opt.optimize(
            KNeighborsClassifier,
            {"p": {"min": 1.0, "max": 2.0}},
            n_calls=2,
            n_initial_points=1,
            random_state=0,
        )
        assert isinstance(result.best_params["p"], float)


class TestOptimizationResult:
    def test_get_best_params_with_defaults(self):
        result = OptimizationResult(
            best_params={"n_neighbors": 3},
            best_score=0.9,
            iteration_results=[],
            classifier_class=KNeighborsClassifier,
        )
        merged = result.get_best_params_with_defaults(
            {"n_neighbors": 5, "weights": "uniform"}
        )
        assert merged == {"n_neighbors": 3, "weights": "uniform"}

    def test_default_metric_name(self):
        result = OptimizationResult(
            best_params={},
            best_score=0.0,
            iteration_results=[],
            classifier_class=KNeighborsClassifier,
        )
        assert result.metric_name == "compute_f1_score"
