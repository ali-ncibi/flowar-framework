from dataclasses import dataclass
from typing import Any

from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real, Space

from backend.annotation.methods import BaseAnnotationMethod
from backend.representation.methods import BaseRepresentationMethod
from backend.segmentation.methods import BaseSegmentationMethod

from ..quality.cv import CrossValidationEvaluator


@dataclass
class OptimizationResult:
    best_params: dict[str, Any]
    best_score: float
    iteration_results: list[dict[str, Any]]
    classifier_class: type
    metric_name: str = "compute_f1_score"

    def get_best_params_with_defaults(self, defaults: dict[str, Any]) -> dict:
        result = dict(defaults)
        result.update(self.best_params)
        return result


class HyperparameterOptimizer:
    def __init__(
        self,
        evaluator: CrossValidationEvaluator,
        segmenter_class: type[BaseSegmentationMethod],
        segmenter_params: dict[str, Any],
        extractor_class: type[BaseRepresentationMethod],
        extractor_params: dict[str, Any],
        annotator_class: type[BaseAnnotationMethod],
    ):
        self.segmenter = segmenter_class(**segmenter_params)
        self.extractor = extractor_class(**extractor_params)
        self.annotator = annotator_class()
        self.evaluator = evaluator
        self.iteration_results: list[dict[str, Any]] = []

    def optimize(
        self,
        classifier_class: type,
        classifier_param_ranges: dict[str, dict[str, Any]],
        n_calls: int = 50,
        n_initial_points: int = 5,
        random_state: int = 42,
    ) -> OptimizationResult:
        self.iteration_results = []
        param_names = []
        space_dimensions = []
        categorical_params = {}

        # Build search space
        for param_name, param_info in classifier_param_ranges.items():
            param_names.append(param_name)
            param_type = param_info.get("type", "float")
            if param_type == "categorical":
                categorical_params[param_name] = param_info.get("values", [])
                space_dimensions.append(
                    Categorical(param_info.get("values", []), name=param_name)
                )
            elif param_type == "int":
                min_val = param_info["min"]
                max_val = param_info["max"]
                space_dimensions.append(
                    Integer(low=int(min_val), high=int(max_val), name=param_name)
                )
            else:  # float
                min_val = param_info["min"]
                max_val = param_info["max"]
                space_dimensions.append(
                    Real(low=float(min_val), high=float(max_val), name=param_name)
                )

        space = Space(space_dimensions)

        def objective(param_values):
            """Objective function to minimize."""
            classifier_params = dict(zip(param_names, param_values, strict=False))

            # Convert types if needed
            for param_name, param_type in [
                (pn, classifier_param_ranges[pn].get("type", "float"))
                for pn in param_names
            ]:
                if param_type == "int":
                    classifier_params[param_name] = int(classifier_params[param_name])

            mean_accuracy, fold_scores = self.evaluator.evaluate_hyperparams(
                classifier_class,
                classifier_params,
                self.segmenter,
                self.extractor,
                self.annotator,
            )
            self.iteration_results.append(
                {
                    "params": dict(classifier_params),
                    "mean_accuracy": mean_accuracy,
                    "fold_scores": fold_scores,
                    "iteration": len(self.iteration_results),
                }
            )
            return -mean_accuracy  # Negative because gp_minimize minimizes

        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            random_state=random_state,
            verbose=False,
            n_jobs=1,
        )
        assert result is not None  # gp_minimize always returns a result

        best_params = {}
        for i, param_name in enumerate(param_names):
            best_params[param_name] = result.x[i]

        # Convert types
        for param_name, param_info in classifier_param_ranges.items():
            param_type = param_info.get("type", "float")
            if param_type == "int":
                best_params[param_name] = int(best_params[param_name])
            elif param_type == "categorical":
                # Already correct type from Categorical space
                pass

        best_score = -result.fun

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            iteration_results=self.iteration_results,
            classifier_class=classifier_class,
            # metric_name=self.evaluator.metric_name,
        )
