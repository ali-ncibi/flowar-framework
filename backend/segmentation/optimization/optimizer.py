from dataclasses import dataclass
from typing import Any

from skopt import gp_minimize
from skopt.space import Integer, Real, Space

from ...segmentation.methods.base import BaseSegmentationMethod
from ..quality.cv import CrossValidationEvaluator


@dataclass
class OptimizationResult:
    best_params: dict[str, Any]
    best_f1: float
    iteration_results: list[dict[str, Any]]
    segmenter_class: type[BaseSegmentationMethod]

    def get_best_params_with_defaults(self, defaults: dict[str, Any]) -> dict:
        result = dict(defaults)
        result.update(self.best_params)
        return result


class HyperparameterOptimizer:
    def __init__(self, evaluator: CrossValidationEvaluator):
        self.evaluator = evaluator
        self.iteration_results: list[dict[str, Any]] = []

    def optimize(
        self,
        segmenter_class: type[BaseSegmentationMethod],
        param_ranges: dict[str, dict[str, Any]],
        n_calls: int = 50,
        n_initial_points: int = 5,
        random_state: int = 42,
    ) -> OptimizationResult:
        self.iteration_results = []
        param_names = []
        space_dimensions = []
        for param_name, param_info in param_ranges.items():
            param_names.append(param_name)
            param_type = param_info.get("type", "float")
            min_val = param_info["min"]
            max_val = param_info["max"]
            if param_type == "int":
                space_dimensions.append(Integer(low=int(min_val), high=int(max_val)))
            else:
                space_dimensions.append(Real(low=float(min_val), high=float(max_val)))
        space = Space(space_dimensions)

        def objective(param_values):
            params = dict(zip(param_names, param_values, strict=False))
            mean_f1, fold_scores = self.evaluator.evaluate_hyperparams(
                segmenter_class, params
            )
            self.iteration_results.append(
                {
                    "params": dict(params),
                    "mean_f1": mean_f1,
                    "fold_scores": fold_scores,
                    "iteration": len(self.iteration_results),
                }
            )
            return -mean_f1

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
        best_params = dict(zip(param_names, result.x, strict=False))
        for param_name, param_info in param_ranges.items():
            param_type = param_info.get("type", "float")
            if param_type == "int":
                best_params[param_name] = int(best_params[param_name])
        best_f1 = -result.fun
        return OptimizationResult(
            best_params=best_params,
            best_f1=best_f1,
            iteration_results=self.iteration_results,
            segmenter_class=segmenter_class,
        )
