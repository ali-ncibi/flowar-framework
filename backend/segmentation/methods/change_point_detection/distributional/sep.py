from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import CpdConfig, _CpdCore, _WindowCpdSegmenter


class _SepScore:
    def validate(self, cfg: CpdConfig) -> None:
        if cfg.alpha_relative is not None:
            raise ValueError("alpha_relative must be None for SEP-CPD")

    def score(
        self,
        core: "_CpdCore",
        past_view: NDArray[np.float64],
        future_view: NDArray[np.float64],
    ) -> float:
        centers = core.select_centers(future_view)
        h_hat = np.zeros(len(centers))
        for center_idx, center in enumerate(centers):
            kernel_vals = np.exp(
                -np.sum((past_view - center) ** 2, axis=1)
                / (2 * core.cfg.kernel_width**2)
            )
            h_hat[center_idx] = np.mean(kernel_vals)
        theta_hat = -h_hat / core.cfg.lambda_reg
        ratio_sum = 0.0
        for x_past in past_view:
            r_j = 0.0
            for center_idx, center in enumerate(centers):
                kernel_val = np.exp(
                    -np.sum((x_past - center) ** 2) / (2 * core.cfg.kernel_width**2)
                )
                r_j += theta_hat[center_idx] * kernel_val
            ratio_sum += r_j
        n_minus = len(past_view)
        return max(0.0, 0.5 - ratio_sum / n_minus)

    def parameters(self, cfg: CpdConfig) -> dict[str, Any]:
        return {
            "local_context_size": cfg.local_context_size,
            "number_of_view_samples": cfg.number_of_view_samples,
            "threshold": cfg.threshold,
        }

    def boundary_type(self) -> str:
        return "sep_cpd"

    def metadata_key(self) -> str:
        return "max_sep_score"


class SepCpd(_WindowCpdSegmenter):
    def __init__(
        self,
        local_context_size: int = 5,
        number_of_view_samples: int = 10,
        number_of_bases: int = 5,
        kernel_width: float = 1.0,
        lambda_reg: float = 0.01,
        threshold: float = 0.3,
        alpha_relative: float | None = None,
        random_state: int = 42,
        features: list[str] | None = None,
    ):
        super().__init__(
            "SepCpd (Aminikhanghahi et al. 2019)",
            CpdConfig(
                local_context_size=local_context_size,
                number_of_view_samples=number_of_view_samples,
                number_of_bases=number_of_bases,
                kernel_width=kernel_width,
                lambda_reg=lambda_reg,
                threshold=threshold,
                alpha_relative=alpha_relative,
                random_state=random_state,
                features=features,
            ),
            _SepScore(),
        )
