from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import CpdConfig, _CpdCore, _WindowCpdSegmenter


class _RulsifPlsbdScore:
    def validate(self, cfg: CpdConfig) -> None:
        if cfg.alpha_relative is None or not (0.0 < cfg.alpha_relative <= 1.0):
            raise ValueError(
                f"alpha_relative must be in (0, 1], got {cfg.alpha_relative}"
            )
        if cfg.alpha_relative == 1.0:
            raise ValueError("alpha_relative must be < 1 for PLsBD-CPD")

    def score(
        self,
        core: "_CpdCore",
        past_view: NDArray[np.float64],
        future_view: NDArray[np.float64],
    ) -> float:
        alpha_relative = float(core.cfg.alpha_relative)  # type: ignore
        centers = core.select_centers(future_view)
        theta_hat = core.rulsif_theta(past_view, future_view, centers, alpha_relative)
        n_minus = len(past_view)
        n_plus = len(future_view)
        ratio_past_sum = 0.0
        for x_past in past_view:
            r_alpha = 0.0
            for ell, center in enumerate(centers):
                kernel_val = np.exp(
                    -np.sum((x_past - center) ** 2) / (2 * core.cfg.kernel_width**2)
                )
                r_alpha += theta_hat[ell] * kernel_val
            ratio_past_sum += r_alpha
        ratio_future_sum = 0.0
        for x_future in future_view:
            r_alpha = 0.0
            for ell, center in enumerate(centers):
                kernel_val = np.exp(
                    -np.sum((x_future - center) ** 2) / (2 * core.cfg.kernel_width**2)
                )
                r_alpha += theta_hat[ell] * kernel_val
            ratio_future_sum += r_alpha
        score = (
            (1.0 / (2.0 * n_minus)) * ratio_past_sum
            - ((2.0 - alpha_relative) / (2.0 * (1.0 - alpha_relative) * n_plus))
            * ratio_future_sum
            + 1.0 / (2.0 * (1.0 - alpha_relative))
        )
        return abs(float(score))

    def parameters(self, cfg: CpdConfig) -> dict[str, Any]:
        return {
            "local_context_size": cfg.local_context_size,
            "number_of_view_samples": cfg.number_of_view_samples,
            "alpha_relative": cfg.alpha_relative,
            "threshold": cfg.threshold,
        }

    def boundary_type(self) -> str:
        return "plsbd_cpd"

    def metadata_key(self) -> str:
        return "max_plsbd_score"


class RulsifPlsbdCpd(_WindowCpdSegmenter):
    def __init__(
        self,
        local_context_size: int = 5,
        number_of_view_samples: int = 10,
        number_of_bases: int = 5,
        kernel_width: float = 1.0,
        lambda_reg: float = 0.01,
        threshold: float = 0.3,
        alpha_relative: float = 0.5,
        random_state: int = 42,
        features: list[str] | None = None,
    ):
        super().__init__(
            "RulsifPlsbdCpd (Sin et al. 2024)",
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
            _RulsifPlsbdScore(),
        )
