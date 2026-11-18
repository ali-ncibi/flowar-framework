from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .....models import EventStream, Interval, IntervalType, Segment, SegmentationResult
from ....features.base import ALL_FEATURES, DEFAULT_FEATURES, extract_window_features
from ....methods.base import BaseSegmentationMethod


@dataclass(frozen=True)
class CpdConfig:
    local_context_size: int = 5
    number_of_view_samples: int = 10
    number_of_bases: int = 5
    kernel_width: float = 1.0
    lambda_reg: float = 0.01
    threshold: float = 0.3
    random_state: int = 42
    alpha_relative: float | None = None
    features: list[str] | None = None


class ScoreStrategy(Protocol):
    def validate(self, cfg: CpdConfig) -> None: ...
    def score(
        self,
        core: "_CpdCore",
        past_view: NDArray[np.float64],
        future_view: NDArray[np.float64],
    ) -> float: ...
    def parameters(self, cfg: CpdConfig) -> dict[str, Any]: ...
    def boundary_type(self) -> str: ...
    def metadata_key(self) -> str: ...


class _CpdCore:
    def __init__(self, cfg: CpdConfig):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.random_state)

    def extract_feature_matrix(self, events: EventStream) -> NDArray[np.float64]:
        feats = []
        features = (
            self.cfg.features if self.cfg.features is not None else DEFAULT_FEATURES
        )
        for j in range(self.cfg.local_context_size, len(events)):
            feats.append(
                extract_window_features(
                    events, j, self.cfg.local_context_size, features
                )
            )
        X = np.array(feats)
        mn = X.min(axis=0)
        mx = X.max(axis=0)
        rng = mx - mn
        rng[rng == 0] = 1.0
        return (X - mn) / rng

    def select_centers(self, future_view: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.cfg.number_of_bases <= len(future_view):
            idx = self.rng.choice(
                len(future_view), self.cfg.number_of_bases, replace=False
            )
        else:
            idx = np.arange(len(future_view))
        return future_view[idx]

    def rulsif_theta(
        self,
        past_view: NDArray[np.float64],
        future_view: NDArray[np.float64],
        centers: NDArray[np.float64],
        alpha_relative: float,
    ) -> NDArray[np.float64]:
        n_minus = len(past_view)
        n_plus = len(future_view)
        number_of_bases = len(centers)
        K_past = np.zeros((n_minus, number_of_bases))
        for i, x_past in enumerate(past_view):
            for ell, center in enumerate(centers):
                K_past[i, ell] = np.exp(
                    -np.sum((x_past - center) ** 2) / (2 * self.cfg.kernel_width**2)
                )
        K_future = np.zeros((n_plus, number_of_bases))
        for j, x_future in enumerate(future_view):
            for ell, center in enumerate(centers):
                K_future[j, ell] = np.exp(
                    -np.sum((x_future - center) ** 2) / (2 * self.cfg.kernel_width**2)
                )
        H = (alpha_relative / n_minus) * (K_past.T @ K_past) + (
            (1.0 - alpha_relative) / n_plus
        ) * (K_future.T @ K_future)
        H_reg = H + self.cfg.lambda_reg * np.eye(number_of_bases)
        h = np.mean(K_past, axis=0)

        try:
            return np.linalg.solve(H_reg, h)
        except (
            np.linalg.LinAlgError
        ):  # pragma: no cover  # H_reg is positive definite (lambda_reg > 0)
            return np.linalg.lstsq(H_reg, h, rcond=None)[0]


class _WindowCpdSegmenter(BaseSegmentationMethod):
    def __init__(self, method_name: str, cfg: CpdConfig, scorer: ScoreStrategy):
        super().__init__(method_name)
        self.cfg = cfg
        self.scorer = scorer
        self.core = _CpdCore(cfg)
        self._validate()

    def _validate(self) -> None:
        if self.cfg.local_context_size < 1:
            raise ValueError(
                f"local_context_size must be >= 1, got {self.cfg.local_context_size}"
            )
        if self.cfg.number_of_view_samples < 1:
            raise ValueError(
                f"number_of_view_samples must be >= 1, got {self.cfg.number_of_view_samples}"
            )
        if self.cfg.number_of_bases < 1:
            raise ValueError(
                f"number_of_bases must be >= 1, got {self.cfg.number_of_bases}"
            )
        if self.cfg.kernel_width <= 0:
            raise ValueError(f"kernel_width must be > 0, got {self.cfg.kernel_width}")
        if self.cfg.lambda_reg <= 0:
            raise ValueError(f"lambda_reg must be > 0, got {self.cfg.lambda_reg}")
        if not 0.0 <= self.cfg.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.cfg.threshold}")
        self.scorer.validate(self.cfg)

    def fit(self, events: EventStream, **kwargs: Any):
        self._is_calibrated = True
        return self

    def transform(self, events: EventStream) -> SegmentationResult:
        if (
            len(events)
            < self.cfg.local_context_size + 2 * self.cfg.number_of_view_samples
        ):
            raise ValueError(
                f"Event stream must have at least {self.cfg.local_context_size + 2 * self.cfg.number_of_view_samples} events, got {len(events)}"
            )
        X = self.core.extract_feature_matrix(events)
        scores: list[float] = []
        valid_indices: list[int] = []
        for idx in range(
            self.cfg.number_of_view_samples, len(X) - self.cfg.number_of_view_samples
        ):
            j = idx + self.cfg.local_context_size
            past_view = X[idx - self.cfg.number_of_view_samples : idx]
            future_view = X[idx : idx + self.cfg.number_of_view_samples]
            scores.append(self.scorer.score(self.core, past_view, future_view))
            valid_indices.append(j)
        boundaries: list[int] = []
        for i in range(1, len(scores) - 1):
            if (
                scores[i] > self.cfg.threshold
                and scores[i] > scores[i - 1]
                and scores[i] > scores[i + 1]
            ):
                boundaries.append(valid_indices[i])
        segments: list[Segment] = []
        boundaries_with_ends = [0] + boundaries + [len(events) - 1]
        for i in range(len(boundaries_with_ends) - 1):
            start_idx = boundaries_with_ends[i]
            end_idx = boundaries_with_ends[i + 1]
            start_t = events.events[start_idx].timestamp
            end_t = events.events[end_idx].timestamp
            segments.append(
                Segment(
                    interval=Interval(start_t, end_t, IntervalType.LEFT_OPEN),
                    event_indices=(start_idx, end_idx),
                    metadata={"boundary_type": self.scorer.boundary_type()},
                )
            )
        return SegmentationResult(
            segments=segments,
            method_name=self.method_name,
            parameters=self.scorer.parameters(self.cfg),
            metadata={
                "num_boundaries": len(boundaries),
                self.scorer.metadata_key(): max(scores) if scores else 0.0,
            },
        )

    @staticmethod
    def get_default_features() -> list[str]:
        return DEFAULT_FEATURES

    @staticmethod
    def get_all_features() -> list[str]:
        return ALL_FEATURES
