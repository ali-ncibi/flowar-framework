from __future__ import annotations

from logging import WARNING, getLogger
from multiprocessing import cpu_count
from warnings import filterwarnings

import numpy as np
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_X_y
from torch import (
    FloatTensor,
    LongTensor,
    Tensor,
    manual_seed,
    no_grad,
    set_float32_matmul_precision,
)
from torch.utils.data import DataLoader, TensorDataset

from ..base import BaseClassificationMethod
from .base import BackboneType, SequenceGridClassifierModule, TaskType

set_float32_matmul_precision("high")


def device_info_filter(record):
    return "PU available: " not in record.getMessage()


getLogger("lightning.pytorch.utilities.rank_zero").addFilter(device_info_filter)
getLogger("pytorch_lightning").setLevel(WARNING)
getLogger("lightning.pyutilities.rank_zero").setLevel(WARNING)
getLogger("lightning.pyaccelerators.cuda").setLevel(WARNING)
getLogger("lightning").setLevel(0)
getLogger("lightning.pytorch.accelerators.cuda").setLevel(WARNING)


class DeepSequenceGridClassifier(
    BaseEstimator, ClassifierMixin, BaseClassificationMethod
):
    def __init__(
        self,
        backbone: BackboneType = "bilstm",
        use_attention: bool = False,
        task: TaskType = "multiclass",
        threshold: float = 0.5,
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        cnn_channels: int = 64,
        cnn_kernel_size: int = 5,
        cnn_blocks: int = 2,
        cnn2d_pool: bool = True,
        dropout: float = 0.1,
        learning_rate_init: float = 1e-3,
        weight_decay: float = 0.0,
        solver: str = "adam",
        max_epochs: int = 100,
        batch_size: int = 32,
        validation_fraction: float = 0.2,
        early_stopping: bool = True,
        n_iter_no_change: int = 10,
        class_weight: str | dict | None = None,
        pos_weight: str | np.ndarray | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        use_masking: bool = True,
    ):
        self.backbone: BackboneType = backbone
        self.use_attention = use_attention
        self.task: TaskType = task
        self.threshold = threshold
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.cnn_channels = cnn_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_blocks = cnn_blocks
        self.cnn2d_pool = cnn2d_pool
        self.dropout = dropout
        self.learning_rate_init = learning_rate_init
        self.weight_decay = weight_decay
        self.solver = solver
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.class_weight = class_weight
        self.pos_weight = pos_weight
        self.random_state = random_state
        self.verbose = verbose
        self.use_masking = use_masking

    def _validate_params(self):
        if self.task not in ("multiclass", "multilabel"):
            raise ValueError("task must be 'multiclass' or 'multilabel'")
        if self.backbone not in ("rnn", "lstm", "bilstm", "cnn1d", "cnn2d"):
            raise ValueError("Invalid backbone")
        if self.hidden_size <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError("dropout must be in [0,1]")
        if not (0.0 <= self.validation_fraction <= 1.0):
            raise ValueError("validation_fraction must be in [0,1]")
        if self.batch_size <= 0 or self.max_epochs <= 0:
            raise ValueError("batch_size and max_epochs must be positive")
        if self.n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be positive")
        if self.task == "multilabel" and not (0.0 < self.threshold < 1.0):
            raise ValueError("threshold must be in (0,1) for multilabel")

    def _setup_random_state(self):
        if self.random_state is not None:
            seed_everything(self.random_state, workers=True)
            manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def _compute_class_weights_multiclass(self, y_encoded: np.ndarray) -> Tensor | None:
        if self.class_weight is None:
            return None
        if isinstance(self.class_weight, str) and self.class_weight == "balanced":
            w = compute_class_weight(
                "balanced", classes=np.arange(self.n_classes_), y=y_encoded
            )
            return FloatTensor(w)
        if isinstance(self.class_weight, dict):
            weights = np.ones(self.n_classes_, dtype=np.float32)
            for k, v in self.class_weight.items():
                weights[int(k)] = float(v)
            return FloatTensor(weights)
        raise ValueError("class_weight must be None, 'balanced', or dict")

    def _compute_pos_weight_multilabel(self, y: np.ndarray) -> Tensor | None:
        """
        pos_weight for BCEWithLogitsLoss: weight for positive examples per label.
        If 'balanced': pos_weight = (n_neg / n_pos) per label.
        """
        if self.pos_weight is None:
            return None
        if isinstance(self.pos_weight, str) and self.pos_weight == "balanced":
            y01 = (y > 0).astype(np.float32)
            n_pos = y01.sum(axis=0)
            n_neg = y01.shape[0] - n_pos
            pos_weight = (n_neg / (n_pos + 1e-8)).astype(np.float32)
            return FloatTensor(pos_weight)
        if isinstance(self.pos_weight, np.ndarray):
            if self.pos_weight.ndim != 1:
                raise ValueError("pos_weight array must be shape (K,)")
            return FloatTensor(self.pos_weight.astype(np.float32))
        raise ValueError("pos_weight must be None, 'balanced', or np.ndarray(K,)")

    def _create_data_loaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
        masks: np.ndarray | None = None,
    ) -> tuple[DataLoader, DataLoader | None]:
        X_tensor = FloatTensor(X)

        y_tensor: Tensor
        if self.task == "multiclass":
            y_tensor = LongTensor(y)
        else:
            y_tensor = FloatTensor(y)

        masks_tensor = FloatTensor(masks) if masks is not None else None

        if self.validation_fraction > 0:
            n = len(X)
            n_val = int(n * self.validation_fraction)
            indices = np.arange(n)
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
            val_idx = indices[:n_val]
            tr_idx = indices[n_val:]

            if masks_tensor is not None:
                train_ds = TensorDataset(
                    X_tensor[tr_idx], y_tensor[tr_idx], masks_tensor[tr_idx]
                )
                val_ds = TensorDataset(
                    X_tensor[val_idx], y_tensor[val_idx], masks_tensor[val_idx]
                )
            else:
                train_ds = TensorDataset(X_tensor[tr_idx], y_tensor[tr_idx])
                val_ds = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])

            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=cpu_count() or 0,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=cpu_count() or 0,
            )
            return train_loader, val_loader
        if masks_tensor is not None:
            train_ds = TensorDataset(X_tensor, y_tensor, masks_tensor)
        else:
            train_ds = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=cpu_count() or 0,
        )
        return train_loader, None

    def fit(self, X, y, **kwargs):
        masks = kwargs.pop("masks", None)
        self._validate_params()
        self._setup_random_state()
        if self.task == "multiclass":
            X, y = check_X_y(X, y, allow_nd=True)
        else:
            X = check_array(X, allow_nd=True)
            y = check_array(y, allow_nd=True)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        elif X.ndim != 3:
            raise ValueError("X must be 2D (n,P) or 3D (n,P,Q)")
        if masks is not None:
            masks = np.asarray(masks)
            if masks.shape[0] != X.shape[0] or masks.shape[1] != X.shape[1]:
                raise ValueError(
                    f"masks shape {masks.shape} incompatible with X shape {X.shape}"
                )
        if self.task == "multiclass":
            self.classes_ = unique_labels(y)
            self.n_classes_ = len(self.classes_)
            self.input_size_ = int(X.shape[2])
            self.label_encoder_ = LabelEncoder()
            y_encoded = self.label_encoder_.fit_transform(y)
            y_encoded = np.asarray(y_encoded, dtype=np.int64)
            class_weights = self._compute_class_weights_multiclass(y_encoded)
            pos_weight = None
            num_outputs = self.n_classes_
            y_train = y_encoded
        else:
            y_arr = np.asarray(y)
            if y_arr.ndim != 2:
                raise ValueError(
                    "For multilabel, y must be 2D array of shape (n_samples, n_labels)"
                )
            self.n_labels_ = int(y_arr.shape[1])
            self.input_size_ = int(X.shape[2])
            self.classes_ = np.arange(self.n_labels_)
            class_weights = None
            pos_weight = self._compute_pos_weight_multilabel(y_arr)
            num_outputs = self.n_labels_
            y_train = y_arr.astype(np.float32)
        self.model_ = SequenceGridClassifierModule(
            backbone=self.backbone,
            use_attention=self.use_attention,
            task=self.task,
            input_size=self.input_size_,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            cnn_channels=self.cnn_channels,
            cnn_kernel_size=self.cnn_kernel_size,
            cnn_blocks=self.cnn_blocks,
            cnn2d_pool=self.cnn2d_pool,
            num_outputs=num_outputs,
            dropout=self.dropout,
            learning_rate_init=self.learning_rate_init,
            weight_decay=self.weight_decay,
            solver=self.solver,
            class_weights=class_weights,
            pos_weight=pos_weight,
            use_masking=self.use_masking,
        )
        train_loader, val_loader = self._create_data_loaders(X, y_train, masks)
        callbacks = []
        if self.early_stopping and val_loader is not None:
            monitor = "val_loss"
            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    patience=self.n_iter_no_change,
                    mode="min",
                    verbose=self.verbose,
                )
            )
        if not self.verbose:
            filterwarnings("ignore", ".*does not have many workers.*")
            self.trainer_ = Trainer(
                accelerator="gpu",
                devices=1,
                default_root_dir="./log",
                max_epochs=self.max_epochs,
                callbacks=callbacks,
                deterministic=self.random_state is not None,
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                enable_checkpointing=False,
                log_every_n_steps=1,
            )
        else:
            self.trainer_ = Trainer(
                accelerator="gpu",
                devices=1,
                default_root_dir="./log",
                max_epochs=self.max_epochs,
                callbacks=callbacks,
                deterministic=self.random_state is not None,
                enable_model_summary=False,
                log_every_n_steps=1,
            )
        self.trainer_.fit(
            self.model_, train_dataloaders=train_loader, val_dataloaders=val_loader
        )
        return self

    def _predict_logits(self, X, masks=None) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("Model must be fitted before prediction")
        X = check_array(X, allow_nd=True)
        if X.ndim == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        elif X.ndim != 3:
            raise ValueError("X must be 2D or 3D array")
        if X.shape[2] != self.input_size_:
            raise ValueError(
                f"X has {X.shape[2]} features, but model expects {self.input_size_}"
            )
        if masks is not None:
            masks = np.asarray(masks)
            if masks.shape[0] != X.shape[0] or masks.shape[1] != X.shape[1]:
                raise ValueError(
                    f"masks shape {masks.shape} incompatible with X shape {X.shape}"
                )
        X_tensor = FloatTensor(X)
        if masks is not None:
            masks_tensor = FloatTensor(masks)
            ds = TensorDataset(X_tensor, masks_tensor)
        else:
            ds = TensorDataset(X_tensor)
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False, num_workers=0
        )
        self.model_.eval()
        logits_all = []
        with no_grad():
            for batch in loader:
                if len(batch) == 2:
                    xb, mb = batch
                    logits = self.model_(xb, mb)
                else:
                    xb = batch[0]
                    logits = self.model_(xb, None)
                logits_all.append(logits.detach().cpu().numpy())
        return np.concatenate(logits_all, axis=0)

    def predict_proba(self, X, masks=None):
        logits = self._predict_logits(X, masks)
        if self.task == "multiclass":
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, X, **kwargs):
        masks = kwargs.pop("masks", None)
        logits = self._predict_logits(X, masks)
        if self.task == "multiclass":
            pred_idx = np.argmax(logits, axis=1)
            return self.label_encoder_.inverse_transform(pred_idx)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return (probs >= self.threshold).astype(int)

    def score(self, X, y, sample_weight=None, masks=None) -> float:
        y_pred = self.predict(X, masks=masks)
        if self.task == "multiclass":
            return float(accuracy_score(y, y_pred, sample_weight=sample_weight))
        y_true = np.asarray(y).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        return float(
            f1_score(y_true, y_pred, average="micro", sample_weight=sample_weight)
        )

    def get_params(self, deep=True):
        return {
            "backbone": self.backbone,
            "use_attention": self.use_attention,
            "task": self.task,
            "threshold": self.threshold,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "cnn_channels": self.cnn_channels,
            "cnn_kernel_size": self.cnn_kernel_size,
            "cnn_blocks": self.cnn_blocks,
            "cnn2d_pool": self.cnn2d_pool,
            "dropout": self.dropout,
            "learning_rate_init": self.learning_rate_init,
            "weight_decay": self.weight_decay,
            "solver": self.solver,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "validation_fraction": self.validation_fraction,
            "early_stopping": self.early_stopping,
            "n_iter_no_change": self.n_iter_no_change,
            "class_weight": self.class_weight,
            "pos_weight": self.pos_weight,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "use_masking": self.use_masking,
        }

    def set_params(self, **params):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"Invalid parameter {k}")
        return self
