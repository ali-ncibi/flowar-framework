from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch import Tensor, argmax, optim

from .layers import (
    AttentionPool1D,
    AttentionPool2D,
    CNN1DBackbone,
    CNN2DBackbone,
    RNNBackbone,
)

TaskType = Literal["multiclass", "multilabel"]
BackboneType = Literal["rnn", "lstm", "bilstm", "cnn1d", "cnn2d"]


class SequenceGridClassifierModule(LightningModule):
    def __init__(
        self,
        backbone: BackboneType,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_outputs: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
        use_attention: bool = False,
        task: TaskType = "multiclass",
        learning_rate_init: float = 1e-3,
        weight_decay: float = 0.0,
        solver: str = "adam",
        class_weights: Tensor | None = None,
        pos_weight: Tensor | None = None,
        use_masking: bool = True,
        cnn_channels: int = 64,
        cnn_kernel_size: int = 5,
        cnn_blocks: int = 2,
        cnn2d_pool: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.use_attention = use_attention
        if backbone in ("rnn", "lstm", "bilstm"):
            kind: Literal["rnn", "lstm"] = "rnn" if backbone == "rnn" else "lstm"
            bi = (backbone == "bilstm") or bidirectional
            self.backbone: Any = RNNBackbone(
                kind=kind,
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bi,
            )
            feat_size = self.backbone.output_size
            self.attn1d = (
                AttentionPool1D(feat_size, dropout=dropout) if use_attention else None
            )

        elif backbone == "cnn1d":
            self.backbone = CNN1DBackbone(
                in_channels=input_size,
                channels=cnn_channels,
                num_blocks=cnn_blocks,
                kernel_size=cnn_kernel_size,
                dropout=dropout,
            )
            feat_size = self.backbone.out_channels
            self.attn1d = (
                AttentionPool1D(feat_size, dropout=dropout) if use_attention else None
            )

        elif backbone == "cnn2d":
            self.backbone = CNN2DBackbone(
                in_channels=1,
                channels=cnn_channels,
                num_blocks=cnn_blocks,
                kernel_size=3,
                dropout=dropout,
                pool=cnn2d_pool,
            )
            feat_size = self.backbone.out_channels
            self.attn2d = (
                AttentionPool2D(feat_size, dropout=dropout) if use_attention else None
            )

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_size, num_outputs)

        if task == "multiclass":
            self.criterion: nn.CrossEntropyLoss | nn.BCEWithLogitsLoss = (
                nn.CrossEntropyLoss(weight=class_weights)
            )
        elif task == "multilabel":
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        bb = self.hparams["backbone"]
        use_masking = bool(self.hparams.get("use_masking", True))

        if bb in ("rnn", "lstm", "bilstm"):
            seq, last = self.backbone(x, mask=mask, use_masking=use_masking)
            if self.use_attention and self.attn1d is not None:
                pooled, _ = self.attn1d(seq, mask=mask if use_masking else None)
            else:
                pooled = last

        elif bb == "cnn1d":
            seq, pooled = self.backbone(x)
            if self.use_attention and self.attn1d is not None:
                pooled, _ = self.attn1d(seq, mask=mask)

        elif bb == "cnn2d":
            if x.dim() == 3:
                x = x.unsqueeze(1)
            featmap, pooled = self.backbone(x)
            if self.use_attention:
                attn2d = getattr(self, "attn2d", None)
                if attn2d is not None:
                    pooled, _ = attn2d(featmap)
        else:
            raise RuntimeError("Invalid backbone configuration")

        pooled = self.dropout(pooled)
        logits = self.head(pooled)
        return logits

    def _shared_step(self, batch, stage: str) -> Tensor:
        if len(batch) == 3:
            x, y, mask = batch
        else:
            x, y = batch
            mask = None

        logits = self(x, mask)

        if self.task == "multiclass":
            loss = self.criterion(logits, y)
            preds = argmax(logits, dim=1)
            acc = (preds == y).float().mean()
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            return loss
        y = y.float()
        loss = self.criterion(logits, y)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        tp = (preds * y).sum()
        fp = (preds * (1 - y)).sum()
        fn = ((1 - preds) * y).sum()
        micro_f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            f"{stage}_micro_f1", micro_f1, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        solver = str(self.hparams["solver"]).lower()
        lr = float(self.hparams["learning_rate_init"])
        wd = float(self.hparams["weight_decay"])
        if solver == "adam":
            return optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        if solver == "adamw":
            return optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        if solver == "sgd":
            return optim.SGD(self.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        raise ValueError(f"Unsupported optimizer: {self.hparams['solver']}")
