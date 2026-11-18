from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor, clamp
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

TaskType = Literal["multiclass", "multilabel"]
BackboneType = Literal["rnn", "lstm", "bilstm", "cnn1d", "cnn2d"]


class AttentionPool1D(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = self.dropout(x)
        logits = self.score(x).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(mask <= 0, -1e9)
        attn = torch.softmax(logits, dim=1)
        pooled = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        return pooled, attn


class AttentionPool2D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.score = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.dropout(x)
        logits = self.score(x)
        b, _, h, w = logits.shape
        logits_flat = logits.view(b, -1)
        attn = torch.softmax(logits_flat, dim=1).view(b, 1, h, w)
        pooled = (x * attn).sum(dim=(2, 3))
        return pooled, attn


class RNNBackbone(nn.Module):
    def __init__(
        self,
        kind: Literal["rnn", "lstm"],
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ):
        super().__init__()
        rnn_cls = nn.RNN if kind == "rnn" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

    @property
    def output_size(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        use_masking: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if mask is not None and use_masking:
            lengths = mask.sum(dim=1).long().cpu()
            lengths = clamp(lengths, min=1)
            packed = pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.rnn(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
            b = x.size(0)
            last = out[torch.arange(b, device=out.device), (lengths.to(out.device) - 1)]
        else:
            out, _ = self.rnn(x)
            last = out[:, -1, :]
        return out, last


class CNN1DBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_blocks: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        c_out = channels
        for _i in range(num_blocks):
            layers += [
                nn.Conv1d(
                    c_in,
                    c_out,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.transpose(1, 2)
        feat = self.net(x)
        seq = feat.transpose(1, 2)
        pooled = feat.mean(dim=2)
        return seq, pooled


class CNN2DBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        channels: int = 32,
        num_blocks: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
        pool: bool = True,
    ):
        super().__init__()
        layers = []
        c_in = in_channels
        c = channels
        for _i in range(num_blocks):
            layers += [
                nn.Conv2d(
                    c_in,
                    c,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
            ]
            if pool:
                layers += [nn.MaxPool2d(kernel_size=2)]
            c_in = c
        self.net = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        feat = self.net(x)
        pooled = feat.mean(dim=(2, 3))
        return feat, pooled
