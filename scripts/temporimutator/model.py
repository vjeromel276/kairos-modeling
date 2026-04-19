"""
Temporimutator model definition (Phase 2).

Architecture (from research plan lines 82-109):
    Input (batch, 252, 4)
        -> Linear projection to d_model=64
        -> + Learnable positional encoding (not sinusoidal)
        -> TransformerEncoder: 3 layers, 4 heads, d_ff=256, dropout=0.1
        -> Global average pool over sequence dim -> (batch, 64)
        -> Concatenate vol_scalar -> (batch, 65)
        -> Direction head: Linear(65,32) -> ReLU -> Linear(32,3)  (logits, softmax at eval)
        -> Force head:     Linear(65,32) -> ReLU -> Linear(32,1)  (raw)

Notes:
  * The direction head returns raw logits so downstream code can pick between
    softmax (inference) and CrossEntropyLoss (training).
  * Attention weights are available via `forward(return_attn=True)` — uses
    `need_weights=True` on each encoder layer's self-attention.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TMConfig:
    seq_len: int = 252
    n_streams: int = 4
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    head_hidden: int = 32
    n_direction_classes: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


class TemporimutatorModel(nn.Module):
    """See module docstring."""

    def __init__(self, cfg: TMConfig | None = None):
        super().__init__()
        self.cfg = cfg or TMConfig()
        c = self.cfg

        self.input_proj = nn.Linear(c.n_streams, c.d_model)
        # Learnable positional encoding: one d_model vector per sequence position
        self.pos_embed = nn.Parameter(torch.zeros(1, c.seq_len, c.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.d_ff,
            dropout=c.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=c.n_layers)

        pooled_dim = c.d_model + 1  # +1 for the appended vol_scalar
        self.dir_head = nn.Sequential(
            nn.Linear(pooled_dim, c.head_hidden),
            nn.ReLU(),
            nn.Linear(c.head_hidden, c.n_direction_classes),
        )
        self.force_head = nn.Sequential(
            nn.Linear(pooled_dim, c.head_hidden),
            nn.ReLU(),
            nn.Linear(c.head_hidden, 1),
        )

    def forward(
        self,
        sequences: torch.Tensor,
        vol_scalar: torch.Tensor,
        return_attn: bool = False,
    ):
        """
        sequences:  (B, seq_len, n_streams) float
        vol_scalar: (B, 1) or (B,) float
        returns:
            dir_logits: (B, 3)
            force:      (B,)
            attn_weights (optional): list of (B, n_heads, seq_len, seq_len)
                                     one entry per encoder layer.
        """
        if vol_scalar.dim() == 1:
            vol_scalar = vol_scalar.unsqueeze(-1)

        x = self.input_proj(sequences) + self.pos_embed  # (B, seq_len, d_model)

        if return_attn:
            # Replicate the encoder manually to capture attention weights.
            attn_list = []
            for layer in self.encoder.layers:
                # Pre-LN path matching TransformerEncoderLayer(norm_first=True)
                # with need_weights=True on self-attention.
                normed = layer.norm1(x)
                attn_out, attn_w = layer.self_attn(
                    normed, normed, normed,
                    need_weights=True, average_attn_weights=False,
                )
                x = x + layer.dropout1(attn_out)
                x2 = layer.norm2(x)
                x = x + layer.dropout2(
                    layer.linear2(layer.dropout(F.gelu(layer.linear1(x2))))
                )
                attn_list.append(attn_w)
            if self.encoder.norm is not None:
                x = self.encoder.norm(x)
        else:
            x = self.encoder(x)

        pooled = x.mean(dim=1)                               # (B, d_model)
        pooled = torch.cat([pooled, vol_scalar], dim=-1)     # (B, d_model + 1)

        dir_logits = self.dir_head(pooled)                   # (B, 3)
        force = self.force_head(pooled).squeeze(-1)          # (B,)

        if return_attn:
            return dir_logits, force, attn_list
        return dir_logits, force

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
