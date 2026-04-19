"""Unit tests for scripts/temporimutator/model.py."""

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.temporimutator.model import TMConfig, TemporimutatorModel


@pytest.fixture
def model():
    torch.manual_seed(0)
    return TemporimutatorModel(TMConfig()).eval()


def test_forward_output_shapes(model):
    B = 7
    seqs = torch.randn(B, 252, 4)
    vs = torch.rand(B, 1) * 0.1
    dir_logits, force = model(seqs, vs)
    assert dir_logits.shape == (B, 3)
    assert force.shape == (B,)


def test_forward_accepts_1d_vol_scalar(model):
    B = 3
    seqs = torch.randn(B, 252, 4)
    vs = torch.rand(B)  # 1-D
    dir_logits, force = model(seqs, vs)
    assert dir_logits.shape == (B, 3)
    assert force.shape == (B,)


def test_return_attn_shapes(model):
    B = 2
    seqs = torch.randn(B, 252, 4)
    vs = torch.rand(B, 1)
    dir_logits, force, attn = model(seqs, vs, return_attn=True)
    cfg = model.cfg
    assert len(attn) == cfg.n_layers
    for w in attn:
        # (B, heads, seq_len, seq_len) when average_attn_weights=False
        assert w.shape == (B, cfg.n_heads, cfg.seq_len, cfg.seq_len)


def test_parameter_count_reasonable(model):
    counts = model.count_parameters()
    # Spec says ~500k params; we allow a generous range.
    assert 100_000 < counts["total"] < 1_000_000, counts
    assert counts["total"] == counts["trainable"]


def test_gradient_flow_nonzero():
    torch.manual_seed(0)
    m = TemporimutatorModel(TMConfig()).train()
    B = 4
    seqs = torch.randn(B, 252, 4, requires_grad=False)
    vs = torch.rand(B, 1)
    targets_dir = torch.randint(0, 3, (B,))
    targets_force = torch.rand(B)

    dir_logits, force = m(seqs, vs)
    loss = torch.nn.functional.cross_entropy(dir_logits, targets_dir) \
         + torch.nn.functional.mse_loss(force, targets_force)
    loss.backward()

    # Every trainable param that participated should have a grad with some energy.
    zero_grads = [n for n, p in m.named_parameters()
                  if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0.0)]
    assert not zero_grads, f"params with zero gradient: {zero_grads[:5]}"


def test_deterministic_with_seed():
    torch.manual_seed(42)
    m1 = TemporimutatorModel(TMConfig()).eval()
    torch.manual_seed(42)
    m2 = TemporimutatorModel(TMConfig()).eval()
    seqs = torch.randn(3, 252, 4)
    vs = torch.rand(3, 1)
    d1, f1 = m1(seqs, vs)
    d2, f2 = m2(seqs, vs)
    assert torch.allclose(d1, d2)
    assert torch.allclose(f1, f2)
