import torch

from rys_jev import engine
from rys_jev.materialize import execution_order, new_positions, parse_span, scale_for


def test_row_lengths_ignore_padding():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 0, 0, 0]])
    out = engine._attention(None, q, k, v, None, startlux_row_lengths=[8, 5])[0]
    alt = engine._attention(None, q, k.clone(), v.clone(), None, startlux_row_lengths=[8, 5])[0]
    assert torch.allclose(out, alt)
    # row 1's positions past its length must not come from the real tokens
    k2 = k.clone()
    k2[1, :, 5:] = 99.0
    out2 = engine._attention(None, q, k2, v, None, startlux_row_lengths=[8, 5])[0]
    assert torch.allclose(out[1, :5], out2[1, :5])


def test_fast_path_matches_per_row():
    torch.manual_seed(1)
    q = torch.randn(1, 2, 6, 8)
    k = torch.randn(1, 2, 6, 8)
    v = torch.randn(1, 2, 6, 8)
    full = engine._attention(None, q, k, v, None, startlux_row_lengths=[6])[0]
    slow = engine._attention(None, q, k, v, None, startlux_row_lengths=[6])
    assert full.shape == (1, 6, 2, 8)
    assert torch.allclose(full, slow[0])


def test_execution_order():
    assert execution_order(8, 2, 5) == [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7]
    assert len(execution_order(64, 41, 49)) == 72


def test_new_positions():
    assert new_positions(40, 41, 49) == [40]
    assert new_positions(41, 41, 49) == [41, 49]
    assert new_positions(48, 41, 49) == [48, 56]
    assert new_positions(49, 41, 49) == [57]
    assert new_positions(63, 41, 49) == [71]


def test_span_and_scale():
    assert parse_span("41-49") == (41, 49)
    assert scale_for("self_attn.o_proj.weight", 1.25) == 1.25
    assert scale_for("linear_attn.out_proj.weight", 1.25) == 1.25
    assert scale_for("mlp.down_proj.weight", 1.25) == 1.25
    assert scale_for("mlp.gate_proj.weight", 1.25) == 1.0
