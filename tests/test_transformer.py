"""Mechanics tests: 

A shape-correct implementation can still have Q/K/V roles scrambled, layers sharing
weights, or no positional signal. Each test below pins down one property that would
break under one of those bugs
"""
import torch

from transformer import (
    MHA,
    Encoder,
    EncoderLayer,
    FullyConnected,
    attention,
    make_model,
)


def test_uniform_attention_returns_mean_of_values():
    """Identical keys -> uniform weights -> output is the plain mean of the values"""
    seq_len, d = 5, 8
    keys = torch.randn(1, d).repeat(seq_len, 1)
    values = torch.randn(seq_len, d)
    queries = torch.randn(seq_len, d)

    _, weights, out = attention(queries, keys, values)

    assert torch.allclose(weights.sum(dim=-1), torch.ones(seq_len), atol=1e-5)
    expected = values.mean(dim=0).unsqueeze(0).expand(seq_len, -1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_encoder_layers_have_independent_weights():
    """clones() must deepcopy, otherwise all N layers share one set of parameters"""
    ffn = FullyConnected(d_model=64, d_ff=256)
    attn = MHA(heads=4, d_model=64)
    enc = Encoder(EncoderLayer(ffn, attn, size=64, dropout=0.1), N=6)

    assert enc.layers[0] is not enc.layers[1]
    single = sum(p.numel() for p in enc.layers[0].parameters())
    total = sum(p.numel() for p in enc.parameters())
    assert total > 5 * single


def test_positional_encoding_breaks_permutation_invariance():
    """Same tokens in a different order must produce different encoder outputs"""
    torch.manual_seed(0)
    model = make_model(11, 11, N=2, d_model=64, d_ff=256, h=4, dropout=0.0).eval()
    src = torch.tensor([[1, 2, 3, 4, 5]])
    shuffled = torch.tensor([[1, 5, 4, 3, 2]])
    mask = torch.ones(1, 1, 1, 5) > 0

    with torch.no_grad():
        a = model.encode(src, mask)
        b = model.encode(shuffled, mask)

    assert not torch.allclose(a, b, atol=1e-4)
