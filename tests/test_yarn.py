"""
Tests for YaRN RoPE context extension (NTK-by-parts frequency interpolation).

Run:
    python -m pytest tests/test_yarn.py -v

Runs on CPU (no GPU / FA3 required).
"""

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache, MLAKVCache

BASE = 100000
DEV = torch.device("cpu")


def _build(rope_scaling=1.0, rope_original_seq_len=0, use_mla=False):
    mla_kw = dict(qk_rope_head_dim=16) if use_mla else {}
    cfg = GPTConfig(
        sequence_len=64, vocab_size=128, n_layer=4, n_head=4, n_kv_head=4, n_embd=128,
        window_pattern="SSSL", rope_scaling=rope_scaling,
        rope_original_seq_len=rope_original_seq_len, use_mla=use_mla, **mla_kw,
    )
    model = GPT(cfg)
    model.init_weights()
    model.eval()
    return model, cfg


def _baseline_inv_freq(head_dim):
    ch = torch.arange(0, head_dim, 2, dtype=torch.float32)
    return 1.0 / (BASE ** (ch / head_dim))


def test_yarn_disabled_is_noop():
    """rope_scaling <= 1 must leave inv_freq exactly equal to standard RoPE."""
    model, _ = _build(rope_scaling=1.0)
    head_dim = 32
    got = model._rope_inv_freq(head_dim, BASE, DEV)
    assert torch.allclose(got, _baseline_inv_freq(head_dim), atol=0, rtol=0)


def test_yarn_interpolates_low_frequencies_only():
    """High-freq dims are kept (extrapolated); low-freq dims are interpolated toward /scale."""
    scale = 8.0
    model, _ = _build(rope_scaling=scale, rope_original_seq_len=64)
    head_dim = 32
    base = _baseline_inv_freq(head_dim)
    got = model._rope_inv_freq(head_dim, BASE, DEV)
    # highest-frequency dim (index 0) is unchanged
    assert torch.allclose(got[0], base[0])
    # lowest-frequency dim is fully interpolated (divided by scale)
    assert torch.allclose(got[-1], base[-1] / scale, rtol=1e-5)
    # interpolation never increases any frequency
    assert torch.all(got <= base + 1e-9)


@pytest.mark.parametrize("use_mla", [False, True])
def test_yarn_forward_parity(use_mla):
    """With YaRN active, prefill+decode still matches a full forward (MHA and MLA)."""
    model, cfg = _build(rope_scaling=4.0, rope_original_seq_len=32, use_mla=use_mla)
    T, prefill = 24, 10
    ids = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.inference_mode():
        full = model(ids)
        if use_mla:
            cache = MLAKVCache(1, cfg.kv_lora_rank, cfg.qk_rope_head_dim, T, cfg.n_layer, DEV, torch.float32)
        else:
            cache = KVCache(1, cfg.n_kv_head, T, cfg.n_embd // cfg.n_head, cfg.n_layer, DEV, torch.float32)
        model(ids[:, :prefill], kv_cache=cache)
        dec = torch.stack([model(ids[:, i:i+1], kv_cache=cache)[:, -1, :] for i in range(prefill, T)], dim=1)
    assert (dec - full[:, prefill:, :]).abs().max().item() < 1e-3
