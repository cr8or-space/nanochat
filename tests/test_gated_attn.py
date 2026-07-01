"""
Tests for gated attention (Qwen3-Next): a per-element sigmoid gate on the attention
output. It must compose cleanly with MHA, MLA, and MTP speculative decoding.

Run:
    python -m pytest tests/test_gated_attn.py -v

Runs on CPU via the SDPA fallback (no GPU / FA3 required).
"""

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache, MLAKVCache


def _build(use_mla=False, use_gated_attn=True, n_mtp=0):
    mla_kw = dict(qk_rope_head_dim=16) if use_mla else {}
    cfg = GPTConfig(
        sequence_len=64, vocab_size=128, n_layer=4, n_head=4, n_kv_head=4, n_embd=128,
        window_pattern="SSSL", use_gated_attn=use_gated_attn, use_mla=use_mla, n_mtp=n_mtp, **mla_kw,
    )
    model = GPT(cfg)
    model.init_weights()
    model.eval()
    return model, cfg


def _make_cache(cfg, seq_len):
    if cfg.use_mla:
        return MLAKVCache(1, cfg.kv_lora_rank, cfg.qk_rope_head_dim, seq_len, cfg.n_layer,
                          torch.device("cpu"), torch.float32)
    return KVCache(1, cfg.n_kv_head, seq_len, cfg.n_embd // cfg.n_head, cfg.n_layer,
                   torch.device("cpu"), torch.float32)


def test_gate_is_created_only_when_enabled():
    on, _ = _build(use_gated_attn=True)
    off, _ = _build(use_gated_attn=False)
    assert all(b.attn.attn_gate is not None for b in on.transformer.h)
    assert all(b.attn.attn_gate is None for b in off.transformer.h)


@pytest.mark.parametrize("use_mla", [False, True])
def test_gated_attn_prefill_decode_parity(use_mla):
    """Gated attention must preserve prefill+decode == full-forward parity (MHA and MLA)."""
    model, cfg = _build(use_mla=use_mla, use_gated_attn=True)
    T, prefill = 24, 10
    ids = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.inference_mode():
        full = model(ids)
        cache = _make_cache(cfg, T)
        model(ids[:, :prefill], kv_cache=cache)
        dec = torch.stack([model(ids[:, i:i+1], kv_cache=cache)[:, -1, :] for i in range(prefill, T)], dim=1)
    maxdiff = (dec - full[:, prefill:, :]).abs().max().item()
    assert maxdiff < 1e-3, f"gated-attn parity failed (use_mla={use_mla}): {maxdiff:.3e}"


@pytest.mark.parametrize("use_mla", [False, True])
def test_gated_attn_composes_with_speculative(use_mla):
    """Gated attention + MTP: speculative decoding still matches greedy exactly."""
    model, _ = _build(use_mla=use_mla, use_gated_attn=True, n_mtp=3)
    prompt = [1, 5, 9, 13, 42, 7]
    greedy = list(model.generate(prompt, max_tokens=32, temperature=0.0))
    spec = list(model.generate_speculative(prompt, max_tokens=32))
    assert greedy == spec
