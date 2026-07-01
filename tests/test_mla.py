"""
Tests for Multi-head Latent Attention (MLA) — the low-rank KV cache variant.

MLA caches only a low-rank latent (plus a small shared decoupled-RoPE key) and
reconstructs full per-head K/V on the fly, maximizing quality-per-cache-byte.

Run:
    python -m pytest tests/test_mla.py -v

These tests run on CPU via the SDPA fallback (no GPU / FA3 required).
"""

from dataclasses import asdict

import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import KVCache, MLAKVCache, Engine
from tests.test_engine import ByteTokenizer


def _build(use_mla, n_layer=4, n_head=4, head_dim=32, vocab_size=262, **mla_kw):
    cfg = GPTConfig(
        sequence_len=64, vocab_size=vocab_size, n_layer=n_layer,
        n_head=n_head, n_kv_head=n_head, n_embd=n_head * head_dim,
        window_pattern="SSSL", use_mla=use_mla, **mla_kw,
    )
    model = GPT(cfg)
    model.init_weights()
    model.eval()
    return model, cfg


def _make_cache(model, cfg, batch_size, seq_len):
    if cfg.use_mla:
        return MLAKVCache(batch_size, cfg.kv_lora_rank, cfg.qk_rope_head_dim,
                          seq_len, cfg.n_layer, torch.device("cpu"), torch.float32)
    return KVCache(batch_size, cfg.n_head, seq_len, cfg.n_embd // cfg.n_head,
                   cfg.n_layer, torch.device("cpu"), torch.float32)


def _parity_maxdiff(model, cfg):
    """Max |Δlogits| between a full forward and incremental prefill+decode."""
    T, prefill = 24, 10
    ids = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.inference_mode():
        full = model(ids)
        cache = _make_cache(model, cfg, batch_size=1, seq_len=T)
        model(ids[:, :prefill], kv_cache=cache)  # prefill
        dec = [model(ids[:, i:i+1], kv_cache=cache)[:, -1, :] for i in range(prefill, T)]
        dec = torch.stack(dec, dim=1)
    return (dec - full[:, prefill:, :]).abs().max().item()


@pytest.mark.parametrize("kv_lora_rank,qk_rope_head_dim", [(64, 16), (32, 16)])
def test_mla_prefill_decode_parity(kv_lora_rank, qk_rope_head_dim):
    """Incremental prefill+decode through the MLA cache must match a full forward."""
    model, cfg = _build(True, kv_lora_rank=kv_lora_rank, qk_rope_head_dim=qk_rope_head_dim)
    maxdiff = _parity_maxdiff(model, cfg)
    assert maxdiff < 1e-3, f"MLA parity failed: max|Δlogits|={maxdiff:.3e}"


def test_mha_parity_regression():
    """The standard MHA path must still match (guards against MLA refactor breakage)."""
    model, cfg = _build(False)
    maxdiff = _parity_maxdiff(model, cfg)
    assert maxdiff < 1e-3, f"MHA parity regressed: max|Δlogits|={maxdiff:.3e}"


def test_mla_disables_value_embeddings():
    """MLA layers must not create value embeddings (cannot be reconstructed from latent)."""
    model_mla, _ = _build(True, kv_lora_rank=64, qk_rope_head_dim=16)
    model_mha, _ = _build(False)
    assert len(model_mla.value_embeds) == 0
    assert len(model_mha.value_embeds) > 0  # sanity: MHA does use them


@pytest.mark.parametrize("preset,rank,expected_ratio", [("4x", 512, 4.0), ("8x", 256, 7.5)])
def test_mla_cache_byte_reduction(preset, rank, expected_ratio):
    """MLA cache must be materially smaller than the MHA cache at real d20 dims."""
    # Real d20-ish geometry: n_head=10, head_dim=128 -> MHA caches 2*10*128=2560 elems/tok/layer
    n_head, head_dim, n_layer, seq = 10, 128, 4, 128
    mha = KVCache(1, n_head, seq, head_dim, n_layer, torch.device("cpu"), torch.float32)
    mla = MLAKVCache(1, rank, 64, seq, n_layer, torch.device("cpu"), torch.float32)

    def nbytes(c, names):
        return sum(getattr(c, n).element_size() * getattr(c, n).nelement() for n in names)
    mha_bytes = nbytes(mha, ["k_cache", "v_cache"])
    mla_bytes = nbytes(mla, ["latent_cache", "krope_cache"])
    ratio = mha_bytes / mla_bytes
    assert ratio >= expected_ratio, f"{preset}: only {ratio:.2f}x reduction (< {expected_ratio}x)"


def test_mla_config_roundtrips_and_generates():
    """asdict->GPTConfig round-trips MLA fields, and Engine generation runs end-to-end."""
    model, cfg = _build(True, kv_lora_rank=64, qk_rope_head_dim=16)
    # Config round-trip (as base_train + checkpoint_manager do it)
    rebuilt = GPTConfig(**asdict(cfg))
    assert rebuilt.use_mla and rebuilt.kv_lora_rank == 64 and rebuilt.qk_rope_head_dim == 16

    # End-to-end batched generation via the Engine with an MLA model
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"
    results, _ = engine.generate_batch(prompt, num_samples=3, max_tokens=5, temperature=0.0)
    assert len(results) == 3
    assert all(len(r) >= len(prompt) for r in results)
