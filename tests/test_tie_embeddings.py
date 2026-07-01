"""
Tests for the opt-in tie_embeddings flag: wte and lm_head share one weight matrix.
Covers the param-accounting (both count asserts), the optimizer grouping, and that a
tied model trains — including composition with MLA/MTP.

Runs on GPU (bf16) when available, else CPU (fp32).
"""

import pytest
import torch
from dataclasses import asdict

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build(tie, **kw):
    cfg = GPTConfig(sequence_len=64, vocab_size=512, n_layer=4, n_head=4, n_kv_head=4,
                    n_embd=256, window_pattern="SSSL", tie_embeddings=tie, **kw)
    model = GPT(cfg)
    model.init_weights()
    model.eval()
    return model.to(DEVICE), cfg


def test_tied_shares_one_tensor():
    model, _ = _build(True)
    assert model.lm_head.weight is model.transformer.wte.weight
    untied, _ = _build(False)
    assert untied.lm_head.weight is not untied.transformer.wte.weight


def test_tying_saves_exactly_one_embedding_matrix():
    tied, cfg = _build(True)
    untied, _ = _build(False)
    saved = sum(p.numel() for p in untied.parameters()) - sum(p.numel() for p in tied.parameters())
    # padded vocab (multiple of 64) times n_embd
    padded_vocab = untied.lm_head.weight.shape[0]
    assert saved == padded_vocab * cfg.n_embd


@pytest.mark.parametrize("kw", [{}, dict(n_mtp=1), dict(use_mla=True, qk_rope_head_dim=16)])
def test_num_scaling_params_and_total_consistent(kw):
    """The exhaustive-total assert inside num_scaling_params must hold when tied (no double count)."""
    model, _ = _build(True, **kw)
    pc = model.num_scaling_params()  # asserts total == sum(params) internally
    assert pc['total'] == sum(p.numel() for p in model.parameters())


def test_optimizer_drops_the_lm_head_group_when_tied():
    tied, _ = _build(True)
    untied, _ = _build(False)
    # tied has one fewer AdamW group (no separate unembedding group)
    assert len(tied.setup_optimizer().param_groups) == len(untied.setup_optimizer().param_groups) - 1


@pytest.mark.parametrize("kw", [{}, dict(n_mtp=1), dict(use_mla=True, qk_rope_head_dim=16)])
def test_tied_model_trains(kw):
    model, cfg = _build(True, **kw)
    ids = torch.randint(0, cfg.vocab_size, (2, 16), device=DEVICE)
    tgt = torch.randint(0, cfg.vocab_size, (2, 16), device=DEVICE)
    model.train()
    loss = model(ids, targets=tgt)
    loss.backward()
    assert torch.isfinite(loss).item()
    # shared weight received a gradient
    assert model.transformer.wte.weight.grad is not None


def test_config_round_trips():
    _, cfg = _build(True)
    assert GPTConfig(**asdict(cfg)).tie_embeddings is True


@pytest.mark.parametrize("kw", [{}, dict(n_mtp=1), dict(use_mla=True, qk_rope_head_dim=16)])
def test_tie_survives_meta_materialization(kw):
    """base_train builds on meta then to_empty()s — which severs shared storage. init_weights must
    re-tie, else num_scaling_params' exhaustive-total assert fires (regression: the pretrain smoke)."""
    cfg = GPTConfig(sequence_len=64, vocab_size=512, n_layer=4, n_head=4, n_kv_head=4,
                    n_embd=256, window_pattern="SSSL", tie_embeddings=True, **kw)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=DEVICE)   # fresh per-param storage, breaks the __init__ tie
    model.init_weights()            # must re-establish it
    assert model.lm_head.weight is model.transformer.wte.weight
    pc = model.num_scaling_params()  # asserts total == sum(params) internally
    assert pc['total'] == sum(p.numel() for p in model.parameters())
