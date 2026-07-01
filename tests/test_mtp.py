"""
Tests for Multi-Token Prediction (MTP) and MTP-based speculative decoding.

MTP adds auxiliary position-wise heads that predict t+2, t+3, ... They sharpen the
training signal and let us draft several tokens per step for speculative decoding,
which is verified to be token-for-token identical to greedy decoding.

Run:
    python -m pytest tests/test_mtp.py -v

Runs on GPU (bf16) when available, else CPU (fp32), via the SDPA fallback on non-Hopper hardware.
"""

import pytest
import torch

from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import GPT, GPTConfig

# Run on GPU when available (matching the real model's compute dtype); else CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# bf16 accumulates more rounding than fp32, so loosen the loss-decomposition tolerance.
LOSS_REL = 1e-4 if COMPUTE_DTYPE == torch.float32 else 5e-3


def _build(use_mla=False, n_mtp=3, n_layer=4):
    mla_kw = dict(qk_rope_head_dim=16) if use_mla else {}
    cfg = GPTConfig(
        sequence_len=64, vocab_size=128, n_layer=n_layer, n_head=4, n_kv_head=4,
        n_embd=128, window_pattern="SSSL", n_mtp=n_mtp, use_mla=use_mla, **mla_kw,
    )
    model = GPT(cfg)
    model.init_weights()
    model.eval()
    return model.to(DEVICE), cfg


@pytest.mark.parametrize("use_mla", [False, True])
def test_speculative_matches_greedy(use_mla):
    """Speculative decoding must produce exactly the greedy sequence (its correctness contract)."""
    model, _ = _build(use_mla=use_mla, n_mtp=3)
    prompt = [1, 5, 9, 13, 42, 7]
    greedy = list(model.generate(prompt, max_tokens=40, temperature=0.0))
    spec = list(model.generate_speculative(prompt, max_tokens=40))
    assert greedy == spec, f"speculative diverged from greedy (use_mla={use_mla})"


def test_speculative_length_respected():
    """Speculative decoding yields exactly max_tokens tokens despite block drafting."""
    model, _ = _build(n_mtp=3)
    prompt = [1, 2, 3]
    for max_tokens in [1, 5, 17, 40]:
        out = list(model.generate_speculative(prompt, max_tokens=max_tokens))
        assert len(out) == max_tokens


def test_speculative_requires_mtp():
    """Speculative decoding is unavailable without MTP heads."""
    model, _ = _build(n_mtp=0)
    with pytest.raises(AssertionError):
        list(model.generate_speculative([1, 2, 3], max_tokens=4))


@pytest.mark.parametrize("use_mla", [False, True])
def test_mtp_loss_is_finite_and_adds_signal(use_mla):
    """Combined MTP loss is finite, and the MTP term is a positive addition to the main loss."""
    model, cfg = _build(use_mla=use_mla, n_mtp=2)
    ids = torch.randint(0, cfg.vocab_size, (2, 32), device=DEVICE)
    x, y = ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
    with torch.inference_mode():
        combined = model(x, targets=y).item()
        # recompute the main-only loss to confirm MTP contributes a strictly positive term
        logits, h0 = model(x, return_hidden=True)
        import torch.nn.functional as F
        main = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1).item()
        mtp = model._mtp_loss(x, h0, y, "mean").item()
    assert torch.isfinite(torch.tensor(combined))
    assert mtp > 0
    assert combined == pytest.approx(main + cfg.mtp_weight * mtp, rel=LOSS_REL)


@pytest.mark.slow
@pytest.mark.parametrize("use_mla", [False, True])
def test_mtp_training_reduces_loss(use_mla):
    """A few optimizer steps reduce the combined loss (MTP params are trainable via Muon)."""
    torch.manual_seed(0)
    model, cfg = _build(use_mla=use_mla, n_mtp=2)
    opt = model.setup_optimizer()
    ids = torch.randint(0, cfg.vocab_size, (4, 33), device=DEVICE)
    x, y = ids[:, :-1].contiguous(), ids[:, 1:].contiguous()
    losses = []
    for _ in range(6):
        loss = model(x, targets=y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
