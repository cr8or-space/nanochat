"""
CPU micro smoke-train for Multi-head Latent Attention (MLA).

This is a feasible, GPU-free sanity check that the MLA low-rank KV bottleneck plus
decoupled RoPE are *train-stable*: it trains a tiny MLA model for a few dozen steps
on a synthetic copy task and asserts the loss goes down. It is NOT a quality
benchmark — it only proves the architecture optimizes.

Run:
    PYTHONPATH=. python dev/mla_smoke_train.py

For a REAL smoke-train (needs a Hopper GPU + tokenized data), use base_train, e.g.:
    torchrun --nproc_per_node=8 scripts/base_train.py --depth 20 --use-mla --mla-preset 4x
    torchrun --nproc_per_node=8 scripts/base_train.py --depth 20 --use-mla --mla-preset 8x --fp8
"""

import torch

from nanochat.gpt import GPT, GPTConfig


def main(steps=60, use_mla=True, preset="4x"):
    torch.manual_seed(0)
    vocab_size = 128
    kv_lora_rank = {"4x": 96, "8x": 48}[preset]  # scaled down for the tiny model
    cfg = GPTConfig(
        sequence_len=64, vocab_size=vocab_size, n_layer=4, n_head=4, n_kv_head=4,
        n_embd=256, window_pattern="SSSL",
        use_mla=use_mla, kv_lora_rank=kv_lora_rank, qk_rope_head_dim=16,
    )
    model = GPT(cfg)
    model.init_weights()
    model.train()
    opt = model.setup_optimizer()

    def batch(bs=8, T=48):
        # Synthetic task: predict a shifted, offset-mixed sequence (learnable but non-trivial)
        base = torch.randint(0, vocab_size, (bs, 1))
        seq = (base + torch.arange(T + 1).view(1, -1)) % vocab_size
        return seq[:, :-1].contiguous(), seq[:, 1:].contiguous()

    print(f"MLA smoke-train (use_mla={use_mla}, preset={preset}, kv_lora_rank={kv_lora_rank})")
    first, last = None, None
    for step in range(steps):
        x, y = batch()
        loss = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
        if step % 10 == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss {loss.item():.4f}")

    print(f"first={first:.4f}  last={last:.4f}  drop={first - last:.4f}")
    assert last < first * 0.6, f"loss did not decrease enough: {first:.4f} -> {last:.4f}"
    print("SMOKE-TRAIN OK: MLA is train-stable and loss decreased.")


if __name__ == "__main__":
    main()
