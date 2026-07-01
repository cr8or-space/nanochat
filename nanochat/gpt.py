"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Multi-head Latent Attention (MLA, DeepSeek-V2/V3): cache a low-rank KV latent
    # instead of full per-head K/V to maximize quality-per-cache-byte. Opt-in; when
    # False the model is the standard MHA/GQA model and none of the fields below apply.
    use_mla: bool = False
    kv_lora_rank: int = 512   # dim of the compressed KV latent (c_KV) that gets cached
    qk_rope_head_dim: int = 64 # decoupled-RoPE dim (shared across heads, also cached)
    q_lora_rank: int = 0      # optional query compression (0 = uncompressed; query is not cached)
    # Multi-Token Prediction (MTP, DeepSeek-V3 / Qwen3-Next): auxiliary heads that predict
    # tokens t+2, t+3, ... to sharpen the training signal and enable speculative decoding.
    # Opt-in; 0 disables it. Modules are position-wise (no attention) so a single-token draft
    # at inference is numerically identical to the training-time computation.
    n_mtp: int = 0            # number of extra prediction depths (0 = disabled)
    mtp_weight: float = 0.3   # weight of the averaged MTP loss relative to the main loss
    # Gated attention (Qwen3-Next): a per-element sigmoid gate on the attention output
    # (before the output projection), computed from the block input. Improves training
    # stability; opt-in. Composes with MHA, GQA and MLA.
    use_gated_attn: bool = False
    # YaRN RoPE context extension (Qwen3): NTK-by-parts frequency interpolation to run at
    # sequences longer than trained. rope_scaling is the extension factor s (1.0 = disabled);
    # rope_original_seq_len is the original context length (0 = use sequence_len). The YaRN
    # attention-temperature (mscale) term is intentionally omitted: it is nullified by QK-norm.
    rope_scaling: float = 1.0
    rope_original_seq_len: int = 0


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.use_mla = config.use_mla
        if self.use_mla:
            # Multi-head Latent Attention: K/V are reconstructed from a low-rank latent.
            # We keep per-head q/k/v dims == head_dim (splitting RoPE within head_dim) so
            # the FA3 uniform-head-dim fast path and c_q/c_proj shapes are unchanged.
            self.qk_rope_head_dim = config.qk_rope_head_dim
            self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
            assert self.qk_nope_head_dim > 0, "qk_rope_head_dim must be < head_dim"
            self.v_head_dim = self.head_dim
            self.kv_lora_rank = config.kv_lora_rank
            self.q_lora_rank = config.q_lora_rank
            if self.q_lora_rank > 0:
                self.q_down = Linear(self.n_embd, self.q_lora_rank, bias=False)
                self.q_up = Linear(self.q_lora_rank, self.n_head * self.head_dim, bias=False)
            else:
                self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
            # down-projection produces [c_KV (latent) | k_rope (shared decoupled-RoPE key)]
            self.kv_down = Linear(self.n_embd, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
            # up-projection reconstructs per-head [k_nope | v] from the latent
            self.kv_up = Linear(self.kv_lora_rank, self.n_head * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
            self.c_proj = Linear(self.n_head * self.v_head_dim, self.n_embd, bias=False)
            # Value embeddings are disabled on MLA layers: their per-token contribution to V
            # cannot be reconstructed from a latent-only cache at decode time. (Folding them
            # into the latent is a natural follow-up.)
            self.ve_gate = None
        else:
            self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
            self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
            self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
            self.ve_gate_channels = 12
            self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        # Gated attention (Qwen3-Next): per-element sigmoid gate on the attention output
        self.attn_gate = Linear(self.n_embd, self.n_head * self.head_dim, bias=False) if config.use_gated_attn else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if self.use_mla:
            return self._forward_mla(x, cos_sin, window_size, kv_cache)
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        if self.attn_gate is not None:
            y = y * torch.sigmoid(self.attn_gate(x))  # gated attention
        y = self.c_proj(y)
        return y

    def _forward_mla(self, x, cos_sin, window_size, kv_cache):
        """Multi-head Latent Attention forward (training and inference).

        Only the low-rank latent c_KV and the shared decoupled-RoPE key k_rope are
        cached; full per-head K and V are reconstructed on the fly via kv_up. The
        RoPE'd k_rope is stored post-rotation so history never needs re-roping.
        """
        B, T, C = x.size()
        cos, sin = cos_sin  # sized to qk_rope_head_dim for MLA models

        # Queries: (B, T, H, head_dim); RoPE applied only to the trailing rope slice
        q = (self.q_up(self.q_down(x)) if self.q_lora_rank > 0 else self.c_q(x))
        q = q.view(B, T, self.n_head, self.head_dim)
        q_nope, q_rope = q[..., :self.qk_nope_head_dim], q[..., self.qk_nope_head_dim:]
        q_rope = apply_rotary_emb(q_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # KV down-projection -> [c_KV (latent) | k_rope (shared, single head)]
        kv = self.kv_down(x)
        c_kv, k_rope = kv[..., :self.kv_lora_rank], kv[..., self.kv_lora_rank:]
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), cos, sin)  # (B, T, 1, rope_dim)

        # Cache the latent + roped shared key, read back the full history
        if kv_cache is not None:
            c_kv, k_rope = kv_cache.update_mla(self.layer_idx, c_kv, k_rope)
        Tk = c_kv.size(1)

        # Up-project the full latent history -> per-head [k_nope | v]
        kv_up = self.kv_up(c_kv).view(B, Tk, self.n_head, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_up[..., :self.qk_nope_head_dim], kv_up[..., self.qk_nope_head_dim:]
        # Assemble full keys: per-head content dims + the shared rope key broadcast to all heads
        k_rope_b = k_rope.expand(B, Tk, self.n_head, self.qk_rope_head_dim)
        k = torch.cat([k_nope, k_rope_b], dim=-1)

        q, k = norm(q), norm(k)  # QK norm, matching the MHA path
        q = q * 1.2
        k = k * 1.2

        # q spans the current T tokens; k/v span the full history (Tk). causal + window
        # masking is handled by flash_attn_func (bottom-right aligned when T != Tk).
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

        if kv_cache is not None and self.layer_idx == kv_cache.n_layers - 1:
            kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        if self.attn_gate is not None:
            y = y * torch.sigmoid(self.attn_gate(x))  # gated attention
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer) and not config.use_mla})
        # Multi-Token Prediction heads: depth d takes [norm(h^{d-1}) ; norm(emb(t_{i+d}))],
        # projects to n_embd, runs an MLP, then the shared lm_head. Position-wise (no attention)
        # so train-time and single-token draft-time computations are identical. Embedding (wte)
        # and output head (lm_head) are shared with the main model.
        self.mtp_proj = nn.ModuleList([Linear(2 * config.n_embd, config.n_embd, bias=False) for _ in range(config.n_mtp)])
        self.mtp_mlp = nn.ModuleList([MLP(config) for _ in range(config.n_mtp)])
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            attn = block.attn
            if attn.use_mla:
                if attn.q_lora_rank > 0:
                    torch.nn.init.uniform_(attn.q_down.weight, -s, s)
                    torch.nn.init.uniform_(attn.q_up.weight, -s, s)
                else:
                    torch.nn.init.uniform_(attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(attn.kv_down.weight, -s, s) # weights use Uniform to avoid outliers
                torch.nn.init.uniform_(attn.kv_up.weight, -s, s)
            else:
                torch.nn.init.uniform_(attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
                torch.nn.init.uniform_(attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(attn.c_proj.weight) # projections are zero
            if attn.attn_gate is not None:
                torch.nn.init.uniform_(attn.attn_gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # MTP heads: same init family as the transformer blocks
        for proj in self.mtp_proj:
            torch.nn.init.uniform_(proj.weight, -s, s)
        for mlp in self.mtp_mlp:
            torch.nn.init.uniform_(mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings. For MLA, RoPE acts only on the decoupled rope slice
        # (qk_rope_head_dim); otherwise it acts on the full head_dim.
        head_dim = self.config.n_embd // self.config.n_head
        rotary_dim = self.config.qk_rope_head_dim if self.config.use_mla else head_dim
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, rotary_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _rope_inv_freq(self, head_dim, base, device):
        """RoPE inverse frequencies, with optional YaRN (NTK-by-parts) interpolation.

        When rope_scaling <= 1 this is the standard 1 / base^(2i/d). When rope_scaling = s > 1,
        high-frequency dims (short wavelength) are left unchanged (extrapolated) while
        low-frequency dims are interpolated toward inv_freq / s, with a linear ramp between —
        letting the model run at s x its trained context length with minimal degradation.
        """
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))  # (head_dim // 2,)
        scale = self.config.rope_scaling
        if scale is None or scale <= 1.0:
            return inv_freq
        import math
        L = self.config.rope_original_seq_len or self.config.sequence_len
        beta_fast, beta_slow = 32.0, 1.0  # standard YaRN band edges (rotations over L)
        def correction_dim(num_rot):
            return (head_dim * math.log(L / (num_rot * 2 * math.pi))) / (2 * math.log(base))
        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), head_dim // 2 - 1)
        high = max(high, low + 1e-3)  # avoid div-by-zero when low == high
        # ramp in [0,1] over the half-dim; 1 => extrapolate (keep), 0 => interpolate (/s)
        ramp = (torch.arange(head_dim // 2, dtype=torch.float32, device=device) - low) / (high - low)
        extrapolation_factor = 1.0 - torch.clamp(ramp, 0.0, 1.0)
        inv_freq_interp = inv_freq / scale
        return inv_freq_interp * (1.0 - extrapolation_factor) + inv_freq * extrapolation_factor

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        inv_freq = self._rope_inv_freq(head_dim, base, device)
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups. MTP projection/MLP weights are matmuls,
        # so they join the Muon matrix group alongside the transformer blocks.
        matrix_params = list(self.transformer.h.parameters()) + list(self.mtp_proj.parameters()) + list(self.mtp_mlp.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Value embeddings group (empty when use_mla, which disables value embeddings)
        if value_embeds_params:
            param_groups.append(dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01))
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _apply_head(self, x_normed):
        """Shared output head: lm_head, crop padding, fp32, logit softcap. Input must be normed."""
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x_normed)
        logits = logits[..., :self.config.vocab_size] # slice to remove vocab padding
        logits = logits.float() # fp32 for softcap and loss
        logits = softcap * torch.tanh(logits / softcap)
        return logits

    def _mtp_step(self, depth, h_prev, tok):
        """One MTP module: combine previous hidden with the next token's embedding.
        h_prev: (B, S, n_embd), tok: (B, S) token ids -> returns h^depth: (B, S, n_embd)."""
        emb = self.transformer.wte(tok).to(h_prev.dtype)
        hp = self.mtp_proj[depth](torch.cat([norm(h_prev), norm(emb)], dim=-1))
        return hp + self.mtp_mlp[depth](norm(hp))

    @staticmethod
    def _shift_left(t, s, fill):
        """out[:, i] = t[:, i+s], with fill for i+s out of range."""
        B, T = t.shape
        out = torch.full_like(t, fill)
        if s < T:
            out[:, :T - s] = t[:, s:]
        return out

    def _mtp_loss(self, idx, h0, main_targets, loss_reduction):
        """Averaged cross-entropy of the MTP depths (predicting t+2, t+3, ...)."""
        losses = []
        h_prev = h0
        for d in range(1, self.config.n_mtp + 1):
            tok = self._shift_left(idx, d, fill=0)          # emb of t_{i+d} (masked positions unused)
            h = self._mtp_step(d - 1, h_prev, tok)
            logits_d = self._apply_head(norm(h))
            tgt = self._shift_left(idx, d + 1, fill=-1)      # target t_{i+d+1}
            tgt = tgt.masked_fill(main_targets == -1, -1)    # respect the main target's masking
            losses.append(F.cross_entropy(logits_d.view(-1, logits_d.size(-1)), tgt.view(-1),
                                          ignore_index=-1, reduction=loss_reduction))
            h_prev = h
        return torch.stack(losses).mean()

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_hidden=False):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
                # Mid-sequence prefill (e.g. speculative verify): position 0 continues from the
                # last committed token, so smear it too. (At the initial prompt prefill pos==0,
                # prev_embedding is None and position 0 correctly gets no smear.)
                if x_pre_smear is not None:
                    g0 = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :1, :24]))
                    x = torch.cat([x[:, :1] + g0 * x_pre_smear, x[:, 1:]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        h0 = x  # trunk hidden (pre final-norm), consumed by the MTP heads
        x = norm(x)

        # Forward the lm_head (compute logits)
        logits = self._apply_head(x) # (B, T, vocab_size)

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            if self.config.n_mtp > 0:
                loss = loss + self.config.mtp_weight * self._mtp_loss(idx, h0, targets, loss_reduction)
            return loss
        else:
            # inference: return logits (and optionally the trunk hidden for MTP drafting)
            return (logits, h0) if return_hidden else logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

    @torch.inference_mode()
    def generate_speculative(self, tokens, max_tokens):
        """Greedy self-speculative decoding using the MTP heads (batch size 1).

        Each step: one main forward proposes t+1 and gives the trunk hidden; the MTP heads
        chain a draft of the next n_mtp tokens; a single verify forward over the draft accepts
        the longest greedy-matching prefix (plus one bonus token when the whole draft matches).
        Output is token-for-token identical to greedy generate(); MTP quality only affects speed.
        Yields ints. Requires n_mtp > 0.
        """
        assert self.config.n_mtp > 0, "speculative decoding requires n_mtp > 0"
        from nanochat.engine import KVCache, MLAKVCache
        device = self.get_device()
        cfg = self.config
        L = cfg.n_mtp + 1  # draft block length (main token + n_mtp speculative)
        capacity = len(tokens) + max_tokens + L + 1
        if cfg.use_mla:
            cache = MLAKVCache(1, cfg.kv_lora_rank, cfg.qk_rope_head_dim, capacity, cfg.n_layer, device, COMPUTE_DTYPE)
        else:
            cache = KVCache(1, cfg.n_kv_head, capacity, cfg.n_embd // cfg.n_head, cfg.n_layer, device, COMPUTE_DTYPE)

        def emb_normed(tok_id):
            e = self.transformer.wte(torch.tensor([[tok_id]], device=device)).to(COMPUTE_DTYPE)
            return norm(e)

        next_input = torch.tensor([tokens], dtype=torch.long, device=device)
        emitted = 0
        while emitted < max_tokens:
            # 1) Advance context; get the greedy next token and the trunk hidden for drafting
            logits, h0 = self.forward(next_input, kv_cache=cache, return_hidden=True)
            committed = cache.get_pos()
            m1 = int(logits[0, -1].argmax())
            # 2) Draft t+2..t+n_mtp+1 by chaining the MTP heads
            draft = [m1]
            h_prev = h0[:, -1:, :]
            tok = torch.tensor([[m1]], dtype=torch.long, device=device)
            for d in range(cfg.n_mtp):
                h = self._mtp_step(d, h_prev, tok)
                nt = int(self._apply_head(norm(h))[0, -1].argmax())
                draft.append(nt)
                tok = torch.tensor([[nt]], dtype=torch.long, device=device)
                h_prev = h
            # 3) Verify the whole draft block in one main forward
            gv = self.forward(torch.tensor([draft], dtype=torch.long, device=device), kv_cache=cache)
            accepted = [m1]
            j = 0
            while j < L - 1:
                v = int(gv[0, j].argmax())
                if v == draft[j + 1]:
                    accepted.append(draft[j + 1]); j += 1
                else:
                    accepted.append(v); break
            else:
                accepted.append(int(gv[0, L - 1].argmax()))  # whole draft matched -> bonus token
            # 4) Roll the cache back to the accepted context length. The last accepted token
            #    (a replacement or bonus) has no cached KV yet, so it becomes the next input.
            n_ctx = len(accepted) - 1  # draft tokens kept as real context (m1..draft[j])
            cache.cache_seqlens.fill_(committed + n_ctx)
            cache.prev_embedding = emb_normed(accepted[-2])  # fix smear state after rollback
            for t in accepted:
                if emitted >= max_tokens:
                    break
                yield t
                emitted += 1
            next_input = torch.tensor([[accepted[-1]]], dtype=torch.long, device=device)
