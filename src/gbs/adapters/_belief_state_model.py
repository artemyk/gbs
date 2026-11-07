"""Checkpoint architecture from artemyk/belief-state (MIT, Artemy Kolchinsky).

The parameter names match upstream state dictionaries. Attention is standard
softmax attention, with RoPE or learned positions and a standard BOS token.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 2
    context: int = 32
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 1
    d_head: int = 8
    d_mlp: int = 256
    rope: bool = False

    def __post_init__(self):
        dimensions = [self.vocab, self.context, self.d_model, self.n_layers, self.n_heads, self.d_head, self.d_mlp]
        if min(dimensions) < 1 or (self.rope and self.d_head % 2):
            raise ValueError("Positive model dimensions and an even RoPE head dimension are required")


def rotate(x, positions):
    half = x.shape[-1] // 2
    if x.shape[-1] % 2:
        raise ValueError("RoPE requires an even head dimension")
    freq = 10000.0 ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = torch.as_tensor(positions, device=x.device)[..., None] * freq
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a*cos-b*sin, a*sin+b*cos], -1)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads, self.d_head, self.rope = cfg.n_heads, cfg.d_head, cfg.rope
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3*cfg.n_heads*cfg.d_head)
        self.out = nn.Linear(cfg.n_heads*cfg.d_head, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.ReLU(), nn.Linear(cfg.d_mlp, cfg.d_model))

    def forward(self, x, *, positions=None, cache=None):
        """Run a full sequence or continue a cache through the same block math.

        Keys and values have shape (batch, heads, positions, head_dim); cached
        keys already include RoPE. Return residuals, normalized inputs, and the
        extended (keys, values), preserving gradients through the supplied cache.
        """
        normalized = self.ln1(x)
        batch, length, _ = x.shape
        past = 0 if cache is None else cache[0].shape[-2]
        if positions is None:
            positions = torch.arange(past, past+length, device=x.device)
        q, k, v = self.qkv(normalized).reshape(batch, length, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = rotate(q, positions), rotate(k, positions)
        if cache is not None:
            k, v = torch.cat([cache[0], k], 2), torch.cat([cache[1], v], 2)
        mask = None
        if past:
            mask = torch.arange(past+length, device=x.device)[None] <= past+torch.arange(length, device=x.device)[:, None]
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        x = x+self.out(attention.transpose(1, 2).flatten(2))
        return x+self.mlp(self.ln2(x)), normalized, (k, v)


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab+1, cfg.d_model)
        self.pos_embed = None if cfg.rope else nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab)

    def forward(self, tokens, features=False):
        if tokens.ndim != 2 or not 1 <= tokens.shape[1] <= self.cfg.context:
            raise ValueError("Provide a nonempty token sequence within the model context")
        inputs = torch.cat([torch.full_like(tokens[:, :1], self.cfg.vocab), tokens[:, :-1]], 1)
        x = self.embed(inputs)
        if self.pos_embed is not None:
            x = x+self.pos_embed(torch.arange(tokens.shape[1], device=tokens.device))
        normalized = []
        for block in self.blocks:
            x, block_input, _ = block(x)
            normalized.append(block_input)
        final = self.ln_final(x)
        logits = self.unembed(final)
        return (logits, torch.cat(normalized+[final], -1)) if features else logits
