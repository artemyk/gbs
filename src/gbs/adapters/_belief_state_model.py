"""Checkpoint architecture from artemyk/belief-state (MIT, Artemy Kolchinsky).

The parameter names match upstream state dictionaries. This inference model
also supports the visible-BOS, log-Z, and raw-(N,Z) attention configurations.
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
    attention_mass: bool = False
    attention_sums: bool = False
    bos_visible: bool = False

    def __post_init__(self):
        dimensions = [self.vocab, self.context, self.d_model, self.n_layers, self.n_heads, self.d_head, self.d_mlp]
        if min(dimensions) < 1 or (self.rope and self.d_head % 2):
            raise ValueError("Positive model dimensions and an even RoPE head dimension are required")
        if self.attention_sums and self.attention_mass:
            raise ValueError("Choose either normalized attention with log Z or raw numerator and denominator")


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
        self.attention_sums = cfg.attention_sums
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3*cfg.n_heads*cfg.d_head)
        self.out = nn.Linear(cfg.n_heads*cfg.d_head, cfg.d_model)
        self.mass_weight = nn.Parameter(torch.zeros(cfg.d_model, cfg.n_heads)) if cfg.attention_mass or cfg.attention_sums else None
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.ReLU(), nn.Linear(cfg.d_mlp, cfg.d_model))

    def forward(self, x, mask):
        normalized = self.ln1(x)
        batch, length, _ = x.shape
        q, k, v = self.qkv(normalized).reshape(batch, length, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        if self.rope:
            positions = torch.arange(length, device=x.device)
            q, k = rotate(q, positions), rotate(k, positions)
        if self.mass_weight is None:
            response = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            denominator = None
        else:
            scores = (q @ k.transpose(-2, -1))*self.d_head**-.5
            scores = scores.masked_fill(~mask, float('-inf'))
            if self.attention_sums:
                weights = scores.exp()
                response, denominator = weights @ v, weights.sum(-1)
            else:
                response, denominator = scores.softmax(-1) @ v, scores.logsumexp(-1)
        x = x + self.out(response.transpose(1, 2).flatten(2))
        if denominator is not None:
            x = x + F.linear(denominator.transpose(1, 2), self.mass_weight)
        return x+self.mlp(self.ln2(x)), normalized


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
        mask = torch.ones(tokens.shape[1], tokens.shape[1], dtype=torch.bool, device=tokens.device).tril()
        if not self.cfg.bos_visible:
            mask[1:, 0] = False
        normalized = []
        for block in self.blocks:
            x, block_input = block(x, mask)
            normalized.append(block_input)
        final = self.ln_final(x)
        logits = self.unembed(final)
        return (logits, torch.cat(normalized+[final], -1)) if features else logits
