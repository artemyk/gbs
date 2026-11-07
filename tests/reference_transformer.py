"""Small independent native forward pass used to validate the adapter."""
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class Config:
    vocab: int = 3
    context: int = 12
    d_model: int = 16
    n_layers: int = 2
    n_heads: int = 2
    d_head: int = 4
    d_mlp: int = 24
    rope: bool = True
    bos_visible: bool = False


def rotary(x, positions):
    half = x.shape[-1]//2
    angles = torch.as_tensor(positions, device=x.device)[..., None] * (
        10000. ** (-torch.arange(half, device=x.device, dtype=torch.float32)/half))
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first*angles.cos()-second*angles.sin(), first*angles.sin()+second*angles.cos()], -1)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3*cfg.n_heads*cfg.d_head)
        self.out = nn.Linear(cfg.n_heads*cfg.d_head, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_mlp), nn.ReLU(), nn.Linear(cfg.d_mlp, cfg.d_model))


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab+1, cfg.d_model)
        self.pos_embed = None if cfg.rope else nn.Embedding(cfg.context, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab)

    def forward(self, tokens):
        cfg = self.cfg
        inputs = torch.cat([torch.full_like(tokens[:, :1], cfg.vocab), tokens[:, :-1]], 1)
        x = self.embed(inputs)
        n = x.shape[1]
        pos = torch.arange(n)
        if self.pos_embed is not None: x = x+self.pos_embed(pos)
        mask = torch.ones(n, n, dtype=torch.bool).tril()
        if not cfg.bos_visible: mask[1:, 0] = False
        for block in self.blocks:
            q, k, v = block.qkv(block.ln1(x)).reshape(len(x), n, 3, cfg.n_heads, cfg.d_head).permute(2, 0, 3, 1, 4)
            if cfg.rope: q, k = rotary(q, pos), rotary(k, pos)
            score = (q @ k.transpose(-2, -1))/cfg.d_head**.5
            score = score.masked_fill(~mask, -torch.inf)
            result = score.softmax(-1) @ v
            x = x+block.out(result.transpose(1, 2).reshape(len(x), n, -1))
            x = x+block.mlp(block.ln2(x))
        return self.unembed(self.ln_final(x))


def make_model(cfg):
    """Production blocks initialized with the independent reference's weights."""
    from gbs.adapters._belief_state_model import Config as ModelConfig, Transformer as Model
    reference = Transformer(cfg)
    with torch.random.fork_rng(devices=[]):
        model = Model(ModelConfig(**asdict(cfg)))
    model.load_state_dict(reference.state_dict())
    return model
