"""Adapter for the architecture in github.com/artemyk/belief-state."""
from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn.functional as F

from .base import CacheModel
from ._belief_state_model import Config, Transformer, attention_projection, rotate
from ..data import prepare_histories


class BeliefStateAdapter(CacheModel):
    """Pre-LayerNorm transformer with configurable vocabulary, size, heads, and BOS.

    State layout: flattened (position, layer, head, key/value, head coordinate),
    current-token coordinates (including BOS), and an explicit history length.
    Length is fixed by the encoder, never inferred from a relaxed token vector.
    This handles empty history separately from the first real token correctly.
    """
    def __init__(self, model):
        super().__init__()
        cfg = model.cfg
        if any(getattr(cfg, name, False) for name in ("attention_mass", "attention_sums")):
            raise ValueError("Only standard softmax attention is supported")
        self.model = model.eval().requires_grad_(False)
        self.vocab_size, self.n_layers = cfg.vocab, cfg.n_layers
        self.n_heads, self.head_dim = cfg.n_heads, cfg.d_head
        self.token_features = cfg.vocab + 1
        self.max_history = cfg.context
        self.bos_visible = getattr(cfg, "bos_visible", False)
        self.rope = cfg.rope
        self.entry_width = self.n_layers*self.n_heads*2*self.head_dim
        if self.dtype not in [torch.float32, torch.float64]:
            raise ValueError("This adapter currently supports float32 and float64 teachers")

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device="cpu"):
        """Load upstream's config + model state dictionary; no source checkout needed.

        The checkpoint contains tensors and ordinary Python data and is loaded
        with weights_only=True. Unknown model fields or mismatched parameter
        names/shapes raise errors rather than silently changing the architecture.
        """
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(saved, dict) or not {"config", "model"} <= saved.keys():
            raise ValueError("Expected a belief-state checkpoint with config and model state dictionary")
        config = dict(saved["config"])
        # Standard upstream checkpoints can contain inactive experimental flags.
        for name in ("attention_mass", "attention_sums"):
            if config.pop(name, False):
                raise ValueError("Only standard softmax attention is supported")
        model = Transformer(Config(**config))
        model.to(dtype=saved["model"]["embed.weight"].dtype)
        model.load_state_dict(saved["model"])
        return cls(model).to(device)

    def train(self, mode=True):
        # Freezing parameters alone does not disable dropout in a custom teacher.
        super().train(False)
        return self

    def spec(self):
        # Keep standard-teacher identity stable for saved abstraction checkpoints.
        config = dict(attention_mass=False, attention_sums=False, bos_visible=False)
        config.update(asdict(self.model.cfg))
        return dict(**super().spec(), layout="kv-token-length-v1", config=config)

    def _unpack(self, states):
        width = states.shape[1] - self.token_features - 1
        if width < 0 or width % self.entry_width:
            raise ValueError("Invalid packed cache width")
        kv = states[:, :width].reshape(len(states), -1, self.n_layers, self.n_heads, 2, self.head_dim)
        positions = states[:, -1].detach()
        if not bool((positions == positions[0]).all()) or not bool((positions == positions.round()).all()):
            raise ValueError("Batch states must share an integer history length")
        n = int(positions[0])
        expected = max(n-1, 0) + int(self.bos_visible and n > 0)
        if kv.shape[1] != expected:
            raise ValueError("Cache length and history-length metadata disagree")
        return kv[..., 0, :], kv[..., 1, :], states[:, width:-1], n

    def _pack(self, keys, values, token, position):
        kv = torch.stack([keys, values], -2).flatten(1)
        return torch.cat([kv, token, token.new_full((len(token), 1), position)], 1)

    def encoder_mask(self, state):
        mask = super().encoder_mask(state)
        mask[-1] = False
        return mask

    def _qkv(self, block, x, positions):
        batch, length, _ = x.shape
        q, k, v = block.qkv(block.ln1(x)).reshape(batch, length, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = rotate(q, positions), rotate(k, positions)
        return q, k, v

    @torch.no_grad()
    def history_state(self, histories):
        histories = prepare_histories(histories, self.vocab_size, device=self.device)
        batch, n = histories.shape
        if n > self.max_history:
            raise ValueError("History exceeds context")
        empty = self.model.embed.weight.new_empty(batch, 0, self.n_layers, self.n_heads, self.head_dim)
        bos = F.one_hot(torch.full((batch,), self.vocab_size, device=self.device), self.token_features).to(self.dtype)
        start = self._pack(empty, empty, bos, 0)
        if n == 0:
            return start
        _, keys, values, _ = self._pass(start, histories)
        if not self.bos_visible:
            keys, values = keys[:, 1:], values[:, 1:]
        return self._pack(keys, values, F.one_hot(histories[:, -1], self.token_features).to(self.dtype), n)

    def _pass(self, states, tokens):
        keys, values, current, n = self._unpack(states)
        batch, horizon = tokens.shape
        if horizon < 1 or n+horizon > self.max_history:
            raise ValueError("Forecast must be nonempty and remain within the teacher context")
        positions = n + torch.arange(horizon, device=states.device)
        x = torch.cat([(current @ self.model.embed.weight)[:, None], self.model.embed(tokens[:, :-1])], 1)
        if self.model.pos_embed is not None:
            x = x + self.model.pos_embed(positions)
        past = keys.shape[1]
        mask = torch.arange(past+horizon, device=states.device)[None] <= past+torch.arange(horizon, device=states.device)[:, None]
        if n == 0 and not self.bos_visible:
            mask[1:, 0] = False
        all_k, all_v = [], []
        for layer, block in enumerate(self.model.blocks):
            q, k, v = self._qkv(block, x, positions)
            k = torch.cat([keys[:, :, layer].transpose(1, 2), k], 2)
            v = torch.cat([values[:, :, layer].transpose(1, 2), v], 2)
            x = x + attention_projection(block, q, k, v, mask)
            x = x + block.mlp(block.ln2(x))
            all_k.append(k.transpose(1, 2))
            all_v.append(v.transpose(1, 2))
        return self.model.unembed(self.model.ln_final(x)).log_softmax(-1), torch.stack(all_k, 2), torch.stack(all_v, 2), n

    def predict(self, states):
        tokens = torch.zeros(len(states), 1, dtype=torch.long, device=states.device)
        return self._pass(states, tokens)[0][:, 0]

    def advance(self, states, tokens):
        lp, keys, values, n = self._pass(states, tokens[:, None])
        if n == 0 and not self.bos_visible:
            keys, values = keys[:, 1:], values[:, 1:]
        return lp[:, 0], self._pack(keys, values, F.one_hot(tokens, self.token_features).to(states.dtype), n+1)

    def trajectory(self, states, tokens):
        lp, keys, values, n = self._pass(states, tokens)
        past = (states.shape[1]-self.token_features-1)//self.entry_width
        successors = []
        for t in range(1, tokens.shape[1]+1):
            start = int(n == 0 and not self.bos_visible)
            k, v = keys[:, start:past+t], values[:, start:past+t]
            successors.append(self._pack(k, v, F.one_hot(tokens[:, t-1], self.token_features).to(states.dtype), n+t))
        return lp, successors

    def features(self, states, queries):
        keys, values, token, n = self._unpack(states)
        count = keys.shape[1]
        if count:
            q = rotate(queries, n) if self.rope else queries
            scores = torch.einsum("lhqd,btlhd->blhqt", q, keys) * self.head_dim**-.5
            response = torch.einsum("blhqt,btlhd->blhqd", scores.softmax(-1), values).flatten(1)
        else:
            response = states.new_zeros(len(states), queries.numel())
        return torch.cat([token, response], 1)
