"""The differentiable recurrent interface required by the fitting code."""
from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
import json

import torch
from torch import Tensor, nn


def words(alphabet: int, length: int, *, device=None, max_outcomes: int = 65536) -> Tensor:
    """All fixed-length words, in lexicographic order; refuse exponential blowups."""
    if alphabet < 2 or length < 0:
        raise ValueError("Require alphabet >= 2 and length >= 0")
    count = alphabet ** length
    if count > max_outcomes:
        raise ValueError(f"{count} futures exceed max_outcomes={max_outcomes}; reduce the horizon")
    if length == 0:
        return torch.empty(1, 0, dtype=torch.long, device=device)
    powers = alphabet ** torch.arange(length - 1, -1, -1, device=device)
    return torch.arange(count, device=device)[:, None] // powers % alphabet


class CacheModel(nn.Module, ABC):
    """Adapter for a frozen autoregressive model with a flat, writable cache state.

    Each state is a floating tensor (batch, state_width). Width may depend on
    history length. ``advance`` predicts *before* observing its token, then
    returns the successor state. Gradients through written states must survive.
    Histories contain ordinary token IDs only, without an explicit BOS.
    """
    vocab_size: int
    n_layers: int
    n_heads: int
    head_dim: int
    token_features: int
    max_history: int | None = None

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @abstractmethod
    def history_state(self, histories: Tensor) -> Tensor:
        """Build natural states for a batch of equally long histories."""

    @abstractmethod
    def advance(self, states: Tensor, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Return next-token log probabilities and states after observing tokens."""

    @abstractmethod
    def features(self, states: Tensor, queries: Tensor) -> Tensor:
        """Read token, attention responses, log(1+Z), and log(1+cache size).

        Queries have shape (layers, heads, queries_per_head, head_dim).
        Return shape (batch, token_features + L H Q (head_dim+1) + 1).
        Use zeros for attention measurements of an empty cache.
        """

    def encoder_mask(self, state: Tensor) -> Tensor:
        """Writable coordinates; override to exclude structural metadata."""
        return torch.ones(state.shape[1], dtype=torch.bool, device=state.device)

    def spec(self) -> dict:
        """JSON metadata identifying state semantics, beyond frozen parameter values."""
        return dict(adapter=type(self).__qualname__, vocab=self.vocab_size,
                    layers=self.n_layers, heads=self.n_heads, head_dim=self.head_dim,
                    token_features=self.token_features, max_history=self.max_history)

    def fingerprint(self) -> str:
        digest = hashlib.sha256(json.dumps(self.spec(), sort_keys=True).encode())
        for name, value in sorted(self.state_dict().items()):
            value = value.detach().cpu().contiguous()
            digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def predict(self, states: Tensor) -> Tensor:
        tokens = torch.zeros(len(states), dtype=torch.long, device=states.device)
        return self.advance(states, tokens)[0]

    def trajectory(self, states: Tensor, tokens: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Teacher-forced predictions and successor states; optional fast override."""
        predictions, successors = [], []
        for token in tokens.unbind(1):
            lp, states = self.advance(states, token)
            predictions.append(lp)
            successors.append(states)
        return torch.stack(predictions, 1), successors

    def future_log_probs(self, states: Tensor, horizon: int, *, max_outcomes=65536) -> Tensor:
        """Joint future log probabilities (batch, vocab_size**horizon)."""
        futures = words(self.vocab_size, horizon, device=states.device, max_outcomes=max_outcomes)
        if horizon == 0:
            return states.new_zeros(len(states), 1)
        expanded = states.repeat_interleave(len(futures), 0)
        tokens = futures.repeat(len(states), 1)
        lp, _ = self.trajectory(expanded, tokens)
        return lp.gather(2, tokens[..., None]).squeeze(2).sum(1).reshape(len(states), -1)

    @torch.no_grad()
    def sample(self, states: Tensor, uniforms: Tensor) -> tuple[Tensor, Tensor]:
        """Sample teacher continuations using caller-supplied uniforms for reproducibility."""
        tokens, probabilities = [], []
        for uniform in uniforms.unbind(1):
            q = self.predict(states).exp()
            token = (uniform[:, None] >= q.cumsum(1)).sum(1).clamp_max(self.vocab_size - 1)
            _, states = self.advance(states, token)
            tokens.append(token)
            probabilities.append(q)
        return torch.stack(tokens, 1), torch.stack(probabilities, 1)
