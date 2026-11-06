"""Validated token histories and fixed teacher continuation banks."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

DEFAULT_EVALUATION_SEED = 1000


def token_ids(tokens, vocab_size, *, device=None):
    """Validate integer IDs before converting them losslessly to long indices."""
    if not isinstance(tokens, Tensor) or tokens.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
        raise ValueError("Token IDs must be an integer tensor")
    tokens = tokens.to(device=device, dtype=torch.long)
    if bool(((tokens < 0) | (tokens >= vocab_size)).any()):
        raise ValueError("Token ID outside the vocabulary")
    return tokens


def prepare_histories(histories, vocab_size, *, device=None, min_rows=1):
    """Normalize a rectangular batch; empty prefixes are valid for GHMM queries."""
    if not isinstance(histories, Tensor) or histories.ndim != 2 or len(histories) < min_rows:
        raise ValueError(f"Provide a two-dimensional batch with at least {min_rows} histories")
    return token_ids(histories, vocab_size, device=device)


def check_history_pair(adapter, train, validation, horizon, allow_shared=False):
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")
    if train.shape[1] != validation.shape[1] or train.shape[1] < 1:
        raise ValueError("This fitter uses one nonempty root history length")
    if adapter.max_history is not None and train.shape[1]+horizon > adapter.max_history:
        raise ValueError("Root length plus forecast horizon exceeds the teacher context")
    if not allow_shared:
        roots = {tuple(row) for row in train.cpu().tolist()}
        if any(tuple(row) in roots for row in validation.cpu().tolist()):
            raise ValueError("History splits overlap; split by distinct root or explicitly allow_shared_histories")


def history_splits(splits, adapter, horizon):
    """Validate and normalize all three splits, including test isolation."""
    names = ["train", "validation", "test"]
    if not isinstance(splits, dict) or not set(names) <= splits.keys():
        raise ValueError("Histories must contain train, validation, and test tensors")
    result = {name: prepare_histories(splits[name], adapter.vocab_size, min_rows=2) for name in names}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        check_history_pair(adapter, result[a], result[b], horizon)
    return result


@dataclass
class DataBank:
    states: Tensor
    tokens: Tensor
    reference: Tensor
    anchors: Tensor | None = None

    def select(self, ids):
        return DataBank(self.states[ids], self.tokens[ids], self.reference[ids],
                        None if self.anchors is None else self.anchors[ids])


@torch.no_grad()
def make_bank(adapter, histories, horizon, *, seed, anchors=None, batch_size=256):
    """Build each natural cache once, in bounded batches, and sample fixed futures."""
    histories = prepare_histories(histories, adapter.vocab_size)
    if horizon < 1 or batch_size < 1:
        raise ValueError("Forecast horizon and preparation batch size must be positive")
    if adapter.max_history is not None and histories.shape[1]+horizon > adapter.max_history:
        raise ValueError("Root length plus forecast horizon exceeds the teacher context")
    rng = np.random.default_rng(seed)
    states, tokens, reference = [], [], []
    for left in range(0, len(histories), batch_size):
        raw = adapter.history_state(histories[left:left+batch_size].to(adapter.device))
        uniforms = torch.tensor(rng.random((len(raw), horizon)), device=adapter.device, dtype=adapter.dtype)
        token, q = adapter.sample(raw, uniforms)
        states.append(raw)
        tokens.append(token)
        reference.append(q)
    return DataBank(torch.cat(states), torch.cat(tokens), torch.cat(reference), anchors)
