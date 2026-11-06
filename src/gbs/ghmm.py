"""Spectral generalized HMM initialization from a teacher's conditional futures."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from .adapters import CacheModel
from .data import prepare_histories, token_ids


class GHMM(nn.Module):
    """Row-vector observable operators with a constant first coordinate.

    Normalization is enforced algebraically; nonnegativity is not guaranteed.
    Predictions are returned as fitted, including negative values. This model
    supplies initialization coordinates; final fitting targets the teacher.
    """
    def __init__(self, operators: Tensor, initial: Tensor):
        super().__init__()
        self.register_buffer("operators", torch.as_tensor(operators))
        self.register_buffer("initial", torch.as_tensor(initial))
        self.dimension = len(initial)

    def probabilities(self, state):
        return state @ self.operators[:, :, 0].T

    def update(self, state, token):
        token = token_ids(torch.as_tensor(token), len(self.operators), device=self.operators.device)
        joint = torch.einsum("...i,...ij->...j", state, self.operators[token])
        if not bool(torch.isfinite(joint).all()) or bool((joint[:, 0].abs() < 1e-12).any()):
            raise ValueError("GHMM normalization is zero or nonfinite")
        return joint/joint[:, :1]

    def states(self, histories):
        histories = prepare_histories(histories, len(self.operators), device=self.initial.device)
        state = self.initial.expand(len(histories), -1)
        for token in histories.unbind(1):
            state = self.update(state, token)
        return state

    def future(self, state, horizon):
        joint = state[:, None]
        for _ in range(horizon):
            joint = torch.einsum("bni,xij->bnxj", joint, self.operators).flatten(1, 2)
        return joint[..., 0]


def spectral_operators(joint: np.ndarray, initial_future: np.ndarray, dimension: int,
                       alphabet: int, horizon: int, *, weights=None, rcond=1e-10):
    """Fit operators from p(x_1,...,x_(horizon+1) | history).

    We center the horizon-future vectors, keep dimension-1 SVD directions,
    and fit token operators by weighted least squares on unnormalized joint
    successors. A full lexicographic future alphabet is assumed.
    """
    joint = np.asarray(joint, dtype=np.float64)
    if alphabet < 2 or horizon < 1 or joint.ndim != 2 or joint.shape[1] != alphabet**(horizon+1):
        raise ValueError("joint must have one column per lexicographic (horizon+1)-token future")
    if not np.isfinite(joint).all() or (joint < 0).any() or not np.allclose(joint.sum(1), 1, atol=1e-5):
        raise ValueError("Teacher joint futures must be finite probability distributions")
    initial_future = np.asarray(initial_future, dtype=np.float64)
    if (initial_future.shape != (alphabet**horizon,) or not np.isfinite(initial_future).all()
            or (initial_future < 0).any() or not np.isclose(initial_future.sum(), 1, atol=1e-5)):
        raise ValueError("Initial future must be a finite probability distribution over horizon-token words")
    if rcond <= 0:
        raise ValueError("rcond must be positive")
    w = np.ones(len(joint)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(joint),) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Invalid spectral history weights")
    w = w/w.sum()
    q = joint.reshape(len(joint), alphabet**horizon, alphabet).sum(2)
    mean = w @ q
    _, singular, vt = np.linalg.svd((q-mean)*np.sqrt(w[:, None]), full_matrices=False)
    k = dimension-1
    if k < 1 or k > len(singular) or singular[k-1] <= rcond*max(singular[0], 1e-30):
        raise ValueError("Requested GHMM dimension exceeds the sampled predictive rank")
    components, scale = vt[:k].T, singular[:k]
    z = (q-mean) @ components/scale
    state = np.column_stack([np.ones(len(q)), z])
    operators = []
    for token in range(alphabet):
        successor_joint = joint.reshape(len(joint), alphabet, -1)[:, token]
        px = successor_joint.sum(1)
        target = np.column_stack([px, (successor_joint-px[:, None]*mean) @ components/scale])
        operators.append(np.linalg.lstsq(state*np.sqrt(w[:, None]), target*np.sqrt(w[:, None]), rcond=rcond)[0])
    operators = np.stack(operators)
    e0 = np.eye(dimension)[0]
    operators[-1, :, 0] = e0-operators[:-1, :, 0].sum(0)
    initial = np.r_[1., (np.asarray(initial_future)-mean) @ components/scale]
    return dict(operators=operators, initial=initial, mean=mean, components=components,
                scale=scale, singular_values=singular)


@torch.no_grad()
def fit_ghmm(adapter: CacheModel, histories: list[Tensor], *, dimension=5, horizon=4,
             batch_size=64, max_outcomes=65536, rcond=1e-10) -> tuple[GHMM, dict]:
    """Fit from batches of histories, which may have different prefix lengths."""
    rows = []
    for history in histories:
        for part in history.split(batch_size):
            state = adapter.history_state(part.to(adapter.device))
            rows.append(adapter.future_log_probs(state, horizon+1, max_outcomes=max_outcomes).exp().cpu().double().numpy())
    if not rows:
        raise ValueError("Provide at least one initialization history")
    empty = torch.empty(1, 0, dtype=torch.long, device=adapter.device)
    initial = adapter.future_log_probs(adapter.history_state(empty), horizon, max_outcomes=max_outcomes).exp()[0].cpu().double().numpy()
    fitted = spectral_operators(np.concatenate(rows), initial, dimension, adapter.vocab_size, horizon, rcond=rcond)
    ghmm = GHMM(torch.tensor(fitted["operators"], device=adapter.device, dtype=adapter.dtype),
                torch.tensor(fitted["initial"], device=adapter.device, dtype=adapter.dtype))
    diagnostics = {key: np.asarray(value).tolist() for key, value in fitted.items()}
    diagnostics["normalization_error"] = float(np.max(abs(fitted["operators"][:, :, 0].sum(0)-np.eye(dimension)[0])))
    return ghmm, diagnostics
