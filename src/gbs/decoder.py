"""One learned-query decoder shared across prefix lengths."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import math
import time

import numpy as np
import torch

from .data import prepare_histories
from .models import Architecture, AttentionDecoder
from ._training import cosine_fraction, require_finite, require_integer


@dataclass
class DecoderConfig(Architecture):
    decoder_steps: int = 40000
    batch_size: int = 64
    decoder_lr: float = 1e-3
    decoder_min_lr_fraction: float = .01
    validate_every: int = 200
    seed: int = 0
    allow_shared_histories: bool = False
    decoder_min_length: int = 0
    decoder_max_length: int | None = None
    decoder_samples: int | None = None
    decoder_validation_samples: int = 256  # Parent rows per length.


def validate_decoder_config(config):
    config.__post_init__()
    for name in ("batch_size", "validate_every", "decoder_validation_samples"):
        require_integer(name, getattr(config, name), minimum=1)
    for name in ("decoder_steps", "decoder_min_length", "seed"):
        require_integer(name, getattr(config, name))
    if config.decoder_max_length is not None:
        require_integer("decoder_max_length", config.decoder_max_length)
    if config.decoder_max_length is not None and config.decoder_max_length < config.decoder_min_length:
        raise ValueError("Invalid decoder prefix-length range")
    if config.decoder_samples is not None:
        require_integer("decoder_samples", config.decoder_samples, minimum=1)
    require_finite("decoder_lr", config.decoder_lr, positive=True)
    require_finite("decoder_min_lr_fraction", config.decoder_min_lr_fraction, positive=True)
    if config.decoder_min_lr_fraction > 1:
        raise ValueError("decoder_min_lr_fraction must be <= 1")


@dataclass
class PrefixBank:
    # Every group has one prefix length, so no padding changes attention.
    groups: dict[int, tuple[torch.Tensor, torch.Tensor]]

    def sample(self, rng, batch_size):
        groups = list(self.groups.values())
        counts = np.array([len(x) for x, _ in groups])
        group = 0 if len(groups) == 1 else rng.choice(len(groups), p=counts/counts.sum())
        x, y = groups[group]
        ids = torch.tensor(rng.integers(len(x), size=batch_size), device=x.device)
        return x[ids], y[ids]

    @torch.no_grad()
    def measure(self, decoder, adapter):
        by_length, total, count = {}, 0., 0
        for n, (x, y) in self.groups.items():
            error = sum(float((decoder(adapter, part)-target).cpu().double().square().sum())
                        for part, target in zip(x.split(256), y.split(256)))
            by_length[str(n)] = error/y.numel()
            total += error
            count += y.numel()
        return total/count, by_length


@dataclass
class DecoderFitResult:
    decoder: AttentionDecoder
    architecture: Architecture
    trace: list[dict]
    diagnostics: dict
    training: PrefixBank = field(repr=False)
    validation: PrefixBank = field(repr=False)


def _prefix_rows(sequences, low, high, count, rng, *, validation=False):
    if validation:
        ids = rng.choice(len(sequences), size=min(len(sequences), count), replace=False)
        return {n: ids for n in range(low, high+1)}
    ids = np.arange(len(sequences)) if count is None else rng.integers(len(sequences), size=count)
    lengths = rng.integers(low, high+1, size=len(ids))
    return {n: ids[lengths == n] for n in range(low, high+1) if np.any(lengths == n)}


@torch.no_grad()
def _prepare(adapter, sequences, rows, coordinates, decoder):
    groups = {}
    for n, ids in rows.items():
        inputs, targets = [], []
        for selected in np.array_split(ids, max(1, (len(ids)+255)//256)):
            histories = sequences[torch.tensor(selected, device=sequences.device), :n]
            target = coordinates(histories).detach().to(device=adapter.device, dtype=adapter.dtype)
            if target.shape != (len(selected), decoder.dimension):
                raise ValueError("coordinates must return one dimension-sized code per prefix")
            if not bool(torch.isfinite(target).all()):
                raise ValueError(f"Nonfinite decoder coordinates at prefix length {n}")
            state = adapter.history_state(histories.to(adapter.device))
            inputs.append(state)
            targets.append(target)
        groups[n] = torch.cat(inputs), torch.cat(targets)
    return PrefixBank(groups)


def fit_decoder(adapter, train_sequences, validation_sequences, coordinates, *, config=None, progress=print):
    """Fit one decoder across prefix lengths using one caller-supplied coordinate map.

    Split parent sequences before prefix expansion. Short prefixes may coincide
    across splits; they are not separate held-out histories. ``coordinates`` must
    use the same basis, center, and whitening at every length. Queries start
    from random initialization and are optimized jointly with the tanh MLP.
    Natural states are retained so attention responses follow the learned queries.
    """
    config = config or DecoderConfig()
    validate_decoder_config(config)
    started = time.perf_counter()
    train_sequences = prepare_histories(train_sequences, adapter.vocab_size)
    validation_sequences = prepare_histories(validation_sequences, adapter.vocab_size)
    high = train_sequences.shape[1] if config.decoder_max_length is None else config.decoder_max_length
    if high > min(train_sequences.shape[1], validation_sequences.shape[1]) or high < config.decoder_min_length:
        raise ValueError("Decoder lengths exceed the supplied parent sequences")
    if adapter.max_history is not None and high > adapter.max_history:
        raise ValueError("Decoder lengths exceed the teacher context")
    if not config.allow_shared_histories:
        roots = {tuple(row) for row in train_sequences.cpu().tolist()}
        if any(tuple(row) in roots for row in validation_sequences.cpu().tolist()):
            raise ValueError("Decoder parent-sequence splits overlap")
    adapter.eval().requires_grad_(False)
    torch.manual_seed(config.seed)
    architecture = Architecture(config.dimension, config.queries_per_head, config.hidden)
    decoder = AttentionDecoder(adapter, architecture)
    rows = _prefix_rows(train_sequences, config.decoder_min_length, high, config.decoder_samples,
                        np.random.default_rng(config.seed+20))
    valid_rows = _prefix_rows(validation_sequences, config.decoder_min_length, high, config.decoder_validation_samples,
                              np.random.default_rng(config.seed+22), validation=True)
    if progress: progress(f"Preparing decoder prefixes {config.decoder_min_length}..{high}")
    training = _prepare(adapter, train_sequences, rows, coordinates, decoder)
    validation = _prepare(adapter, validation_sequences, valid_rows, coordinates, decoder)
    prepared = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.decoder_lr)
    trace, best, state = [], float('inf'), None
    for step in range(config.decoder_steps+1):
        if step % config.validate_every == 0 or step == config.decoder_steps:
            score, _ = validation.measure(decoder, adapter)
            if not math.isfinite(score):
                raise ValueError("Nonfinite decoder validation MSE")
            trace.append(dict(stage='decoder', step=step, validation_mse=score, seconds=time.perf_counter()-started))
            if score < best:
                best, state = score, copy.deepcopy(decoder.state_dict())
            if progress: progress(f"decoder {step}/{config.decoder_steps}: validation MSE {score:.6g}")
        if step == config.decoder_steps:
            break
        fraction = cosine_fraction(step, config.decoder_steps, config.decoder_min_lr_fraction)
        optimizer.param_groups[0]['lr'] = config.decoder_lr*fraction
        x, y = training.sample(rng, config.batch_size)
        loss = (decoder(adapter, x)-y).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite decoder training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Check gradients without changing their magnitude or the fitted schedule.
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), float("inf"),
                                      norm_type=float("inf"), error_if_nonfinite=True)
        optimizer.step()
    decoder.load_state_dict(state)
    score, by_length = validation.measure(decoder, adapter)
    diagnostics = dict(lengths=[config.decoder_min_length, high],
        training_rows=sum(len(x) for x, _ in training.groups.values()),
        validation_rows=sum(len(x) for x, _ in validation.groups.values()),
        validation_mse=score,
        validation_by_length=by_length, timing_seconds=dict(preparation=prepared-started,
                                                          fitting=time.perf_counter()-prepared,
                                                          total=time.perf_counter()-started))
    return DecoderFitResult(decoder.eval(), architecture, trace, diagnostics, training, validation)
