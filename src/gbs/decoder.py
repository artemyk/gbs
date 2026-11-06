"""One decoder across prefix lengths, with learned or fixed observed queries."""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import time

import numpy as np
import torch

from .data import prepare_histories
from .models import Architecture, AttentionDecoder


@dataclass
class DecoderConfig:
    dimension: int = 4
    queries_per_head: int = 4
    hidden: int | None = 64
    decoder_steps: int = 5000
    batch_size: int = 64
    decoder_lr: float = 1e-3
    decoder_min_lr_fraction: float = 1.0
    validate_every: int = 200
    seed: int = 0
    allow_shared_histories: bool = False
    decoder_method: str = "learned"
    decoder_min_length: int = 0
    decoder_max_length: int | None = None
    decoder_samples: int | None = None
    decoder_validation_samples: int = 256  # Parent rows per length.
    decoder_ridges: tuple[float, ...] = (1e-10, 1e-8, 1e-6, 1e-4, .01, 1.)


def validate_decoder_config(config):
    if config.decoder_method not in ["learned", "observed"]:
        raise ValueError("decoder_method must be learned or observed")
    if (config.decoder_min_length < 0 or
            (config.decoder_max_length is not None and config.decoder_max_length < config.decoder_min_length)):
        raise ValueError("Invalid decoder prefix-length range")
    if config.decoder_samples is not None and config.decoder_samples < 1:
        raise ValueError("decoder_samples must be positive")
    if config.decoder_validation_samples < 1 or config.batch_size < 1 or config.validate_every < 1:
        raise ValueError("Decoder batch and validation sizes must be positive")
    if config.decoder_steps < 0 or config.decoder_lr <= 0 or not 0 < config.decoder_min_lr_fraction <= 1:
        raise ValueError("Invalid decoder_steps, decoder_lr, or decoder_min_lr_fraction")
    if not config.decoder_ridges or any(not np.isfinite(x) or x <= 0 for x in config.decoder_ridges):
        raise ValueError("decoder_ridges must contain positive finite penalties")


@dataclass
class PrefixBank:
    # Every group has one prefix length, so no padding changes attention.
    groups: dict[int, tuple[torch.Tensor, torch.Tensor]]
    cached_features: bool

    def sample(self, rng, batch_size):
        groups = list(self.groups.values())
        counts = np.array([len(x) for x, _ in groups])
        group = 0 if len(groups) == 1 else rng.choice(len(groups), p=counts/counts.sum())
        x, y = groups[group]
        ids = torch.tensor(rng.integers(len(x), size=batch_size), device=x.device)
        return x[ids], y[ids]

    def prediction(self, decoder, adapter, x):
        return decoder.net(x) if self.cached_features else decoder(adapter, x)

    @torch.no_grad()
    def measure(self, decoder, adapter):
        by_length, total, count = {}, 0., 0
        for n, (x, y) in self.groups.items():
            error = sum(float((self.prediction(decoder, adapter, part)-target).cpu().double().square().sum())
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
def _observed_bank(adapter, sequences, rows, queries_per_head, rng):
    # Sample iid from the empirical training-prefix distribution. The terminal
    # context state has no native next-token query, but remains a decoder target.
    allowed = {n: ids for n, ids in rows.items() if adapter.max_history is None or n < adapter.max_history}
    if not allowed:
        raise ValueError("No training prefix has a native next-token query")
    lengths = list(allowed)
    counts = np.array([len(allowed[n]) for n in lengths])
    choices = rng.choice(len(lengths), size=queries_per_head, p=counts/counts.sum())
    bank = torch.empty(adapter.n_layers, adapter.n_heads, queries_per_head, adapter.head_dim,
                       device=adapter.device, dtype=adapter.dtype)
    sampled = []
    for group, n in enumerate(lengths):
        slots = np.flatnonzero(choices == group)
        if not len(slots):
            continue
        ids = rng.choice(allowed[n], size=len(slots), replace=True)
        for left in range(0, len(ids), 256):
            selected = ids[left:left+256]
            histories = sequences[torch.tensor(selected, device=sequences.device), :n]
            native = adapter.observed_queries(adapter.history_state(histories.to(adapter.device)))
            if native.shape != (len(selected), adapter.n_layers, adapter.n_heads, adapter.head_dim):
                raise ValueError("observed_queries returned an incompatible shape")
            bank[:, :, torch.tensor(slots[left:left+256], device=bank.device)] = native.permute(1, 2, 0, 3)
        sampled.extend(dict(slot=int(slot), length=n, parent_row=int(parent)) for slot, parent in zip(slots, ids))
    return bank, sorted(sampled, key=lambda x: x['slot'])


@torch.no_grad()
def _prepare(adapter, sequences, rows, coordinates, decoder, cached):
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
            inputs.append(adapter.features(state, decoder.queries) if cached else state)
            targets.append(target)
        groups[n] = torch.cat(inputs), torch.cat(targets)
    return PrefixBank(groups, cached)


@torch.no_grad()
def _moments(bank):
    count = 0
    sx = sy = xx = xy = None
    for x, y in bank.groups.values():
        for part, target in zip(x.split(512), y.split(512)):
            a, b = part.cpu().double(), target.cpu().double()
            if sx is None:
                sx, sy = a.sum(0), b.sum(0)
                xx, xy = a.T@a, a.T@b
            else:
                sx += a.sum(0); sy += b.sum(0)
                xx += a.T@a; xy += a.T@b
            count += len(a)
    mx, my = sx/count, sy/count
    return mx, my, xx/count-torch.outer(mx, mx), xy/count-torch.outer(mx, my)


def _ridge(decoder, adapter, training, validation, penalties):
    mean, target_mean, covariance, cross = _moments(training)
    scale = covariance.diag().clamp_min(0).sqrt()
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = covariance/scale[:, None]/scale[None]
    eigen, vectors = torch.linalg.eigh((normalized+normalized.T)*.5)
    rhs = vectors.T@(cross/scale[:, None])
    scores, best, state = [], float('inf'), None
    for penalty in penalties:
        coefficient = vectors@(rhs/(eigen.clamp_min(0)+penalty)[:, None])
        weight = coefficient.T/scale
        bias = target_mean-weight@mean
        with torch.no_grad():
            decoder.net.weight.copy_(weight.to(decoder.net.weight))
            decoder.net.bias.copy_(bias.to(decoder.net.bias))
        score, _ = validation.measure(decoder, adapter)
        scores.append(dict(penalty=float(penalty), validation_mse=score))
        if score < best:
            best, state = score, (weight.clone(), bias.clone())
    with torch.no_grad():
        decoder.net.weight.copy_(state[0].to(decoder.net.weight))
        decoder.net.bias.copy_(state[1].to(decoder.net.bias))
    return scores


def fit_decoder(adapter, train_sequences, validation_sequences, coordinates, *, config=None, progress=print):
    """Fit one decoder across prefix lengths using one caller-supplied coordinate map.

    Split parent sequences before prefix expansion. Short prefixes may coincide
    across splits; they are not separate held-out histories. ``coordinates`` must
    use the same basis, center, and whitening at every length. The observed
    decoder method samples training-prefix queries before positional rotation
    and freezes them. With hidden=None its readout is fitted by ridge regression;
    otherwise only the MLP is trained on cached features. Learned queries start
    from the decoder's random initialization and use raw states throughout.
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
    architecture = Architecture(config.dimension, config.queries_per_head, config.hidden, config.decoder_method == 'learned')
    decoder = AttentionDecoder(adapter, architecture)
    rows = _prefix_rows(train_sequences, config.decoder_min_length, high, config.decoder_samples,
                        np.random.default_rng(config.seed+20))
    valid_rows = _prefix_rows(validation_sequences, config.decoder_min_length, high, config.decoder_validation_samples,
                              np.random.default_rng(config.seed+22), validation=True)
    cached = config.decoder_method == 'observed'
    sampled = []
    if cached:
        queries, sampled = _observed_bank(adapter, train_sequences, rows, config.queries_per_head,
                                           np.random.default_rng(config.seed+21))
        decoder.queries.copy_(queries)
    if progress: progress(f"Preparing decoder prefixes {config.decoder_min_length}..{high}; method={config.decoder_method}")
    training = _prepare(adapter, train_sequences, rows, coordinates, decoder, cached)
    validation = _prepare(adapter, validation_sequences, valid_rows, coordinates, decoder, cached)
    prepared = time.perf_counter()
    trace, ridge_scores = [], []
    if cached and config.hidden is None:
        ridge_scores = _ridge(decoder, adapter, training, validation, config.decoder_ridges)
        score, _ = validation.measure(decoder, adapter)
        trace.append(dict(stage='decoder', step=0, validation_mse=score, seconds=time.perf_counter()-started))
        if progress: progress(f"decoder ridge: validation MSE {score:.6g}")
    else:
        rng = np.random.default_rng(config.seed)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=config.decoder_lr)
        best, state = float('inf'), None
        for step in range(config.decoder_steps+1):
            if step % config.validate_every == 0 or step == config.decoder_steps:
                score, _ = validation.measure(decoder, adapter)
                trace.append(dict(stage='decoder', step=step, validation_mse=score, seconds=time.perf_counter()-started))
                if score < best:
                    best, state = score, copy.deepcopy(decoder.state_dict())
                if progress: progress(f"decoder {step}/{config.decoder_steps}: validation MSE {score:.6g}")
            if step == config.decoder_steps:
                break
            fraction = config.decoder_min_lr_fraction+(1-config.decoder_min_lr_fraction)*.5*(1+np.cos(np.pi*step/config.decoder_steps))
            optimizer.param_groups[0]['lr'] = config.decoder_lr*fraction
            x, y = training.sample(rng, config.batch_size)
            loss = (training.prediction(decoder, adapter, x)-y).square().mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        decoder.load_state_dict(state)
    score, by_length = validation.measure(decoder, adapter)
    diagnostics = dict(method=config.decoder_method, lengths=[config.decoder_min_length, high],
        training_rows=sum(len(x) for x, _ in training.groups.values()),
        validation_rows=sum(len(x) for x, _ in validation.groups.values()),
        query_samples=sampled, ridge_scores=ridge_scores, validation_mse=score,
        validation_by_length=by_length, timing_seconds=dict(preparation=prepared-started,
                                                          fitting=time.perf_counter()-prepared,
                                                          total=time.perf_counter()-started))
    return DecoderFitResult(decoder.eval(), architecture, trace, diagnostics, training, validation)
