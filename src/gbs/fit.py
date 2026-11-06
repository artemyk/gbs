"""GHMM initialization and predictive/recursive fitting on frozen teacher data."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .adapters import CacheModel
from .ghmm import GHMM, fit_ghmm
from .models import Architecture, AttentionDecoder, AffineEncoder, GBSAbstraction


@dataclass
class FitConfig:
    dimension: int = 4
    queries_per_head: int = 4
    hidden: int = 64
    horizon: int = 16
    future_length: int = 4
    spectral_samples: int = 512
    decoder_steps: int = 5000
    encoder_steps: int = 2000
    joint_steps: int = 4000
    batch_size: int = 64
    decoder_lr: float = 1e-3
    joint_decoder_lr: float = 1e-4
    encoder_lr: float = 3e-4
    tracking_weight: float = .1
    coordinate_weight: float = .1
    roundtrip_weight: float = .1
    validate_every: int = 200
    seed: int = 0
    max_outcomes: int = 65536
    allow_shared_histories: bool = False


@dataclass
class RefineConfig:
    steps: int = 2000
    horizon: int = 16
    batch_size: int = 64
    lr: float = 3e-4
    tracking_weight: float = .1
    roundtrip_weight: float = .1
    preservation_slack: float = .05
    validate_every: int = 200
    seed: int = 0


@dataclass
class DataBank:
    states: Tensor
    tokens: Tensor
    reference: Tensor
    anchors: Tensor | None = None

    def select(self, ids):
        return DataBank(self.states[ids], self.tokens[ids], self.reference[ids],
                        None if self.anchors is None else self.anchors[ids])


@dataclass
class FitResult:
    abstraction: GBSAbstraction
    config: dict
    trace: list[dict]
    initialization: dict = field(default_factory=dict)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.abstraction.save(directory/"abstraction.pt")
        for name, value in [("config", self.config), ("trace", self.trace), ("initialization", self.initialization)]:
            (directory/f"{name}.json").write_text(json.dumps(value, indent=2)+"\n")


def _validate_config(config):
    if config.batch_size < 1 or config.validate_every < 1 or config.horizon < 1:
        raise ValueError("Batch size, validation interval, and forecast horizon must be positive")
    for key, value in asdict(config).items():
        if (key.endswith("_steps") or key == "steps") and value < 0:
            raise ValueError("Training steps cannot be negative")
        if (key.endswith("_lr") or key == "lr") and value <= 0:
            raise ValueError("Learning rates must be positive")
        if key.endswith("_weight") and value < 0:
            raise ValueError("Loss weights cannot be negative")


def _check_histories(adapter, train, validation, horizon, allow_shared):
    if train.ndim != 2 or validation.ndim != 2 or not len(train) or not len(validation):
        raise ValueError("Provide nonempty, two-dimensional training and validation histories")
    if train.shape[1] != validation.shape[1] or train.shape[1] < 1:
        raise ValueError("This fitter uses one nonempty root history length")
    if adapter.max_history is not None and train.shape[1]+horizon > adapter.max_history:
        raise ValueError("Root length plus forecast horizon exceeds the teacher context")
    if not allow_shared:
        roots = {tuple(row) for row in train.cpu().tolist()}
        if any(tuple(row) in roots for row in validation.cpu().tolist()):
            raise ValueError("Training/validation histories overlap; split by distinct root or explicitly allow_shared_histories")


@torch.no_grad()
def make_bank(adapter, histories, horizon, *, seed, anchors=None, batch_size=256):
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


def conditional_kl(reference, log_probabilities):
    return (reference*(reference.clamp_min(torch.finfo(reference.dtype).tiny).log()-log_probabilities)).sum(-1)


def abstraction_loss(model, bank, *, tracking_weight=.1, coordinate_weight=.1, roundtrip_weight=.1):
    """Mean token KLs plus tracking, coordinate anchoring, and explicit root round trip."""
    code = model.decode(bank.states)
    written = model.encode(code)
    single_lp, written_states = model.adapter.trajectory(written, bank.tokens)
    with torch.no_grad():
        _, natural_states = model.adapter.trajectory(bank.states, bank.tokens)
        targets = [model.decode(state) for state in natural_states]
    root_error = (model.decode(written)-code.detach()).square().mean()
    tracking = torch.stack([root_error]+[(model.decode(state)-target).square().mean()
                                      for state, target in zip(written_states, targets)]).mean()
    single = conditional_kl(bank.reference, single_lp).mean()
    repeated = conditional_kl(bank.reference, model.repeated(code, bank.tokens)).mean()
    coordinate = code.new_zeros(()) if bank.anchors is None else (code-bank.anchors).square().mean()
    total = .5*(single+repeated)+tracking_weight*tracking+coordinate_weight*coordinate+roundtrip_weight*root_error
    return dict(total=total, single=single, repeated=repeated, tracking=tracking,
                coordinates=coordinate, roundtrip=root_error)


def _interface_state(model):
    return copy.deepcopy((model.decoder.state_dict(), model.encoder.state_dict()))


def _restore(model, state):
    model.decoder.load_state_dict(state[0])
    model.encoder.load_state_dict(state[1])


def _weighted_coordinate_transform(codes):
    x = codes.cpu().double()
    center = x.mean(0)
    covariance = (x-center).T@(x-center)/len(x)
    eigen, vectors = torch.linalg.eigh(covariance)
    if eigen[0] <= max(float(eigen[-1]), 1e-30)*1e-10:
        raise ValueError("Root histories do not span the requested initialization coordinates")
    return center.to(codes), (vectors@torch.diag(eigen.rsqrt())@vectors.T).to(codes)


def fit_abstraction(adapter: CacheModel, train_histories: Tensor, validation_histories: Tensor, *,
                    config: FitConfig | None = None, initializer: Callable[[Tensor], Tensor] | GHMM | None = None,
                    initialization_histories: list[Tensor] | None = None, progress=print) -> FitResult:
    """Fit a fixed-length abstraction, selecting checkpoints on validation data only.

    By default a GHMM supplies coordinates from the teacher's conditional futures.
    A supplied GHMM or callable can instead return initial coordinates. Callable
    outputs must have ``config.dimension`` columns; no HMM labels are required.
    Root rows determine the averaging distribution, e.g. process-sampled rows.
    """
    config = config or FitConfig()
    _validate_config(config)
    _check_histories(adapter, train_histories, validation_histories, config.horizon, config.allow_shared_histories)
    adapter.eval().requires_grad_(False)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    diagnostics = {}
    if initializer is None:
        if initialization_histories is None:
            if config.spectral_samples < config.dimension+1:
                raise ValueError("Too few spectral initialization histories")
            longest = train_histories.shape[1]
            if adapter.max_history is not None:
                longest = min(longest, adapter.max_history-config.future_length-1)
            if longest < 0:
                raise ValueError("GHMM future queries exceed the teacher context")
            lengths = rng.integers(longest+1, size=config.spectral_samples)
            ids = rng.integers(len(train_histories), size=config.spectral_samples)
            initialization_histories = [train_histories[torch.tensor(ids[lengths == n]), :n]
                                        for n in range(longest+1) if np.any(lengths == n)]
        if progress: progress("Fitting GHMM initialization from teacher future probabilities")
        initializer, diagnostics = fit_ghmm(adapter, initialization_histories, dimension=config.dimension+1,
            horizon=config.future_length, max_outcomes=config.max_outcomes)
    with torch.no_grad():
        if isinstance(initializer, GHMM):
            target = lambda h: initializer.states(h.to(adapter.device))[:, 1:]
        else:
            target = initializer
        train_target = target(train_histories.to(adapter.device)).detach().to(device=adapter.device, dtype=adapter.dtype)
        validation_target = target(validation_histories.to(adapter.device)).detach().to(train_target)
        if train_target.shape != (len(train_histories), config.dimension) or validation_target.shape != (len(validation_histories), config.dimension):
            raise ValueError("Initializer must return one dimension-sized code per history")
        if not bool(torch.isfinite(train_target).all() and torch.isfinite(validation_target).all()):
            raise ValueError("Initialization produced nonfinite coordinates")
        center, whitener = _weighted_coordinate_transform(train_target)
        train_target, validation_target = (train_target-center)@whitener, (validation_target-center)@whitener
        train_states = adapter.history_state(train_histories.to(adapter.device))
        validation_states = adapter.history_state(validation_histories.to(adapter.device))
    architecture = Architecture(config.dimension, config.queries_per_head, config.hidden)
    decoder = AttentionDecoder(adapter, architecture)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.decoder_lr)
    trace, best, best_state = [], float("inf"), None
    for step in range(config.decoder_steps+1):
        if step % config.validate_every == 0 or step == config.decoder_steps:
            with torch.no_grad():
                score = float((decoder(adapter, validation_states)-validation_target).square().mean())
            trace.append(dict(stage="decoder", step=step, validation_mse=score))
            if score < best: best, best_state = score, copy.deepcopy(decoder.state_dict())
            if progress: progress(f"decoder {step}/{config.decoder_steps}: validation MSE {score:.6g}")
        if step == config.decoder_steps: break
        ids = torch.tensor(rng.integers(len(train_states), size=config.batch_size), device=adapter.device)
        loss = (decoder(adapter, train_states[ids])-train_target[ids]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    decoder.load_state_dict(best_state)
    encoder = AffineEncoder.initialize(adapter, train_states, train_target)
    model = GBSAbstraction(adapter, decoder, encoder, architecture)
    if progress: progress("Sampling fixed teacher continuation banks")
    train = make_bank(adapter, train_histories, config.horizon, seed=config.seed+10, anchors=train_target)
    valid = make_bank(adapter, validation_histories, config.horizon, seed=config.seed+11, anchors=validation_target)
    for stage, steps in [("encoder", config.encoder_steps), ("joint", config.joint_steps)]:
        decoder.requires_grad_(stage == "joint")
        optimizer = torch.optim.Adam([dict(params=encoder.parameters(), lr=config.encoder_lr),
                                      dict(params=decoder.parameters(), lr=config.joint_decoder_lr)])
        best, best_state = float("inf"), None
        for step in range(steps+1):
            if step % config.validate_every == 0 or step == steps:
                with torch.no_grad():
                    totals = [float(abstraction_loss(model, valid.select(slice(left, left+256)),
                        tracking_weight=config.tracking_weight, coordinate_weight=config.coordinate_weight,
                        roundtrip_weight=config.roundtrip_weight)["total"])*min(256, len(valid.states)-left)
                        for left in range(0, len(valid.states), 256)]
                score = sum(totals)/len(valid.states)
                trace.append(dict(stage=stage, step=step, validation_loss=score))
                if score < best: best, best_state = score, _interface_state(model)
                if progress: progress(f"{stage} {step}/{steps}: validation loss {score:.6g}")
            if step == steps: break
            fraction = .1+.45*(1+np.cos(np.pi*step/steps))
            optimizer.param_groups[0]["lr"] = config.encoder_lr*fraction
            optimizer.param_groups[1]["lr"] = config.joint_decoder_lr*fraction
            ids = torch.tensor(rng.integers(len(train.states), size=config.batch_size), device=adapter.device)
            terms = abstraction_loss(model, train.select(ids), tracking_weight=config.tracking_weight,
                coordinate_weight=config.coordinate_weight, roundtrip_weight=config.roundtrip_weight)
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *decoder.parameters()], 1., error_if_nonfinite=True)
            optimizer.step()
        _restore(model, best_state)
    diagnostics.update(center=center.cpu().tolist(), whitener=whitener.cpu().tolist())
    return FitResult(model.eval(), asdict(config), trace, diagnostics)


@torch.no_grad()
def _evaluate_bank(model, bank):
    rows = {key: [] for key in ["roundtrip_mse", "single_kl", "repeated_kl", "tracking_mse"]}
    for left in range(0, len(bank.states), 256):
        data = bank.select(slice(left, left+256))
        code = model.decode(data.states)
        written = model.encode(code)
        lp, trajectory = model.adapter.trajectory(written, data.tokens)
        _, natural = model.adapter.trajectory(data.states, data.tokens)
        rows["roundtrip_mse"].append((model.decode(written)-code).square().mean(1))
        rows["single_kl"].append(conditional_kl(data.reference, lp).sum(1))
        rows["repeated_kl"].append(conditional_kl(data.reference, model.repeated(code, data.tokens)).sum(1))
        rows["tracking_mse"].append(torch.stack([(model.decode(a)-model.decode(b)).square().mean(1)
                                                for a, b in zip(trajectory, natural)], 1).mean(1))
    summary = {}
    for key, parts in rows.items():
        value = torch.cat(parts)
        if not bool(torch.isfinite(value).all()): raise ValueError("Nonfinite evaluation")
        summary[key] = dict(mean=float(value.mean()), standard_error=float(value.std(correction=1)/len(value)**.5))
    return summary


def evaluate(model, histories, *, horizon=16, seed=1000):
    """Evaluate on independent teacher draws; predictive KL is summed over horizon."""
    if histories.ndim != 2 or len(histories) < 2 or horizon < 1:
        raise ValueError("Evaluation requires at least two histories and a positive horizon")
    model.eval()
    bank = make_bank(model.adapter, histories, horizon, seed=seed)
    return _evaluate_bank(model, bank)


def refine_encoder(model, train_histories, validation_histories, *, config=None, progress=print):
    """Improve round trips with decoder and teacher fixed and validation-only limits.

    This starts fresh Adam and samples new continuation banks. The starting
    encoder is always an eligible checkpoint. Existing model weights are
    updated in place; save the original abstraction first to retain it.
    """
    config = config or RefineConfig()
    _validate_config(config)
    if config.preservation_slack < 0: raise ValueError("preservation_slack cannot be negative")
    _check_histories(model.adapter, train_histories, validation_histories, config.horizon, False)
    model.adapter.eval().requires_grad_(False)
    model.decoder.requires_grad_(False)
    initial_decoder = copy.deepcopy(model.decoder.state_dict())
    train = make_bank(model.adapter, train_histories, config.horizon, seed=config.seed+10)
    valid = make_bank(model.adapter, validation_histories, config.horizon, seed=config.seed+11)
    starting = _evaluate_bank(model, valid)
    limits = {key: starting[key]["mean"]*(1+config.preservation_slack) for key in ["single_kl", "repeated_kl", "tracking_mse"]}
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=config.lr)
    rng = np.random.default_rng(config.seed)
    trace, best, best_state = [], float("inf"), None
    for step in range(config.steps+1):
        if step % config.validate_every == 0 or step == config.steps:
            measured = _evaluate_bank(model, valid)
            feasible = all(measured[key]["mean"] <= value+1e-12 for key, value in limits.items())
            trace.append(dict(stage="refine", step=step, feasible=feasible, metrics=measured))
            if feasible and measured["roundtrip_mse"]["mean"] < best:
                best, best_state = measured["roundtrip_mse"]["mean"], _interface_state(model)
            if progress: progress(f"refine {step}/{config.steps}: round-trip MSE {measured['roundtrip_mse']['mean']:.6g}; feasible={feasible}")
        if step == config.steps: break
        optimizer.param_groups[0]["lr"] = config.lr*(.1+.45*(1+np.cos(np.pi*step/config.steps)))
        ids = torch.tensor(rng.integers(len(train.states), size=config.batch_size), device=model.adapter.device)
        loss = abstraction_loss(model, train.select(ids), tracking_weight=config.tracking_weight,
                                coordinate_weight=0, roundtrip_weight=config.roundtrip_weight)["total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    _restore(model, best_state)
    assert all(torch.equal(value, model.decoder.state_dict()[key]) for key, value in initial_decoder.items())
    return FitResult(model.eval(), asdict(config), trace, dict(validation_limits=limits))
