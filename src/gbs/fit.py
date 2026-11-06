"""GHMM initialization and predictive/recursive fitting on frozen teacher data."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .adapters import CacheModel
from .ghmm import GHMM, fit_ghmm
from .models import Architecture, AttentionDecoder, AffineEncoder, GBSAbstraction
from .data import DEFAULT_EVALUATION_SEED, DataBank, EncodedBank, check_history_pair, encode_bank, make_bank, prepare_histories


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
    decoder_min_lr_fraction: float = 1.0
    joint_decoder_lr: float = 1e-4
    encoder_lr: float = 3e-4
    tracking_weight: float = .1
    coordinate_weight: float = .1
    roundtrip_weight: float = .1
    validate_every: int = 200
    seed: int = 0
    max_outcomes: int = 65536
    allow_shared_histories: bool = False
    teacher_every: int = 1


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
    teacher_every: int = 1


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
    if not isinstance(config.teacher_every, int) or config.teacher_every < 1:
        raise ValueError("teacher_every must be a positive integer")
    if config.teacher_every > 1 and config.roundtrip_weight + config.tracking_weight == 0:
        raise ValueError("Alternating updates require a positive round-trip or tracking weight")
    if isinstance(config, FitConfig) and not 0 < config.decoder_min_lr_fraction <= 1:
        raise ValueError("decoder_min_lr_fraction must be in (0, 1]")
    for key, value in asdict(config).items():
        if (key.endswith("_steps") or key == "steps") and value < 0:
            raise ValueError("Training steps cannot be negative")
        if (key.endswith("_lr") or key == "lr") and value <= 0:
            raise ValueError("Learning rates must be positive")
        if key.endswith("_weight") and value < 0:
            raise ValueError("Loss weights cannot be negative")


def conditional_kl(reference, log_probabilities):
    return (reference*(reference.clamp_min(torch.finfo(reference.dtype).tiny).log()-log_probabilities)).sum(-1)


def _path_terms(model, bank: DataBank | EncodedBank):
    """Per-root errors shared by training and evaluation; natural targets are detached."""
    if isinstance(bank, EncodedBank) and any(parameter.requires_grad for parameter in model.decoder.parameters()):
        raise ValueError("Cached natural targets require a frozen decoder")
    code = bank.codes if isinstance(bank, EncodedBank) else model.decode(bank.states)
    written = model.encode(code)
    single_lp, written_states = model.adapter.trajectory(written, bank.tokens)
    if isinstance(bank, EncodedBank):
        targets = bank.targets.unbind(1)
    else:
        with torch.no_grad():
            _, natural_states = model.adapter.trajectory(bank.states, bank.tokens)
            targets = [model.decode(state) for state in natural_states]
    return dict(roundtrip=(model.decode(written)-code.detach()).square().mean(1),
        tracking=torch.stack([(model.decode(state)-target).square().mean(1)
                              for state, target in zip(written_states, targets)], 1),
        single=conditional_kl(bank.reference, single_lp),
        repeated=conditional_kl(bank.reference, model.repeated(code, bank.tokens)),
        coordinates=code.new_zeros(len(code)) if bank.anchors is None else (code-bank.anchors).square().mean(1))


def abstraction_loss(model, bank: DataBank | EncodedBank, *, tracking_weight=.1, coordinate_weight=.1, roundtrip_weight=.1):
    """Mean token KLs plus tracking, coordinate anchoring, and explicit root round trip."""
    terms = _path_terms(model, bank)
    root_error = terms["roundtrip"].mean()
    tracking = torch.cat([terms["roundtrip"][:, None], terms["tracking"]], 1).mean()
    single, repeated, coordinate = (terms[key].mean() for key in ["single", "repeated", "coordinates"])
    total = .5*(single+repeated)+tracking_weight*tracking+coordinate_weight*coordinate+roundtrip_weight*root_error
    return dict(total=total, single=single, repeated=repeated, tracking=tracking,
                coordinates=coordinate, roundtrip=root_error)


def _encoder_loss(model, bank: EncodedBank, config, step, steps):
    """Use the full objective periodically; other updates fit only its root round trip."""
    if (step+1) % config.teacher_every == 0 or step+1 == steps:
        return abstraction_loss(model, bank, tracking_weight=config.tracking_weight,
            coordinate_weight=0, roundtrip_weight=config.roundtrip_weight)["total"], True
    weight = config.roundtrip_weight + config.tracking_weight/(bank.tokens.shape[1]+1)
    return weight*(model.decode(model.encode(bank.codes))-bank.codes).square().mean(), False


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
    train_histories = prepare_histories(train_histories, adapter.vocab_size)
    validation_histories = prepare_histories(validation_histories, adapter.vocab_size)
    check_history_pair(adapter, train_histories, validation_histories, config.horizon, config.allow_shared_histories)
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
            target = lambda h: initializer.states(h)[:, 1:]
            target_device = initializer.initial.device
        else:
            target = initializer
            target_device = adapter.device
        train_target = target(train_histories.to(target_device)).detach().to(device=adapter.device, dtype=adapter.dtype)
        validation_target = target(validation_histories.to(target_device)).detach().to(train_target)
        if train_target.shape != (len(train_histories), config.dimension) or validation_target.shape != (len(validation_histories), config.dimension):
            raise ValueError("Initializer must return one dimension-sized code per history")
        if not bool(torch.isfinite(train_target).all() and torch.isfinite(validation_target).all()):
            raise ValueError("Initialization produced nonfinite coordinates")
        center, whitener = _weighted_coordinate_transform(train_target)
        train_target, validation_target = (train_target-center)@whitener, (validation_target-center)@whitener
    if progress: progress("Preparing natural caches and fixed teacher continuation banks")
    train = make_bank(adapter, train_histories, config.horizon, seed=config.seed+10, anchors=train_target)
    valid = make_bank(adapter, validation_histories, config.horizon, seed=config.seed+11, anchors=validation_target)
    train_states, validation_states = train.states, valid.states
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
        fraction = config.decoder_min_lr_fraction + (1-config.decoder_min_lr_fraction)*.5*(
            1+np.cos(np.pi*step/config.decoder_steps))
        optimizer.param_groups[0]["lr"] = config.decoder_lr*fraction
        ids = torch.tensor(rng.integers(len(train_states), size=config.batch_size), device=adapter.device)
        loss = (decoder(adapter, train_states[ids])-train_target[ids]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    decoder.load_state_dict(best_state)
    encoder = AffineEncoder.initialize(adapter, train_states, train_target)
    model = GBSAbstraction(adapter, decoder, encoder, architecture)
    for stage, steps in [("encoder", config.encoder_steps), ("joint", config.joint_steps)]:
        decoder.requires_grad_(stage == "joint")
        stage_train = encode_bank(adapter, decoder, train) if stage == "encoder" else train
        stage_valid = encode_bank(adapter, decoder, valid) if stage == "encoder" else valid
        optimizer = torch.optim.Adam([dict(params=encoder.parameters(), lr=config.encoder_lr),
                                      dict(params=decoder.parameters(), lr=config.joint_decoder_lr)])
        best, best_state = float("inf"), None
        for step in range(steps+1):
            if step % config.validate_every == 0 or step == steps:
                with torch.no_grad():
                    totals = [float(abstraction_loss(model, stage_valid.select(slice(left, left+256)),
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
            if stage == "encoder":
                loss, _ = _encoder_loss(model, stage_train.select(ids), config, step, steps)
            else:
                loss = abstraction_loss(model, stage_train.select(ids), tracking_weight=config.tracking_weight,
                    coordinate_weight=config.coordinate_weight, roundtrip_weight=config.roundtrip_weight)["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *decoder.parameters()], 1., error_if_nonfinite=True)
            optimizer.step()
        _restore(model, best_state)
    diagnostics.update(center=center.cpu().tolist(), whitener=whitener.cpu().tolist())
    return FitResult(model.eval(), asdict(config), trace, diagnostics)


@torch.no_grad()
def _evaluate_bank(model, bank: DataBank | EncodedBank):
    rows = {key: [] for key in ["roundtrip_mse", "single_kl", "repeated_kl", "tracking_mse"]}
    for left in range(0, len(bank.tokens), 256):
        data = bank.select(slice(left, left+256))
        terms = _path_terms(model, data)
        rows["roundtrip_mse"].append(terms["roundtrip"])
        rows["single_kl"].append(terms["single"].sum(1))
        rows["repeated_kl"].append(terms["repeated"].sum(1))
        rows["tracking_mse"].append(terms["tracking"].mean(1))
    summary = {}
    for key, parts in rows.items():
        value = torch.cat(parts)
        if not bool(torch.isfinite(value).all()): raise ValueError("Nonfinite evaluation")
        summary[key] = dict(mean=float(value.mean()), standard_error=float(value.std(correction=1)/len(value)**.5))
    return summary


def evaluate(model, histories, *, horizon=16, seed=DEFAULT_EVALUATION_SEED):
    """Evaluate on independent teacher draws; predictive KL is summed over horizon."""
    histories = prepare_histories(histories, model.adapter.vocab_size, min_rows=2)
    if horizon < 1:
        raise ValueError("Evaluation requires at least two histories and a positive horizon")
    model.eval()
    bank = make_bank(model.adapter, histories, horizon, seed=seed)
    return _evaluate_bank(model, bank)


def refine_encoder(model, train_histories, validation_histories, *, config=None, progress=print,
                   checkpoint_directory=None):
    """Improve round trips with decoder and teacher fixed and validation-only limits.

    This starts fresh Adam and samples new continuation banks. The starting
    encoder is always an eligible checkpoint. Existing model weights are
    updated in place; save the original abstraction first to retain it.
    Natural targets are cached once with the decoder frozen. teacher_every=1
    uses the full loss on every update. Larger values interleave cheap root
    round-trip updates and a full teacher update at that interval (and the end).
    Infeasible validation checkpoints roll back the encoder and reset Adam
    when alternating updates are enabled. Final selection always uses the full
    validation metrics, with the starting encoder retained as a fallback.
    If checkpoint_directory is supplied, create a new directory and save each
    validation-selected best abstraction there, including the starting model.
    """
    config = config or RefineConfig()
    started = time.perf_counter()
    _validate_config(config)
    if config.preservation_slack < 0: raise ValueError("preservation_slack cannot be negative")
    train_histories = prepare_histories(train_histories, model.adapter.vocab_size)
    validation_histories = prepare_histories(validation_histories, model.adapter.vocab_size, min_rows=2)
    check_history_pair(model.adapter, train_histories, validation_histories, config.horizon)
    model.adapter.eval().requires_grad_(False)
    model.decoder.requires_grad_(False)
    initial_decoder = copy.deepcopy(model.decoder.state_dict())
    train = make_bank(model.adapter, train_histories, config.horizon, seed=config.seed+10, decoder=model.decoder)
    valid = make_bank(model.adapter, validation_histories, config.horizon, seed=config.seed+11, decoder=model.decoder)
    prepared = time.perf_counter()
    starting = _evaluate_bank(model, valid)
    validation_seconds = time.perf_counter()-prepared
    limits = {key: max(0., starting[key]["mean"])*(1+config.preservation_slack)
              for key in ["single_kl", "repeated_kl", "tracking_mse"]}
    # KL is summed over time, so allow one machine epsilon per token.
    epsilon = torch.finfo(model.adapter.dtype).eps
    tolerances = {key: epsilon*(config.horizon if key.endswith("kl") else 1) for key in limits}
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=config.lr)
    rng = np.random.default_rng(config.seed)
    trace, best, best_state = [], starting["roundtrip_mse"]["mean"], copy.deepcopy(model.encoder.state_dict())
    selected_step, teacher_steps, roundtrip_steps, rollbacks = 0, 0, 0, 0
    if checkpoint_directory is not None:
        checkpoint_directory = Path(checkpoint_directory)
        checkpoint_directory.mkdir(parents=True, exist_ok=False)
    for step in range(config.steps+1):
        if step % config.validate_every == 0 or step == config.steps:
            validation_started = time.perf_counter()
            measured = starting if step == 0 else _evaluate_bank(model, valid)
            validation_seconds += time.perf_counter()-validation_started
            feasible = all(measured[key]["mean"] <= value+tolerances[key] for key, value in limits.items())
            rollback = config.teacher_every > 1 and not feasible
            trace.append(dict(stage="refine", step=step, feasible=feasible, metrics=measured,
                elapsed_seconds=time.perf_counter()-started, teacher_steps=teacher_steps,
                roundtrip_steps=roundtrip_steps, rolled_back=rollback))
            improved = feasible and measured["roundtrip_mse"]["mean"] < best
            if improved:
                best, best_state = measured["roundtrip_mse"]["mean"], copy.deepcopy(model.encoder.state_dict())
                selected_step = step
            if rollback:
                model.encoder.load_state_dict(best_state)
                optimizer.state.clear()
                rollbacks += 1
            if checkpoint_directory is not None and (step == 0 or improved):
                model.save(checkpoint_directory/f"step-{step:06d}.pt")
                (checkpoint_directory/f"step-{step:06d}.json").write_text(json.dumps(dict(
                    trace=trace, config=asdict(config), validation_limits=limits,
                    validation_tolerances=tolerances), indent=2)+"\n")
            if progress: progress(f"refine {step}/{config.steps}: round-trip MSE {measured['roundtrip_mse']['mean']:.6g}; feasible={feasible}")
        if step == config.steps: break
        optimizer.param_groups[0]["lr"] = config.lr*(.1+.45*(1+np.cos(np.pi*step/config.steps)))
        ids = torch.tensor(rng.integers(len(train.tokens), size=config.batch_size), device=model.adapter.device)
        loss, teacher_step = _encoder_loss(model, train.select(ids), config, step, config.steps)
        teacher_steps += int(teacher_step)
        roundtrip_steps += int(not teacher_step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    model.encoder.load_state_dict(best_state)
    assert all(torch.equal(value, model.decoder.state_dict()[key]) for key, value in initial_decoder.items())
    return FitResult(model.eval(), asdict(config), trace,
                     dict(validation_limits=limits, validation_tolerances=tolerances,
                         selected_step=selected_step, teacher_steps=teacher_steps,
                         roundtrip_steps=roundtrip_steps, rollbacks=rollbacks,
                         timing_seconds=dict(preparation=prepared-started, validation=validation_seconds,
                                             total=time.perf_counter()-started)))
