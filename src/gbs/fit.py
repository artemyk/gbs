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
from .models import AffineEncoder, GBSAbstraction
from .decoder import DecoderConfig, fit_decoder, validate_decoder_config
from .data import DEFAULT_EVALUATION_SEED, DataBank, check_history_pair, make_bank, prepare_histories
from ._training import cosine_fraction, require_finite, require_integer


@dataclass
class FitConfig(DecoderConfig):
    horizon: int = 16
    future_length: int = 4
    spectral_samples: int = 512
    joint_steps: int = 10000
    joint_decoder_lr: float = 1e-4
    encoder_lr: float = 3e-4
    tracking_weight: float = .1
    coordinate_weight: float = .1
    roundtrip_weight: float = .1
    max_outcomes: int = 65536
    teacher_every: int = 10


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
    validate_decoder_config(config)
    for key, value in asdict(config).items():
        if key in DecoderConfig.__dataclass_fields__:
            continue
        if key.endswith("_steps"):
            require_integer(key, value)
        elif key in ("horizon", "teacher_every",
                     "future_length", "spectral_samples", "max_outcomes"):
            require_integer(key, value, minimum=1)
        elif key.endswith("_lr"):
            require_finite(key, value, positive=True)
        elif key.endswith("_weight"):
            require_finite(key, value)
    if config.teacher_every > 1 and config.roundtrip_weight + config.tracking_weight == 0:
        raise ValueError("Alternating updates require a positive round-trip or tracking weight")


def conditional_kl(reference, log_probabilities):
    return (reference*(reference.clamp_min(torch.finfo(reference.dtype).tiny).log()-log_probabilities)).sum(-1)


def _path_terms(model, bank: DataBank):
    """Per-root errors shared by training and evaluation; natural targets are detached."""
    code = model.decode(bank.states)
    written = model.encode(code)
    single_lp, written_states = model.adapter.trajectory(written, bank.tokens)
    with torch.no_grad():
        _, natural_states = model.adapter.trajectory(bank.states, bank.tokens)
        targets = [model.decode(state) for state in natural_states]
    return dict(roundtrip=(model.decode(written)-code.detach()).square().mean(1),
        tracking=torch.stack([(model.decode(state)-target).square().mean(1)
                              for state, target in zip(written_states, targets)], 1),
        single=conditional_kl(bank.reference, single_lp),
        repeated=conditional_kl(bank.reference, model.repeated(code, bank.tokens)))


def abstraction_loss(model, bank: DataBank, *, tracking_weight=.1, roundtrip_weight=.1):
    """Mean token KLs, successor tracking including the root, and root round trip.

    The multi-prefix coordinate anchor is added separately by the fitter.
    """
    terms = _path_terms(model, bank)
    root_error = terms["roundtrip"].mean()
    tracking = torch.cat([terms["roundtrip"][:, None], terms["tracking"]], 1).mean()
    single, repeated = (terms[key].mean() for key in ["single", "repeated"])
    total = .5*(single+repeated)+tracking_weight*tracking+roundtrip_weight*root_error
    return dict(total=total, single=single, repeated=repeated, tracking=tracking, roundtrip=root_error)


def _joint_loss(model, bank: DataBank, config, step):
    """Use current decoder codes on every update; only natural targets detach.

    A full rollout occurs at each teacher interval and the final update. Cheap
    steps avoid transformer rollouts and retain the root coefficient of the
    full objective, including its contribution to tracking. Both maps train.
    """
    if (step+1) % config.teacher_every == 0 or step+1 == config.joint_steps:
        return abstraction_loss(model, bank, tracking_weight=config.tracking_weight,
            roundtrip_weight=config.roundtrip_weight)["total"], True
    code = model.decode(bank.states)
    weight = config.roundtrip_weight + config.tracking_weight/(bank.tokens.shape[1]+1)
    return weight*(model.decode(model.encode(code))-code.detach()).square().mean(), False


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
    """Fit a shared decoder, initialize an affine encoder, then fit both jointly.

    A GHMM supplies default coordinates from the teacher's conditional futures;
    a supplied GHMM or callable may replace it. Callable outputs have exactly
    ``config.dimension`` columns. The root rows define the sampling measure.
    After decoder regression across positions and least-squares encoder
    initialization, both maps train in one joint phase. Nine cheap round trips
    alternate with one full prediction/tracking update by default. The GHMM
    coordinate anchor is active on every update. Natural codes are recomputed
    with the current decoder, and full validation selects both maps together.
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
        train_target = (train_target-center)@whitener
    if progress: progress("Preparing natural caches and fixed teacher continuation banks")
    train = make_bank(adapter, train_histories, config.horizon, seed=config.seed+10)
    valid = make_bank(adapter, validation_histories, config.horizon, seed=config.seed+11)
    def coordinates(histories):
        code = target(histories.to(target_device)).detach().to(device=adapter.device, dtype=adapter.dtype)
        return (code-center)@whitener
    train_sequences = torch.cat([train_histories.cpu(), train.tokens.cpu()], 1)
    validation_sequences = torch.cat([validation_histories.cpu(), valid.tokens.cpu()], 1)
    decoder_result = fit_decoder(adapter, train_sequences, validation_sequences, coordinates,
                                 config=config, progress=progress)
    decoder, architecture = decoder_result.decoder, decoder_result.architecture
    trace = decoder_result.trace.copy()
    diagnostics['decoder'] = decoder_result.diagnostics
    encoder = AffineEncoder.initialize(adapter, train.states, train_target)
    model = GBSAbstraction(adapter, decoder, encoder, architecture)
    started = time.perf_counter()
    decoder.requires_grad_(True)
    encoder.requires_grad_(True)
    rng = np.random.default_rng(config.seed)
    modules = (encoder, decoder)
    rates = (config.encoder_lr, config.joint_decoder_lr)
    parameters = [parameter for module in modules for parameter in module.parameters()]
    optimizer = torch.optim.Adam([dict(params=module.parameters(), lr=rate)
                                  for module, rate in zip(modules, rates)])
    best, best_state, selected_step = float("inf"), None, 0
    teacher_steps, roundtrip_steps = 0, 0
    for step in range(config.joint_steps+1):
        if step % config.validate_every == 0 or step == config.joint_steps:
            with torch.no_grad():
                totals = [float(abstraction_loss(model, valid.select(slice(left, left+256)),
                    tracking_weight=config.tracking_weight,
                    roundtrip_weight=config.roundtrip_weight)["total"])*min(256, len(valid.states)-left)
                    for left in range(0, len(valid.states), 256)]
            coordinate_score, by_length = decoder_result.validation.measure(decoder, adapter)
            score = sum(totals)/len(valid.states)+config.coordinate_weight*coordinate_score
            if not np.isfinite(score):
                raise ValueError("Nonfinite joint validation loss")
            trace.append(dict(stage="joint", step=step, validation_loss=score,
                              validation_coordinate_mse=coordinate_score,
                              validation_coordinate_by_length=by_length,
                              elapsed_seconds=time.perf_counter()-started,
                              teacher_steps=teacher_steps, roundtrip_steps=roundtrip_steps))
            if score < best:
                best, selected_step = score, step
                best_state = copy.deepcopy([module.state_dict() for module in modules])
            if progress: progress(f"joint {step}/{config.joint_steps}: validation loss {score:.6g}")
        if step == config.joint_steps: break
        fraction = cosine_fraction(step, config.joint_steps)
        for group, rate in zip(optimizer.param_groups, rates):
            group["lr"] = rate*fraction
        ids = torch.tensor(rng.integers(len(train.states), size=config.batch_size), device=adapter.device)
        loss, teacher_step = _joint_loss(model, train.select(ids), config, step)
        # Always draw the prefix minibatch, so changing an auxiliary weight
        # does not change the subsequent root/prefix sampling sequence.
        x, y = decoder_result.training.sample(rng, config.batch_size)
        if config.coordinate_weight:
            loss = loss+config.coordinate_weight*(model.decode(x)-y).square().mean()
        teacher_steps += int(teacher_step)
        roundtrip_steps += int(not teacher_step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
    for module, state in zip(modules, best_state):
        module.load_state_dict(state)
    diagnostics["joint"] = dict(selected_step=selected_step, validation_loss=best,
        teacher_steps=teacher_steps, roundtrip_steps=roundtrip_steps,
        total_seconds=time.perf_counter()-started)
    diagnostics.update(center=center.cpu().tolist(), whitener=whitener.cpu().tolist())
    return FitResult(model.eval(), asdict(config), trace, diagnostics)


@torch.no_grad()
def _evaluate_bank(model, bank: DataBank):
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
