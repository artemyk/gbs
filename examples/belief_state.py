"""Fit a GBS abstraction to a large RoPE model from artemyk/belief-state.

Run from an installed checkout; see README.md for the complete commands.
The process below supplies histories only. No latent states or exact Bayesian
beliefs are passed to the fitter.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from gbs import (BeliefStateAdapter, FitConfig, GBSAbstraction, RefineConfig,
                 evaluate, fit_abstraction, refine_encoder)


def sample_histories(size, length, epsilon, seed):
    """Stationary MRRXOR histories, with the jump applied before each emission."""
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    emit = np.zeros((2, 5, 5))
    emit[0, 0, 1] = emit[1, 0, 2] = .5
    emit[0, 1, 4] = emit[1, 1, 3] = .5
    emit[0, 2, 3] = emit[1, 2, 4] = .5
    emit[1, 3, 0] = emit[0, 4, 0] = 1
    jump = np.full((5, 5), epsilon/4)
    np.fill_diagonal(jump, 1-epsilon)
    transition = jump @ emit
    values, vectors = np.linalg.eig(transition.sum(0).T)
    stationary = vectors[:, np.argmin(abs(values-1))].real
    stationary /= stationary.sum()
    rng = np.random.default_rng(seed)
    hidden = rng.choice(5, size=size, p=stationary)
    cdf = transition.transpose(1, 0, 2).reshape(5, 10).cumsum(1)[:, :-1]
    tokens = np.empty((size, length), dtype=np.int64)
    for position in range(length):
        outcome = (rng.random(size)[:, None] >= cdf[hidden]).sum(1)
        tokens[:, position], hidden = outcome//5, outcome % 5
    return torch.from_numpy(tokens)


def split_histories(histories, seed):
    """Assign distinct roots to disjoint splits; retain their sampling multiplicity."""
    unique, inverse = np.unique(histories.numpy(), axis=0, return_inverse=True)
    if len(unique) < 30:
        raise ValueError("Too few distinct roots for training, validation, and test")
    order = np.random.default_rng(seed).permutation(len(unique))
    group = np.empty(len(unique), dtype=int)
    train_end, val_end = int(.8*len(unique)), int(.9*len(unique))
    group[order[:train_end]], group[order[train_end:val_end]], group[order[val_end:]] = 0, 1, 2
    return tuple(histories[torch.from_numpy(group[inverse] == i)] for i in range(3))


def geometry(model, histories, output):
    import matplotlib.pyplot as plt

    with torch.no_grad():
        code = model.decode(model.adapter.history_state(histories.to(model.adapter.device)))
        roundtrip = model.decode(model.encode(code))
    code, roundtrip = code.cpu().numpy(), roundtrip.cpu().numpy()
    color = histories[:, -2:].numpy() @ np.array([2, 1])
    figure, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for row, (a, b) in enumerate([(0, 1), (2, 3)]):
        for column, (points, title) in enumerate([(code, "Decoded teacher"), (roundtrip, "Encoder round trip")]):
            axes[row, column].scatter(points[:, a], points[:, b], c=color, cmap="viridis", s=7, alpha=.5)
            axes[row, column].set(xlabel=f"z{a+1}", ylabel=f"z{b+1}", title=title, aspect="equal")
        for axis in ["x", "y"]:
            limits = [getattr(ax, f"get_{axis}lim")() for ax in axes[row]]
            for ax in axes[row]:
                getattr(ax, f"set_{axis}lim")(min(x[0] for x in limits), max(x[1] for x in limits))
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--belief-state", type=Path, required=True, help="Local artemyk/belief-state checkout")
    parser.add_argument("--checkpoint", type=Path, help="Defaults to runs/L4_d64_rope/eps0.05/s0.pt")
    parser.add_argument("--model-source", type=Path, help="Defaults to MODEL_CHECKOUT/model.py; use matching source for variants")
    parser.add_argument("--output", type=Path, default=Path("outputs/rope"))
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--samples", type=int, help="4096 by default; 256 with --quick")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=.05, help="MRRXOR history distribution; match the checkpoint")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="Small end-to-end smoke run, not a converged fit")
    parser.add_argument("--plot", action="store_true", help="Save decoded and round-trip coordinate projections")
    parser.add_argument("--refine-steps", type=int, default=0, help="Optional frozen-decoder refinement after saving the initial fit")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new directory")
    if args.refine_steps < 0:
        parser.error("refine-steps cannot be negative")
    torch.set_num_threads(args.threads)
    checkpoint = args.checkpoint or args.belief_state/"runs/L4_d64_rope/eps0.05/s0.pt"
    source = args.model_source or args.belief_state/"model.py"
    adapter = BeliefStateAdapter.from_checkpoint(checkpoint, model_source=source, device=args.device)
    config = FitConfig(seed=args.seed)
    if args.quick:
        config = FitConfig(seed=args.seed, spectral_samples=128, decoder_steps=200,
            encoder_steps=100, joint_steps=200, batch_size=32, horizon=8, validate_every=100)
    histories = sample_histories(args.samples or (256 if args.quick else 4096), 16, args.epsilon, args.seed)
    train, validation, test = split_histories(histories, args.seed+1)
    print(f"Device: {adapter.device}; train/validation/test rows: {len(train)}/{len(validation)}/{len(test)}", flush=True)
    progress = lambda message: print(message, flush=True)
    result = fit_abstraction(adapter, train, validation, config=config, progress=progress)
    result.save(args.output)
    metrics = evaluate(result.abstraction, test, horizon=config.horizon, seed=args.seed+1000)
    (args.output/"test_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    torch.save(dict(train=train, validation=validation, test=test), args.output/"histories.pt")
    # Verify the saved artifact against the exact teacher used for this run.
    loaded = GBSAbstraction.load(args.output/"abstraction.pt", adapter)
    with torch.no_grad():
        states = adapter.history_state(test[:8].to(adapter.device))
        torch.testing.assert_close(loaded.decode(states), result.abstraction.decode(states), rtol=0, atol=0)
    commit = subprocess.run(["git", "-C", str(args.belief_state), "rev-parse", "HEAD"], capture_output=True, text=True)
    metadata = dict(seed=args.seed, epsilon=args.epsilon, device=args.device,
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        teacher_commit=commit.stdout.strip() if commit.returncode == 0 else None,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        split_sizes=dict(train=len(train), validation=len(validation), test=len(test)), fit=asdict(config))
    (args.output/"demo.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metrics, indent=2), flush=True)
    if args.plot:
        geometry(result.abstraction, test, args.output/"geometry.png")
    if args.refine_steps:
        refined = refine_encoder(result.abstraction, train, validation, config=RefineConfig(
            steps=args.refine_steps, horizon=config.horizon, batch_size=config.batch_size,
            validate_every=config.validate_every, roundtrip_weight=.3, seed=args.seed+2000), progress=progress)
        refined.save(args.output/"refined")
        metrics = evaluate(refined.abstraction, test, horizon=config.horizon, seed=args.seed+1000)
        (args.output/"refined/test_metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    print(f"Saved fit to {args.output}", flush=True)


if __name__ == "__main__":
    main()
