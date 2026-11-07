"""Shared fit and evaluation commands for belief-state checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import torch

from .adapters import BeliefStateAdapter
from .data import DEFAULT_EVALUATION_SEED, history_splits
from .fit import FitConfig, fit_abstraction
from .models import GBSAbstraction
from .runs import save_run


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gbs", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["fit", "evaluate"]:
        command = commands.add_parser(name)
        command.add_argument("--teacher", type=Path, required=True, help="Transformer config + state-dictionary checkpoint")
        command.add_argument("--histories", type=Path, required=True, help="Tensor dictionary containing train/validation/test histories")
        command.add_argument("--output", type=Path, required=True, help="New output directory")
        command.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
        command.add_argument("--threads", type=int, default=2)
        command.add_argument("--evaluation-seed", type=int, default=DEFAULT_EVALUATION_SEED)
        if name != "fit":
            command.add_argument("--abstraction", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--horizon", type=int, default=16)
        else:
            command.add_argument("--config", type=Path, help="JSON fields for FitConfig")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(f"Output directory already exists: {args.output}")
        if args.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)
        adapter = BeliefStateAdapter.from_checkpoint(args.teacher, device=args.device)
        if args.command == "evaluate":
            config, horizon = None, args.horizon
        else:
            fields = json.loads(args.config.read_text()) if args.config else {}
            config = FitConfig(**fields)
            horizon = config.horizon
        splits = history_splits(torch.load(args.histories, map_location="cpu", weights_only=True), adapter, horizon)
        result = None
        if args.command == "fit":
            result = fit_abstraction(adapter, splits["train"], splits["validation"], config=config)
            model = result.abstraction
        else:
            model = GBSAbstraction.load(args.abstraction, adapter)
        metrics = save_run(args.output, model, splits, teacher_checkpoint=args.teacher, horizon=horizon,
            evaluation_seed=args.evaluation_seed, result=result,
            metadata=dict(command=args.command, input_histories_sha256=hashlib.sha256(args.histories.read_bytes()).hexdigest()),
            source_abstraction=None if args.command == "fit" else args.abstraction)
        print(json.dumps(metrics, indent=2), flush=True)
    except (ValueError, TypeError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
