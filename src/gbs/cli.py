"""Shared fit, refinement, and evaluation commands for belief-state checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .adapters import BeliefStateAdapter
from .fit import FitConfig, RefineConfig, _check_histories, evaluate, fit_abstraction, refine_encoder
from .models import GBSAbstraction


def _histories(path, adapter, horizon):
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")
    splits = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(splits, dict) or not {"train", "validation", "test"} <= splits.keys():
        raise ValueError("Histories must contain train, validation, and test tensors")
    for name in ["train", "validation", "test"]:
        h = splits[name]
        if (not isinstance(h, torch.Tensor) or h.ndim != 2 or len(h) < 2
                or h.dtype not in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]):
            raise ValueError(f"{name} must contain at least two integer token histories")
        if bool(((h < 0) | (h >= adapter.vocab_size)).any()):
            raise ValueError(f"Invalid token in {name} histories")
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        _check_histories(adapter, splits[a], splits[b], horizon, False)
    return splits


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gbs", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["fit", "refine", "evaluate"]:
        command = commands.add_parser(name)
        command.add_argument("--teacher", type=Path, required=True, help="Transformer config + state-dictionary checkpoint")
        command.add_argument("--histories", type=Path, required=True, help="Tensor dictionary containing train/validation/test histories")
        command.add_argument("--output", type=Path, required=True, help="New output directory")
        command.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
        command.add_argument("--threads", type=int, default=2)
        command.add_argument("--evaluation-seed", type=int, default=1000)
        if name != "fit":
            command.add_argument("--abstraction", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--horizon", type=int, default=16)
        else:
            command.add_argument("--config", type=Path, help=f"JSON fields for {'FitConfig' if name == 'fit' else 'RefineConfig'}")
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
            config = (FitConfig if args.command == "fit" else RefineConfig)(**fields)
            horizon = config.horizon
        splits = _histories(args.histories, adapter, horizon)
        if args.command == "fit":
            result = fit_abstraction(adapter, splits["train"], splits["validation"], config=config)
            model = result.abstraction
        else:
            model = GBSAbstraction.load(args.abstraction, adapter)
            if args.command == "refine":
                result = refine_encoder(model, splits["train"], splits["validation"], config=config)
        metrics = evaluate(model, splits["test"], horizon=horizon, seed=args.evaluation_seed)
        if args.command == "evaluate":
            args.output.mkdir(parents=True, exist_ok=False)
        else:
            result.save(args.output)
        shutil.copyfile(args.histories, args.output/"histories.pt")
        metadata = dict(command=args.command, teacher_sha256=hashlib.sha256(args.teacher.read_bytes()).hexdigest(),
            teacher_fingerprint=adapter.fingerprint(), teacher_config=adapter.spec()["config"],
            histories_sha256=hashlib.sha256(args.histories.read_bytes()).hexdigest(),
            horizon=horizon, evaluation_seed=args.evaluation_seed, device=args.device, threads=args.threads,
            torch_version=str(torch.__version__), numpy_version=np.__version__)
        if args.command != "fit":
            metadata["initial_abstraction_sha256"] = hashlib.sha256(args.abstraction.read_bytes()).hexdigest()
        for name, value in [("run", metadata), ("test_metrics", metrics)]:
            (args.output/f"{name}.json").write_text(json.dumps(value, indent=2)+"\n")
        print(json.dumps(metrics, indent=2), flush=True)
    except (ValueError, TypeError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
