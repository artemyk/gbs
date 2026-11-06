"""Shared artifact and provenance output for the demo and command line."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from .fit import evaluate
from .data import DEFAULT_EVALUATION_SEED


def checkpoint_hash(path):
    """Hash a single checkpoint or a native model directory, including relative filenames."""
    path = Path(path)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(file.relative_to(path).as_posix().encode()+b"\0")
        digest.update(hashlib.sha256(file.read_bytes()).digest())
    return digest.hexdigest()


def save_run(directory, model, splits, *, teacher_checkpoint, horizon, evaluation_seed=DEFAULT_EVALUATION_SEED,
             result=None, metadata=None, initial_abstraction=None):
    """Evaluate on test histories and save one consistent experiment record.

    ``result`` supplies fitted weights, configuration, and trace. Without it,
    write an evaluation-only record. Existing directories are never overwritten.
    The stored history hash identifies the saved, normalized tensor dictionary.
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    if result is not None and result.abstraction is not model:
        raise ValueError("The fit result and evaluated model must be the same object")
    metrics = evaluate(model, splits["test"], horizon=horizon, seed=evaluation_seed)
    if result is None:
        directory.mkdir(parents=True, exist_ok=False)
    else:
        result.save(directory)
    torch.save(splits, directory/"histories.pt")
    record = dict(metadata or {})
    record.update(teacher_sha256=checkpoint_hash(teacher_checkpoint),
        teacher_fingerprint=model.adapter.fingerprint(), adapter_spec=model.adapter.spec(),
        histories_sha256=hashlib.sha256((directory/"histories.pt").read_bytes()).hexdigest(),
        split_sizes={name: len(value) for name, value in splits.items()},
        horizon=horizon, evaluation_seed=evaluation_seed, device=str(model.adapter.device),
        threads=torch.get_num_threads(), torch_version=str(torch.__version__), numpy_version=np.__version__,
        transformers_version=transformers.__version__)
    if initial_abstraction is not None:
        record["initial_abstraction_sha256"] = hashlib.sha256(Path(initial_abstraction).read_bytes()).hexdigest()
    for name, value in [("run", record), ("test_metrics", metrics)]:
        (directory/f"{name}.json").write_text(json.dumps(value, indent=2)+"\n")
    return metrics
