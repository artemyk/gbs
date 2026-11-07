from dataclasses import asdict
import hashlib
import json

import pytest
import torch

from gbs import BeliefStateAdapter, GBSAbstraction
from gbs.adapters import words
from gbs.cli import main
from reference_transformer import Config, Transformer


@pytest.fixture(params=[False, True])
def inputs(request, tmp_path):
    torch.manual_seed(5)
    rope = request.param
    teacher = Transformer(Config(rope=rope))
    checkpoint = tmp_path/"teacher.pt"
    torch.save(dict(config=asdict(teacher.cfg), model=teacher.state_dict()), checkpoint)
    roots = words(teacher.cfg.vocab, 3).to(torch.uint8)
    splits = dict(train=roots[:12], validation=roots[12:20], test=roots[20:])
    histories = tmp_path/"histories.pt"
    torch.save(splits, histories)
    config = tmp_path/"fit.json"
    config.write_text(json.dumps(dict(dimension=1, queries_per_head=1, hidden=4, horizon=2,
        future_length=1, spectral_samples=16, decoder_steps=2, joint_steps=10,
        batch_size=2, validate_every=1, seed=10)))
    return checkpoint, histories, config


def test_shared_fit_and_evaluate_commands(inputs, tmp_path):
    checkpoint, histories, config = inputs
    common = ["--teacher", str(checkpoint), "--histories", str(histories)]
    fitted = tmp_path/"fit"
    main(["fit", *common, "--output", str(fitted), "--config", str(config)])
    adapter = BeliefStateAdapter.from_checkpoint(checkpoint)
    model = GBSAbstraction.load(fitted/"abstraction.pt", adapter)
    trace = json.loads((fitted/"trace.json").read_text())
    assert {row["stage"] for row in trace} == {
        "decoder", "joint"}
    assert bool(model.encoder.writable[:-1].all())
    assert not bool(model.encoder.writable[-1])
    evaluated = tmp_path/"evaluation"
    main(["evaluate", *common, "--abstraction", str(fitted/"abstraction.pt"),
          "--output", str(evaluated), "--horizon", "2"])
    assert json.loads((evaluated/"test_metrics.json").read_text()) == json.loads((fitted/"test_metrics.json").read_text())
    metadata = json.loads((fitted/"run.json").read_text())
    assert metadata["teacher_fingerprint"] == adapter.fingerprint()
    assert metadata["command"] == "fit"
    assert metadata["histories_sha256"] == hashlib.sha256((fitted/"histories.pt").read_bytes()).hexdigest()
    assert metadata["input_histories_sha256"] == hashlib.sha256(histories.read_bytes()).hexdigest()
    assert metadata["evaluation_seed"] == 1000
    evaluation_metadata = json.loads((evaluated/"run.json").read_text())
    assert evaluation_metadata["source_abstraction_sha256"] == hashlib.sha256((fitted/"abstraction.pt").read_bytes()).hexdigest()
    with pytest.raises(SystemExit):
        main(["fit", *common, "--output", str(fitted), "--config", str(config)])


def test_cli_rejects_heldout_overlap_before_training(inputs, tmp_path):
    checkpoint, histories, config = inputs
    splits = torch.load(histories, weights_only=True)
    splits["test"] = splits["train"][:2]
    torch.save(splits, histories)
    output = tmp_path/"should-not-exist"
    with pytest.raises(SystemExit):
        main(["fit", "--teacher", str(checkpoint), "--histories", str(histories),
              "--output", str(output), "--config", str(config)])
    assert not output.exists()
