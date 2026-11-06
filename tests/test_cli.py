import itertools
import hashlib
import json

import pytest
import torch

from gbs import TransformersAdapter, GBSAbstraction
from gbs.adapters import words
from gbs.cli import main
from native_transformer import make_model, native_log_probs


@pytest.fixture(params=list(itertools.product([False, True], [False, True])))
def inputs(request, tmp_path):
    torch.manual_seed(5)
    rope, bos_visible = request.param
    teacher = make_model(rope=rope, bos_visible=bos_visible)
    checkpoint = tmp_path/"teacher"
    teacher.save_pretrained(checkpoint)
    roots = words(len(teacher.config.gbs["token_ids"]), 3).to(torch.uint8)
    splits = dict(train=roots[:12], validation=roots[12:20], test=roots[20:])
    histories = tmp_path/"histories.pt"
    torch.save(splits, histories)
    config = tmp_path/"fit.json"
    config.write_text(json.dumps(dict(dimension=1, queries_per_head=1, hidden=4, horizon=2,
        future_length=1, spectral_samples=16, decoder_steps=2, encoder_steps=1,
        encoder_refine_steps=1, joint_steps=1, joint_refine_steps=1,
        batch_size=2, validate_every=1, seed=10)))
    return checkpoint, histories, config


def test_shared_fit_refine_and_evaluate_commands(inputs, tmp_path):
    checkpoint, histories, config = inputs
    common = ["--teacher", str(checkpoint), "--histories", str(histories)]
    fitted = tmp_path/"fit"
    main(["fit", *common, "--output", str(fitted), "--config", str(config)])
    adapter = TransformersAdapter.from_checkpoint(checkpoint)
    model = GBSAbstraction.load(fitted/"abstraction.pt", adapter)
    trace = json.loads((fitted/"trace.json").read_text())
    assert {row["stage"] for row in trace} == {
        "decoder", "encoder", "encoder_refine", "joint", "joint_refine"}
    assert bool(model.encoder.writable.all())
    before = {key: value.clone() for key, value in model.decoder.state_dict().items()}
    refine_config = tmp_path/"refine.json"
    refine_config.write_text(json.dumps(dict(steps=1, horizon=2, batch_size=2, validate_every=1, seed=10)))
    refined = tmp_path/"refine"
    main(["refine", *common, "--abstraction", str(fitted/"abstraction.pt"),
          "--output", str(refined), "--config", str(refine_config)])
    loaded = GBSAbstraction.load(refined/"abstraction.pt", adapter)
    assert all(torch.equal(value, loaded.decoder.state_dict()[key]) for key, value in before.items())
    evaluated = tmp_path/"evaluation"
    main(["evaluate", *common, "--abstraction", str(refined/"abstraction.pt"),
          "--output", str(evaluated), "--horizon", "2"])
    assert json.loads((evaluated/"test_metrics.json").read_text()) == json.loads((refined/"test_metrics.json").read_text())
    metadata = json.loads((refined/"run.json").read_text())
    assert metadata["teacher_fingerprint"] == adapter.fingerprint()
    assert metadata["command"] == "refine"
    assert metadata["histories_sha256"] == hashlib.sha256((refined/"histories.pt").read_bytes()).hexdigest()
    assert metadata["input_histories_sha256"] == hashlib.sha256(histories.read_bytes()).hexdigest()
    assert metadata["evaluation_seed"] == 1000
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
