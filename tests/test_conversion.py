"""Checkpoint conversion preserves the original process without custom execution."""
from dataclasses import asdict

import pytest
import torch

from gbs import TransformersAdapter, convert_belief_state_checkpoint, convert_belief_state_abstraction
from reference_transformer import Config, Transformer, rotary


@pytest.mark.parametrize("bos_visible", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rope_conversion_preserves_native_predictions_and_save_pretrained(bos_visible, dtype, tmp_path):
    torch.manual_seed(7)
    source = Transformer(Config(rope=True, bos_visible=bos_visible)).to(dtype).eval()
    path = tmp_path/"teacher.pt"
    torch.save(dict(config=asdict(source.cfg), model=source.state_dict()), path)
    converted = convert_belief_state_checkpoint(path)
    adapter = TransformersAdapter(converted)
    histories = torch.randint(source.cfg.vocab, (3, source.cfg.context))
    with torch.no_grad():
        expected = source(histories).log_softmax(-1)
        for length in [0, 1, 5, 11]:
            actual = adapter.predict(adapter.history_state(histories[:, :length]))
            torch.testing.assert_close(actual, expected[:, length], atol=2e-6, rtol=2e-6)
    assert converted.model.layers[0].self_attn.q_proj.out_features == source.cfg.n_heads*source.cfg.d_head
    folder = tmp_path/"native"
    converted.save_pretrained(folder)
    restored = TransformersAdapter.from_checkpoint(folder)
    assert restored.dtype == dtype
    assert restored.fingerprint() == adapter.fingerprint()
    torch.testing.assert_close(restored.predict(restored.history_state(histories[:, :5])),
                               adapter.predict(adapter.history_state(histories[:, :5])))


def test_conversion_rejects_unknown_or_incompatible_checkpoint_data(tmp_path):
    source = Transformer(Config())
    path = tmp_path/"bad.pt"
    torch.save(dict(weights={}), path)
    with pytest.raises(ValueError, match="config and model"):
        convert_belief_state_checkpoint(path)
    for field in ["attention_mass", "attention_sums"]:
        torch.save(dict(config=dict(asdict(source.cfg), **{field: True}), model=source.state_dict()), path)
        with pytest.raises(ValueError, match="standard softmax attention"):
            convert_belief_state_checkpoint(path)
    torch.save(dict(config=dict(asdict(source.cfg), unknown_architecture=True), model=source.state_dict()), path)
    with pytest.raises(ValueError, match="Unknown architecture fields"):
        convert_belief_state_checkpoint(path)


@pytest.mark.parametrize("bos_visible", [False, True])
def test_fitted_abstraction_conversion_preserves_roundtrips_and_code_gradients(bos_visible, tmp_path):
    torch.manual_seed(19)
    source = Transformer(Config(rope=True, bos_visible=bos_visible)).double().eval()
    path = tmp_path/"teacher.pt"
    torch.save(dict(config=asdict(source.cfg), model=source.state_dict()), path)
    adapter = TransformersAdapter.from_checkpoint(path)
    length, dimension, hidden, token_width = 4, 2, 4, source.cfg.vocab+1
    cache_width = (length if bos_visible else length-1)*adapter.entry_width
    width = cache_width+token_width
    mean, scale = torch.randn(width+1, dtype=torch.float64)*.1, torch.ones(width+1, dtype=torch.float64)
    mean[-1] = length
    writable = torch.ones(width+1, dtype=torch.bool)
    writable[-1] = False
    writer = torch.randn(width, dimension, dtype=torch.float64)*.1
    bias = torch.randn(width, dtype=torch.float64)*.1
    queries = torch.randn(adapter.n_layers, adapter.n_heads, 1, adapter.head_dim, dtype=torch.float64)
    readout = torch.randn(hidden, token_width+queries.numel(), dtype=torch.float64)*.1
    b1 = torch.randn(hidden, dtype=torch.float64)*.1
    w2, b2 = torch.randn(dimension, hidden, dtype=torch.float64), torch.randn(dimension, dtype=torch.float64)
    original = dict(format_version=2, teacher_fingerprint=adapter.model.config.gbs["source_fingerprint"],
        architecture=dict(dimension=dimension, queries_per_head=1, hidden=hidden, learn_queries=True),
        decoder={"queries": queries, "net.0.weight": readout, "net.0.bias": b1,
                 "net.2.weight": w2, "net.2.bias": b2},
        encoder={"mean": mean, "scale": scale, "writable": writable,
                 "linear.weight": writer, "linear.bias": bias})
    checkpoint = tmp_path/"abstraction.pt"
    torch.save(original, checkpoint)
    migrated = convert_belief_state_abstraction(checkpoint, adapter)
    code = torch.randn(3, dimension, dtype=torch.float64, requires_grad=True)
    old_state = mean[:-1]+torch.nn.functional.linear(code, writer, bias)
    kv = old_state[:, :cache_width].reshape(3, -1, adapter.n_layers, adapter.n_heads, 2, adapter.head_dim)
    q = rotary(queries, length)
    scores = torch.einsum("lhqd,btlhd->blhqt", q, kv[..., 0, :])/adapter.head_dim**.5
    response = torch.einsum("blhqt,btlhd->blhqd", scores.softmax(-1), kv[..., 1, :]).flatten(1)
    measurements = torch.cat([old_state[:, cache_width:], response], 1)
    expected = torch.nn.functional.linear(torch.nn.functional.linear(measurements, readout, b1).tanh(), w2, b2)
    actual = migrated.decode(migrated.encode(code))
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    a = torch.autograd.grad(actual.square().sum(), code)[0]
    b = torch.autograd.grad(expected.square().sum(), code)[0]
    torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-8)
