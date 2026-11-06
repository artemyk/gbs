import copy

import pytest
import torch
from torch import nn

from gbs import CacheModel, FitConfig, RefineConfig, fit_abstraction, refine_encoder, evaluate, GBSAbstraction
from gbs.adapters import words
from gbs.fit import abstraction_loss, conditional_kl
from gbs.data import make_bank


class CoinMixture(CacheModel):
    """Analytic Bayesian mixture with a one-dimensional log-odds state."""
    vocab_size, n_layers, n_heads, head_dim, token_features = 2, 1, 1, 1, 1
    max_history = 20

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.float64), requires_grad=False)
        self.register_buffer("likelihood", torch.tensor([[.8, .2], [.3, .7]], dtype=torch.float64))

    def history_state(self, h):
        z = self.likelihood[1, h].log().sum(1)-self.likelihood[0, h].log().sum(1)
        return torch.stack([z, z.new_full(z.shape, h.shape[1])], 1)

    def advance(self, states, tokens):
        b = states[:, 0].sigmoid()
        q = (1-b[:, None])*self.likelihood[0]+b[:, None]*self.likelihood[1]
        increment = (self.likelihood[1]/self.likelihood[0]).log()[tokens]
        return q.log(), torch.stack([states[:, 0]+increment, states[:, 1]+1], 1)

    def features(self, states, queries):
        z = states[:, :1]
        response = z.expand(-1, queries.numel())
        return torch.cat([z, response], 1)

    def encoder_mask(self, states):
        return torch.tensor([True, False])


def test_full_fit_refinement_and_safe_serialization(tmp_path):
    adapter = CoinMixture()
    histories = words(2, 5)
    train, valid = histories[::2], histories[1::2]
    config = FitConfig(dimension=1, queries_per_head=2, hidden=8, horizon=3, future_length=2,
                       spectral_samples=32, decoder_steps=100, encoder_steps=10, joint_steps=10,
                       batch_size=8, validate_every=10, seed=41)
    result = fit_abstraction(adapter, train, valid, config=config, progress=None)
    decoder_rows = [v for v in result.trace if v["stage"]=="decoder"]
    assert min(v["validation_mse"] for v in decoder_rows) < decoder_rows[0]["validation_mse"]
    output = tmp_path/"fit"
    result.save(output)
    restored = GBSAbstraction.load(output/"abstraction.pt", adapter)
    torch.testing.assert_close(restored.encode(torch.tensor([[.1]], dtype=torch.float64)), result.abstraction.encode(torch.tensor([[.1]], dtype=torch.float64)))
    before = copy.deepcopy(restored.decoder.state_dict())
    refined = refine_encoder(restored, train, valid,
        config=RefineConfig(steps=2, horizon=3, batch_size=8, validate_every=1), progress=None,
        checkpoint_directory=tmp_path/"encoder-checkpoints")
    assert all(torch.equal(v, restored.decoder.state_dict()[k]) for k, v in before.items())
    assert any(v["feasible"] for v in refined.trace)
    starting_checkpoint = GBSAbstraction.load(tmp_path/"encoder-checkpoints/step-000000.pt", adapter)
    assert all(torch.equal(v, starting_checkpoint.decoder.state_dict()[k]) for k, v in before.items())
    checkpoints = sorted((tmp_path/"encoder-checkpoints").glob("*.pt"))
    assert len(checkpoints) >= 1
    for checkpoint in checkpoints:
        assert checkpoint.with_suffix(".json").is_file()
        GBSAbstraction.load(checkpoint, adapter)
    latest = GBSAbstraction.load(checkpoints[-1], adapter)
    assert all(torch.equal(v, latest.encoder.state_dict()[k])
               for k, v in restored.encoder.state_dict().items())
    metrics = evaluate(restored, valid, horizon=3, seed=131)
    assert all(torch.isfinite(torch.tensor(v["mean"])) for v in metrics.values())
    with pytest.raises(ValueError, match="overlap"):
        fit_abstraction(adapter, train, train, config=config, progress=None)


def test_repeated_compression_kl_matches_exact_joint_kl():
    adapter = CoinMixture()
    histories = words(2, 4)
    result = fit_abstraction(adapter, histories[::2], histories[1::2],
        config=FitConfig(dimension=1, queries_per_head=1, hidden=4, horizon=3, future_length=2,
                         spectral_samples=20, decoder_steps=2, encoder_steps=0, joint_steps=0,
                         batch_size=4, validate_every=1), progress=None)
    model = result.abstraction
    root = adapter.history_state(torch.tensor([[1, 0, 1, 1]]))
    future = words(2, 3)
    expanded = root.expand(len(future), -1)
    true_lp = adapter.future_log_probs(root, 3)[0]
    compressed = model.repeated(model.decode(expanded), future)
    compressed_joint = compressed.gather(2, future[..., None]).squeeze(2).sum(1)
    torch.testing.assert_close(compressed_joint.exp().sum(), torch.tensor(1., dtype=torch.float64), atol=1e-12, rtol=1e-12)
    reference = adapter.trajectory(expanded, future)[0].exp()
    joint_kl = (true_lp.exp()*(true_lp-compressed_joint)).sum()
    chain_kl = (true_lp.exp()*conditional_kl(reference, compressed).sum(1)).sum()
    torch.testing.assert_close(joint_kl, chain_kl, atol=1e-12, rtol=1e-12)
    joint_kl.backward()
    assert model.encoder.linear.weight.grad.norm() > 0
    assert adapter.anchor.grad is None


def test_fitting_builds_each_root_cache_once_in_bounded_batches():
    class CountingTeacher(CoinMixture):
        def __init__(self):
            super().__init__()
            self.prepared = []

        def history_state(self, history):
            self.prepared.append(len(history))
            return super().history_state(history)

    adapter = CountingTeacher()
    histories = words(2, 10)
    fit_abstraction(adapter, histories[:512], histories[512:],
        initializer=lambda h: h.double().sum(1, keepdim=True), config=FitConfig(
            dimension=1, queries_per_head=1, hidden=4, horizon=2,
            decoder_steps=0, encoder_steps=0, joint_steps=0), progress=None)
    assert sum(adapter.prepared) == len(histories)
    assert max(adapter.prepared) <= 256


def test_loss_and_gradients_match_an_independent_sequential_rollout():
    adapter = CoinMixture()
    histories = words(2, 4)
    result = fit_abstraction(adapter, histories[::2], histories[1::2], config=FitConfig(
        dimension=1, queries_per_head=1, hidden=4, horizon=3, future_length=2,
        spectral_samples=20, decoder_steps=2, encoder_steps=0, joint_steps=0), progress=None)
    model = result.abstraction
    bank = make_bank(adapter, histories[:4], 3, seed=17)
    bank.anchors = torch.zeros(4, 1, dtype=adapter.dtype)
    code = model.decode(bank.states)
    written = model.encode(code)
    natural, single, repeated = bank.states, written, code
    tracking = [(model.decode(written)-code.detach()).square().mean()]
    single_kl, repeated_kl = [], []
    for j, token in enumerate(bank.tokens.unbind(1)):
        lp, single = adapter.advance(single, token)
        with torch.no_grad():
            _, natural = adapter.advance(natural, token)
            target = model.decode(natural)
        tracking.append((model.decode(single)-target).square().mean())
        induced_lp, successor = adapter.advance(model.encode(repeated), token)
        repeated = model.decode(successor)
        single_kl.append(conditional_kl(bank.reference[:, j], lp).mean())
        repeated_kl.append(conditional_kl(bank.reference[:, j], induced_lp).mean())
    expected = (.5*(torch.stack(single_kl).mean()+torch.stack(repeated_kl).mean())
                +.1*torch.stack(tracking).mean()+.1*code.square().mean()+.1*tracking[0])
    actual = abstraction_loss(model, bank)["total"]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    parameters = [p for p in model.parameters() if p.requires_grad]
    a = torch.autograd.grad(actual, parameters, allow_unused=True)
    b = torch.autograd.grad(expected, parameters, allow_unused=True)
    for first, second in zip(a, b):
        if first is None or second is None:
            assert first is None and second is None
        else:
            torch.testing.assert_close(first, second, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("force_rollback", [False, True])
def test_alternating_refinement_checks_full_metrics_and_keeps_a_safe_fallback(force_rollback, monkeypatch):
    import gbs.fit as fitting
    adapter = CoinMixture()
    histories = words(2, 4)
    model = fit_abstraction(adapter, histories[::2], histories[1::2], config=FitConfig(
        dimension=1, queries_per_head=1, hidden=4, horizon=2, future_length=2,
        spectral_samples=20, decoder_steps=2, encoder_steps=0, joint_steps=0), progress=None).abstraction
    original = copy.deepcopy(model.encoder.state_dict())
    evaluate_bank = fitting._evaluate_bank
    calls = []
    def measured(*args):
        metrics = evaluate_bank(*args)
        calls.append(metrics)
        if force_rollback and len(calls) > 1:
            metrics["tracking_mse"]["mean"] += calls[0]["tracking_mse"]["mean"]+1
        return metrics
    monkeypatch.setattr(fitting, "_evaluate_bank", measured)
    result = refine_encoder(model, histories[::2], histories[1::2], config=RefineConfig(
        steps=5, horizon=2, batch_size=4, validate_every=2, teacher_every=3), progress=None)
    assert result.initialization["teacher_steps"] == 2
    assert result.initialization["roundtrip_steps"] == 3
    assert len(calls) == 4
    assert result.initialization["timing_seconds"]["preparation"] >= 0
    if force_rollback:
        assert result.initialization["rollbacks"] == 3
        assert result.initialization["selected_step"] == 0
        assert all(torch.equal(value, model.encoder.state_dict()[key]) for key, value in original.items())
    else:
        selected = next(row for row in result.trace if row["step"] == result.initialization["selected_step"])
        assert selected["feasible"]


@pytest.mark.parametrize("interval", [0, -1, 1.5])
def test_invalid_teacher_update_intervals_are_rejected(interval):
    with pytest.raises(ValueError, match="teacher_every"):
        refine_encoder(None, None, None, config=RefineConfig(teacher_every=interval), progress=None)


def test_cheap_encoder_update_uses_no_teacher_rollout(monkeypatch):
    from gbs import Architecture, AttentionDecoder, AffineEncoder
    from gbs.data import encode_bank
    from gbs.fit import _encoder_loss
    adapter = CoinMixture()
    roots = words(2, 3)
    bank = make_bank(adapter, roots, 2, seed=17)
    architecture = Architecture(1, 1, 4)
    decoder = AttentionDecoder(adapter, architecture).requires_grad_(False)
    encoder = AffineEncoder.initialize(adapter, bank.states, decoder(adapter, bank.states))
    model = GBSAbstraction(adapter, decoder, encoder, architecture)
    cached = encode_bank(adapter, decoder, bank)
    calls = []
    advance = adapter.advance
    def counted(*args):
        calls.append(1)
        return advance(*args)
    monkeypatch.setattr(adapter, "advance", counted)
    config = RefineConfig(teacher_every=3)
    loss, full = _encoder_loss(model, cached, config, 0, 3)
    loss.backward()
    assert not full and not calls
    assert encoder.linear.weight.grad is not None
    _, full = _encoder_loss(model, cached, config, 2, 3)
    assert full and calls


def test_encoded_bank_preparation_streams_before_building_the_next_raw_batch():
    from gbs import Architecture, AttentionDecoder
    class CountingTeacher(CoinMixture):
        def __init__(self):
            super().__init__()
            self.prepared = []
        def history_state(self, histories):
            self.prepared.append(len(histories))
            return super().history_state(histories)
    adapter = CountingTeacher()
    decoder = AttentionDecoder(adapter, Architecture(1, 1, 4)).requires_grad_(False)
    decoded_after = []
    hook = decoder.register_forward_pre_hook(lambda *args: decoded_after.append(sum(adapter.prepared)))
    try:
        bank = make_bank(adapter, words(2, 9), 2, seed=5, batch_size=64, decoder=decoder)
    finally:
        hook.remove()
    assert decoded_after[0] == 64
    assert sum(adapter.prepared) == 512 and max(adapter.prepared) == 64
    assert len(bank.codes) == 512 and not hasattr(bank, "states")
