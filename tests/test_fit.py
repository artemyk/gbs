import copy

import pytest
import torch
from torch import nn

from gbs import CacheModel, FitConfig, fit_abstraction, evaluate, GBSAbstraction
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


def test_full_joint_fit_and_safe_serialization(tmp_path):
    adapter = CoinMixture()
    histories = words(2, 5)
    train, valid = histories[::2], histories[1::2]
    config = FitConfig(dimension=1, queries_per_head=2, hidden=8, horizon=3, future_length=2,
                       spectral_samples=32, decoder_steps=100, joint_steps=10,
                       batch_size=8, validate_every=10, seed=41)
    result = fit_abstraction(adapter, train, valid, config=config, progress=None)
    decoder_rows = [v for v in result.trace if v["stage"]=="decoder"]
    assert min(v["validation_mse"] for v in decoder_rows) < decoder_rows[0]["validation_mse"]
    output = tmp_path/"fit"
    result.save(output)
    restored = GBSAbstraction.load(output/"abstraction.pt", adapter)
    torch.testing.assert_close(restored.encode(torch.tensor([[.1]], dtype=torch.float64)), result.abstraction.encode(torch.tensor([[.1]], dtype=torch.float64)))
    assert {row['stage'] for row in result.trace} == {'decoder', 'joint'}
    assert result.initialization['joint']['teacher_steps'] == 1
    assert result.initialization['joint']['roundtrip_steps'] == 9
    assert all(p.grad is None for p in adapter.parameters())
    metrics = evaluate(restored, valid, horizon=3, seed=131)
    assert all(torch.isfinite(torch.tensor(v["mean"])) for v in metrics.values())
    with pytest.raises(ValueError, match="overlap"):
        fit_abstraction(adapter, train, train, config=config, progress=None)


def test_repeated_compression_kl_matches_exact_joint_kl():
    adapter = CoinMixture()
    histories = words(2, 4)
    result = fit_abstraction(adapter, histories[::2], histories[1::2],
        config=FitConfig(dimension=1, queries_per_head=1, hidden=4, horizon=3, future_length=2,
                         spectral_samples=20, decoder_steps=2, joint_steps=0,
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


def test_fitting_prepares_root_and_prefix_caches_in_bounded_batches():
    class CountingTeacher(CoinMixture):
        def __init__(self):
            super().__init__()
            self.prepared = []

        def history_state(self, history):
            self.prepared.append((len(history), history.shape[1]))
            return super().history_state(history)

    adapter = CountingTeacher()
    histories = words(2, 10)
    fit_abstraction(adapter, histories[:512], histories[512:],
        initializer=lambda h: h.double().sum(1, keepdim=True), config=FitConfig(
            dimension=1, queries_per_head=1, hidden=4, horizon=2,
            decoder_steps=0, joint_steps=0), progress=None)
    # Root banks, one randomly assigned prefix per training parent, and a
    # shared 256-parent validation sample at each of lengths 0 through 12.
    assert sum(count for count, _ in adapter.prepared) == 1024+512+256*13
    assert max(count for count, _ in adapter.prepared) <= 256
    assert {length for _, length in adapter.prepared} == set(range(13))


def test_loss_and_gradients_match_an_independent_sequential_rollout():
    adapter = CoinMixture()
    histories = words(2, 4)
    result = fit_abstraction(adapter, histories[::2], histories[1::2], config=FitConfig(
        dimension=1, queries_per_head=1, hidden=4, horizon=3, future_length=2,
        spectral_samples=20, decoder_steps=2, joint_steps=0), progress=None)
    model = result.abstraction
    bank = make_bank(adapter, histories[:4], 3, seed=17)
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
                +.1*torch.stack(tracking).mean()+.1*tracking[0])
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


@pytest.mark.parametrize("interval", [0, -1, 1.5])
def test_invalid_teacher_update_intervals_are_rejected(interval):
    with pytest.raises(ValueError, match="teacher_every"):
        fit_abstraction(None, None, None, config=FitConfig(teacher_every=interval), progress=None)


def test_cheap_joint_update_uses_current_codes_and_trains_both_maps_without_rollouts(monkeypatch):
    from gbs import Architecture, AttentionDecoder, AffineEncoder, BeliefStateAdapter
    from gbs.fit import _joint_loss
    from reference_transformer import Config, make_model
    torch.manual_seed(17)
    adapter = BeliefStateAdapter(make_model(Config(vocab=2, bos_visible=True)).double())
    bank = make_bank(adapter, words(2, 3), 2, seed=17)
    architecture = Architecture(2, 1, 4)
    decoder = AttentionDecoder(adapter, architecture)
    encoder = AffineEncoder.initialize(adapter, bank.states, decoder(adapter, bank.states))
    model = GBSAbstraction(adapter, decoder, encoder, architecture)
    config = FitConfig(joint_steps=20)
    calls = []
    advance = adapter.advance
    def counted(*args):
        calls.append(1)
        return advance(*args)
    monkeypatch.setattr(adapter, 'advance', counted)
    parameters = [*encoder.parameters(), *decoder.parameters()]
    for iteration in range(2):
        code = model.decode(bank.states)
        expected = (.1+.1/3)*(model.decode(model.encode(code))-code.detach()).square().mean()
        loss, full = _joint_loss(model, bank, config, iteration)
        assert not full and not calls
        torch.testing.assert_close(loss, expected)
        actual_grad = torch.autograd.grad(loss, parameters)
        expected_grad = torch.autograd.grad(expected, parameters)
        for actual, wanted in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, wanted)
        assert sum(g.square().sum() for g in actual_grad[:2]) > 0
        assert sum(g.square().sum() for g in actual_grad[2:]) > 0
        assert actual_grad[2].norm() > 0  # Learned attention queries also train.
        with torch.no_grad():
            for parameter, gradient in zip(parameters, actual_grad):
                parameter.sub_(.001*gradient)
    _, full = _joint_loss(model, bank, config, 9)
    assert full and calls
    calls.clear()
    _, full = _joint_loss(model, bank, FitConfig(joint_steps=13), 12)
    assert full and calls  # A partial final cycle also uses the full loss.


def test_joint_phase_reuses_continuations_anchors_every_update_and_restores_both_maps(monkeypatch):
    import gbs.fit as fitting
    from gbs.decoder import PrefixBank
    original_bank, original_loss = fitting.make_bank, fitting._joint_loss
    original_step, original_sample = torch.optim.Adam.step, PrefixBank.sample
    banks, full_updates, snapshots, samples = [], [], {}, []
    current = None
    def bank(*args, **kwargs):
        value = original_bank(*args, **kwargs)
        banks.append(value)
        return value
    def loss(model, data, config, step):
        nonlocal current
        current = model
        assert all(p.requires_grad for p in [*model.decoder.parameters(), *model.encoder.parameters()])
        assert all(not p.requires_grad for p in model.adapter.parameters())
        if step == 0:
            snapshots[0] = copy.deepcopy((model.decoder.state_dict(), model.encoder.state_dict()))
        value, full = original_loss(model, data, config, step)
        full_updates.append(full)
        return value, full
    def update(optimizer, *args, **kwargs):
        assert len(optimizer.param_groups) == 2
        result = original_step(optimizer, *args, **kwargs)
        snapshots[len(full_updates)] = copy.deepcopy((current.decoder.state_dict(), current.encoder.state_dict()))
        return result
    def sample(*args):
        samples.append(1)
        return original_sample(*args)
    monkeypatch.setattr(fitting, 'make_bank', bank)
    monkeypatch.setattr(fitting, '_joint_loss', loss)
    monkeypatch.setattr(torch.optim.Adam, 'step', update)
    monkeypatch.setattr(PrefixBank, 'sample', sample)
    roots = words(2, 4)
    config = FitConfig(dimension=1, queries_per_head=1, hidden=8, horizon=2,
        decoder_steps=0, joint_steps=13, batch_size=8, validate_every=2, seed=41)
    result = fit_abstraction(CoinMixture(), roots[::2], roots[1::2], config=config,
        initializer=lambda h: h.double().sum(1, keepdim=True), progress=None)
    assert len(banks) == 2 and len(samples) == 13
    assert full_updates == [False]*9+[True, False, False, True]
    rows = [row for row in result.trace if row['stage'] == 'joint']
    assert {row['stage'] for row in result.trace} == {'decoder', 'joint'}
    selected = min(rows, key=lambda row: row['validation_loss'])
    diagnostics = result.initialization['joint']
    assert diagnostics['selected_step'] == selected['step']
    assert diagnostics['teacher_steps'] == 2 and diagnostics['roundtrip_steps'] == 11
    for module, saved in zip((result.abstraction.decoder, result.abstraction.encoder), snapshots[selected['step']]):
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, saved[name])
    assert any(not torch.equal(snapshots[0][0][key], value) for key, value in snapshots[13][0].items())
    assert any(not torch.equal(snapshots[0][1][key], value) for key, value in snapshots[13][1].items())


def test_default_joint_schedule():
    config = FitConfig()
    assert config.joint_steps == 10000 and config.teacher_every == 10
    assert config.encoder_lr == 3e-4 and config.joint_decoder_lr == 1e-4


@pytest.mark.parametrize('field', ['joint_steps', 'future_length', 'teacher_every', 'batch_size'])
@pytest.mark.parametrize('value', [1.5, float('nan'), True])
def test_noninteger_counts_fail_before_teacher_preparation(field, value):
    with pytest.raises(ValueError, match=field):
        fit_abstraction(None, None, None, config=FitConfig(**{field: value}), progress=None)


def test_cheap_updates_require_a_roundtrip_loss():
    with pytest.raises(ValueError, match='round-trip'):
        fit_abstraction(None, None, None,
            config=FitConfig(tracking_weight=0, roundtrip_weight=0), progress=None)


def test_joint_optimizer_decays_both_learning_rates(monkeypatch):
    rates = []
    original = torch.optim.Adam.step
    def recorded(optimizer, *args, **kwargs):
        rates.append([group['lr'] for group in optimizer.param_groups])
        return original(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, 'step', recorded)
    roots = words(2, 4)
    fit_abstraction(CoinMixture(), roots[::2], roots[1::2], config=FitConfig(
        dimension=1, queries_per_head=1, hidden=4, horizon=2, decoder_steps=0,
        joint_steps=4, validate_every=1),
        initializer=lambda h: h.double().sum(1, keepdim=True), progress=None)
    assert len(rates) == 4 and rates[0] == [3e-4, 1e-4]
    assert rates[-1] == pytest.approx([rate*.23180194846605365 for rate in rates[0]])


@pytest.mark.parametrize("steps, fraction", [(0, .01), (4, .01), (4, 1.)])
def test_decoder_learning_rate_schedule_reaches_the_optimizer(steps, fraction, monkeypatch):
    rates = []
    original = torch.optim.Adam.step
    def recorded(optimizer, *args, **kwargs):
        rates.append(float(optimizer.param_groups[0]["lr"]))
        return original(optimizer, *args, **kwargs)
    monkeypatch.setattr(torch.optim.Adam, "step", recorded)
    histories = words(2, 4)
    result = fit_abstraction(CoinMixture(), histories[::2], histories[1::2],
        initializer=lambda h: h.double().sum(1, keepdim=True), config=FitConfig(
            dimension=1, queries_per_head=1, hidden=4, horizon=2,
            decoder_steps=steps, joint_steps=0,
            decoder_min_lr_fraction=fraction), progress=None)
    assert result.config["decoder_min_lr_fraction"] == fraction
    assert len(rates) == steps
    if steps:
        assert rates[0] == result.config["decoder_lr"]
        if fraction < 1:
            assert rates[-1] < rates[0]
            assert all(a >= b for a, b in zip(rates, rates[1:]))
        else:
            assert all(rate == rates[0] for rate in rates)


@pytest.mark.parametrize("fraction", [0, -.1, 1.1, float("nan")])
def test_invalid_decoder_learning_rate_fractions_are_rejected(fraction):
    with pytest.raises(ValueError, match="decoder_min_lr_fraction"):
        fit_abstraction(None, None, None, config=FitConfig(decoder_min_lr_fraction=fraction), progress=None)


@pytest.mark.parametrize("field", ["tracking_weight", "coordinate_weight", "roundtrip_weight",
    "encoder_lr", "joint_decoder_lr"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_fit_settings_fail_before_teacher_preparation(field, value):
    with pytest.raises(ValueError, match=field):
        fit_abstraction(None, None, None, config=FitConfig(**{field: value}), progress=None)
