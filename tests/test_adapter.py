import itertools
from dataclasses import asdict

import pytest
import torch

from gbs import BeliefStateAdapter, Architecture, AttentionDecoder, AffineEncoder, GBSAbstraction
from gbs.adapters import CacheModel
from reference_transformer import Config, Transformer, rotary
from gbs.data import encode_bank, make_bank
from gbs.fit import abstraction_loss


@pytest.fixture(params=list(itertools.product([False, True], [False, True])))
def adapter(request):
    rope, bos = request.param
    torch.manual_seed(42)
    return BeliefStateAdapter(Transformer(Config(rope=rope, bos_visible=bos)).double())


def test_native_predictions_cache_states_and_context_bounds(adapter):
    torch.manual_seed(1)
    tokens = torch.randint(adapter.vocab_size, (2, 12))
    native = adapter.model(tokens).log_softmax(-1)
    for n in [0, 1, 4, 11]:
        state = adapter.history_state(tokens[:, :n])
        torch.testing.assert_close(adapter.predict(state), native[:, n], atol=1e-10, rtol=1e-10)
        horizon = min(3, 12-n)
        lp, states = adapter.trajectory(state, tokens[:, n:n+horizon])
        sequential, successive = CacheModel.trajectory(adapter, state, tokens[:, n:n+horizon])
        torch.testing.assert_close(lp, native[:, n:n+horizon], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(lp, sequential, atol=1e-10, rtol=1e-10)
        for offset, (a, b) in enumerate(zip(states, successive), 1):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
            torch.testing.assert_close(a, adapter.history_state(tokens[:, :n+offset]), atol=1e-10, rtol=1e-10)
    state = adapter.history_state(tokens)
    with pytest.raises(ValueError, match="context"): adapter.predict(state)


def test_intervention_gradients_and_normalized_decoder_inputs(adapter):
    h = torch.tensor([[0, 1, 2, 1], [2, 1, 0, 2]])
    state = adapter.history_state(h).requires_grad_(True)
    continuation = torch.tensor([[2, 1, 0], [0, 2, 1]])
    parallel = adapter.trajectory(state, continuation)[0]
    sequential = CacheModel.trajectory(adapter, state, continuation)[0]
    a = torch.autograd.grad(parallel.square().sum(), state, retain_graph=True)[0]
    b = torch.autograd.grad(sequential.square().sum(), state)[0]
    torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
    assert a.norm() > 0 and all(p.grad is None for p in adapter.model.parameters())
    q = torch.randn(2, 2, 2, 4, dtype=torch.float64)
    features = adapter.features(state.detach(), q)
    keys, values, _, n = adapter._unpack(state.detach())
    query = rotary(q, n) if adapter.rope else q
    weights = torch.exp(torch.einsum("lhqd,btlhd->blhqt", query, keys)/2)
    z = weights.sum(-1)
    numerator = torch.einsum("blhqt,btlhd->blhqd", weights, values)
    response = features[:, 4:36].reshape(2, 2, 2, 2, 4)
    assert features.shape == (2, adapter.token_features+q.numel())
    torch.testing.assert_close(features[:, :4], state.detach()[:, -5:-1])
    torch.testing.assert_close(response, numerator/z[..., None])
    # A common key shift changes each denominator but cancels in softmax.
    shifted = adapter._pack(keys+0.7, values, state.detach()[:, -5:-1], n)
    torch.testing.assert_close(adapter.features(shifted, q), features)
    empty = adapter.history_state(h[:, :0])
    empty_features = adapter.features(empty, q)
    assert empty_features.shape == features.shape
    torch.testing.assert_close(empty_features[:, :4], empty[:, :4])
    assert torch.count_nonzero(empty_features[:, 4:]) == 0


def test_encoder_fixed_coordinates_and_checkpoint_roundtrip(adapter, tmp_path):
    words = torch.randint(3, (20, 4))
    raw = adapter.history_state(words)
    architecture = Architecture(2, 2, 8)
    decoder = AttentionDecoder(adapter, architecture)
    codes = torch.randn(20, 2, dtype=torch.float64)
    encoder = AffineEncoder.initialize(adapter, raw, codes)
    model = GBSAbstraction(adapter, decoder, encoder, architecture)
    written = model.encode(codes)
    mask = adapter.encoder_mask(raw)
    assert bool(mask[:-1].all()) and not bool(mask[-1])
    torch.testing.assert_close(written[:, ~mask], raw[:, ~mask])
    # All cache entries use the same writer, including the first (BOS if visible).
    loss = adapter.predict(written).square().sum()
    gradient = torch.autograd.grad(loss, encoder.linear.weight)[0]
    assert gradient[:adapter.entry_width].norm() > 0
    model.train()
    assert not adapter.model.training
    path = tmp_path/"abstraction.pt"
    model.save(path)
    restored = GBSAbstraction.load(path, adapter)
    torch.testing.assert_close(restored.decode(written), model.decode(written), atol=0, rtol=0)
    torch.testing.assert_close(restored.encode(codes), model.encode(codes), atol=0, rtol=0)
    with pytest.raises(FileExistsError): model.save(path)
    obsolete = torch.load(path, weights_only=True)
    obsolete["format_version"] = 1
    old_path = tmp_path/"obsolete.pt"
    torch.save(obsolete, old_path)
    with pytest.raises(ValueError, match="normalized-attention decoder inputs"):
        GBSAbstraction.load(old_path, adapter)
    with torch.no_grad(): adapter.model.unembed.bias.add_(.01)
    with pytest.raises(ValueError, match="match"): GBSAbstraction.load(path, adapter)


def test_checkpoint_load_without_source_preserves_predictions_and_dtype(adapter, tmp_path):
    path = tmp_path/"teacher.pt"
    # Inactive experimental flags appear in some standard upstream checkpoints.
    config = dict(asdict(adapter.model.cfg), attention_mass=False, attention_sums=False)
    torch.save(dict(config=config, model=adapter.model.state_dict()), path)
    loaded = BeliefStateAdapter.from_checkpoint(path)
    assert loaded.dtype == adapter.dtype
    assert loaded.fingerprint() == adapter.fingerprint()
    tokens = torch.randint(adapter.vocab_size, (3, adapter.max_history))
    torch.testing.assert_close(loaded.model(tokens), adapter.model(tokens), atol=1e-10, rtol=1e-10)
    raw = loaded.history_state(tokens[:, :4])
    torch.testing.assert_close(loaded.predict(raw), adapter.predict(adapter.history_state(tokens[:, :4])), atol=1e-10, rtol=1e-10)


def test_checkpoint_rejects_missing_or_mismatched_architecture(tmp_path):
    path = tmp_path/"bad.pt"
    torch.save(dict(weights={}), path)
    with pytest.raises(ValueError, match="config and model"):
        BeliefStateAdapter.from_checkpoint(path)
    model = Transformer(Config())
    config = asdict(model.cfg)
    torch.save(dict(config=dict(config, unknown_architecture=True), model=model.state_dict()), path)
    with pytest.raises(TypeError, match="unknown_architecture"):
        BeliefStateAdapter.from_checkpoint(path)
    torch.save(dict(config=dict(config, d_model=32), model=model.state_dict()), path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        BeliefStateAdapter.from_checkpoint(path)
    for name in ("attention_mass", "attention_sums"):
        torch.save(dict(config=dict(config, **{name: True}), model=model.state_dict()), path)
        with pytest.raises(ValueError, match="Only standard softmax attention"):
            BeliefStateAdapter.from_checkpoint(path)
        setattr(model.cfg, name, True)
        with pytest.raises(ValueError, match="Only standard softmax attention"):
            BeliefStateAdapter(model)
        delattr(model.cfg, name)


def test_single_pass_sampling_matches_generic_sampling(adapter, monkeypatch):
    original = adapter._pass
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(adapter, "_pass", counted)
    for length in [0, 4]:
        state = adapter.history_state(torch.randint(adapter.vocab_size, (3, length)))
        uniforms = torch.rand(3, 3, dtype=adapter.dtype)
        uniforms[0] = 0
        uniforms[1] = 1
        calls.clear()
        expected_tokens, expected_probabilities = CacheModel.sample(adapter, state, uniforms)
        assert len(calls) == 6
        calls.clear()
        tokens, probabilities = adapter.sample(state, uniforms)
        assert len(calls) == 3
        torch.testing.assert_close(tokens, expected_tokens)
        torch.testing.assert_close(probabilities, expected_probabilities, atol=0, rtol=0)


def test_cached_natural_targets_preserve_loss_and_encoder_gradients(adapter):
    histories = torch.randint(adapter.vocab_size, (12, 4))
    bank = make_bank(adapter, histories, 3, seed=7)
    architecture = Architecture(2, 2, 8)
    decoder = AttentionDecoder(adapter, architecture).requires_grad_(False)
    codes = decoder(adapter, bank.states).detach()
    encoder = AffineEncoder.initialize(adapter, bank.states, codes)
    model = GBSAbstraction(adapter, decoder, encoder, architecture)
    bank.anchors = torch.randn_like(codes)
    cached = encode_bank(adapter, decoder, bank)
    streamed = make_bank(adapter, histories, 3, seed=7, decoder=decoder)
    for name in ["codes", "targets", "tokens", "reference"]:
        torch.testing.assert_close(getattr(cached, name), getattr(streamed, name), atol=0, rtol=0)
    assert not hasattr(cached, "states")
    raw = abstraction_loss(model, bank)
    encoded = abstraction_loss(model, cached)
    for name in raw:
        torch.testing.assert_close(raw[name], encoded[name], atol=1e-11, rtol=1e-11)
    first = torch.autograd.grad(raw["total"], tuple(encoder.parameters()))
    second = torch.autograd.grad(encoded["total"], tuple(encoder.parameters()))
    for a, b in zip(first, second):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    decoder.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen decoder"):
        encode_bank(adapter, decoder, bank)
    with pytest.raises(ValueError, match="frozen decoder"):
        abstraction_loss(model, cached)


def test_observed_queries_match_native_projection_before_rope(adapter):
    tokens = torch.randint(adapter.vocab_size, (3, 12))
    native = []
    handles = [block.qkv.register_forward_hook(
        lambda module, inputs, output: native.append(output.detach().clone()))
        for block in adapter.model.blocks]
    adapter.model(tokens)
    for handle in handles:
        handle.remove()
    for n in [0, 1, 5, 11]:
        queries = adapter.observed_queries(adapter.history_state(tokens[:, :n]))
        expected = torch.stack([value.reshape(3, 12, 3, adapter.n_heads, adapter.head_dim)
                                [:, n, 0] for value in native], 1)
        torch.testing.assert_close(queries, expected, atol=1e-10, rtol=1e-10)
        if adapter.rope and n:
            assert not torch.allclose(queries, rotary(queries, n))
