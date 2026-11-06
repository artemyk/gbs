import itertools
import copy

import pytest
import torch

from gbs import TransformersAdapter, Architecture, AttentionDecoder, AffineEncoder, GBSAbstraction
from gbs.adapters import CacheModel
from native_transformer import make_model, native_log_probs
from reference_transformer import rotary
from gbs.data import encode_bank, make_bank
from gbs.fit import abstraction_loss
from transformers import DynamicCache


@pytest.fixture(params=list(itertools.product([False, True], [False, True])))
def adapter(request):
    rope, bos = request.param
    torch.manual_seed(42)
    return TransformersAdapter(make_model(rope=rope, bos_visible=bos).double())


def test_native_predictions_cache_states_and_context_bounds(adapter):
    torch.manual_seed(1)
    tokens = torch.randint(adapter.vocab_size, (2, 12))
    native = native_log_probs(adapter.model, tokens)
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
    with pytest.raises(ValueError, match="nonempty"):
        adapter.trajectory(state, tokens[:, :0])


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
    keys, values, current = adapter._unpack(state.detach())
    n = keys.shape[1]
    query = rotary(q, n) if adapter.rope else q
    weights = torch.exp(torch.einsum("lhqd,btlhd->blhqt", query, keys)/2)
    if not adapter.attend_to_bos:
        weights[..., 0] = 0
    z = weights.sum(-1)
    numerator = torch.einsum("blhqt,btlhd->blhqd", weights, values)
    response = features[:, adapter.token_features:].reshape(2, 2, 2, 2, 4)
    assert features.shape == (2, adapter.token_features+q.numel())
    torch.testing.assert_close(features[:, :adapter.token_features], current)
    torch.testing.assert_close(response, numerator/z[..., None])
    # A common key shift changes each denominator but cancels in softmax.
    cache, current = adapter.cache_from_state(state.detach())
    shifted_cache = DynamicCache(ddp_cache_data=[(layer.keys+.7, layer.values) for layer in cache.layers])
    shifted = adapter.state_from_cache(shifted_cache, current)
    torch.testing.assert_close(adapter.features(shifted, q), features)
    empty = adapter.history_state(h[:, :0])
    empty_features = adapter.features(empty, q)
    assert empty_features.shape == features.shape
    torch.testing.assert_close(empty_features[:, :adapter.token_features], empty)
    assert torch.count_nonzero(empty_features[:, adapter.token_features:]) == 0


def test_encoder_writes_all_cache_and_embedding_coordinates_and_checkpoint_roundtrip(adapter, tmp_path):
    words = torch.randint(3, (20, 4))
    raw = adapter.history_state(words)
    architecture = Architecture(2, 2, 8)
    decoder = AttentionDecoder(adapter, architecture)
    codes = torch.randn(20, 2, dtype=torch.float64)
    encoder = AffineEncoder.initialize(adapter, raw, codes)
    model = GBSAbstraction(adapter, decoder, encoder, architecture)
    written = model.encode(codes)
    mask = adapter.encoder_mask(raw)
    assert bool(mask.all())
    cache, current = adapter.cache_from_state(written)
    assert cache.get_seq_length() == 4
    assert written.shape[1] == 4*adapter.entry_width+adapter.token_features
    # All cache entries use the same writer, including the first (BOS if visible).
    loss = adapter.predict(written).square().sum()
    gradient = torch.autograd.grad(loss, encoder.linear.weight)[0]
    assert gradient[:4*adapter.entry_width].norm() > 0
    assert gradient[-adapter.token_features:].norm() > 0
    model.train()
    assert not adapter.model.training
    path = tmp_path/"abstraction.pt"
    model.save(path)
    restored = GBSAbstraction.load(path, adapter)
    torch.testing.assert_close(restored.decode(written), model.decode(written), atol=0, rtol=0)
    torch.testing.assert_close(restored.encode(codes), model.encode(codes), atol=0, rtol=0)
    with pytest.raises(FileExistsError): model.save(path)
    malformed = torch.load(path, weights_only=True)
    malformed["format_version"] = 999
    malformed_path = tmp_path/"malformed.pt"
    torch.save(malformed, malformed_path)
    with pytest.raises(ValueError, match="Unsupported abstraction checkpoint format"):
        GBSAbstraction.load(malformed_path, adapter)
    with torch.no_grad(): adapter.model.get_output_embeddings().weight[0, 0].add_(.01)
    with pytest.raises(ValueError, match="match"): GBSAbstraction.load(path, adapter)


def test_checkpoint_load_without_source_preserves_predictions_and_dtype(adapter, tmp_path):
    path = tmp_path/"teacher"
    adapter.model.save_pretrained(path)
    loaded = TransformersAdapter.from_checkpoint(path)
    assert loaded.dtype == adapter.dtype
    assert loaded.fingerprint() == adapter.fingerprint()
    tokens = torch.randint(adapter.vocab_size, (3, adapter.max_history))
    torch.testing.assert_close(native_log_probs(loaded.model, tokens), native_log_probs(adapter.model, tokens), atol=1e-10, rtol=1e-10)
    raw = loaded.history_state(tokens[:, :4])
    torch.testing.assert_close(loaded.predict(raw), adapter.predict(adapter.history_state(tokens[:, :4])), atol=1e-10, rtol=1e-10)



def test_single_pass_sampling_matches_generic_sampling(adapter):
    calls = []
    handle = adapter.model.register_forward_pre_hook(lambda *args: calls.append(1))
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
    handle.remove()


def test_native_cache_intervention_can_replace_a_shorter_history_without_a_position_counter():
    torch.manual_seed(11)
    adapter = TransformersAdapter(make_model(bos_visible=True, context=32))
    original = adapter.history_state(torch.randint(3, (2, 10)))
    replacement = adapter.history_state(torch.randint(3, (2, 16))).requires_grad_(True)
    cache, current = adapter.cache_from_state(replacement)
    assert isinstance(cache, DynamicCache) and cache.get_seq_length() == 16
    assert replacement.shape[1] == 16*adapter.entry_width+adapter.token_features
    assert original.shape[1] == 10*adapter.entry_width+adapter.token_features
    reconstructed = adapter.state_from_cache(cache, current)
    torch.testing.assert_close(reconstructed, replacement, atol=0, rtol=0)
    native = adapter.model(inputs_embeds=current[:, None], past_key_values=cache,
                           use_cache=True).logits[:, 0, :3].log_softmax(-1)
    actual, successor = adapter.advance(replacement, torch.tensor([0, 1]))
    torch.testing.assert_close(actual, native)
    assert adapter.cache_from_state(successor)[0].get_seq_length() == 17
    assert adapter.cache_from_state(replacement)[0].get_seq_length() == 16
    actual.square().mean().backward()
    assert replacement.grad[:, :16*adapter.entry_width].norm() > 0
    assert replacement.grad[:, -adapter.token_features:].norm() > 0


def test_native_eager_and_sdpa_backends_preserve_written_cache_predictions_and_gradients():
    torch.manual_seed(27)
    sdpa = TransformersAdapter(make_model(bos_visible=True))
    eager = TransformersAdapter(copy.deepcopy(sdpa.model), attn_implementation="eager")
    assert sdpa.fingerprint() == eager.fingerprint()
    state = sdpa.history_state(torch.randint(3, (3, 4))).requires_grad_(True)
    a, b = sdpa.predict(state), eager.predict(state)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
    first = torch.autograd.grad(a.square().mean(), state)[0]
    second = torch.autograd.grad(b.square().mean(), state)[0]
    torch.testing.assert_close(first, second, atol=2e-6, rtol=2e-6)


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
