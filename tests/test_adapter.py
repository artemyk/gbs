import itertools

import pytest
import torch

from gbs import BeliefStateAdapter, Architecture, AttentionDecoder, AffineEncoder, GBSAbstraction
from gbs.adapters import CacheModel
from reference_transformer import Config, Transformer, rotary


@pytest.fixture(params=list(itertools.product([False, True], [False, True], ["standard", "log", "raw"])))
def adapter(request):
    rope, bos, mode = request.param
    torch.manual_seed(42)
    return BeliefStateAdapter(Transformer(Config(rope=rope, bos_visible=bos,
        attention_mass=mode=="log", attention_sums=mode=="raw")).double())


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


def test_intervention_gradients_and_true_attention_sums(adapter):
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
    recovered_z = features[:, 36:44].expm1().reshape(2, 2, 2, 2)
    torch.testing.assert_close(recovered_z, z)
    torch.testing.assert_close(response*recovered_z[..., None], numerator)


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
    with torch.no_grad(): adapter.model.unembed.bias.add_(.01)
    with pytest.raises(ValueError, match="match"): GBSAbstraction.load(path, adapter)
