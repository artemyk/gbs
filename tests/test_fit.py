import copy

import pytest
import torch
from torch import nn

from gbs import CacheModel, FitConfig, RefineConfig, fit_abstraction, refine_encoder, evaluate, GBSAbstraction
from gbs.adapters import words
from gbs.fit import abstraction_loss, make_bank, conditional_kl


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
        return torch.cat([z, response, response.new_full(response.shape, .6931471805599453), z.new_full(z.shape, .6931471805599453)], 1)

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
        config=RefineConfig(steps=2, horizon=3, batch_size=8, validate_every=1), progress=None)
    assert all(torch.equal(v, restored.decoder.state_dict()[k]) for k, v in before.items())
    assert any(v["feasible"] for v in refined.trace)
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
