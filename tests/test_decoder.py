"""Prefix fitting, native-query freezing, and ridge regression invariants."""
import numpy as np
import pytest
import torch

from gbs import (AffineEncoder, Architecture, AttentionDecoder, BeliefStateAdapter,
                 DecoderConfig, GBSAbstraction, fit_decoder)
from gbs.adapters import words
from gbs.decoder import PrefixBank, _ridge
from reference_transformer import Config, Transformer


@pytest.mark.parametrize('hidden', [None, 8])
def test_observed_prefix_fit_uses_one_frozen_training_bank_and_roundtrips(hidden, tmp_path):
    torch.manual_seed(4)
    adapter = BeliefStateAdapter(Transformer(Config(bos_visible=True)).double())
    parents = words(3, 5)
    train, valid = parents[:100], parents[100:120]
    def coordinates(h):
        return torch.stack([h.sum(1), (h == 1).sum(1)], 1).double()
    result = fit_decoder(adapter, train, valid, coordinates, config=DecoderConfig(
        dimension=2, queries_per_head=3, hidden=hidden, decoder_method='observed',
        decoder_steps=3, validate_every=2, decoder_validation_samples=10), progress=None)
    assert set(result.validation.groups) == set(range(6))
    assert result.diagnostics['training_rows'] == len(train)
    assert result.diagnostics['validation_rows'] == 60
    queries = result.decoder.queries.clone()
    for sample in result.diagnostics['query_samples']:
        n, row, slot = sample['length'], sample['parent_row'], sample['slot']
        expected = adapter.observed_queries(adapter.history_state(train[row:row+1, :n]))[0]
        torch.testing.assert_close(queries[:, :, slot], expected)
    # Joint fitting can unfreeze the readout, but must preserve the feature bank.
    result.decoder.requires_grad_(True)
    assert 'queries' not in dict(result.decoder.named_parameters())
    assert not result.decoder.queries.requires_grad
    roots = adapter.history_state(train[:, :4])
    code = result.decoder(adapter, roots).detach()
    encoder = AffineEncoder.initialize(adapter, roots, code)
    model = GBSAbstraction(adapter, result.decoder, encoder, result.architecture)
    written = model.encode(code)
    model.decode(written).square().mean().backward()
    assert encoder.linear.weight.grad.norm() > 0
    torch.testing.assert_close(queries, result.decoder.queries, atol=0, rtol=0)
    path = tmp_path/'abstraction.pt'; model.save(path)
    restored = GBSAbstraction.load(path, adapter)
    assert 'queries' not in dict(restored.decoder.named_parameters())
    torch.testing.assert_close(restored.decode(written), model.decode(written), atol=0, rtol=0)


def test_ridge_matches_standardized_regularized_least_squares_with_collinearity():
    torch.manual_seed(7)
    adapter = BeliefStateAdapter(Transformer(Config()).double())
    decoder = AttentionDecoder(adapter, Architecture(2, 1, None, False))
    width = decoder.net.in_features
    x = torch.randn(60, width, dtype=torch.float64)
    x[:, 0] = 1  # Constant token coordinate.
    x[:, 1] = x[:, 2]  # Dependent observed-query responses.
    y = torch.randn(60, 2, dtype=torch.float64)
    training = PrefixBank({1: (x[:30], y[:30]), 3: (x[30:], y[30:])}, True)
    validation = PrefixBank({2: (x, y)}, True)
    penalty = .01
    _ridge(decoder, adapter, training, validation, [penalty])
    mean, scale = x.mean(0), x.std(0, correction=0)
    scale[scale == 0] = 1
    z = (x-mean)/scale
    beta = torch.linalg.solve(z.T@z/len(x)+penalty*torch.eye(width, dtype=x.dtype), z.T@(y-y.mean(0))/len(x))
    torch.testing.assert_close(decoder.net(x), z@beta+y.mean(0), atol=1e-10, rtol=1e-10)


def test_prefix_domain_and_parent_split_validation():
    adapter = BeliefStateAdapter(Transformer(Config()))
    parents = words(3, 3)
    target = lambda h: h.float().sum(1, keepdim=True)
    with pytest.raises(ValueError, match='overlap'):
        fit_decoder(adapter, parents, parents[:2], target, progress=None)
    with pytest.raises(ValueError, match='supplied parent'):
        fit_decoder(adapter, parents[:10], parents[10:], target,
                    config=DecoderConfig(decoder_max_length=4), progress=None)
    # Root-only is an explicit control; the default spans all supplied prefixes.
    result = fit_decoder(adapter, parents[:10], parents[10:], target,
        config=DecoderConfig(dimension=1, queries_per_head=1, hidden=4, decoder_steps=0,
                             decoder_min_length=3, decoder_max_length=3), progress=None)
    assert set(result.training.groups) == set(result.validation.groups) == {3}


def test_full_fit_anchors_prefixes_and_keeps_observed_queries_frozen():
    from gbs import FitConfig, fit_abstraction
    torch.manual_seed(5)
    adapter = BeliefStateAdapter(Transformer(Config(bos_visible=True)).double())
    roots = words(3, 3)
    result = fit_abstraction(adapter, roots[:15], roots[15:],
        initializer=lambda h: h.double().sum(1, keepdim=True), config=FitConfig(encoder_refine_steps=0, joint_refine_steps=0,
            dimension=1, queries_per_head=2, hidden=None, decoder_method='observed',
            horizon=2, decoder_steps=0, encoder_steps=2, joint_steps=2,
            validate_every=2), progress=None)
    joint = [r for r in result.trace if r['stage'] == 'joint']
    assert len(joint) == 2
    assert set(joint[-1]['validation_coordinate_by_length']) == {str(n) for n in range(6)}
    assert np.isfinite(joint[-1]['validation_loss'])
    assert not result.abstraction.decoder.queries.requires_grad
    assert result.abstraction.encoder.mean[-1] == 3


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS hardware required")
def test_observed_query_ridge_with_an_mps_teacher():
    adapter = BeliefStateAdapter(Transformer(Config(vocab=2, bos_visible=True))).to('mps')
    parents = words(2, 4)
    result = fit_decoder(adapter, parents[:8], parents[8:],
        lambda h: h.float().sum(1, keepdim=True), config=DecoderConfig(
            dimension=1, queries_per_head=2, hidden=None, decoder_method='observed'), progress=None)
    assert result.decoder.queries.device.type == 'mps'
    assert np.isfinite(result.diagnostics['validation_mse'])


def test_frozen_query_bank_is_collected_in_bounded_batches():
    class CountingAdapter(BeliefStateAdapter):
        def __init__(self, model):
            super().__init__(model)
            self.query_batches = []
        def observed_queries(self, states):
            self.query_batches.append(len(states))
            return super().observed_queries(states)
    adapter = CountingAdapter(Transformer(Config(vocab=2)))
    parents = words(2, 4)
    fit_decoder(adapter, parents[:8], parents[8:], lambda h: h.float().sum(1, keepdim=True),
        config=DecoderConfig(dimension=1, queries_per_head=513, hidden=4, decoder_steps=0,
            decoder_method='observed',
            decoder_min_length=3, decoder_max_length=3), progress=None)
    assert adapter.query_batches == [256, 256, 1]
