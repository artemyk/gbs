import numpy as np
import pytest
import torch

from gbs import GHMM
from gbs.adapters import words
from gbs.ghmm import spectral_operators


def test_spectral_fit_recovers_a_three_symbol_coin_mixture_without_empty_training_row():
    probabilities = np.array([[.6, .3, .1], [.1, .2, .7]])
    prior = np.array([.65, .35])
    histories = words(3, 3).numpy()
    likelihood = np.prod(probabilities[:, histories], axis=2).T
    beliefs = likelihood*prior
    beliefs /= beliefs.sum(1, keepdims=True)
    futures = words(3, 3).numpy()
    future_likelihood = np.prod(probabilities[:, futures], axis=2)
    joint = beliefs @ future_likelihood
    initial = prior @ np.prod(probabilities[:, words(3, 2).numpy()], axis=2)
    fit = spectral_operators(joint, initial, 2, 3, 2)
    model = GHMM(torch.tensor(fit["operators"]), torch.tensor(fit["initial"]))
    test = words(3, 4)
    likelihood = np.prod(probabilities[:, test.numpy()], axis=2).T
    posterior = likelihood*prior
    posterior /= posterior.sum(1, keepdims=True)
    expected = posterior @ np.prod(probabilities[:, words(3, 4).numpy()], axis=2)
    np.testing.assert_allclose(model.future(model.states(test), 4).numpy(), expected, atol=1e-11, rtol=1e-10)
    np.testing.assert_allclose(model.probabilities(model.initial[None])[0].numpy(), prior@probabilities, atol=1e-12)
    np.testing.assert_allclose(fit["operators"][:, :, 0].sum(0), [1, 0], atol=1e-14)
    with pytest.raises(ValueError, match="rank"): spectral_operators(joint, initial, 3, 3, 2)


def test_invalid_probabilities_are_not_hidden():
    model = GHMM(torch.tensor([[[1.2]], [[-.2]]]), torch.tensor([1.]))
    assert model.future(model.initial[None], 1)[0, 1] < 0
    with pytest.raises(ValueError, match="max_outcomes"): words(10, 8)
    with pytest.raises(ValueError, match="probability"):
        spectral_operators(np.array([[-.1, .2, .3, .6]]), [.5, .5], 2, 2, 1)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8])
def test_integer_tokens_have_the_same_categorical_meaning(dtype):
    model = GHMM(torch.tensor([[[.2, 0.], [.5, .7]], [[.8, 0.], [-.5, .3]]]), torch.tensor([1., .5]))
    histories = torch.tensor([[0, 1], [1, 0], [0, 0]], dtype=dtype)
    torch.testing.assert_close(model.states(histories), model.states(histories.long()))
    initial = model.initial.expand(3, -1)
    torch.testing.assert_close(model.update(initial, histories[:, 0]), model.update(initial, histories[:, 0].long()))
    torch.testing.assert_close(model.update(initial, torch.tensor(0, dtype=dtype)), model.update(initial, 0))
    with pytest.raises(ValueError, match="integer"):
        model.update(initial, True)
