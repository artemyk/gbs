# Generalized belief states

Method and experiments: [Generalized belief states in transformers](https://www.lesswrong.com/posts/4f9p7brLdnTMFuwfB/generalized-belief-states-in-transformers-1).

Fit a low-dimensional predictive state to a trained, frozen transformer. A
**decoder** reads the transformer's KV cache and current token into coordinates
`z`. An **encoder** writes those coordinates into a representative KV cache and
a continuous current-token embedding. Running the original transformer on a
written state supplies the prediction and update maps of a recurrent model.

The default initialization is a generalized hidden Markov model (GHMM) fitted
to the transformer's conditional future probabilities. Fitting requires token
histories and access to the trained model. It does not require the process's
hidden states or exact Bayesian beliefs.

## Install and test

Requires Python 3.10 or later, PyTorch, and NumPy. From this repository:

```sh
python -m pip install -e '.[test,demo]'
python -m pytest -q
```

The import name is `gbs`; the distribution name is `generalized-belief-states`.
The `demo` extra adds Matplotlib for optional geometry plots. This repository
does not include pretrained transformer weights or publish a PyPI release.

## Demo: large RoPE models

The supplied adapter supports the transformers in
[artemyk/belief-state](https://github.com/artemyk/belief-state). Start with its
four-layer, width-64 RoPE model, trained on MRRXOR with mixing probability 0.05:

```sh
python examples/belief_state.py --quick --plot
```

This performs a small smoke run and writes `outputs/rope/`. For a full fit,
omit `--quick` and choose a new output directory:

```sh
python examples/belief_state.py --output outputs/rope-full --plot
```

The demo downloads only the roughly 0.6 MB
[checkpoint](https://github.com/artemyk/belief-state/blob/61c2a5570ce7df8ca5a739b2ecc0a27716cea5c4/runs/L4_d64_rope/eps0.05/s0.pt),
pinned to upstream commit `61c2a5570ce7df8ca5a739b2ecc0a27716cea5c4`, and verifies
its SHA-256 checksum. It caches the file at
`checkpoints/L4_d64_rope_eps0.05_s0.pt`; subsequent runs reuse the verified file.
No source checkout is needed. Use `--checkpoint path/to/model.pt` for a local
checkpoint. The adapter also supports visible-BOS, log-normalizer, raw
numerator/denominator, learned-position, and multi-head configurations.

### Checkpoint format

The existing belief-state files already use the usual
[PyTorch checkpoint pattern](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html):
an ordinary dictionary containing a configuration dictionary and a
model `state_dict`, rather than a pickled model object:

```python
torch.save({"config": config_dict, "model": model.state_dict()}, "model.pt")
```

The loader reads `config` and `model`, constructs the supported architecture
included in this package, and loads weights strictly with
`torch.load(..., weights_only=True, map_location="cpu")`. Training metadata in
the checkpoint is ignored. There is no format conversion or remote Python
source execution. Configurations must specify all trained attention and
position options; unknown fields and mismatched weights raise errors. This
loader is specific to the belief-state architecture; other model families use
their own `CacheModel` adapter. The encoder and fitting code are shared.

The demo samples length-16 MRRXOR histories. Distinct histories are assigned to
disjoint training, validation, and test sets, retaining each history's sampling
multiplicity within its split. Only tokens enter the fitter. Continuations and
prediction targets come from the transformer. `--epsilon` controls the history
distribution and should match the chosen experiment. `--device cpu`, `mps`, or
`cuda` selects execution; spectral decompositions and initial least squares run
on CPU in double precision. Transformer execution currently supports float32
and float64.

Outputs include:

- `abstraction.pt`: decoder, encoder, architecture, and teacher fingerprint;
- `config.json`, `initialization.json`, and `trace.json`: settings, fitted GHMM
  and coordinate transformation, and validation checkpoints;
- `test_metrics.json`: evaluation on the test histories with independent teacher
  continuations;
- `histories.pt` and `demo.json`: data splits, seeds, checkpoint hash and URL,
  source commit for the pinned demo, model configuration, and library versions;
- `geometry.png` with `--plot`: projections of decoded states and their encoder
  round trips, colored by the last two tokens.

The quick run checks the workflow; it is not a converged fit. No convergence or
prediction-accuracy threshold is assumed by the fitter.

## Fit a model

```python
import torch
from gbs import BeliefStateAdapter, FitConfig, fit_abstraction, evaluate

teacher = BeliefStateAdapter.from_checkpoint(
    "checkpoints/L4_d64_rope_eps0.05_s0.pt",
    device="cpu",
)
# Supply disjoint integer tensors (number_of_histories, history_length).
splits = torch.load("histories.pt", weights_only=True)
train, validation, test = (splits[key] for key in ["train", "validation", "test"])
result = fit_abstraction(teacher, train, validation, config=FitConfig(
    dimension=4, horizon=16, future_length=4, seed=0,
))
result.save("outputs/my-fit")
metrics = evaluate(result.abstraction, test, horizon=16, seed=1000)
```

Use `GBSAbstraction.load(path, teacher)` to reload an abstraction. The frozen
teacher is supplied separately; loading checks a hash of its parameters and
adapter specification. Save methods refuse to overwrite existing artifacts.

The fitted interface can be used directly:

```python
model = result.abstraction
with torch.no_grad():
    state = teacher.history_state(test.to(teacher.device))
    z = model.decode(state)
    z_roundtrip = model.decode(model.encode(z))
    log_next = model.predict(z)
    log_next, z_next = model.advance(z, test[:, -1].to(teacher.device))
```

`advance` returns predictions **before** consuming its token. `predict` and
`advance` return normalized log probabilities over ordinary tokens, excluding
BOS. `advance` implements write, transformer update, then decode.

## Fitting procedure

1. **Fit a GHMM.** Query the teacher's joint probabilities of short futures.
   Center their vectors, retain the leading singular directions, and fit one
   observable operator per token by least squares on joint successor vectors.
   An explicit constant coordinate enforces normalization. A four-dimensional
   abstraction starts with a five-dimensional GHMM. Its initial state is
   obtained from a separate empty-history query.
2. **Fit the decoder.** Whiten GHMM coordinates using training histories and
   regress them from attention measurements of natural transformer states.
   The decoder learns queries for every layer and head. It reads normalized
   attention responses, `log(1 + Z)` for attention denominators, the cache size,
   and current-token coordinates, followed by a tanh MLP.
3. **Initialize and fit the encoder.** An affine map writes a representative
   cache and relaxed token. Initialize it by least squares against natural
   states, then fit with the decoder fixed.
4. **Fit both maps jointly.** Optimize predictive preservation, recurrent
   consistency, coordinate anchoring, and the root round trip. Select the
   checkpoint with the smallest validation objective at each stage.

There are two predictive losses. **Single insertion** writes the root once and
lets the teacher's cache grow along a continuation. **Repeated compression**
writes, advances, and decodes at every token. Both compare against the original
teacher's conditional distributions along teacher-sampled continuations.
Their expected summed conditional KL equals the corresponding joint sequence
KL by the chain rule. Finite continuation banks provide Monte Carlo estimates.

Training averages KL per token and coordinate errors per dimension. The
objective is

$$
\mathcal{L} = \frac{1}{2}(L_{\mathrm{single}} + L_{\mathrm{repeated}}) + \lambda_{\mathrm{track}} L_{\mathrm{track}} + \lambda_{\mathrm{coord}} L_{\mathrm{coord}} + \lambda_{\mathrm{rt}} L_{\mathrm{rt}}.
$$

Tracking compares the decoded written trajectory to the decoded natural
trajectory, including the root and all successors; natural targets are detached
from gradients. The coordinate term anchors natural states to the initial
whitened GHMM coordinates. The explicit root term measures `D(E(z)) - z` with
the target `z` detached. Default weights are 0.1 for each auxiliary term.
The frozen teacher receives no parameter updates; gradients flow through its
operations into written states. Configuration dataclasses expose dimensions,
optimization steps, learning rates, loss weights, seeds, and validation cadence.

Evaluation reports means and standard errors over root rows. `single_kl` and
`repeated_kl` are **summed** over the forecast horizon, in nats. `roundtrip_mse`
is the root error per coordinate; `tracking_mse` averages successor errors per
coordinate and time step, excluding the root. Test data do not select weights.

## Improve encoder round trips

Keep the teacher and decoder fixed and optimize the encoder further:

```python
from gbs import RefineConfig, refine_encoder

# Save the starting fit before refinement, which updates model in place.
refined = refine_encoder(model, train, validation, config=RefineConfig(
    steps=10000, horizon=16, roundtrip_weight=0.3,
    preservation_slack=0.05, seed=2000,
))
refined.save("outputs/refined")
```

At each validation checkpoint, single-insertion KL, repeated-compression KL,
and tracking MSE must each remain within 5% of their initial validation values.
Among eligible checkpoints, select the smallest round-trip MSE; the starting
encoder remains eligible. These are empirical validation constraints, not a
guarantee on unseen histories. Report independent test metrics afterward.
Refinement starts fresh Adam and new fixed continuation banks; it does not
resume optimizer state. The demo's `--refine-steps` option runs this workflow.

## Other transformers and initializations

The fitting code depends on `CacheModel`, not a particular transformer class.
The included `BeliefStateAdapter` provides a concrete implementation. To use
another architecture, implement:

- `history_state(histories)`: natural flat floating states for equal-length
  histories;
- `advance(states, tokens)`: differentiable log probabilities and successor
  states, preserving gradients through arbitrary written states;
- `features(states, queries)`: attention measurements in the documented layout;
- `encoder_mask(states)`: coordinates the encoder may write;
- model dimensions, frozen parameters, and a `spec()` identifying state semantics.

Generic implementations supply sampling, sequential rollouts, future queries,
and checkpoint fingerprints. A vectorized `trajectory` override is optional.
Keep the teacher in evaluation mode, with gradients disabled on its parameters.
This is an adapter interface, not an automatic loader for arbitrary transformer
libraries.

For the supplied adapter, state includes past KV entries, a current token, and
explicit position metadata. With a length-16 history, the past cache has 15
ordinary-token positions, plus a BOS entry when BOS is visible. The encoder
writes every KV entry with the same affine map, including the BOS entry, and
leaves only structural position metadata fixed. It writes
post-position-encoding keys directly and uses a continuous token vector, so
written states need not correspond to any discrete token history. The fitted
encoder uses one root length; advancing increases the length until the next
compression resets it to that representative length. Context limits still apply.

Pass `initializer=ghmm` to reuse a `GHMM`, or a callable mapping integer
histories to initial coordinates, to bypass spectral fitting. A GHMM's first,
constant coordinate is omitted from the decoder targets. Custom callables
return exactly `FitConfig.dimension` coordinates. Whitening and all encoder
scaling are fitted on training roots only.

The spectral initializer enumerates all vocabulary-sized futures up to
`future_length + 1`. This is intended for small alphabets and short horizons;
`max_outcomes` bounds enumeration. It is not practical for a language-model
vocabulary without a different initializer. Spectral GHMM operators enforce
normalization but may produce negative probabilities; these are not clipped.
The GHMM supplies initial coordinates, while subsequent predictive losses use
the transformer's normalized distributions. Rank-deficient initialization or
zero GHMM normalizers raise errors.

## Tests and provenance

Tests compare cached and native transformer predictions, parallel and
sequential continuation gradients, attention measurements, BOS and position
handling, affine-writer constraints, checkpoint matching, and spectral recovery
of an analytic three-symbol process. An end-to-end fit checks default GHMM
initialization, split isolation, artifact reload, and frozen-decoder refinement.
Exact enumeration also checks the sequence-KL chain rule for repeated
compression. CI runs the suite on Python 3.10 and 3.12.

The included checkpoint architecture follows
[belief-state](https://github.com/artemyk/belief-state); the demo uses its MRRXOR
process. Both repositories are available under the MIT license. This package
contains fitting code, not the earlier experiment directories or their results.
