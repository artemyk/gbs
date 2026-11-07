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
checkpoint, including a visible-BOS RoPE model. The adapter supports standard
softmax attention with RoPE or learned positions, optional visible BOS, and
multiple heads.

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
- `histories.pt` and `run.json`: data splits, seeds, checkpoint hash and URL,
  source commit for the pinned demo, model configuration, and library versions;
- `geometry.png` with `--plot`: projections of decoded states and their encoder
  round trips, colored by the last two tokens.

The demo and CLI use the same artifact writer. Both default to evaluation seed
1000, independent of the training seed; use `--evaluation-seed` to change it.
The default fit initializes the decoder from GHMM coordinates across prefix
lengths, initializes the affine encoder by least squares, then trains both
maps together. Nine cheap round-trip updates alternate with one full
prediction/tracking update. The quick run uses the same procedure with small
budgets; it is not a converged fit. No convergence or prediction-accuracy
threshold is assumed by the fitter.

## Shared experiment commands

Use the same commands for every supported teacher configuration. Supply a
checkpoint and a `histories.pt` dictionary with disjoint `train`, `validation`,
and `test` integer tensors. The demo saves a file in this format. Optional JSON
configuration files contain fields of `FitConfig`.

```sh
python -m gbs fit --teacher model.pt --histories histories.pt --output outputs/fit
python -m gbs evaluate --teacher model.pt --histories histories.pt \
  --abstraction outputs/fit/abstraction.pt --horizon 16 --output outputs/evaluation
```

After installation, `gbs` is also available as a console command. For example,
pass `--config fit.json` with:

```json
{"joint_steps": 10000, "teacher_every": 10, "seed": 0}
```

Each command requires a new output directory and records test metrics, input
hashes, teacher identity, the history splits, numerical library versions, and
the evaluation seed. The fit command also saves the abstraction,
configuration, and validation trace. The same default evaluation seed is used
across commands; set `--evaluation-seed` to change it. `--device` and `--threads`
control execution. The commands call the shared Python fitting API below;
there are no separate encoder or fitting branches for BOS and attention variants.
Integer histories are validated and converted to `torch.long` before indexing;
floating-point histories are rejected. The history hash in `run.json` identifies
the saved normalized splits; the CLI also records the original input-file hash.

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

`fit_abstraction` runs the complete procedure used for the final decoder/encoder
fit:

1. **Fit a GHMM.** Query the teacher's joint probabilities of short futures.
   Center their vectors, retain the leading singular directions, and fit one
   observable operator per token by least squares on joint successor vectors.
   An explicit constant coordinate enforces normalization. A four-dimensional
   abstraction starts with a five-dimensional GHMM. Its initial state is
   obtained from a separate empty-history query.
2. **Fit one decoder across positions.** Drop the GHMM's constant coordinate,
   whiten the remaining coordinates using training roots, and reuse that
   transform at every prefix length. Regress these coordinates from normalized
   attention responses and current-token coordinates. Initialize queries
   randomly and optimize them together with the readout. Select by validation
   coordinate MSE.
3. **Initialize the encoder.** Fit an affine map by least squares from whitened
   GHMM coordinates to natural transformer states, with scaling fitted only on
   the training roots.
4. **Fit both maps jointly.** Keep both maps trainable throughout one phase.
   Alternate nine cheap root round-trip updates with one full
   prediction/tracking update. Keep multi-prefix GHMM coordinate anchoring
   active on every update. Select both maps together by the full validation
   objective.

The teacher and GHMM coordinate system stay fixed. Sample teacher continuations
once for training and validation; recompute natural codes with the current
decoder as it changes. Only validation data select weights. Defaults are:

| Optimization | Updates | Learning rate | Selection |
| --- | ---: | --- | --- |
| Decoder regression | 40,000 | 0.001 → 0.00001 | Coordinate MSE |
| Joint fitting | 10,000 | Decoder 0.0001 → 0.00001; encoder 0.0003 → 0.00003 | Full objective |

Learning rates follow cosine schedules. The decoder has eight learned queries
per head and 128 tanh hidden units. Batches contain 64 rows, validation runs every
200 updates, and the demo's default root length/horizon is 16/16. Set
`teacher_every` to change the interval between full updates; the final update
always uses the full objective, even for a partial cycle. With the defaults,
9,000 updates are cheap round trips and 1,000 are full prediction/tracking
updates. All auxiliary loss weights are 0.1.

`FitConfig` exposes these settings. `teacher_every=1` uses the full objective on
every update, and `decoder_min_lr_fraction=1` disables decoder learning-rate
decay. The trace labels optimization phases `decoder` and `joint`;
`initialization.json` records selections and timings.

Decoder pretraining samples prefixes from length 0 through root length plus
`horizon` by default. Each training parent contributes one uniformly sampled
length; `decoder_samples` changes the total number of samples. Validation uses
up to `decoder_validation_samples` parent rows at **every** length. A minibatch
contains one length, avoiding attention padding. Set `decoder_min_length` and
`decoder_max_length` to restrict the range. The encoder still writes a fixed
representative root length. One set of decoder queries and one readout are
shared across all lengths, with a single train-derived coordinate transform.
Split parent histories first; short prefixes may coincide across splits.

The decoder uses random query initialization and a tanh MLP with a positive
integer hidden width. Queries and readout are optimized together across prefix
lengths, and both can change during joint fitting. For RoPE, the adapter rotates
the same learned query templates to the position being decoded. Attention
responses are recomputed as queries move. For decoder-only experiments,
`fit_decoder` accepts `DecoderConfig`, parent sequences, and a callable giving
coordinates in one shared basis.
Decoder validation trace entries include elapsed seconds from the start of fitting,
including query initialization and cache preparation.
Configuration validation rejects nonfinite learning rates, loss weights, and
sampling settings before preparing teacher data. Step counts, dimensions,
and sampling counts must be integers in their documented ranges.

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
whitened GHMM coordinates across the decoder's training-prefix distribution,
including during joint fitting. The explicit root term measures `D(E(z)) - z` with
the target `z` detached. Default weights are 0.1 for each auxiliary term.
The frozen teacher receives no parameter updates; gradients flow through its
operations into written states. Configuration dataclasses expose dimensions,
optimization steps, learning rates, loss weights, seeds, and validation cadence.

Cheap updates recompute `z = D(natural_state)` and optimize the root round trip
with coefficient `roundtrip_weight + tracking_weight / (horizon + 1)`, plus the
same multi-prefix coordinate anchor. They avoid teacher rollouts. Gradients
flow through the source decoder, encoder, and read-back decoder; only the
target code is detached. Full updates additionally evaluate both predictive
losses and successor tracking. Changing the update interval changes the
relative optimization effort devoted to these terms, so compare prediction
errors as well as round-trip errors.

Natural root caches are constructed in batches of at most 256 roots and reused
for encoder initialization and the continuation banks. Decoder regression
separately prepares its sampled prefixes, retaining raw states so attention
responses can change with the learned queries. There is no cache of decoder
codes during joint fitting. Training and evaluation share the per-root rollout
calculations; their mean versus summed KL conventions remain distinct.

Evaluation reports means and standard errors over root rows. `single_kl` and
`repeated_kl` are **summed** over the forecast horizon, in nats. `roundtrip_mse`
is the root error per coordinate; `tracking_mse` averages successor errors per
coordinate and time step, excluding the root. Test data do not select weights.

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
Reused GHMMs are evaluated on their own device and in their stored precision;
the resulting coordinates are then transferred to the teacher's device and dtype.

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
initialization, split isolation, artifact reload, and joint checkpoint selection.
Additional tests check learned-query updates, prefix-length coverage, bounded
prefix preparation, and prediction preservation after reload. Joint fitting
tests check continuation reuse, current-decoder targets,
gradients through both maps on cheap updates, multi-prefix anchoring on every
update, and full updates at the configured interval and the end. Configuration tests
check that invalid numeric settings fail before data preparation.
Exact enumeration also checks the sequence-KL chain rule for repeated
compression. CI runs the suite on Python 3.10 and 3.12.

Native and cached execution share the teacher's attention projection, and
natural states are built using the same cache execution as interventions.
The included checkpoint architecture follows
[belief-state](https://github.com/artemyk/belief-state); the demo uses its MRRXOR
process. Both repositories are available under the MIT license. This package
contains fitting code, not the earlier experiment directories or their results.
