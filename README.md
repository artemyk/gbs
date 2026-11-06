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
checkpoint. The adapter supports standard softmax attention with RoPE or learned
positions, optional visible BOS, and multiple heads.

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
Refinement writes a complete record in the `refined/` subdirectory, including
its history splits and the starting abstraction's hash.

The quick run checks the workflow; it is not a converged fit. No convergence or
prediction-accuracy threshold is assumed by the fitter.

## Shared experiment commands

Use the same commands for every supported teacher configuration. Supply a
checkpoint and a `histories.pt` dictionary with disjoint `train`, `validation`,
and `test` integer tensors. The demo saves a file in this format. Optional JSON
configuration files contain fields of `FitConfig` or `RefineConfig`.

```sh
python -m gbs fit --teacher model.pt --histories histories.pt --output outputs/fit
python -m gbs refine --teacher model.pt --histories histories.pt \
  --abstraction outputs/fit/abstraction.pt --config refine.json --output outputs/refined
python -m gbs evaluate --teacher model.pt --histories histories.pt \
  --abstraction outputs/refined/abstraction.pt --horizon 16 --output outputs/evaluation
```

After installation, `gbs` is also available as a console command. For example,
`refine.json` can contain:

```json
{"steps": 10000, "horizon": 16, "roundtrip_weight": 0.3, "seed": 2000}
```

Each command requires a new output directory and records test metrics, input
hashes, teacher identity, the history splits, numerical library versions, and
the evaluation seed. Fit and refinement commands also save the abstraction,
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

Decoder regression uses `FitConfig.decoder_lr`. Set `decoder_min_lr_fraction`
below 1 to apply cosine decay toward that fraction of the initial learning
rate. Its default of 1 keeps the learning rate constant. For example,
`decoder_lr=0.001, decoder_min_lr_fraction=0.01` decays toward 0.00001.
`queries_per_head`, `hidden`, and `decoder_steps` control decoder capacity and
training budget. Decoder checkpoints are selected by validation coordinate MSE.

1. **Fit a GHMM.** Query the teacher's joint probabilities of short futures.
   Center their vectors, retain the leading singular directions, and fit one
   observable operator per token by least squares on joint successor vectors.
   An explicit constant coordinate enforces normalization. A four-dimensional
   abstraction starts with a five-dimensional GHMM. Its initial state is
   obtained from a separate empty-history query.
2. **Fit the decoder.** Whiten GHMM coordinates using training histories and
   regress them from attention measurements of natural transformer states.
   The decoder learns queries for every layer and head. It reads normalized
   attention responses and current-token coordinates, followed by a tanh MLP.
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

Natural caches are constructed once in batches of at most 256 roots and reused
for decoder regression, encoder initialization, and the continuation banks.
Training and evaluation share the per-root rollout calculations; their mean
versus summed KL conventions remain distinct.

While the decoder is frozen, its natural root and successor codes are cached
once. Refinement streams preparation in bounded batches and retains these
compact targets instead of accumulating natural KV states across the dataset.
Joint fitting recomputes targets as the decoder changes. Continuation sampling
in the belief-state adapter uses one teacher pass per observed token.

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
For roundoff near zero, preservation limits use a nonnegative baseline plus an
absolute allowance of one machine epsilon per forecast token for summed KL,
and one epsilon for tracking MSE. Raw measured KL values are retained, including
tiny negative roundoff values. Limits and allowances are recorded in
`initialization.json`; the starting encoder is always retained as a fallback.
Refinement starts fresh Adam and new fixed continuation banks; it does not
resume optimizer state. The demo's `--refine-steps` option runs this workflow.

To reduce teacher evaluations, set `teacher_every=20` in `RefineConfig` (or in
the CLI's refinement JSON). Nineteen updates then optimize only the root round
trip, followed by one update with the full teacher objective. The final update
also uses the full objective. The default, `teacher_every=1`, uses it on every
update. Cheap updates use the root coefficient from the full objective,
`roundtrip_weight + tracking_weight / (horizon + 1)`.

All validation checks still use the full forecast horizon and all preservation
limits. With alternating updates, an infeasible checkpoint restores the best
eligible encoder and resets Adam's moments. Final selection retains the
starting encoder as a fallback. The trace records update counts, elapsed time,
and rollbacks; `initialization.json` records the selected step and timings.
`FitConfig.teacher_every` also controls the frozen-decoder encoder stage of an
initial fit; joint fitting always uses the full objective and selects by its
full validation loss. Alternating updates change the optimization schedule,
so compare prediction metrics as well as round-trip error.

To retain intermediate refinement results, pass a new `checkpoint_directory`
to `refine_encoder`. It saves the starting abstraction and each
validation-selected improvement, with its configuration and validation trace.

## Selected layers and an inverse encoder

Set `FitConfig(layers=(2, 3))` to fit a decoder using zero-based layers 2 and 3.
The current-token coordinates remain part of its input. All layers are used by
default. A selected-layer `Architecture` is retained in abstraction checkpoints.

For a known, frozen decoder, an alternative to a purely affine writer is a
value-only affine writer followed by differentiable inverse corrections:

```python
from gbs import AffineEncoder, InverseEncoder

model.decoder.requires_grad_(False)
states = model.adapter.history_state(train.to(model.adapter.device))
codes = model.decode(states).detach()
mask = model.adapter.value_mask(states, layers=model.architecture.layers)
base = AffineEncoder.initialize(model.adapter, states, codes, writable=mask)
model.encoder = InverseEncoder.from_decoder(
    model.adapter, model.decoder, base, steps=12, damping=1e-3,
)
```

Unwritten coordinates are fixed to their training means, including keys and
current-token coordinates. The inverse stores normalized attention weights and
MLP factors from the frozen decoder. It applies damped Gauss--Newton corrections
to values, solving a code-sized system at each iteration. The operations are
differentiable through both the input code and the base writer's parameters;
the base writer can therefore be optimized with prediction and tracking losses.
Rebuild the inverse whenever the decoder or fixed reference state changes.

This encoder generally changes the state nonlinearly. It can substantially
reduce round-trip error, while prediction and trajectory preservation still
require independent measurement. Solver steps and damping trade execution cost
against convergence. For inverse encoders with nearly zero round-trip error,
selecting predictive fits by the full validation objective is more informative
than selecting numerical fluctuations of round-trip MSE.

The supplied belief-state adapter exposes `value_mask` and `attention_weights`
for this construction. Other adapters can implement the same optional methods;
value-mask ordering is position, selected layer, head, then value coordinate.

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
initialization, split isolation, artifact reload, and frozen-decoder refinement.
Exact enumeration also checks the sequence-KL chain rule for repeated
compression. CI runs the suite on Python 3.10 and 3.12.

Native and cached execution share the teacher's attention projection, and
natural states are built using the same cache execution as interventions.
The included checkpoint architecture follows
[belief-state](https://github.com/artemyk/belief-state); the demo uses its MRRXOR
process. Both repositories are available under the MIT license. This package
contains fitting code, not the earlier experiment directories or their results.
