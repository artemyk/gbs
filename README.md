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

Requires Python 3.10 or later, PyTorch 2.6+, NumPy, and Transformers 5.20+.
From this repository:

```sh
python -m pip install -e '.[test,demo]'
python -m pytest -q
```

The import name is `gbs`; the distribution name is `generalized-belief-states`.
The `demo` extra adds Matplotlib for optional geometry plots. This repository
does not include pretrained transformer weights or publish a PyPI release.

## Native Hugging Face models and caches

Transformer execution uses unmodified Hugging Face causal LMs and their
[`DynamicCache`](https://huggingface.co/docs/transformers/en/kv_cache), with
PyTorch SDPA as the default attention backend. There is no transformer forward
implementation in `gbs` and no separately stored position counter.

```python
from transformers import AutoModelForCausalLM
from gbs import TransformersAdapter

native_model = AutoModelForCausalLM.from_pretrained(
    "path/to/local/model", attn_implementation="sdpa", dtype="auto",
)
teacher = TransformersAdapter(native_model)
```

The adapter supports full, uniform attention caches for Llama, Starcoder2,
GPT-2, GPT-NeoX, and GPTBigCode on one device, including grouped-query attention.
Sliding, hybrid, quantized, and state-space caches are not supported. A valid BOS token
is required for the empty-history initial state; pass `bos_token_id` if the
model configuration does not supply one. By default the modeled alphabet is
the entire model vocabulary. Optional `token_ids` selects and renormalizes a
restricted alphabet; the belief-state converter uses this for its input-only
BOS. Token histories passed to the fitter then index this selected alphabet.
Supply histories without padding or an already prepended BOS; the adapter
supplies the initial BOS itself.

The fitting representation flattens the native KV tensors followed
by the current input embedding. Every numerical coordinate is writable. Cache
size comes from the tensor shapes, and the native model determines the next
position from its cache. To exchange states with normal Hugging Face calls:

```python
state = teacher.history_state(histories)
past_key_values, current_embedding = teacher.cache_from_state(state)
# past_key_values is a standard DynamicCache, not a gbs cache class.
state = teacher.state_from_cache(past_key_values, current_embedding)
```

These operations preserve gradients. Each execution creates a fresh cache
object, so updating it cannot mutate another stored fitting state. The encoder
can replace a length-10 history with a representative length-16 cache; native
execution continues from position 16. No position coordinate is stored or fed
separately into the model. Context limits still apply.

SDPA is an execution backend, not a guarantee of a speedup. Its fused kernels
depend on hardware, dtype, and shapes. The adapter requires gradients through
written caches and inputs; inference-only serving engines are not substitutes
for this fitting path. Existing CUDA backends may be selected through
`attn_implementation`; CPU tests exercise SDPA and eager attention.

## Demo: large RoPE models

RoPE checkpoints from [artemyk/belief-state](https://github.com/artemyk/belief-state)
are converted to a native Hugging Face Starcoder2 model when loaded. Start with its
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
RoPE checkpoint, including a visible-BOS model, or a native `save_pretrained`
directory. Learned-position models use native Hugging Face checkpoints.

### Checkpoint format

The existing belief-state files already use the usual
[PyTorch checkpoint pattern](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html):
an ordinary dictionary containing a configuration dictionary and a
model `state_dict`, rather than a pickled model object:

```python
torch.save({"config": config_dict, "model": model.state_dict()}, "model.pt")
```

`TransformersAdapter.from_checkpoint` accepts a native `save_pretrained`
directory or a belief-state RoPE file. A file is read with
`torch.load(..., weights_only=True, map_location="cpu")` and converted once in
memory into the library's `Starcoder2ForCausalLM`, preserving narrow head sizes,
LayerNorm, ReLU MLPs, and standard softmax attention. The output bias is absorbed
into final LayerNorm bias when algebraically representable. Unsupported fields,
missing weights, and incompatible output biases fail explicitly. No remote
Python source is executed.

The converted model can be saved normally:

```python
from gbs import convert_belief_state_checkpoint

native_model = convert_belief_state_checkpoint("teacher.pt")
native_model.save_pretrained("checkpoints/native-teacher")
```

For an existing decoder/encoder fit, convert its state parameterization too:

```python
from gbs import convert_belief_state_abstraction

teacher = TransformersAdapter.from_checkpoint("checkpoints/native-teacher")
model = convert_belief_state_abstraction("old-fit/abstraction.pt", teacher)
model.save("native-abstraction.pt")
```

This maps the old token writer into input embeddings and transforms the
corresponding decoder readout; it does not retrain the fit or overwrite source
checkpoints. New native fits use the same generic encoder and decoder. A
historical masked-BOS teacher retains its BOS entry in the standard cache and
uses the native model's attention-mask input, so cache length still determines
position without an additional counter.

The demo samples length-16 MRRXOR histories. Distinct histories are assigned to
disjoint training, validation, and test sets, retaining each history's sampling
multiplicity within its split. Only tokens enter the fitter. Continuations and
prediction targets come from the transformer. `--epsilon` controls the history
distribution and should match the chosen experiment. `--device cpu`, `mps`, or
`cuda` selects execution; spectral decompositions and initial least squares run
on CPU in double precision. Tests cover float32 and float64 execution;
accelerator-specific reduced-precision kernels require matching model support.

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

The default fit includes both encoder refinement stages and joint fine-tuning;
no separate refinement command is needed. The quick run exercises these same
stages with small budgets; it is not a converged fit. No convergence or
prediction-accuracy threshold is assumed by the fitter.

## Shared experiment commands

Use the same commands for every supported teacher configuration. Supply a
native model directory (or belief-state RoPE checkpoint) and a `histories.pt`
dictionary with disjoint `train`, `validation`, and `test` integer tensors.
The demo saves a file in this format. Optional JSON
configuration files contain fields of `FitConfig` or `RefineConfig`.

```sh
python -m gbs fit --teacher checkpoints/native-teacher --histories histories.pt --output outputs/fit
python -m gbs refine --teacher checkpoints/native-teacher --histories histories.pt \
  --abstraction outputs/fit/abstraction.pt --config refine.json --output outputs/refined
python -m gbs evaluate --teacher checkpoints/native-teacher --histories histories.pt \
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
from gbs import TransformersAdapter, FitConfig, fit_abstraction, evaluate

teacher = TransformersAdapter.from_checkpoint(
    "checkpoints/native-teacher",
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
`advance` return normalized log probabilities over the selected modeled alphabet.
Converted belief-state checkpoints exclude their input-only BOS.
`advance` implements write, transformer update, then decode.

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
3. **Initialize and fit the encoder.** Initialize an affine map by least squares
   from whitened GHMM coordinates to natural transformer states. With the
   decoder fixed, cache its natural root and successor codes, then alternate
   cheap round-trip updates with full prediction/tracking updates. Select by
   the full validation objective.
4. **Refine the encoder.** Keep the decoder fixed and select the smallest
   validation round-trip error subject to prediction and tracking limits.
5. **Fit both maps jointly.** Optimize predictive preservation, recurrent
   consistency, multi-prefix GHMM coordinate anchoring, and the root round trip.
   Use the full teacher objective on every update and select by the full
   validation objective.
6. **Refine the encoder again.** Freeze the jointly fitted decoder, rebuild its
   cached natural codes, and repeat constrained round-trip refinement.

The teacher and GHMM coordinate system stay fixed throughout. All stages reuse
the same teacher-sampled continuations, and only validation data select weights.
The defaults are:

| Stage | Updates | Learning rate | Selection |
| --- | ---: | --- | --- |
| Decoder regression | 40,000 | 0.001 → 0.00001 | Coordinate MSE |
| Encoder fitting | 10,000 | 0.0003 → 0.00003 | Full objective |
| Pre-joint encoder refinement | 2,000 | 0.0001 → 0.00001 | Constrained round-trip MSE |
| Joint fitting | 1,000 | Decoder 0.0001 → 0.00001; encoder 0.0003 → 0.00003 | Full objective |
| Final encoder refinement | 2,000 | 0.0001 → 0.00001 | Constrained round-trip MSE |

Learning rates follow cosine schedules. The default decoder has eight learned
queries per head and 128 tanh hidden units. Batches contain 64 rows, validation
runs every 200 updates, and the default root length/horizon in the demo are
16/16. Frozen-decoder stages use `teacher_every=20`: nineteen cheap updates,
then one full update; their final update always uses the full objective.
Refinement uses round-trip weight 0.3 and allows at most 5% degradation in each
validation prediction/tracking metric relative to that refinement's starting
checkpoint. Other auxiliary weights are 0.1.

`FitConfig` exposes all these settings. Set `encoder_refine_steps=0` or
`joint_refine_steps=0` to omit the corresponding refinement, and `teacher_every=1`
to use full teacher updates throughout. `decoder_min_lr_fraction=1` disables
decoder learning-rate decay. The trace labels the five optimization stages
`decoder`, `encoder`, `encoder_refine`, `joint`, and `joint_refine`;
`initialization.json` records selections, refinement constraints, and timings.

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
preservation tolerances before preparing teacher data. Step counts, dimensions,
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

Natural root caches are constructed in batches of at most 256 roots and reused
for encoder initialization and the continuation banks. Decoder pretraining
separately prepares its sampled prefixes, retaining raw states for learned
queries. The frozen-decoder encoder stage reuses the decoder's selected
coordinate-validation scores and optimizes and checkpoints only the encoder.
Training and evaluation share the per-root rollout calculations; their mean
versus summed KL conventions remain distinct.

While the decoder is frozen, its natural root and successor codes are cached
once and reused for initial encoder fitting and pre-joint refinement. Joint
fitting recomputes targets as the decoder changes; final refinement rebuilds
them from the selected joint checkpoint. Standalone refinement streams
preparation in bounded batches and retains compact targets instead of
accumulating natural KV states across the dataset. Continuation sampling
in the belief-state adapter uses one teacher pass per observed token.

Evaluation reports means and standard errors over root rows. `single_kl` and
`repeated_kl` are **summed** over the forecast horizon, in nats. `roundtrip_mse`
is the root error per coordinate; `tracking_mse` averages successor errors per
coordinate and time step, excluding the root. Test data do not select weights.

## Further encoder refinement

The complete fit already includes constrained refinement before and after joint
fitting. To extend it, keep the teacher and decoder fixed and run:

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
Every refinement starts fresh Adam. Standalone `refine_encoder` samples new
fixed continuation banks; the integrated fitting stages reuse the fit's banks.
The demo's `--refine-steps` option requests further refinement after the complete
fit.

The default `teacher_every=20` in both `FitConfig` and `RefineConfig` reduces
teacher evaluations. Nineteen updates optimize only the root round
trip, followed by one update with the full teacher objective. The final update
also uses the full objective. Set `teacher_every=1` to use it on every update.
Cheap updates use the root coefficient from the full objective,
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

## Other transformers and initializations

The fitting code depends on `CacheModel`, not a particular transformer class.
The included `TransformersAdapter` provides a concrete implementation. To use
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

The encoder writes all KV tensors and the continuous current input embedding.
It writes keys in the native cache representation, after RoPE when used. These
states need not correspond to any discrete token history. One representative root
length fixes the encoder's output shape; advancing grows the cache until the
next compression replaces it. Neither prediction nor decoding uses a stored
position coordinate.

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
handling, full-cache/embedding writers, checkpoint matching, and spectral recovery
of an analytic three-symbol process. An end-to-end fit checks default GHMM
initialization, split isolation, artifact reload, and frozen-decoder refinement.
Additional tests check learned-query updates, prefix-length coverage, bounded
prefix preparation, and prediction preservation after reload. The complete
staged fit checks continuation reuse, refreshed natural codes after joint
fitting, and validation constraints in both refinements. Configuration tests
check that invalid numeric settings fail before data preparation.
Exact enumeration also checks the sequence-KL chain rule for repeated
compression. CI runs the suite on Python 3.10 and 3.12.

Natural states and interventions both run through the same unmodified Hugging
Face model. Conversion tests compare against a small independent reference
implementation kept only in the test suite. The demo uses the MRRXOR process
from [belief-state](https://github.com/artemyk/belief-state). This package contains
fitting and conversion code, not the earlier experiment directories or results.
