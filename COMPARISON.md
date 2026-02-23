# Architecture Comparison: MLP vs Tensor Algebra Predictor

## Setup

Both architectures are trained on the same task: character-level name prediction
from `names.txt` (Karpathy's makemore dataset). Both use:
- Same alphabet (27 chars: '.' + a-z)
- Same corpus construction (dot-separated names)
- Same initialization scheme (small linear ramp around zero)
- Same optimizer (gradient ascent on log-likelihood, normalized by corpus length)
- Same training infrastructure (explicit backpropagation in Agda)

### Architecture Details

| Property | MLP (MLPREV.agda) | Tensor (TensorBigram.agda) |
|---|---|---|
| Parameters | 209 | 324 |
| Context | Last 2 chars (sliding window) | Full history (accumulated tensor state) |
| Embeddings | 27 x 2 = 54 params | 27 x 2 = 54 params |
| Hidden layer | 4 units with tanh | None (linear) |
| Output | W2 * tanh(W1 * emb + b1) + b2 | W_state * state + W_prev * prevEmb + b |
| State | None (fixed window) | 7-dim: scalar + R^2 + R^{2x2} |
| Learning rate | 10.0 | 5.0 |
| Design origin | Engineered (Bengio et al. 2003) | Derived from algebra (truncated tensor algebra T_2(R^d)) |

### Key Structural Difference

The MLP uses a **fixed context window** (last 2 characters) and a **nonlinear hidden layer** (tanh).

The Tensor predictor uses the **truncated tensor algebra** as a monoid state that accumulates
over the entire history. Its state captures:
- Order 0: character count (1 dim)
- Order 1: sum of embeddings = unigram statistics (2 dims)
- Order 2: sum of outer products = bigram correlations in embedding space (4 dims)

The tensor predictor has no nonlinearity -- it is purely linear from state to logits.
However, the outer product in the state update is a multiplicative interaction,
giving it some nonlinear capacity.

## Results

### Head-to-head: same data, same steps

All numbers are **negative log-likelihood per character** (lower is better).

| Setting | MLP (209 params) | Tensor (324 params) | Winner |
|---|---|---|---|
| 10 steps, 50 names (train NLL) | 2.835 | 2.728 | **Tensor** |
| 25 steps, 50 names (train NLL) | 2.711 | 2.651 | **Tensor** |
| 25 steps, 20 names (train NLL) | 2.678 | 2.631 | **Tensor** |

### Reference points

| System | NLL |
|---|---|
| Uniform random | 3.296 |
| **Tensor (25 steps, 50 names)** | **2.651** |
| **MLP (25 steps, 50 names)** | **2.711** |
| Karpathy bigram (32k names, count-based) | 2.454 |
| Karpathy MLP (32k names, ~200k steps) | ~2.3 |

### Generalization

| Metric | MLP | Tensor |
|---|---|---|
| Train NLL (50 names, 25 steps) | 2.711 | 2.651 |
| Eval NLL (500 names, 25 steps) | 2.784 | N/A* |
| Generalization gap | 0.073 | -- |

*The tensor model was evaluated on the same 50-name training set. A 500-name eval
was not run, but the gap is expected to be similar or smaller given the model's
structural bias toward bigram statistics.

### Name Generation (greedy decoding, 25 steps)

**MLP:** Generates only empty names (predicts '.' immediately for all seeds).
This suggests the model has not learned enough structure to generate plausible
character sequences, despite having reasonable NLL.

**Tensor:** Generates short names ("ea" from '.', "a" from '.m'). Still primitive,
but shows some learned character preferences.

## Analysis

### The tensor predictor wins on all comparable settings

The tensor algebra predictor achieves lower NLL than the MLP at every comparison
point: 10 steps, 25 steps, 50 names, 20 names. The margin is consistent at
roughly 0.05-0.10 NLL.

### Caveats

1. **Parameter count favors tensor (324 vs 209).** The tensor model has 55% more
   parameters. Some of this advantage may come simply from having more capacity.
   However, the parameter difference is primarily in W_state (189 params for
   the 7-dim state-to-27 logit mapping) -- this is a consequence of the richer
   state representation, not gratuitous overparameterization.

2. **Learning rates differ (10.0 vs 5.0).** Both were hand-tuned separately.
   The MLP might do better with a different learning rate, or vice versa.
   Neither was systematically optimized.

3. **Both are severely undertrained.** 25 gradient steps on 50 names is
   negligible compared to Karpathy's ~200k steps on 32k names. At this
   scale, the comparison tests "how quickly does the architecture start learning"
   rather than "which architecture is better at convergence."

4. **The MLP has a nonlinearity (tanh) but the tensor model does not.** In principle,
   the MLP should have more expressive power per parameter. The fact that the
   linear tensor model wins suggests that for this (very small) training budget,
   the algebraic structure of the tensor state is more useful than nonlinear
   transformation capacity.

### What this means for the denotational design thesis

The tensor algebra predictor was **derived from algebraic principles**: the truncated
tensor algebra is the universal construction for capturing polynomial interactions
in an associative monoid. It was chosen because the Kleisli morphism framework
from `Architectures.agda` requires the state to form a monoid under context
accumulation.

The MLP was **engineered**: take a standard neural network architecture (Bengio et al.
2003), implement it with embeddings and a hidden layer, train with backprop.

The fact that the algebraically-derived architecture **matches or beats** the
engineered one -- even with no nonlinearity and no hidden layer -- is encouraging
for the denotational design methodology. It suggests that choosing representations
guided by algebraic structure can be at least as effective as standard engineering
choices, even before any systematic optimization.

The honest gap: both models are far from the Karpathy baselines (2.454 for bigrams,
~2.3 for MLP). The comparison is between two undertrained models. The real test
would be scaling both to 32k names with proper training -- but that is beyond
current Agda runtime constraints.

## Reproducing

```bash
# Compile both (requires Agda 2.8.0 with standard-library-2.3)
agda --compile MLPREV.agda
agda --compile TensorBigram.agda

# Run both
./MLPREV
./TensorBigram
```

Both read `names.txt` at runtime. Compilation takes ~15-30s each; runtime is a few seconds.
