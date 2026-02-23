# Attention as Algebra: From Tensor State to Linear Attention

## The monoid homomorphism constraint

The foundational result in `Kleisli.agda` is that score decomposes as an **indexed monoid homomorphism**:

```
scoreFrom p h (xs ++ ys) = scoreFrom p h xs + scoreFrom p (h ++ xs) ys
```

Any architecture that processes text sequentially must maintain some state `S` with:

1. An initial state `s_0 : S`
2. A transition `step : S -> Char -> S`
3. An output `read : S -> Char -> R`

The state is computed by folding `step` over the history. Score decomposition then requires that the state accumulation respects concatenation -- the state after processing `xs ++ ys` must be determinable from the states after `xs` and `ys` separately.

This is the **monoid constraint**: `(S, combine, s_0)` must be a monoid, and the map from `(List Char, ++)` to `(S, combine)` must be a monoid homomorphism.


## T_2(R^d) as a concrete monoid

In `TensorBigram.agda`, we chose the **truncated tensor algebra** at order 2 as our state monoid:

```
T_2(R^d) = R x R^d x R^{d x d}
```

Each character `c` with embedding `e_c in R^d` maps to an element of `T_2`:

```
embed(c) = (1, e_c, e_prev (x) e_c)
```

where `(x)` denotes the outer product and `e_prev` is the embedding of the preceding character. The state accumulates by component-wise addition:

- **Order 0** (scalar): character count
- **Order 1** (vector): sum of embeddings (unigram statistics)
- **Order 2** (matrix): sum of consecutive outer products (bigram correlations in embedding space)

Output is a linear readout from the (normalized) state plus a direct bigram term:

```
logits = W_state . normalize(state) + W_prev . e_prev + b
```

This achieves **NLL = 2.210** on 32k names with T_2(R^16), beating Karpathy's MLP baseline (~2.3) while using fewer parameters.

The key property: state update is a monoid homomorphism **by construction**. The tensor algebra is the free associative algebra -- it is precisely the universal monoid over a vector space.


## Linear attention is generalized T_2

Linear attention (Katharopoulos et al., 2020) maintains state:

```
S_t = sum_{i=1}^{t} phi(k_i) (x) v_i^T
```

queried at each step by:

```
output_t = q_t^T . S_t
```

where `phi(k)` is a feature map applied to keys, and `k`, `v`, `q` are learned linear projections of the input.

This is **also** a monoid homomorphism. Each token contributes `phi(k_i) (x) v_i^T` to the state, and states combine by matrix addition. The monoid is `(R^{d_k x d_v}, +, 0)`.

The connection to our tensor model is direct:

| Component | T_2(R^d) | Linear attention |
|-----------|----------|------------------|
| State space | R x R^d x R^{d x d} | R^{d_k x d_v} |
| Per-token contribution | (1, e_c, e_prev (x) e_c) | phi(k_i) (x) v_i^T |
| State combination | component-wise + | matrix + |
| Readout | linear map from state | q^T . S |
| Key/value | raw embeddings | learned projections |

**T_2 is the special case** where:
- `phi(k) = e_prev` (previous character's raw embedding)
- `v = e_c` (current character's raw embedding)
- `q = W_state[k,:]` (a row of the output weight matrix)

Linear attention **generalizes** this by learning separate key, value, and query projections. The algebraic structure -- a monoid of matrices under addition, queried linearly -- is identical.


## The algebraic hierarchy

This gives us a hierarchy of architectures ordered by both algebraic structure and expressiveness:

### Level 1: Bigram (diagonal state)
- State: `R^{|Sigma|}` (count vector)
- Each character adds a one-hot vector
- Readout: look up the row for the previous character
- Monoid: `(R^{|Sigma|}, +, 0)`
- NLL ~ 2.454

### Level 2: T_2 with raw embeddings (our model)
- State: `R x R^d x R^{d x d}`
- Each character adds `(1, e_c, e_prev (x) e_c)`
- Readout: linear map from normalized state + direct bigram
- Monoid: `(T_2(R^d), +, 0)`
- NLL ~ 2.210

### Level 3: Linear attention (T_2 with learned projections)
- State: `R^{d_k x d_v}`
- Each token adds `phi(W_k x) (x) (W_v x)^T`
- Readout: `(W_q x)^T . S`
- Monoid: `(R^{d_k x d_v}, +, 0)`
- Same algebra as Level 2, but the projections into the algebra are learned rather than fixed

### Level 4: Softmax attention (breaks the monoid)
- No fixed-size state; attends over all positions
- The softmax normalization factor depends on the full sequence
- **Not a monoid homomorphism** -- you cannot combine states from two subsequences

The transition from Level 3 to Level 4 is where the monoid structure breaks. This is not incidental -- it is the source of both the power and the cost of softmax attention.


## What softmax attention adds, and what it costs

Softmax attention computes:

```
Attn(Q, K, V) = softmax(Q K^T / sqrt(d)) V
```

The softmax normalization creates a **non-compositional** dependency: the attention weight for position `i` depends on the scores at *all other positions*. If you split a sequence into `xs ++ ys`, the attention weights for tokens in `xs` change when `ys` is appended (because the normalization denominator grows).

This is precisely why:
- **Transformers cannot be "chunked"** the way RNNs or linear attention models can. There is no fixed-size state that summarizes a prefix.
- **KV-caching works** only for autoregressive (causal) attention, and even then you must store the full key-value history, not a compressed summary.
- **Linear attention approximations** (which restore the monoid property) lose some expressiveness but gain the ability to process sequences in streaming fashion.

The algebra makes the tradeoff crisp: softmax attention sacrifices the monoid homomorphism property in exchange for data-dependent, normalized weighting over the full history.


## Analogy to automatic differentiation

In Conal Elliott's AD framework:
- Forward-mode and reverse-mode are different **representations** of the same mathematical object (the derivative, which is a linear map).
- The choice of representation affects efficiency but not correctness.
- The functor equation `D(f . g) = D(f) . D(g)` holds for both modes.

Our hierarchy has a partial analogy:
- T_2 and linear attention are different **parameterizations** of the same algebraic structure (a matrix monoid under addition, queried linearly).
- The monoid homomorphism property holds for both.
- More expressive parameterizations (learned projections) capture more functions but preserve the algebra.

The analogy is **strong** between Levels 1-3: these are genuinely different parameterizations of the same underlying monoid structure, just as forward-mode and reverse-mode are different representations of the same linear map.

The analogy **breaks** at Level 4. Softmax attention computes a fundamentally *different* function, not just a different representation of the same one. In AD terms, this would be like switching from computing derivatives to computing something that is not a derivative at all. There is no sense in which softmax attention is "another representation" of the matrix-addition monoid.


## Honest assessment

### What the algebra forces
The score decomposition theorem requires state to be a monoid. Any architecture that processes sequences left-to-right with a fixed-size state must use a monoid homomorphism from `(List Char, ++)` to some `(S, *, s_0)`. This is not a design choice; it is a consequence of the specification.

### What the algebra suggests
Given that the state must be a monoid, **matrices under addition** are a natural and expressive choice. The tensor algebra construction tells us *which* matrices: those built from outer products of embeddings, capturing k-gram correlations in a compressed space. Learning the projections into this space (as linear attention does) is the natural generalization.

### What the algebra does not derive
- **Why softmax?** The specific choice of softmax normalization is not algebraically motivated. It breaks the monoid property. Its effectiveness is empirical.
- **Multi-head structure.** The idea of running multiple independent attention heads and concatenating is an engineering choice, not an algebraic consequence.
- **Positional encodings.** The monoid framework is position-agnostic (history is accumulated, not indexed). Positional encodings add information that the monoid structure deliberately discards.
- **The gap between Level 3 and Level 4.** The algebra tells us that softmax attention is non-compositional but not *why* this non-compositionality helps. Understanding this gap -- what exactly the normalization buys -- remains open.

### The imperfect analogy
In AD, forward-mode and reverse-mode compute the **same derivative** (proven: both modes agree on all primitives, and functoriality preserves agreement through composition). In text prediction, different levels of our hierarchy compute **different functions** with different expressive power. The algebraic structure is shared, but the functions are not. This is a weaker relationship than what Conal's methodology achieves for AD.

The honest summary: the algebra *constrains* the design space (state must be a monoid), *suggests* good points within it (tensor algebra, linear attention), and *explains* what is lost when you leave it (softmax attention breaks compositionality). But it does not *derive* the full transformer architecture the way the algebra of linear maps derives both modes of AD.


## Empirical results

We tested the algebraic hierarchy on 32k names (Karpathy's makemore dataset), 200 training steps:

| Model | Params | NLL | Notes |
|-------|--------|-----|-------|
| Count-based bigram | - | 2.455 | Optimal MLE baseline |
| T2(R^8) raw embeddings | 2,430 | **2.279** | Param-matched to linear attn |
| T2(R^16) raw embeddings | 8,262 | **2.247** | Our best algebraic model |
| LinAttn De=16 Dk=Dv=8 | 1,734 | 2.366 | Learned projections |
| LinAttn De=16 Dk=Dv=16 | 2,550 | 2.367 | Matched dim to T2 |
| Karpathy MLP | ~200 | ~2.3 | Reference |

**T2 beats linear attention at matched parameter count** (2.279 vs 2.367). But there is an important confound: our online training (no backprop through time) cannot train the key/value projection matrices `W_k` and `W_v`. These remain at random initialization. In T2, this is not a problem because the state uses the same embeddings `E` that receive gradient from the direct bigram path `W_p * E[prev]`. In linear attention, the projections are separate from `E` and have no gradient-receiving path.

This reveals a **training asymmetry** between T2 and linear attention:
- T2: embeddings are shared between the gradient-receiving path (direct bigram) and the state path. Training the bigram improves the state for free.
- Linear attention: projections only affect the state path. Without BPTT, they cannot be trained.

The algebra connects T2 to linear attention as special case to generalization. But the training dynamics favor the special case: **parameter sharing between the readout and state paths is a structural advantage** that the algebra does not predict. This is analogous to weight tying in language models -- a practical benefit that falls outside the algebraic framework.

A proper comparison would require either BPTT (to train projections) or a reformulation where projections appear in the gradient-receiving path. This remains future work.
