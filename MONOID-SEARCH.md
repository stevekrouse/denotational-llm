# The Monoid Search: Finding Novel Target Monoids for Text Prediction

## Working notes -- not for publication

---

## 0. The precise question

From `EXPLORATION.md` and `Architectures.agda`: an architecture for text
prediction is determined by a monoid M plus an assignment of the 27
character generators to elements of M, plus an output function
`M -> (Char -> R)`. The free monoid on Char maps homomorphically to M,
and the predictor factors as:

```
List Char --phi--> M --out--> (Char -> R)
predict h c = out(phi(h), c)
```

where `phi(h) = phi(c_1) * phi(c_2) * ... * phi(c_n)` for `h = [c_1, ..., c_n]`.

Known monoids:

| Monoid M | dim(M) | Architecture | Parallelizable? |
|----------|--------|-------------|-----------------|
| `{*}` (trivial) | 0 | Unigram | Yes |
| `Z/27Z` | 1 | Bigram | Yes |
| `(Z/27Z)^n` (shift register) | n | n-gram | Yes |
| `R^{d x d}` (matrix mult) | d^2 | Linear RNN / SSM | Yes (parallel scan) |
| `End(R^d)` via nonlinear map | d | General RNN | No |

We seek M such that:
1. Natural homomorphism from `Free(27)` to M
2. Computationally efficient (finite-dimensional, fast multiplication)
3. Expressive enough for natural language conditional distributions
4. Novel as an architecture

---

## 1. Truncated tensor algebra (path signatures)

### The construction

The free monoid on 27 generators embeds naturally into the tensor algebra
`T(R^27) = R + R^27 + R^{27x27} + R^{27x27x27} + ...`

Each character `c` maps to a vector `e_c` in `R^27`. A history
`[c_1, c_2, ..., c_n]` maps to the tensor product
`e_{c_1} (x) e_{c_2} (x) ... (x) e_{c_n}`.

The full tensor algebra is infinite-dimensional (it IS the free algebra,
so it's just a re-encoding of the free monoid). The interesting move is
to **truncate** at degree k:

```
T_k(R^27) = R + R^27 + R^{27x27} + ... + R^{27^k}
```

with the multiplication rule: take the tensor product, then discard
all components of degree > k. This gives a finite-dimensional algebra
of dimension `1 + 27 + 27^2 + ... + 27^k = (27^{k+1} - 1) / 26`.

### What the predictor looks like

- **State**: a tuple `(s_0, s_1, ..., s_k)` where `s_j` is a tensor in `R^{27^j}`.
- **Update**: upon seeing character c, the new state is:
  ```
  s'_j = s_j + s_{j-1} (x) e_c    (truncated at degree k)
  ```
  More precisely, the truncated tensor product shifts information up
  one degree and adds it.
- **Output**: a learned function from the state to `Char -> R`.

The degree-j component `s_j` captures all j-th order correlations
among the characters seen so far. Specifically:
- `s_0` = 1 (scalar, just tracks that we're running)
- `s_1` = sum of all character embeddings (bag of characters)
- `s_2` = sum of all pairs `e_{c_i} (x) e_{c_j}` for `i < j`
  (ordered bigram counts, essentially)
- `s_k` = sum of all ordered k-tuples (k-gram-like statistics)

### Comparison to known architectures

**vs n-gram**: An n-gram model stores the exact last n characters.
The truncated tensor algebra at degree k stores all k-th order
correlations over the ENTIRE history, not just the last k positions.
These are fundamentally different:
- n-gram knows the exact recent context but nothing about distant past
- Truncated tensor knows aggregate statistics of all orders up to k
  but not exact positions

For example, with k=2, the tensor model knows how many times each
bigram appeared anywhere in the history, but doesn't know which bigram
was most recent. The n-gram with n=2 knows the most recent bigram but
nothing about earlier ones.

**vs linear RNN (matrix state)**: A linear RNN with state in `R^d`
and transition `s' = A_c * s` for character c is a homomorphism into
`End(R^d) = R^{d x d}`. The truncated tensor algebra is a specific
algebraic structure that End(R^d) does not naturally have (it has
matrix multiplication, not tensor product). However, one could embed
the truncated tensor algebra into a matrix algebra via a faithful
representation. The tensor algebra `T_k(R^27)` acts on itself by
left multiplication, giving an embedding into `End(T_k(R^27))`,
which is a matrix algebra of dimension `((27^{k+1}-1)/26)^2`.
This is a HUGE matrix -- for k=2 it's 757^2 = 573,049 parameters
just for the state space.

### The path signature connection

**This is NOT novel.** The truncated tensor algebra applied to
sequential data is precisely the **path signature** method developed
by Terry Lyons and collaborators. The signature of a path is the
sequence of iterated integrals, which in the discrete case reduces
to exactly the truncated tensor products described above.

Path signatures have been applied to time series classification,
financial data analysis, and handwriting recognition. The mathematical
framework (truncated tensor algebra with shuffle product) is
well-established. Several machine learning papers have used signatures
as features for sequential data.

However, path signatures have NOT been widely used as a **language
model** architecture specifically. The typical application is
feature extraction for classification, not autoregressive prediction.
There is a gap here, but it's an engineering gap, not a conceptual one.

### Honest assessment

| Criterion | Rating |
|-----------|--------|
| Natural homomorphism | Perfect -- it's the universal one |
| Computational efficiency | Poor for k >= 3 (dimension 27^k) |
| Expressiveness | Good for aggregate statistics, poor for position-sensitive patterns |
| Novelty as LM architecture | Moderate -- signatures are known, but not as autoregressive LMs |

**Verdict**: The truncated tensor algebra is mathematically natural
and well-studied (as path signatures). It captures different information
than n-grams or RNNs -- aggregate correlations rather than recent
context or compressed state. The fatal problem is dimensionality:
27^k grows too fast. For k=3, the state has dimension ~20,000.
For k=4, ~500,000. This makes it impractical for large alphabets.

One mitigation: use **random projections** (randomized signatures).
Project the high-dimensional tensor into a lower-dimensional space
via random linear maps. This has been explored in the signature
literature and gives approximate versions with controllable dimension.
But this essentially converts it into a linear RNN with structured
random transitions, which is close to existing approaches like
random feature attention or S4-style models.

**The truncated tensor algebra is the mathematically "right" answer
to "what monoid captures k-th order correlations" but it's
computationally intractable without approximation, and the
approximations converge toward known architectures.**

---

## 2. Semidirect products

### The construction

A semidirect product `M = N ><| H` consists of two monoids N and H
where H acts on N by automorphisms. Elements are pairs (n, h), and
multiplication is:

```
(n_1, h_1) * (n_2, h_2) = (n_1 * h_1(n_2), h_1 * h_2)
```

The idea: the state carries both a "summary" (in N) and a "transition
rule" (in H). The transition rule can change over time, modeling
non-stationarity.

### Concrete example

Let `N = R^d` (state vector) and `H = R^{d x d}` (transition matrices).
H acts on N by matrix-vector multiplication. Then:

```
(v_1, A_1) * (v_2, A_2) = (v_1 + A_1 * v_2, A_1 * A_2)
```

Each character c maps to a pair `(b_c, A_c)` where:
- `A_c` is a d x d matrix (how this character transforms the transition)
- `b_c` is a d-vector (the character's direct contribution to state)

After processing `[c_1, ..., c_n]`:
- The H component is `A_{c_1} * A_{c_2} * ... * A_{c_n}` -- the composed transition
- The N component is `b_{c_1} + A_{c_1} * b_{c_2} + A_{c_1} * A_{c_2} * b_{c_3} + ...`

The N component is exactly the state of a **linear RNN** with
transition matrices A_c and biases b_c! The semidirect product
`R^d ><| R^{d x d}` naturally encodes the affine recurrence:

```
s_{t+1} = A_{c_t} * s_t + b_{c_t}
```

This is well-known. In fact, the standard parallel scan algorithm for
linear RNNs / SSMs works precisely because the affine updates
`s -> A*s + b` form a monoid under composition, which IS the
semidirect product `R^d ><| R^{d x d}`.

### More exotic semidirect products

Could we use non-standard semidirect products? Some possibilities:

**(a) H acts on N nonlinearly.**
If H = {nonlinear functions N -> N}, this is just a general RNN.
Not a monoid homomorphism anymore (composition of nonlinear functions
doesn't factor as a product of individually computed pieces).

**(b) H is a finite group.**
Let H = S_n (permutation group) acting on N = R^n by permuting
coordinates. Then the state carries both a vector AND a permutation.
The permutation component tracks how characters have "rearranged"
the state dimensions. This is related to equivariant neural networks.
The monoid is `R^n ><| S_n`, with dimension n + log(n!). It's
finite (for the H component) and efficient. But it's unclear why
permuting state dimensions would capture useful language structure.

**(c) Wreath products (iterated semidirect products).**
The Krohn-Rhodes decomposition theorem says every finite semigroup
decomposes as an iterated wreath product of simple groups and
aperiodic semigroups. This is the semigroup-theoretic analog of the
Jordan-Holder theorem. In principle, one could build a "Krohn-Rhodes
architecture" that mirrors this decomposition. Each layer would be
either a group layer (capturing cyclic/permutation-like patterns)
or an aperiodic layer (capturing "reset" / "forget" behavior).

This is intriguing in principle. But in practice:
- The decomposition depends on the target semigroup, which we don't know
- The wreath product of even small groups gets large quickly
- It's unclear how to learn the decomposition from data

### Connection to Mamba / selective SSMs

Mamba and related selective state space models make the transition
matrices input-dependent: `A_t = f(x_t)`. In our framework, this
means the monoid M changes with each input -- it's NOT a fixed
homomorphism from the free monoid, but a data-dependent one. This
breaks the monoid homomorphism structure, which is exactly why Mamba
cannot use the simple parallel scan (it uses a modified selective scan
instead).

The semidirect product perspective clarifies why: a linear SSM (fixed A)
is a homomorphism into `R^d ><| R^{d x d}`. A selective SSM (varying A)
is a homomorphism into a LARGER monoid where the transition matrix
itself is parameterized by the input, which means the effective monoid
is closer to `End(State)` (general RNN territory) than to the
semidirect product.

### Honest assessment

| Criterion | Rating |
|-----------|--------|
| Natural homomorphism | Yes -- semidirect products accept homomorphisms componentwise |
| Computational efficiency | Good for affine case (same as linear RNN) |
| Expressiveness | Affine case = linear RNN; more exotic cases unclear |
| Novelty | Low -- affine semidirect product IS the linear RNN/SSM |

**Verdict**: The semidirect product `R^d ><| R^{d x d}` is exactly the
monoid underlying linear RNNs and state space models. This is a clean
algebraic description of known architecture. More exotic semidirect
products (with finite group actions, wreath products) are theoretically
possible but don't obviously capture useful language structure.
The Krohn-Rhodes direction is interesting but too abstract to be
immediately practical.

---

## 3. Polynomial functors / species (multiset monoids)

### The construction

Instead of tracking exact history or a compressed state, track
**statistics** of the history -- counts, frequencies, or other
combinatorial summaries.

The simplest version: map each character to a basis vector in `R^27`,
and let the monoid operation be **addition**. Then `phi(h)` is the
vector of character frequencies. The monoid is `(R^27, +, 0)` --
commutative! This means the state is a **bag of characters** with
no order information.

More generally, we can track multisets of k-grams:

```
M_k = R^{27^k}   (counts of each k-gram)
```

with addition as the monoid operation. After processing history h,
`phi(h)` records how many times each k-gram appears in h.

Wait -- this is NOT a monoid homomorphism from the free monoid.
The k-gram counts of `h_1 ++ h_2` are NOT the sum of k-gram counts
of h_1 and h_2, because k-grams can span the boundary. For example,
the bigram "ab" in "xab" doesn't appear in "x" or "ab" separately.

The boundary issue means this only works for the bag-of-characters
case (k=1), which IS a homomorphism because single characters don't
span boundaries.

For k >= 2, we need to also track the boundary: the last (k-1)
characters. This gives a monoid:

```
M = R^{27^k}  x  (Z/27Z)^{k-1}
```

where the first component tracks k-gram counts and the second tracks
the last (k-1) characters (for boundary handling). The multiplication
must account for boundary k-grams. This is a semidirect product of
the count space by the shift register.

### What the predictor looks like

For the bag-of-characters case (M = (R^27, +, 0)):
- State: a 27-dimensional vector of character counts
- Update: `s' = s + e_c` (increment count of character c)
- Output: predict next character from frequency distribution of past

This is a **unigram model with memory** -- it knows the overall
character distribution but not any sequential information. For names,
it would learn that 'a' is common and 'z' is rare, but not that
'q' is usually followed by 'u'.

For the k-gram count case:
- State: vector of k-gram counts + last (k-1) characters
- Update: form new k-gram from boundary + new char, increment its count
- Output: predict based on recent context (from boundary) weighted by
  k-gram statistics

This is essentially a **count-based n-gram model** -- exactly what
`BigramCount.agda` implements! The monoid structure is the one
underlying maximum likelihood estimation for n-grams.

### Species-theoretic perspective

The theory of combinatorial species provides a systematic way to
enumerate and combine data structures. The species perspective
on our problem:
- A "bag" (multiset) is the species `E` (exponential)
- A "sequence" is the species `L` (list)
- Our state monoids correspond to functors from sequences to bags

The species framework doesn't obviously suggest a novel monoid beyond
what we've already identified. It provides a language for describing
the same constructions (bags of k-grams, etc.) but doesn't generate
genuinely new ones for this problem.

### Honest assessment

| Criterion | Rating |
|-----------|--------|
| Natural homomorphism | Only for bag-of-characters; k-gram counts need boundary handling |
| Computational efficiency | Excellent (just count accumulation) |
| Expressiveness | Poor -- loses all ordering information (bag case) or reduces to n-gram (with boundary) |
| Novelty | None -- this is the count-based n-gram model |

**Verdict**: The commutative monoid approach gives count-based models.
With boundary handling, it gives exactly the classical n-gram. The
species framework is a nice language for these constructions but
doesn't suggest anything beyond what statisticians figured out in
the 1980s.

---

## 4. Tropical (max-plus) algebra

### The construction

The tropical semiring replaces `(R, +, x)` with `(R_max, max, +)`:
- "Addition" is `max`
- "Multiplication" is `+`
- Zero (additive identity) is `-inf`
- One (multiplicative identity) is `0`

Tropical matrices multiply by: `(A *_trop B)_{ij} = max_k (A_{ik} + B_{kj})`.
This computes shortest paths (or, with max, longest paths).

A "tropical RNN" would have state in `R^d` and transitions:
```
s'_i = max_j (A_{c,ij} + s_j)     for each character c
```

This is a max-plus linear recurrence. Each character c has a
`d x d` tropical matrix `A_c`, and the state updates by tropical
matrix-vector multiplication.

The monoid: `M = (R^{d x d}, *_trop, I_trop)` where `I_trop` has
0 on diagonal, `-inf` off diagonal.

### What the predictor looks like

- **State**: d-dimensional vector in `R^d` (initialized to some `s_0`)
- **Update**: `s' = A_c *_trop s` (tropical matrix-vector product)
- **Interpretation**: `s_i` represents the "maximum accumulated score
  along any path that ends in state i." The transition adds the
  character-specific score and takes the max over incoming states.
- **Output**: predict from the tropical state

This is closely related to:
- **Viterbi algorithm** for HMMs (finding most likely state sequence)
- **Weighted finite automata** over the tropical semiring
- **Dynamic programming** (Bellman equation with max)

### Expressiveness analysis

Tropical matrix multiplication captures the same class of
computations as shortest-path problems. For language modeling,
this means the tropical RNN can track "what is the best
explanation for the history so far" where "best" is in the
max-sum sense.

However, for predicting probability distributions, we need the
OUTPUT to be a probability distribution, not a tropical value.
The tropical state would need to be converted to probabilities
somehow (e.g., softmax of the tropical state vector). This is
awkward -- the tropical algebra doesn't naturally produce
probability distributions.

More fundamentally, tropical algebra captures "max" structure
but not "sum" structure. Language modeling needs expected
log-probabilities (sums), not maximum log-probabilities (max).
The tropical approach optimizes the wrong objective: it finds
the single best explanation rather than marginalizing over all
explanations.

### Connection to existing work

Tropical geometry has been studied extensively in connection with
neural networks:
- ReLU networks compute tropical rational functions
- The decision boundaries of ReLU networks are tropical hypersurfaces
- Tropical methods have been used for network compression and analysis

However, using tropical algebra as the RECURRENCE structure (rather
than as the activation function) is less explored. There are some
papers on "min-max-plus neural networks" that use tropical operations
as layers, but not specifically as the state transition monoid for
autoregressive language modeling.

"Tropical attention" has been proposed very recently (2025) for
combinatorial algorithms, but this applies tropical operations to
the attention mechanism, not to the recurrent state.

### Honest assessment

| Criterion | Rating |
|-----------|--------|
| Natural homomorphism | Yes -- tropical matrix monoid works fine |
| Computational efficiency | Good (tropical mat-mul is O(d^2) per step, parallelizable) |
| Expressiveness | Limited -- captures max-paths but not marginals |
| Novelty | Moderate -- tropical RNN as LM state is not well-explored |

**Verdict**: A tropical RNN computes max-plus combinations, which is
natural for Viterbi-style "best path" computations but awkward for
probabilistic language modeling. The fundamental mismatch is that
log-likelihood scoring uses sums (expectations), while tropical
algebra uses maxima. The tropical approach would be more natural
for a "maximum a posteriori" predictor than for a Bayesian one.
As a language model architecture, it's novel but likely inferior
to standard (sum-product) approaches for the standard log-likelihood
objective.

**An interesting exception**: if the goal were MAP prediction
(predicting the single most likely continuation) rather than full
distribution prediction, tropical algebra would be exactly right.
The algebra of MAP prediction IS tropical. But the standard LM
objective is log-likelihood, which is additive (not max-plus).

---

## 5. Group algebras of finite groups

### The construction

Let G be a finite group. The group algebra `R[G]` is the vector
space of formal linear combinations `sum_{g in G} a_g * g` with
multiplication extending the group operation bilinearly:

```
(sum_g a_g * g) * (sum_h b_h * h) = sum_{g,h} (a_g * b_h) * (g * h)
                                   = sum_k (sum_{g*h=k} a_g * b_h) * k
```

This is convolution on the group.

Each character c maps to an element of R[G]: a function `G -> R`
assigning a real weight to each group element. Processing a history
amounts to convolving these functions.

By the representation theory of finite groups, R[G] decomposes as
a direct sum of matrix algebras:

```
R[G] = M_{d_1}(R) + M_{d_2}(R) + ... + M_{d_r}(R)
```

where d_1, ..., d_r are the dimensions of the irreducible
representations, and `sum d_i^2 = |G|`.

Via the Fourier transform on G, convolution in R[G] becomes
**independent matrix multiplication** in each block:

```
f_hat(rho_i) = sum_{g in G} f(g) * rho_i(g)
```

where rho_i is the i-th irreducible representation (a d_i-dimensional
matrix-valued function on G). Convolution `f * g` transforms to
`f_hat(rho_i) * g_hat(rho_i)` -- pointwise matrix product in each
irreducible.

### What the predictor looks like

- **State**: a collection of matrices `(S_1, ..., S_r)` where `S_i`
  is `d_i x d_i`. Total parameters: `sum d_i^2 = |G|`.
- **Update**: upon seeing character c (which maps to matrices
  `C_1, ..., C_r` via the Fourier transform), update:
  `S'_i = S_i * C_i` for each i.
- **Output**: either read from one block or combine all blocks.

Each irreducible representation captures a different "frequency" of
the input. Low-dimensional representations capture coarse structure;
higher-dimensional ones capture finer distinctions.

### Choice of group

The group G is a design choice. Some candidates:

**(a) Cyclic group Z/nZ:**
R[Z/nZ] = R^n with circular convolution. The irreducible
representations are the n-th roots of unity. This gives a model
that tracks periodic patterns of period dividing n. State dimension = n.
The "Fourier transform" is the standard DFT.

**(b) Symmetric group S_n:**
|S_n| = n!, which grows very fast. The irreducible representations
correspond to Young diagrams. This is very expressive but
computationally expensive.

**(c) Dihedral group D_n:**
|D_n| = 2n. Captures both cyclic patterns and mirror symmetry.
Irreducible representations are 1- or 2-dimensional.

**(d) Direct product of small groups:**
E.g., `(Z/2Z)^k` gives an abelian group of order 2^k with
all 1-dimensional irreducibles. Convolution is the Walsh-Hadamard
transform. State dimension = 2^k.

### The key question: why a group algebra?

The group algebra R[G] is a specific kind of matrix algebra
(it's a direct sum of full matrix algebras via Wedderburn's theorem).
A general d x d matrix monoid End(R^d) is ALSO a collection of
matrices under multiplication.

The difference: R[G] has additional algebraic structure. Specifically,
it has an involution `g -> g^{-1}` (since G is a group), and the
decomposition into irreducible blocks is canonical (independent of
the particular elements).

For language modeling, this additional structure means:
- Each irreducible representation evolves independently
- The representations capture "frequency components" of the history
- The group inverse provides a notion of "forgetting" or "undoing"

The independent evolution is the key feature: unlike a general
matrix RNN where all dimensions interact, the group algebra RNN
has independent blocks. This is **exactly the same structure as
diagonal SSMs** (like S4 with diagonal state matrix), but with
potentially higher-dimensional blocks.

### Comparison to diagonal SSMs

A diagonal SSM has state `s in C^d` with update `s' = diag(a_c) * s`
(componentwise multiplication by complex numbers). This is a
representation of the free monoid into `C^d` under componentwise
multiplication -- which is the same as a homomorphism into `(C^*)^d`,
the direct product of d copies of the multiplicative group of
nonzero complex numbers.

The group algebra approach generalizes this: instead of d copies
of a 1-dimensional representation (diagonal), we allow blocks of
varying dimension. A block of dimension k corresponds to a
k-dimensional irreducible representation, which can track
k-dimensional "correlations" that a diagonal model cannot.

Recent research (2024-2025) has shown that diagonal LRNNs have
significant expressiveness limitations: they cannot solve the parity
problem or track even simple finite-state machines. The fix proposed
in the literature is to allow negative eigenvalues (extending from
[0,1] to [-1,1]). The group algebra perspective suggests a more
principled fix: allow **non-abelian groups**, giving irreducible
blocks of dimension > 1 that can track state-machine-like behavior.

### Honest assessment

| Criterion | Rating |
|-----------|--------|
| Natural homomorphism | Yes -- map generators to elements of R[G] |
| Computational efficiency | Good -- block-diagonal structure, FFT for abelian groups |
| Expressiveness | Depends on G; larger G = more expressive but more parameters |
| Novelty | Low-moderate -- diagonal SSMs are the abelian case; non-abelian blocks are less explored but related to existing work on structured matrices |

**Verdict**: The group algebra approach is a principled generalization
of diagonal state space models. Diagonal SSMs = abelian group algebras.
Non-abelian group algebras give block-diagonal SSMs with potentially
richer expressiveness. This is a clean algebraic description but
not dramatically novel -- it's the algebraic way to say "use
block-diagonal transition matrices with blocks of varying size."
The specific insight about which groups and which irreducible
representations to use for language modeling is still open.

**The most promising aspect**: the group algebra forces a specific
decomposition into independent blocks, each of which has a
well-understood algebraic structure. This could be useful for
interpretability -- different blocks capture different "frequencies"
of the language, analogous to how Fourier analysis decomposes a
signal into independent frequency components.

---

## 6. Two additional directions

### 6a. Clifford algebras

The Clifford algebra `Cl(p,q)` is generated by vectors `e_1, ..., e_n`
subject to `e_i * e_j + e_j * e_i = 2 * g_{ij}` where g is a
quadratic form. `Cl(n,0)` has dimension `2^n`.

Map each character to a Clifford algebra element. The state is a
general element of the Clifford algebra, which has `2^n` components
corresponding to scalars, vectors, bivectors, etc.

This is related to the truncated tensor algebra (direction 1) but
with the anticommutation relation imposed. The Clifford algebra is
a quotient of the tensor algebra by the ideal generated by
`v (x) v - Q(v)` for all vectors v.

Clifford algebras have been used in neural networks (Geometric
Clifford Algebra Networks, 2023) but primarily for geometric data
(3D rotations, point clouds), not for text. For 27-dimensional input,
`Cl(27,0)` has dimension `2^27 = 134 million` -- far too large.
One would need to work with a much smaller algebra, which means
projecting the 27-dimensional character space into a low-dimensional
space first. But then the Clifford structure isn't really buying
anything over a standard embedding + matrix multiplication.

**Verdict**: Clifford algebras are interesting for geometric problems
but poorly suited to high-dimensional discrete inputs like text.
The dimension explosion (`2^n`) is worse than the tensor algebra
(`(n^{k+1}-1)/(n-1)` for truncation at degree k).

### 6b. Semigroup algebras of the syntactic monoid

For any language L (set of strings), the syntactic monoid `Syn(L)` is
the quotient of the free monoid by the Myhill-Nerode equivalence
relation. It's the minimal monoid that recognizes L.

For natural language, the "language" is the set of well-formed
strings (or, more precisely, the support of the distribution we're
trying to model). The syntactic monoid captures exactly the
distinctions that matter for membership in L.

The problem: for natural language, the syntactic monoid is either
infinite (if the language is not regular) or enormous (if we
approximate it as regular). Natural language is context-free (or
mildly context-sensitive), so its syntactic monoid is not finite.

Even for simplified versions, the syntactic monoid approach requires
knowing the target language in advance, which defeats the purpose of
learning a language model.

**Verdict**: Theoretically beautiful but impractical. We'd need to
know the language to build the monoid, but the point of a language
model is to learn the language.

---

## 7. Synthesis: what did we find?

### The landscape

After working through five directions, a clear picture emerges:

```
                      Expressiveness
                           |
        Full predictor     |
        (no monoid)        |
                           |
        Attention          |     (not a monoid -- looks at all positions)
                           |
        General RNN        |     End(State), nonlinear -- not parallelizable
                           |
        ---- the monoid frontier ----
                           |
        Group algebra RNN  |     R[G] -- block-diagonal, parallelizable
        (non-abelian)      |
                           |
        Linear RNN / SSM   |     R^{dxd} or diagonal -- parallelizable
        (affine semidirect)|
                           |
        Truncated tensor   |     T_k(R^27) -- huge but captures k-order stats
                           |
        Tropical RNN       |     max-plus matrices -- wrong objective
                           |
        n-gram (counts)    |     commutative + boundary = classical n-gram
                           |
        Bag of chars       |     (R^27, +) -- no order information
                           |
        Unigram            |
                           |
                      Computational cost
```

### The honest conclusion

**None of the five directions produced a genuinely novel, practical
architecture.** Each direction either:

(a) **Converges to a known architecture** under closer analysis:
    - Semidirect product -> linear RNN / SSM
    - Commutative monoid -> n-gram
    - Abelian group algebra -> diagonal SSM

(b) **Is mathematically natural but computationally intractable**:
    - Full tensor algebra (dimension 27^k)
    - Clifford algebra (dimension 2^n)
    - Large group algebras (dimension |G|)

(c) **Has a fundamental mismatch with the LM objective**:
    - Tropical algebra optimizes max, not sum

### Why this happened

The fundamental constraint is that the monoid must be both:
1. **Finite-dimensional** (for computational tractability)
2. **Rich enough** to distinguish histories that produce different
   conditional distributions

These pull in opposite directions. The free monoid (= full history)
satisfies (2) perfectly but has infinite dimension. Any
finite-dimensional quotient loses some distinctions.

The known finite-dimensional monoids that preserve useful distinctions
are precisely the ones already used as architectures:
- Matrices (linear RNNs) -- capture linear recurrences
- Diagonal matrices (diagonal SSMs) -- capture independent oscillations
- Shift registers (n-grams) -- capture recent context

The tensor algebra, group algebra, and Clifford algebra are all
attempts to find a "sweet spot" between these, but the dimension
grows too fast to be practical.

### What IS somewhat new

Despite the negative conclusion about genuinely novel architectures,
several observations are worth recording:

**1. Non-abelian group algebra SSMs.** The observation that diagonal
SSMs = abelian group algebras suggests that using non-abelian
groups (giving block-diagonal transition matrices with blocks of
dimension > 1) could address the known expressiveness limitations
of diagonal SSMs. The specific prediction: a block-diagonal SSM
where block sizes correspond to dimensions of irreducible
representations of a chosen non-abelian group should be able to
solve the parity problem and track finite-state machines that
diagonal SSMs cannot. This isn't "novel" in the strong sense
(block-diagonal matrices are well-known) but the algebraic
framing suggests principled choices of block structure.

**2. Tensor-train monoid = structured linear RNN.** The tensor-train
language model (TTLM) from recent work (2024) is precisely the
monoid obtained by representing the truncated tensor algebra via
a low-rank tensor decomposition. In our framework, this is a
homomorphism from the free monoid into a structured matrix
monoid -- specifically, the monoid of matrices with tensor-train
structure. The algebraic perspective clarifies WHY tensor trains
work for language modeling: they approximate the universal
(truncated tensor) monoid with a computationally tractable one.

**3. The parallel scan = monoid homomorphism.** The parallel scan
algorithm used in S4/S5/Mamba for efficient training is precisely
the computation of a monoid homomorphism via parallel prefix sum.
This is explicit in some papers but the connection to the
architecture-as-monoid-choice framework is not usually made.
Our framework says: any architecture that can be expressed as a
monoid homomorphism automatically supports parallel training via
scan. This is a SELECTION CRITERION for architectures, not just
a computational trick.

**4. The expressiveness hierarchy has algebraic characterizations.**
Recent work (2024-2025) on the expressiveness of linear RNNs
uses the theory of transformation monoids and Krohn-Rhodes
decomposition to characterize which formal languages different
architectures can recognize. This is exactly our
"architecture = monoid" framework, independently discovered in
the formal language theory community. The state-tracking problem
IS the monoid word problem.

### The most promising concrete direction

If I had to bet on one direction being fruitful, it would be:

**Non-abelian block-diagonal SSMs with group-theoretic structure.**

The recipe:
1. Choose a finite non-abelian group G (e.g., S_3, S_4, A_4, or
   a dihedral group D_n)
2. Compute the irreducible representations of G
3. Build a state space model where the state lives in the
   direct sum of these irreducible representation spaces
4. Each character maps to a block-diagonal matrix with one block
   per irreducible representation
5. Train the character-to-matrix mapping

This gives a model that:
- Is parallelizable (via scan on block-diagonal matrices)
- Has well-understood algebraic structure (representation theory of G)
- Overcomes the diagonal SSM limitation (blocks of dim > 1)
- Has a principled choice of block sizes (from rep theory of G)

The open question: which group G, and which assignment of characters
to group algebra elements, gives the best language model?

For a 27-character alphabet, reasonable choices might include:
- `S_3` (|G| = 6, irreducible dims 1, 1, 2) -- 6 parameters per step
- `S_4` (|G| = 24, irreducible dims 1, 1, 2, 3, 3) -- 24 parameters
- `A_5` (|G| = 60, irreducible dims 1, 3, 3, 4, 5) -- 60 parameters
- Direct products like `S_3 x S_3` (|G| = 36) for more flexibility

This is testable: implement the group algebra SSM for small groups
and compare against diagonal SSMs on the makemore task.

---

## 8. Summary table

| Direction | Monoid M | Dimension | Expressiveness | Novel? | Practical? |
|-----------|----------|-----------|----------------|--------|------------|
| Truncated tensor `T_k(R^27)` | Tensor product (truncated) | `(27^{k+1}-1)/26` | k-order correlations | No (= path signatures) | No (dim explosion) |
| Semidirect `R^d ><| R^{dxd}` | Affine maps | `d^2 + d` | Linear recurrences | No (= linear RNN) | Yes |
| Commutative `(R^{27^k}, +)` | Addition (with boundary) | `27^k + k-1` | k-gram counts | No (= n-gram MLE) | Yes (for small k) |
| Tropical `R^{dxd}_{max-plus}` | Max-plus matrices | `d^2` | Max-paths | Moderate | Mismatched objective |
| Group algebra `R[G]` | Convolution | `|G|` | Depends on G | Moderate | Yes (via FFT) |
| Non-abelian block-diagonal | Block matrices | `sum d_i^2` | Richer than diagonal | Yes-ish | Yes |
| Clifford `Cl(n,0)` | Clifford product | `2^n` | Geometric | No (for text) | No (dim explosion) |

---

## 9. Connections to the Agda formalization

How would the most promising directions connect back to the codebase?

**In `Architectures.agda`**: A new architecture record for the
group algebra SSM would look like:

```agda
record GroupAlgRep (G : FiniteGroup) : Set where
  field
    -- For each irreducible representation rho_i of G,
    -- a matrix S_i tracking the state in that block
    state-blocks : (i : Fin (num-irreps G)) -> Matrix (dim-irrep G i) (dim-irrep G i)

    -- Assignment of characters to group algebra elements
    char-map : Char -> GroupAlgebra G

    -- Output function from state to prediction
    output : (blocks : ...) -> Char -> R
```

The embedding into `Predictor` would fold the block-diagonal
matrix multiplication over the history, then apply the output function.

**In `Kleisli.agda`**: The score homomorphism transfers automatically
since the embedding produces a valid `Predictor`. The parallel scan
property would need a new lemma showing that block-diagonal matrix
multiplication is associative (which it is, trivially).

**In `Parameterize.agda`**: The parameters would be the character-to-block
mappings (27 x |G| real numbers) plus the output function parameters.
Gradient ascent validity transfers via the existing theorems.

**Executable test**: Implement a group algebra bigram on the
makemore task. Use `S_3` (smallest non-abelian group, dimension 6).
Compare NLL against the count-based bigram (NLL = 2.454) and
gradient-descent bigram. This would be a concrete, testable
prediction of the algebraic framework.

---

## References and sources

- Path signatures and tensor algebra: Terry Lyons, "A Primer on the Signature Method in Machine Learning" (2016). See also recent work on [randomized signatures](https://arxiv.org/pdf/1603.03788).
- Tensor-train language models: [Language Modeling Using Tensor Trains](https://arxiv.org/abs/2405.04590) (2024), which shows TTLM generalizes second-order RNNs and multiplicative RNNs.
- Tropical geometry and neural networks: [Tropical Geometry of Deep Neural Networks](https://www.stat.uchicago.edu/~lekheng/work/tropical.pdf); [Tropical Attention](https://arxiv.org/html/2505.17190v2) (2025).
- Diagonal SSM expressiveness limitations: [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/html/2411.12537v2) (2024); [On the Expressiveness of Selective SSMs on Regular Languages](https://arxiv.org/abs/2412.19350) (2024).
- Mamba and selective state spaces: [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (2024).
- Parallel scan and monoid homomorphisms: [Simplified State Space Layers for Sequence Modeling](https://arxiv.org/pdf/2208.04933) (S5, 2022). Also: [Parallel reductions using semidirect products](https://juliafolds.github.io/data-parallelism/explanation/semidirect-products/).
- Krohn-Rhodes theory: [On Krohn-Rhodes Theory for Semiautomata](https://arxiv.org/pdf/2010.16235) (2020).
- Clifford algebra neural networks: [Geometric Clifford Algebra Networks](https://arxiv.org/abs/2302.06594) (2023).
- Group algebras and Fourier analysis on finite groups: [Fourier transform on finite groups](https://en.wikipedia.org/wiki/Fourier_transform_on_finite_groups); [Fourier analysis on finite abelian groups](https://www.math.ucla.edu/~tao/247b.1.07w/notes9.pdf) (Tao).
