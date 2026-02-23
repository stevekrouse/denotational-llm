# Can the Algebra Force a Novel Architecture?

## Working notes -- not for publication

---

## 1. What we have: the algebraic structure, precisely

The specification gives us:

```
Predictor = List Char -> Char -> R
scoreFrom p h [] = 0
scoreFrom p h (c :: cs) = log(p h c) + scoreFrom p (h ++ [c]) cs
```

And the core theorem (score-split):

```
scoreFrom p h (xs ++ ys) = scoreFrom p h xs + scoreFrom p (h ++ xs) ys
```

This makes `scoreFrom p` an **indexed monoid homomorphism** from
`(Corpus, ++, [])` to `(R, +, 0)`, where the index is the history `h`
and the index shifts by `xs` after processing `xs`.

The `IndexedHomomorphism` record in Kleisli.agda packages this:
- `F h [] = 0`
- `F h (xs ++ ys) = F h xs + F(h ++ xs) ys`

---

## 2. What the analogy promises vs what it delivers

### The AD side (Conal's success story)

In AD, the abstract type is **linear maps** (the derivative `D(f)(x) : V -> W`).
The key insight: different *representations* of linear maps yield different
algorithms for computing the same mathematical object:

| Representation of linear map | AD algorithm |
|------------------------------|-------------|
| Matrix (W x V array)        | Jacobian    |
| Function V -> W             | Forward-mode |
| Continuation W -> V         | Reverse-mode |

Critically: all three compute **the same derivative**. The representation
doesn't change what is computed, only how. The algebra of linear maps
(composition, addition) transfers to each representation, giving correctness
for free.

The genuine surprise: **continuations = reverse-mode**. This wasn't obvious
before seeing it through the algebraic lens. The algebra of linear maps said
"anything that implements compose and add is a valid representation," and
continuations happened to do so -- yielding backpropagation.

### Our side (the gap)

In text prediction, the abstract type is **predictors** (`List Char -> Char -> R`).
Different representations yield different architectures:

| Representation of predictor | Architecture |
|-----------------------------|-------------|
| `Char -> Char -> R`         | Bigram      |
| `List Char -> Char -> R` (last n) | n-gram |
| `State x (State -> Char -> State) x (State -> Char -> R)` | RNN |
| `List (N x Char) -> Char -> R` | Attention |

**But here is the honest gap**: these representations do NOT compute the same
thing. A bigram literally cannot express what an attention model can. In AD,
forward-mode and reverse-mode compute the same derivative -- they're just
different ways to evaluate the same mathematical function. In text prediction,
bigram and attention are *different functions*.

So the analogy is:
- **Strong** for the homomorphism structure (score decomposition = chain rule)
- **Strong** for optimization (gradient ascent validity transfers to any representation)
- **Weak** for architecture choice (representations have genuinely different expressive power)

This means the algebra can't force a novel architecture the same way it forced
reverse-mode AD. The algebra doesn't tell you what function to compute -- it
tells you that whatever function you compute, it decomposes correctly.

**Honesty check: this is the single most important observation. The rest of
this document tries to find what the algebra CAN still tell us, despite this
gap.**

---

## 3. What the indexed homomorphism structure DOES constrain

Even though the algebra can't force one representation over another, the
indexed homomorphism structure does constrain what a valid "incremental
scoring" scheme must look like. Let's be precise.

### 3.1 The state-passing obligation

Any implementation of `scoreFrom` must handle the fact that the index (history)
shifts. When you process `xs ++ ys`, the second fragment `ys` is scored
relative to history `h ++ xs`. This means any architecture must either:

(a) **Store enough of the history** to compute `scoreFrom p (h ++ xs) ys`
    given only the summary of `h ++ xs` (the RNN approach), or

(b) **Re-derive the relevant history** from the full sequence (the attention
    approach), or

(c) **Ignore parts of the history** and accept reduced expressiveness
    (the n-gram approach).

This is the **context accumulation problem**: the indexed homomorphism forces
any architecture to deal with the shifting index somehow.

### 3.2 The decomposition into local contributions

Score-split says scoring is additive over corpus segments. This is
deeply structural: it says that **prediction quality decomposes locally**.
Each position contributes `log(p h c)` to the total, and these contributions
are independent given the history.

This rules out certain hypothetical scoring functions. For instance, you
couldn't have a score that depends on *global* statistics of the predictor's
behavior across the corpus (like "how diverse are its predictions?"). The
spec says: score is a sum of pointwise terms. Period.

### 3.3 What the decomposition suggests about architecture

The score splits over concatenation. What if the **architecture itself**
mirrors this split? Specifically:

The score decomposition says:
```
score(p, xs ++ ys) = score(p, xs) + scoreShifted(p, xs, ys)
```

This is a **left fold** -- we process the corpus left to right, accumulating
score. But there's no reason the folding direction has to be left-to-right.
Could we also fold right-to-left? Or could we fold in a tree structure
(divide-and-conquer)?

**Tree-structured scoring.** For a corpus `w = w1 w2 ... wn`, instead of:
```
score = s(w1) + s(w1, w2) + s(w1w2, w3) + ... + s(w1...wn-1, wn)
```
we could split at the midpoint:
```
score = score(w1...wk) + scoreShifted(w1...wk, wk+1...wn)
```
and recurse on both halves. The score-split theorem guarantees this gives the
same answer. This suggests a **balanced binary tree** structure for scoring.

But does this suggest a novel *architecture*? Not really -- it suggests a
novel *evaluation strategy* for the same architecture. The predictor itself
still needs to handle arbitrary histories. The tree structure is about how
we *organize the computation*, not about what the predictor *is*.

**This parallels something in AD**: forward-mode and reverse-mode aren't
different derivatives -- they're different evaluation strategies for the
same linear map. Similarly, left-fold vs. tree-fold scoring are different
evaluation strategies for the same predictor.

Wait. This parallel might actually be productive. Let me push on it.

---

## 4. Forward and reverse modes of text prediction

### 4.1 The AD analogy more carefully

In AD for f : R^n -> R:
- **Forward-mode**: seed one input with tangent 1. One pass gives one
  directional derivative. Need n passes for the full gradient.
- **Reverse-mode**: seed the output with cotangent 1. One backward pass
  gives the full gradient. Need 1 pass.

The key: forward-mode and reverse-mode are dual evaluation strategies for
the same bilinear form (the derivative as a linear map).

### 4.2 What would "reverse-mode prediction" mean?

In forward scoring, we process the corpus left-to-right:
```
score = log p(c1|[]) + log p(c2|c1) + log p(c3|c1c2) + ...
```

Each term depends on the *past*. The history grows as we move right.

What if we processed right-to-left?
```
score = ... + log p(c3|c1c2) + log p(c2|c1) + log p(c1|[])
```

Same sum, different order. But the interesting question is: does the
**predictor** benefit from a different decomposition of the joint probability?

The chain rule of probability gives:
```
P(c1, c2, ..., cn) = P(c1) * P(c2|c1) * P(c3|c1,c2) * ...  [forward]
                    = P(cn) * P(cn-1|cn) * P(cn-2|cn-1,cn) * ... [reverse]
```

The "reverse" factorization conditions on the *future*, not the past. A
"reverse predictor" would be:
```
ReversePredictor = List Char -> Char -> R  -- but the List Char is the FUTURE
```
where `p suffix c` means "probability of c given that the *following*
characters are `suffix`."

**Is this useful?** In isolation, probably not -- text is generated left-to-right.
But in **bidirectional** models (like BERT), we actually do use both directions.
The algebraic observation here is that score-split is symmetric: you can
decompose the score starting from either end.

**Verdict: this is a known technique (bidirectional models), not a novel insight.
The algebra confirms it's valid but doesn't reveal anything new.**

### 4.3 Something potentially less obvious: the "continuation" of a predictor

In reverse-mode AD, the key object is the continuation: given the result's
adjoint, compute the input's adjoint. The continuation for `f : R -> R`
at point `x` is `delta -> delta * f'(x)`.

What's the analogous object for prediction? The "score continuation" at
position `i` in corpus `c1...cn` would be:

```
K_i(h) = scoreFrom p (h ++ c1...ci) (ci+1...cn)
```

This is a function `List Char -> R` that takes the "past summary" and
returns the "future score." It's the **suffix score** -- how well the
predictor will do on the rest of the corpus, given that the history is `h`
plus whatever came before.

The score-split theorem says:
```
total_score = scoreFrom p h (c1...ci) + K_i(h)(remaining)
```

But `K_i` depends on the specific corpus suffix. It's not a general object --
it's tied to the specific test data.

However, in the **TrueSpec** formulation (expected score under the true
distribution), the suffix score becomes an expectation:
```
K_i(h) = E_{future ~ D|h} [scoreFrom p h future]
```

This IS a general object: for each history, it tells you the expected future
score. Optimizing this is exactly what the predictor should do at each step.

**The Bellman equation connection.** The expected suffix score satisfies:
```
K_n(h) = 0
K_i(h) = E_{c ~ D(h)} [log p(h, c) + K_{i+1}(h ++ [c])]
```

This is a **Bellman equation** for the "prediction value function." The
predictor at each step contributes `log p(h, c)` to the immediate "reward,"
and the remaining value is `K_{i+1}`.

**This is actually interesting.** It says: the score decomposition gives text
prediction the structure of a sequential decision problem, where the
"action" at each step is the probability assigned by the predictor, and
the "reward" is log-probability.

But does this suggest a novel architecture? The RL connection suggests
using value-function approximation to estimate K_i. But K_i depends on the
predictor p itself (it's the expected future score *of p*), so this is
circular -- you'd be using the predictor to evaluate itself.

**Verdict: the Bellman equation structure is a real algebraic observation.
It connects text prediction to sequential decision theory. But it doesn't
obviously suggest a novel architecture -- it's more of a theoretical
insight.**

---

## 5. The group action perspective

### 5.1 History shift as a monoid action

The indexed homomorphism has the signature:
```
F : H -> Corpus -> R
```
where H = List Char (histories), and the index shifts: `F h (xs ++ ys) =
F h xs + F(h ++ xs) ys`.

The key operation is `h |-> h ++ xs`: extending the history by a corpus
fragment. This is a **monoid action** of `(Corpus, ++, [])` on `H = List Char`.
The monoid of corpus fragments acts on histories by appending.

The indexed homomorphism condition says the score is compatible with this
action: the score of a concatenation decomposes into a sum where the
"shift" in the second term corresponds to the monoid action.

### 5.2 Quotient by the architecture's equivalence relation

Each architecture imposes an equivalence relation on histories: two histories
are equivalent if the architecture can't distinguish them.

- **Bigram**: h1 ~ h2 iff lastChar(h1) = lastChar(h2). Quotient H/~ has 27 elements.
- **n-gram**: h1 ~ h2 iff lastN(n, h1) = lastN(n, h2). Quotient has 27^n elements.
- **RNN**: h1 ~ h2 iff runRNN(h1) = runRNN(h2). Quotient has |State| elements.
- **Attention**: (no identification -- all histories are distinguished).

The monoid action descends to the quotient: extending a history by a character
maps equivalence classes to equivalence classes. For bigram, the action of
character c on state s is simply `s |-> c`. For n-gram, it's a shift register.
For RNN, it's the state transition function.

**The architecture IS the quotient.** An architecture is completely determined by:
1. The quotient space H/~ (the "state space")
2. How the monoid action descends to this quotient (the "transition function")
3. A function from the quotient to distributions over Char (the "output function")

### 5.3 What quotients are "natural"?

Now the question becomes: are there quotients of the history monoid that are
algebraically natural, and that haven't been tried as architectures?

**The minimal sufficient statistic.** For a given true distribution D, there's
a minimal sufficient statistic T(h) of the history: the coarsest equivalence
relation such that P(next | h) = P(next | T(h)) for all histories h. For a
k-th order Markov process, T(h) = lastK(h). For a non-Markov process, T(h)
might need the full history.

But we're not asking what the true distribution requires -- we're asking what
the algebra suggests.

**Observation: the homomorphism condition constrains valid quotients.**

For the score to decompose correctly on the quotient, we need:
```
score on quotient state s of (xs ++ ys)
= score on s of xs + score on (s * xs) of ys
```
where `s * xs` is the action of xs on state s. This is automatically satisfied
by any quotient that's compatible with the monoid action (i.e., the action
descends to the quotient). So ALL architectures work -- the algebra doesn't
rule any out.

BUT: the algebra says something about **efficiency** of the quotient.

### 5.4 Fixed-point iteration on histories

Here's an observation I haven't seen emphasized: the history `h` after
processing a long corpus is very long, but for many true distributions,
the *conditional distribution* P(next | h) depends on only a limited
aspect of h. As h grows, the conditional distributions stabilize.

For a stationary ergodic source, the conditional distributions converge:
as |h| -> infinity, P(next | h) approaches a stationary conditional.

This means the **state sequence of an RNN** should converge to a periodic
orbit (or fixed point) for stationary input. An architecture that explicitly
models this convergence might be more efficient.

**Potential novel architecture: contractive state maps.** Require the state
transition function to be contractive:
```
||step(s1, c) - step(s2, c)|| < k * ||s1 - s2||  for some k < 1
```

This guarantees that the state converges regardless of initial conditions,
which mirrors the ergodic property of natural text. Standard RNNs don't
enforce contractivity, which is related to the vanishing/exploding gradient
problem.

**Verdict: this is a real architectural constraint suggested by the algebra
(stationary conditional distributions -> contractive state maps). But it's
essentially the idea behind echo state networks and certain regularized
RNNs. Known territory, though arrived at from a different angle.**

---

## 6. The representation theory angle

### 6.1 Representations of the history monoid

The free monoid on Char is `(List Char, ++, [])`. An architecture is a
monoid homomorphism from this free monoid to some target monoid, plus an
output function:

```
phi : List Char -> S           (state summary -- monoid homomorphism)
out : S -> Char -> R           (output distribution)
predict h c = out (phi h) c
```

For this to be a monoid homomorphism: `phi(h ++ xs) = phi(h) * phi(xs)`
for some multiplication on S.

**Wait -- this is stronger than what architectures actually satisfy.**

For a bigram: `phi(h) = lastChar(h)`. Is this a monoid homomorphism?
`phi(h ++ xs)` = lastChar(h ++ xs) = lastChar(xs) (if xs nonempty)
= phi(xs). So phi(h ++ xs) != phi(h) * phi(xs) in general.

Actually, bigram state update IS a monoid action, but NOT a homomorphism
from the free monoid to a group. It's a **semigroup action** where the state
only depends on the last character.

For an RNN: `phi(h) = foldl step init h`. Is this a monoid homomorphism?
`phi(h ++ xs) = foldl step (foldl step init h) xs = foldl step (phi(h)) xs`.
This is NOT the same as `phi(h) * phi(xs)` because the fold over xs starts
from phi(h), not from init.

So phi is NOT a monoid homomorphism -- it's a monoid ACTION. The state
space S is acted on by the free monoid, but S doesn't have its own monoid
structure that makes phi a homomorphism.

### 6.2 When IS phi a monoid homomorphism?

If we require phi to be a monoid homomorphism, we need:
```
phi(h ++ xs) = phi(h) * phi(xs)
```

This means: the state after processing h then xs equals the "product" of
the state after h and the state after xs (starting fresh). In particular,
phi(xs) must be computed **without knowing h**.

This is a very strong condition. It says the state contribution of xs is
independent of preceding context. For text prediction, this seems too strong
-- the whole point is that context matters.

UNLESS... the monoid S encodes the context-dependence internally. For example:

**S = the monoid of functions State -> State.** Then phi(c) for a single
character c is the function `s |-> step(s, c)`, and phi(xs) = phi(xn) . ... . phi(x1)
is the composed transition function. This IS a monoid homomorphism from
List Char to (State -> State, composition, id).

And the predictor becomes:
```
predict h c = out(phi(h)(init), c)
```

This is just the standard RNN, but viewed as a homomorphism into the
**function monoid** End(State).

**This is known**: the representation of a free monoid into End(S) is exactly
what an RNN computes. The "state" of the RNN is the image of init under
the composed transition, and the transitions are the representation of the
generators.

### 6.3 What representations of the free monoid exist?

The question "what architectures are possible" becomes "what representations
of the free monoid on 27 generators exist."

By the universal property of the free monoid, ANY function from 27 generators
to a monoid M extends uniquely to a monoid homomorphism List Char -> M.
So we need to choose:
1. A target monoid M
2. An assignment of the 27 characters to elements of M
3. An output function M -> (Char -> R)

Different choices of M give different architectures:

| Monoid M | Architecture |
|----------|-------------|
| Trivial monoid {*} | Unigram (ignore all history) |
| Z/27Z (last char only) | Bigram |
| (Z/27Z)^n | n-gram (via shift register) |
| End(R^d) = R^{d x d} matrices | Linear RNN |
| End(State) for nonlinear State | General RNN |
| ??? | ??? |

**The question: are there monoids M with interesting properties that
haven't been used as architectures?**

### 6.4 Candidates for novel target monoids

**(a) The tropical semiring.** Instead of real matrices, use the tropical
semiring (R, max, +). Tropical matrix multiplication is used in shortest-path
algorithms. A "tropical RNN" would compute max-plus combinations instead of
linear combinations. This has been explored in some optimization contexts
but not widely as a text prediction architecture.

**(b) Free groups / hyperbolic groups.** Instead of mapping to matrices (which
can lose information through compression), map to elements of a free group
or hyperbolic group. The word problem is decidable, and the group has
infinite "memory" (no two distinct histories collapse). But the output function
would need to handle infinitely many states.

**(c) Polynomial monoids.** Map each character to a polynomial, and compose
by substitution. The "state" is a polynomial whose evaluation at specific
points recovers useful features of the history.

**(d) The monoid of distributions.** Map each character to a *distribution
transformer*: `Dist -> Dist`. The state is the current belief distribution
over some latent variable. This is essentially a hidden Markov model -- the
state is a belief state updated by Bayesian conditioning.

Actually, (d) is interesting. Let me think about it more carefully.

---

## 7. The belief-state architecture (Bayesian predictor)

### 7.1 The idea

Suppose there's a latent variable Z (which could be "the type of name,"
"the language," "the speaker," etc.). The true distribution factors as:

```
P(c | h) = sum_z P(c | z, h) * P(z | h)
```

A Bayesian predictor maintains a belief state B(h) = P(z | h) and updates
it by Bayes' rule after each character:

```
B(h ++ [c])(z) = P(c | z, h) * B(h)(z) / sum_z' P(c | z', h) * B(h)(z')
```

The predictor then marginalizes:
```
p(h, c) = sum_z P(c | z, h) * B(h)(z)
```

### 7.2 Does this fit the algebraic framework?

The state space S = Dist(Z) (probability distributions over Z).
The transition: given current belief B and observed character c,
update B via Bayes' rule. The output: marginalize over Z.

This IS an RNN in the sense of Architectures.agda -- the state is
a distribution over Z, updated recurrently. But the update rule is
SPECIFICALLY Bayesian, not a generic learned function.

The algebraic interest: this architecture has a **canonical update rule**
(Bayes' rule) rather than a learned one. The only parameters are the
emission model P(c | z, h) and the prior P(z).

### 7.3 Is this novel?

No -- this is essentially a Hidden Markov Model (HMM), which predates
neural language models. HMMs are well-studied and known to be less
expressive than neural models for text.

But there's a twist: what if Z is continuous and high-dimensional? Then the
belief state is a continuous distribution, and exact Bayesian updating is
intractable. You'd need to approximate -- which brings you back to something
like a variational autoencoder (VAE) for text.

**Verdict: the algebra suggests Bayesian belief updating as a natural state
transition, but this is the well-known HMM/belief-state framework. Not novel.**

---

## 8. What IS genuinely novel here?

After thorough analysis, I think the honest answer is: **the algebraic
structure does not straightforwardly force a novel architecture, because
the analogy breaks at the critical point.** In AD, representations of
linear maps all compute the same derivative. In text prediction,
representations of predictors compute different functions.

However, there are a few observations that, while not complete novel
architectures, might be worth pursuing:

### 8.1 The homomorphism decomposition principle

The score decomposition says `score(xs ++ ys) = score(xs) + shifted_score(ys)`.
This is exploited implicitly by all sequential models. But what about
exploiting it **explicitly** in the architecture?

**Idea: hierarchical decomposition.** Instead of processing text character
by character, recursively split the corpus and compute scores on the halves.
The score-split theorem guarantees correctness. The architecture would be:

```
TreePredictor:
  - For a single character: use a base predictor
  - For a sequence: split at midpoint, recursively compute
  - Combine: add the scores (which is algebraically valid by score-split)
```

But the issue is that the second half's score depends on the first half's
history. So you need to pass the "history summary" from the first half to
the second. This is exactly what a **transformer with hierarchical attention**
does -- but now arrived at from the algebraic decomposition.

**The novelty, if any**: the score-split theorem tells you that this
hierarchical decomposition is EXACT (not an approximation). Most hierarchical
models in NLP (like hierarchical attention or tree-structured LSTMs) are
heuristic -- they choose tree structures for computational reasons. Here,
the algebra says any binary tree decomposition gives the same score, as
long as you pass the history correctly.

This suggests: **the tree structure doesn't matter for correctness -- it only
matters for computational efficiency.** You could parallelize scoring by
processing the two halves simultaneously (given the history from the first half),
similar to how tree-reduction parallelizes summation.

### 8.2 Score as a derivation (the Leibniz rule perspective)

The score of a single character is `log(p h c)`. The score decomposes
additively. But what about the multiplicative structure?

Before taking log, the joint probability is:
```
P(xs) = prod_i p(h_i, x_i)
```

Taking log converts this product to a sum. But the product structure satisfies
a Leibniz rule when we differentiate with respect to parameters:

```
d/dtheta [prod_i p_theta(h_i, x_i)]
  = sum_i [prod_{j != i} p(h_j, x_j)] * dp/dtheta(h_i, x_i)
```

In log space, this becomes:
```
d/dtheta [sum_i log p(h_i, x_i)] = sum_i (1/p(h_i, x_i)) * dp/dtheta(h_i, x_i)
```

The gradient decomposes as a sum over positions. Each position's contribution
to the gradient is **independent** given the history. This is why minibatch
training works: you can estimate the gradient from a subset of positions.

**Algebraic observation**: the gradient of the score is itself an indexed
homomorphism (it decomposes over concatenation). So gradient computation
has the same algebraic structure as scoring itself. This is the text-prediction
analog of "the derivative of a composition is a composition of derivatives."

### 8.3 The "dual" of a predictor (information-geometric perspective)

In the AD story, forward and reverse modes are related by duality of linear
maps (transpose). Is there a "dual" of a predictor?

A predictor maps `(history, char) -> probability`. The "dual" would map
`(future, char) -> probability` -- i.e., P(c | what comes after c).

In information geometry, the dual of the exponential family (log-probabilities)
is the moment family (expected sufficient statistics). A predictor in
exponential form is:

```
log p(h, c) = theta(h) . phi(c) - A(theta(h))
```

where theta(h) are natural parameters depending on history, phi(c) are
sufficient statistics of the character, and A is the log-partition function.

The dual parameterization would use **expected sufficient statistics**
mu(h) = E_{c ~ p(h,·)} [phi(c)] instead of theta(h).

**This might be interesting**: the natural gradient (Fisher information metric)
makes optimization in the mu-parameterization equivalent to mirror descent
in the theta-parameterization. Different parameterizations of the same
predictor family lead to different optimization algorithms, just as different
representations of linear maps lead to different AD algorithms.

**But this is about optimization, not architecture.** It's the
Parameterize.agda side of the story, not the Architectures.agda side.

### 8.4 Idempotent approximation (the novel candidate)

Here is perhaps the most genuinely novel observation I can extract.

Consider what happens when a history `h` is very long. For a k-gram model,
`phi(h) = lastK(h)` -- the state depends only on the last k characters.
This means that after k characters, the state is **completely determined by
the recent input**, regardless of anything earlier.

In monoid terms: for the k-gram monoid, every element `m` satisfies
`m * g^k = g^k` for sufficiently long generator products `g^k`. The
state becomes "absorbing" after k steps.

For an RNN, this doesn't hold in general -- the state retains information
about the entire history. But for *practical* text, the conditional
distribution P(next | history) becomes approximately independent of ancient
history.

**The algebraic formulation**: we want a monoid M where long products converge:
```
lim_{n -> infty} phi(c1) * phi(c2) * ... * phi(cn) * phi(c_{n+1}) * ... * phi(c_{n+k})
```
depends only on `c_{n+1}, ..., c_{n+k}` (for any prefix `c1, ..., cn`).

This is the **profinite completion** perspective: the free monoid's completion
with respect to the family of quotients that forget ancient history.

**A concrete architecture**: instead of a standard RNN state in R^d, use a
state space with built-in "forgetting":

```
State = (R^d1, R^d2, ..., R^dL)   -- L "timescale" components
step((s1, ..., sL), c) =
  (fast_update(s1, c),             -- updates quickly, forgets quickly
   slow_update(s2, s1, c),         -- updates slowly, retains longer
   ...,
   slowest_update(sL, s_{L-1}, c)) -- updates very slowly, retains longest
```

where each component `si` has a characteristic timescale that determines
how quickly it "forgets." The output function attends to all components:
```
out(s1, ..., sL, c) = f(s1, ..., sL, c)
```

This is essentially a **multi-timescale RNN**, which has been explored
(clockwork RNN, hierarchical RNN). But the algebraic motivation is different:
the timescale hierarchy is suggested by the **filtration of the history
monoid** -- each quotient M/~k (ignoring history older than k) gives a
different level of approximation, and the full architecture stacks these
levels.

**Is this novel enough?** Multi-timescale RNNs exist, but the specific
algebraic derivation (as a compatible family of quotients of the free monoid)
is not standard. It gives a principled way to choose the timescales and
their interactions.

---

## 9. The most promising direction: score decomposition as a design principle

Let me refocus on what the algebra gives us most clearly.

The score-split theorem says: for ANY way you split a corpus into segments,
the total score is the sum of segment scores (with history threading).
This is the **compositionality** of prediction quality.

In current practice, this compositionality is exploited in one direction
only: left-to-right sequential processing. But score-split holds for
ANY split, including:

1. **Right-to-left** (reverse prediction)
2. **Binary tree** (hierarchical prediction)
3. **Random chunks** (shuffled prediction -- like masked language modeling)
4. **Overlapping segments** with correction terms

Direction (3) is interesting: if you score on random chunks, the total
score is STILL the sum of chunk scores plus appropriate history-dependent
terms. Masked language models (BERT) essentially do this -- they score
on random masked positions. The score decomposition theorem says this
is algebraically valid as a scoring scheme, not just a heuristic.

**The frontier question**: Can we design architectures where the compositional
structure of scoring is baked into the architecture itself?

For example: an architecture where the predictor at position i doesn't
just get a "state summary" from the left (as in an RNN) but gets
independent summaries from multiple directions/scales, combined according
to the score decomposition formula. This would be a **score-aware** architecture
that uses the algebraic structure of the score to organize its computation.

This is vaguely transformer-like (attention lets you look at multiple
positions), but with a specific algebraic structure governing how information
from different positions is combined.

---

## 10. Summary of findings

### What the algebra forces:
1. Score decomposes additively over any corpus split (score-split theorem)
2. Any architecture must handle the shifting index (context accumulation)
3. An architecture = a quotient of the history monoid + output function
4. The gradient also decomposes (making minibatch training algebraically valid)

### What the algebra suggests but doesn't force:
1. Multi-timescale state spaces (from the filtration of the history monoid)
2. Tree-structured evaluation (from score-split applied to binary trees)
3. Bidirectional prediction (from the symmetry of score decomposition)
4. Contractive state maps (from the convergence of conditional distributions)

### What is genuinely novel vs repackaged:
- **Repackaged**: bigram/n-gram/RNN/attention as quotients (this is just naming what we know)
- **Repackaged**: bidirectional models, multi-timescale RNNs, HMMs as belief states
- **Somewhat novel**: the algebraic derivation of multi-timescale architectures from monoid filtrations
- **Somewhat novel**: tree-structured scoring as a parallelism strategy justified by score-split
- **Somewhat novel**: the Bellman equation structure of expected score
- **Honest gap**: the algebra can't force a novel architecture the way it forced reverse-mode AD, because architecture choice genuinely changes expressiveness (unlike AD representation choice)

### The most honest conclusion:
The algebra of text prediction is beautiful and clarifying. It unifies
known architectures under a single framework (representations of the
Kleisli morphism). It proves correctness properties transfer automatically.
But it does not -- in its current form -- derive a genuinely novel
architecture the way Conal's algebra derived reverse-mode AD from
continuations. The reason is structural: in AD, all representations compute
the SAME derivative; in text prediction, different representations compute
DIFFERENT predictors.

The closest thing to a "surprise from the algebra" would be finding a
monoid M such that:
1. The free monoid on Char has a canonical homomorphism to M
2. M is computationally efficient (finite-dimensional, fast multiplication)
3. M is expressive enough to capture the true conditional distributions
   of natural language
4. Nobody has used M as an architecture

This is the open question. The algebraic framework tells us exactly
what kind of mathematical object to look for. Whether such an M exists
and is practically useful remains to be seen.
