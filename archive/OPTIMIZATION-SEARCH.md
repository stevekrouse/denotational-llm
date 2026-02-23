# Can the Algebra Force Novel Optimization?

## Working notes -- not for publication

---

Premise: EXPLORATION.md showed that the algebra cannot straightforwardly force
a novel architecture because representations have genuinely different
expressiveness (unlike AD, where forward/reverse compute the same derivative).
But the algebra might still have untapped content on the OPTIMIZATION side:
how we search within a fixed architecture. This document explores five
directions and is honest about what is new vs. repackaged.

---

## 1. Score decomposition and training algorithms

### 1.1 What the algebra gives us

The score-split theorem (Spec.agda, line 89--134):

```
scoreFrom p h (xs ++ ys) = scoreFrom p h xs + scoreFrom p (h ++ xs) ys
```

And from EXPLORATION.md section 8.2, the gradient inherits this decomposition:

```
d/dtheta score(p_theta, xs ++ ys)
  = d/dtheta scoreFrom(p_theta, h, xs)
  + d/dtheta scoreFrom(p_theta, h ++ xs, ys)
```

The second term depends on the parameters both through the scoring of `ys`
AND through the history `h ++ xs` (since `p_theta` determines how the history
influences the score of `ys`). But for architectures where the state summary
is a deterministic function of the history and the parameters (all of ours),
the gradient decomposes cleanly: each corpus position contributes a term
`(1/p_theta(h_i, c_i)) * d/dtheta p_theta(h_i, c_i)` to the total gradient.

This means:

```
grad_theta score(theta, corpus) = sum_i grad_theta log p_theta(h_i, c_i)
```

A SUM of per-position gradient contributions.

### 1.2 Can we exploit the decomposition more aggressively than SGD?

Standard SGD already exploits this: a minibatch of size B computes
`(1/B) sum_{i in batch} grad_theta log p_theta(h_i, c_i)`, which is an
unbiased estimator of the full gradient. The score decomposition is what
makes this valid.

**Direction: segment-parallel training.** Split the corpus into K segments.
Compute full gradients on each segment independently in parallel. Average the
gradients. The score-split theorem guarantees:

```
grad score(whole) = grad score(seg_1) + grad score(seg_2) + ... + grad score(seg_K)
```

So averaging the segment gradients gives `(1/K) * grad score(whole)`, which
is the true gradient scaled by `1/K`. This is EXACT, not an approximation.

**Is this novel?** No. This is data-parallel distributed SGD, which is the
standard approach (used in every large-scale training system: Horovod,
PyTorch DDP, etc.). The gradient decomposition over data points is the
foundational reason data-parallel training works. We are just naming the
algebraic structure that everyone already exploits.

**What if we don't average, but compose?** The score-split theorem says
the total score is a SUM. But the index shifts: the second segment is scored
relative to the history from the first. In data-parallel training, we typically
ignore the cross-boundary history dependence (each GPU processes its segment
independently with its own history). The algebra tells us this is WRONG --
the second segment's score depends on the full history from the first.

For architectures with limited context (n-grams), the cross-boundary error
is bounded: only the first n positions of each segment are affected. For
RNNs, the error decays geometrically if the state transition is contractive
(EXPLORATION.md section 5.4). For attention, the error is unbounded in
principle.

**Algebraically-correct segment training:** Process segments sequentially
for the history state, but compute gradients in parallel for the non-boundary
positions. Specifically:

1. Forward-pass the full corpus to compute all history states h_i.
2. Distribute segments to workers, each receiving its initial state h_{start}.
3. Each worker computes gradients for its segment in parallel.
4. Sum the gradients.

This is the standard "gradient accumulation with proper state" approach used
in sequence-parallel training (Megatron-LM's sequence parallelism). The
algebra validates it but doesn't suggest it -- practitioners discovered it
from computational necessity.

**Verdict: the gradient decomposition from score-split is the algebraic
foundation of data-parallel and sequence-parallel training. This is real
but completely standard. The algebra names what is already known.**

### 1.3 A genuinely different decomposition: tree-structured gradients

The score-split theorem holds for ANY binary split, not just left-to-right.
For a corpus of length n, we could split at the midpoint:

```
score(w_1...w_n) = score(w_1...w_{n/2}) + score_{shifted}(w_{n/2+1}...w_n)
```

And recurse on both halves. The gradient inherits this tree structure:

```
grad score(w_1...w_n) = grad score(left half) + grad score_{shifted}(right half)
```

In a tree with depth log(n), we could compute the gradient with O(log n) sequential
steps (the depth of the tree) instead of O(n) (the length of the sequence).
Each level of the tree can be parallelized.

**The catch:** computing `score_{shifted}(right half)` requires the history
state from the left half. So we can't parallelize the two halves independently.
The history must flow from left to right at each level.

But wait -- what if we separate the "state computation" from the "gradient
computation"?

1. Forward pass: compute all history states h_0, h_1, ..., h_n sequentially (O(n)).
2. Gradient pass: given all states, compute per-position gradients in parallel (O(1) depth with n processors).

This is... just the standard approach. The forward pass must be sequential
(for RNNs) or can be parallelized (for transformers, which don't have
sequential state). The gradient pass is embarrassingly parallel given the
states.

For transformers (attention architecture), the forward pass IS parallelizable
because there's no sequential state dependency -- the history at each position
is just the raw input tokens, not a computed state. So transformers already
achieve O(1)-depth forward and backward passes.

**Verdict: tree-structured gradient computation is only interesting for
sequential architectures (RNNs), and even there it requires pre-computed
states. For transformers, it's irrelevant. Not novel.**

---

## 2. The Bellman equation and reinforcement learning

### 2.1 The structure from EXPLORATION.md section 4.3

Under the true distribution D (from TrueSpec.agda), the expected future score
from position i satisfies:

```
K_n(h) = 0
K_i(h) = E_{c ~ D(h)} [log p(h, c) + K_{i+1}(h ++ [c])]
```

This is a Bellman equation. The "state" is the history h, the "action" is
the distribution p(h, -) assigned by the predictor, the "reward" is log p(h, c)
where c is drawn from the true distribution, and the "value function" is K_i.

### 2.2 Could value-function methods replace cross-entropy training?

In standard training, we minimize cross-entropy (maximize score), which is
equivalent to supervised learning on (history, next-char) pairs. The gradient is:

```
grad_theta E_{c ~ D(h)} [log p_theta(h, c)]
  = E_{c ~ D(h)} [grad_theta log p_theta(h, c)]
```

In RL, policy gradient methods optimize a similar objective but with a
different estimator:

```
grad_theta E_{c ~ p_theta(h,-)} [R(c)]
  = E_{c ~ p_theta(h,-)} [R(c) * grad_theta log p_theta(h, c)]
```

**Critical difference:** In supervised text prediction, we sample c from
the TRUE distribution D (the training data). In RL, we sample from the
POLICY p_theta. The supervised gradient is:

```
sum_c D(h,c) * grad log p_theta(h,c)
  = sum_c D(h,c) * (1/p_theta(h,c)) * grad p_theta(h,c)
```

The RL (policy gradient) version would be:

```
sum_c p_theta(h,c) * log p_theta(h,c) * grad log p_theta(h,c)
```

These are DIFFERENT objectives. The supervised version drives p_theta toward D.
The RL version drives p_theta toward maximizing its own log-probability, which
is degenerate -- the optimizer would make p_theta a point mass (put all
probability on one character).

**The Bellman structure is real but misleading here.** The "reward" at each
step is log p(h, c) where c comes from the DATA, not from the policy. So
this is NOT a standard RL problem where the agent's actions influence the
trajectory. The training data is fixed; the predictor doesn't get to choose
what characters appear.

### 2.3 Where the Bellman structure IS useful: value-function baselines

Even though this isn't RL in the standard sense, the Bellman equation gives
a value function K_i(h) = "expected future score given history h." This is
useful as a BASELINE for variance reduction.

The per-position gradient contribution is `grad_theta log p_theta(h_i, c_i)`.
Adding a baseline b(h_i) doesn't change the expected gradient (it's a control
variate), but can reduce variance:

```
grad_theta [log p_theta(h_i, c_i) - b(h_i)]
```

The OPTIMAL baseline is the expected reward, which is K_i(h_i). In RLHF-style
training, value function baselines are standard. But for supervised language
modeling, the gradient is already low-variance (it's a supervised gradient
on observed data), so baselines are unnecessary.

### 2.4 TD learning for training predictors?

Temporal difference (TD) learning updates the value function using:

```
V(h) <- V(h) + alpha * [r + V(h') - V(h)]
```

where r is the immediate reward and h' is the next state.

In our setting, r = log p(h, c) and h' = h ++ [c]. So:

```
V(h) <- V(h) + alpha * [log p(h, c) + V(h ++ [c]) - V(h)]
```

But V is the expected future score of the PREDICTOR, which depends on both
the predictor p and the true distribution D. If we're training p, then V
changes as p changes. TD learning would be simultaneously updating p and V,
which is the actor-critic framework.

**Is this useful for language modeling?** Actor-critic methods are used in
RLHF (Proximal Policy Optimization), but there the reward signal is sparse
and comes from a separate reward model. In supervised language modeling,
we have dense supervision (the true next character at every position), so
there's no need for value-function bootstrapping.

**However:** There IS one scenario where the Bellman structure matters.
When the training objective is not the simple cross-entropy but a more
complex objective that involves future consequences of predictions (e.g.,
"predict well not just on the next token but on the next K tokens"), then
the Bellman equation gives a principled way to decompose the objective into
per-step contributions. This is exactly what happens in beam search training,
sequence-level training, and minimum risk training.

The algebraic insight: the score-split theorem decomposes the TOTAL score
into per-position contributions. The Bellman equation says that at each
position, the OPTIMAL action (prediction) is the one that maximizes
immediate reward plus future value. For cross-entropy training, the optimal
action at each position is the true conditional distribution (by Gibbs
inequality from TrueSpec.agda), REGARDLESS of what happens at future
positions. This is because log-probability decomposes additively, so each
position can be optimized independently.

**This is a genuine algebraic insight:** The additive decomposition of score
means that the globally optimal predictor is the one that is locally optimal
at EVERY position independently. There is no tension between short-term and
long-term optimization. This is why greedy (per-token) cross-entropy training
finds the global optimum -- the score decomposition makes the problem
separable.

**Comparison to RL:** In RL, the reward function typically does NOT decompose
additively over time steps, which is why RL needs dynamic programming
(Bellman equations, value functions). In text prediction with log-likelihood,
the reward DOES decompose additively (by score-split), which is why we DON'T
need RL machinery. The Bellman equation collapses to pointwise optimization.

**Verdict: The Bellman equation structure is real and algebraically
grounded. But it simplifies rather than complicates: it tells us that RL
methods are UNNECESSARY for cross-entropy training because the additive
decomposition makes the problem separable. The interesting exceptions
(sequence-level training, RLHF) involve reward functions that do NOT
decompose additively, breaking the score-split structure. This is not a
novel training algorithm but rather a theoretical explanation of why the
standard approach is optimal. Knowing this rigorously is valuable --
it tells us exactly when RL methods become necessary (when the objective
stops being a sum of per-position terms).**

---

## 3. Natural gradient from the Kleisli structure

### 3.1 The geometric setup

A parameterized family `f : Params -> Predictor` traces out a manifold M in
the space of predictors. At each parameter theta, the tangent space T_theta(M)
represents infinitesimal changes to the predictor. The Euclidean gradient
`grad_theta S` lives in the parameter space, but the "natural" direction of
steepest ascent on the manifold uses the Fisher Information Matrix (FIM).

The natural gradient is:

```
nat_grad_theta S = F(theta)^{-1} * grad_theta S
```

where F(theta) is the FIM:

```
F(theta)_{ij} = E_{c ~ p_theta(h,-)} [
  (d/d_theta_i log p_theta(h,c)) * (d/d_theta_j log p_theta(h,c))
]
```

### 3.2 Does the Kleisli structure give a canonical metric?

The Kleisli category structure says the predictor lives in a specific
categorical framework. The question is whether this framework canonically
determines a Riemannian metric on the space of predictors.

**Answer: yes, and it's the Fisher metric.** The Kleisli category of the
probability monad (Dist) has a natural metric: the Fisher-Rao metric. This
is the unique (up to scale) Riemannian metric that is invariant under
sufficient statistics / reparameterization. The probability monad's Kleisli
arrows are distributions, and distributions have a canonical information
geometry.

So the Kleisli structure does single out a specific metric, and it IS the
Fisher metric. The natural gradient follows this metric.

### 3.3 Does this give a specific optimizer?

The natural gradient update is:

```
theta_{t+1} = theta_t + eta * F(theta_t)^{-1} * grad_theta S(theta_t)
```

This is Fisher natural gradient / the natural policy gradient. For specific
parameterizations:

- For softmax outputs (exponential family): the natural gradient has closed-form
  expressions involving the covariance of sufficient statistics.
- For neural networks: the FIM is too large to invert directly (it's
  n_params x n_params). Approximations include K-FAC, diagonal Fisher, etc.

**From the Kleisli perspective:** The score `scoreFrom p h` is a sum of
`log p(h_i, c_i)` terms. The FIM for the full score is therefore a sum
of per-position FIMs:

```
F(theta) = sum_i F_i(theta)
```

where `F_i(theta)_{jk} = E_{c ~ p_theta(h_i,-)} [(d/d_j log p)(d/d_k log p)]`.

The score-split theorem (extended to second-order information) says:

```
F_{whole}(theta) = F_{first_half}(theta) + F_{second_half}(theta)
```

**The Fisher metric also decomposes additively over corpus segments.** This
is the second-order analog of the gradient decomposition.

### 3.4 What's actually novel here?

The observation that the Fisher metric decomposes over corpus segments is
real but unsurprising -- the FIM is an expectation, and expectations are
linear. The algebraic content is:

1. The Kleisli structure canonically determines the Fisher metric (known in
   information geometry since Amari 1985).
2. The score decomposition extends to the FIM (follows from linearity).
3. Natural gradient is the "right" optimizer for the Kleisli category's
   intrinsic geometry.

**What the algebra suggests concretely:** Instead of computing the FIM
over the full corpus (expensive), compute it over segments and add. This
gives an exact decomposition of F into per-segment contributions, enabling
distributed natural gradient computation.

But this is just the standard approach -- K-FAC and other approximate
natural gradient methods already decompose the FIM computation over
minibatches.

### 3.5 A less obvious angle: the score's exponential family structure

For a predictor using softmax output (which is essentially all of them),
the per-position log-probability is:

```
log p_theta(h, c) = logit_theta(h, c) - log_sum_exp(logit_theta(h, -))
```

This is a canonical exponential family form where `logit_theta(h, c)` is
the natural parameter and `c` is a one-hot sufficient statistic. The natural
gradient for exponential families is particularly simple:

```
nat_grad_theta log p_theta(h, c) = (indicator(c) - p_theta(h, -))'s
                                    pushforward through the parameter Jacobian
```

The term `indicator(c) - p_theta(h, -)` is the "residual" in the moment
parameterization. This is exactly what backpropagation computes as the
output layer gradient for softmax cross-entropy loss! The standard
`output - target` gradient signal for cross-entropy is the natural gradient
of the exponential family.

**This IS a genuine algebraic observation:** For softmax predictors, standard
SGD with the cross-entropy loss IS natural gradient descent with respect to
the last layer. The Kleisli structure (which gives us the Fisher metric)
combined with the exponential family form (softmax) means that the
"obvious" training algorithm is already natural gradient at the output level.

This was observed by Martens (2014, "New insights and perspectives on the
natural gradient method") and Pascanu & Bengio (2013): for the output layer
of a softmax classifier, the cross-entropy gradient is already the natural
gradient. The Kleisli perspective gives a categorical explanation of WHY.

**But for the hidden layers, standard SGD is NOT natural gradient.** The
FIM for the full network is not diagonal, and the per-layer blocks interact.
This is where K-FAC and related methods try to approximate the natural gradient.

### 3.6 Algebraic suggestion: layer-wise natural gradient from the Kleisli decomposition

Here's a potentially less standard observation. In the Kleisli category,
the predictor at position i is a composition of two morphisms:

1. History -> State (the encoding, done by the architecture)
2. State -> Dist(Char) (the output layer)

The score factorizes through this composition. The Fisher metric on the
composed predictor can be decomposed into contributions from each "layer"
of the composition, analogous to how the chain rule decomposes the derivative.

For a two-layer decomposition `p = out . enc`:

```
F_{total} = J_enc^T * F_out * J_enc + (correction from enc's own geometry)
```

where J_enc is the Jacobian of the encoding and F_out is the FIM of the
output layer.

The Kleisli composition structure suggests a BLOCK-DIAGONAL approximation
to the FIM that respects the compositional structure: treat each layer of
the composition as having an independent Fisher block. This is essentially
the K-FAC approximation (Kronecker-Factored Approximate Curvature), which
decomposes the FIM into per-layer blocks.

**Verdict: The Kleisli structure canonically determines the Fisher metric,
which gives the natural gradient. The score decomposition extends to the FIM
(second-order information also decomposes over corpus segments). The
compositional structure of the Kleisli category suggests block-diagonal FIM
approximations (K-FAC). All of these are known. The categorical perspective
gives a clean explanation but does not suggest a genuinely novel optimizer.**

---

## 4. Sufficient statistics from the spec

### 4.1 The TrueSpec constraint

TrueSpec.agda establishes: the optimal predictor IS the true conditional
distribution `D(h, c) = P(c | h)`. The Gibbs inequality (line 136) proves
this is the unique maximizer of the true score.

For an exponential family model, the sufficient statistics T(h) determine
the optimal predictor completely: `P(c | h) = f(T(h), c)` where f is a
known function of the sufficient statistics and the character.

### 4.2 What can the algebra tell us about sufficient statistics?

The score decomposition says:

```
score(xs ++ ys) = score(xs) + score_{h ++ xs}(ys)
```

The second term depends on the history only through whatever information
`h ++ xs` contains that is relevant for scoring `ys`. For a Markov source
of order k, the relevant information is `lastK(h ++ xs)`, and the sufficient
statistic is the last k characters.

**The algebraic constraint on sufficient statistics:** For the score to
decompose as a sum of per-position contributions (which it does, by the
score-split theorem), each position's contribution `log p(h_i, c_i)` must
depend on h_i only through some computable function of the history. The
minimal such function is the minimal sufficient statistic.

But the algebra doesn't tell us WHAT the minimal sufficient statistic IS
for a given distribution. That depends on the true distribution D, which is
empirical/unknown. The algebra only tells us that WHATEVER the sufficient
statistic is, it must be compatible with the monoid action (appending characters
to the history).

### 4.3 A concrete observation about exponential family predictors

Suppose the predictor has exponential family form:

```
log p_theta(h, c) = theta(h) . phi(c) - A(theta(h))
```

where theta(h) are natural parameters depending on history, phi(c) are
features of the character, and A is the log-partition function.

The score on a corpus is:

```
score = sum_i [theta(h_i) . phi(c_i) - A(theta(h_i))]
```

The gradient with respect to the natural parameters theta is:

```
d score / d theta(h_i) = phi(c_i) - E_{c ~ p_theta(h_i,-)}[phi(c)]
                        = phi(c_i) - mu(theta(h_i))
```

where mu = dA/dtheta is the mean parameter (expected sufficient statistic).

**The sufficient statistic of the data for this predictor is the collection
of {phi(c_i)} values.** The features phi(c) of the characters are the sufficient
statistics. For a simple character-level model where phi(c) = one-hot(c), the
sufficient statistic is just which character appeared.

**For the HISTORY's influence:** The architecture determines theta(h), which
is the "summary" of the history relevant for prediction. The algebra says that
theta(h) must depend on h in a way that's compatible with the monoid action
(history extension). But it doesn't tell us what theta should be -- that's
the architecture choice.

### 4.4 Can we read off novel sufficient statistics?

**The honest answer: no.** The algebra tells us the STRUCTURE of how
sufficient statistics must interact with the history (compatibility with
the monoid action), but not what they ARE for natural language. The sufficient
statistics of English text are a property of English, not of the algebraic
framework.

What the algebra does tell us:

1. The sufficient statistic must be updatable: given T(h) and a new character c,
   we can compute T(h ++ [c]) without knowing h. (This is the monoid action on
   the state space from Architectures.agda.)

2. The sufficient statistic should be MINIMAL: it should not retain information
   about h that is irrelevant for predicting future characters. (This is the
   expressiveness-efficiency tradeoff from Architectures.agda section 8.)

3. For exponential family predictors, the gradient at each position depends
   only on the RESIDUAL `phi(c_i) - mu(theta(h_i))`, which is the difference
   between the observed sufficient statistic and the expected one.

Point 3 is standard (it's the basis of the EM algorithm and natural gradient
for exponential families). Points 1 and 2 restate what Architectures.agda
already formalizes. No new sufficient statistics emerge.

**Verdict: The algebra constrains the form of sufficient statistics
(updatable, compatible with monoid action) but cannot determine their content
for a specific language. This is the same gap as the architecture question:
the algebra tells us what KIND of thing to look for, not what it IS.**

---

## 5. The "dual" optimization

### 5.1 Primal and dual formulations

The score is a sum of log-probabilities:

```
score(theta) = sum_i log p_theta(h_i, c_i)
```

Maximizing this is the PRIMAL optimization problem. The dual formulation
considers the Lagrangian of the constrained problem:

```
maximize sum_i log p_theta(h_i, c_i)
subject to: sum_c p_theta(h, c) = 1 for all h
            p_theta(h, c) >= 0 for all h, c
```

The Lagrangian is:

```
L = sum_i log p_theta(h_i, c_i)
  - sum_h lambda_h * (sum_c p_theta(h,c) - 1)
  + sum_{h,c} mu_{h,c} * p_theta(h,c)
```

Setting dL/dp_theta(h,c) = 0:

```
(count(h,c) / p_theta(h,c)) - lambda_h + mu_{h,c} = 0
```

where count(h,c) is the number of times (h,c) appears in the corpus.
At the optimum, p_theta(h,c) = count(h,c) / count(h), which is the
maximum likelihood estimate. The dual variables lambda_h are the normalizing
constants.

### 5.2 Dual = KL divergence minimization

The dual of maximizing sum of log-probabilities (with the normalization
constraint) is minimizing the KL divergence from the empirical distribution
to the model:

```
KL(empirical || model) = sum_{h,c} p_hat(h,c) * log(p_hat(h,c) / p_theta(h,c))
                        = const - sum_{h,c} p_hat(h,c) * log p_theta(h,c)
                        = const - (1/N) * score(theta)
```

So maximizing score IS minimizing KL divergence (up to a constant and
scaling). They are the same problem.

### 5.3 Does the dual formulation suggest a different algorithm?

In convex optimization, the dual problem sometimes has different structure
that makes it easier to solve. Here:

- Primal: maximize `sum_i log p_theta(h_i, c_i)` over theta.
- Dual: minimize `KL(empirical || p_theta)` over theta.

These are identical (up to constant). No algorithmic difference.

But there's another duality: **forward KL vs. reverse KL.**

- Forward KL: `KL(data || model)` -- this is what cross-entropy training
  minimizes. It's "mode-covering" (the model spreads probability to cover
  all modes of the data).

- Reverse KL: `KL(model || data)` -- this is "mode-seeking" (the model
  concentrates on the highest-probability modes of the data). Minimizing
  reverse KL is equivalent to maximizing `E_{c ~ p_theta} [log D(h,c)]`,
  which requires knowing the true distribution D.

**The reverse KL is what RL-based training (RLHF) effectively minimizes:**
sample from the model, evaluate with a reward (which approximates log D),
and update. So the "dual optimization" direction leads back to RL methods,
which we analyzed in Section 2.

### 5.4 The log-domain duality

The score uses log probabilities: `log p(h,c)`. There's a "dual" view
using the probability domain directly.

Before taking log, the joint probability is (from Kleisli.agda, `probProduct`):

```
P(c_1, ..., c_n | h) = prod_i p(h_i, c_i)
```

Maximizing this product is equivalent to maximizing its log (the score),
since log is monotone. But the GRADIENT in the product domain is different:

```
d/dtheta [prod_i p_theta(h_i, c_i)]
  = [prod_i p_theta(h_i, c_i)] * sum_i [d/dtheta log p_theta(h_i, c_i)]
  = P_theta(corpus) * grad_theta score(theta)
```

The product-domain gradient is the log-domain gradient SCALED by the
likelihood P_theta(corpus). For small likelihoods (long corpora), this
scaling factor is astronomically small, making product-domain optimization
numerically impossible. This is why everyone uses log-domain (score) instead.

The log-domain formulation turns the product into a sum (the score
decomposition). This is the fundamental reason the log transform appears:
it converts the multiplicative structure of probability into the additive
structure that the score-split theorem exploits.

### 5.5 Mirror descent and the dual parameterization

In information geometry, exponential families have two parameterizations:

- **Natural parameters** theta: `log p(c) = theta . phi(c) - A(theta)`
- **Mean parameters** mu: `mu = E_{p_theta}[phi(c)] = dA/dtheta`

The Legendre transform maps between them: `A*(mu) = sup_theta [theta.mu - A(theta)]`.

Mirror descent in the natural parameter space is equivalent to online
gradient descent in the mean parameter space:

- Natural param update: `theta_{t+1} = theta_t + eta * grad`
  (this is standard gradient ascent)
- Mean param update: `mu_{t+1} = mu_t + eta * (phi(c_t) - mu_t)`
  (this is an online moment update)

The mean parameter update is simply tracking the running average of the
sufficient statistics. For a bigram model, this is count-based MLE (exactly
what BigramCount.agda does). For more complex models, it's the online EM
algorithm.

**The "dual optimization" for text prediction is online EM / moment matching.**
For exponential families, this is a known equivalence. For neural networks
(which are not exponential families), the dual parameterization doesn't have
a clean form.

### 5.6 Something potentially interesting: the Bregman divergence perspective

The score `S(theta)` defines a convex function on the parameter space (assuming
the model is an exponential family). The Bregman divergence associated with
`-S` is:

```
D_S(theta, theta') = S(theta') - S(theta) - grad S(theta) . (theta' - theta)
```

This is non-negative (by convexity) and measures how much the linear
approximation understimates the improvement. It's the "curvature" of the
score surface.

The Bregman divergence gives a way to measure progress that is INTRINSIC
to the score surface, not dependent on the Euclidean structure of the
parameter space. An optimization algorithm based on Bregman divergence
would be:

```
theta_{t+1} = argmin_theta' [D_S(theta', theta_t) - eta * grad S(theta_t) . theta']
```

This is mirror descent with the Bregman divergence as the proximal term.
For the KL divergence (which is a Bregman divergence with A as the
generating function), this gives the natural gradient.

**This is known (mirror descent / natural gradient via Bregman
divergences is a well-established framework). The algebraic contribution
is recognizing that the score function's Bregman divergence is the canonical
one for the Kleisli category's geometry.**

**Verdict: The primal-dual perspective does not yield a novel training
algorithm. Maximizing score and minimizing KL divergence are the same
problem. The "dual parameterization" (mean parameters vs natural parameters)
gives mirror descent / natural gradient, which is known. The log-domain
formulation (score) is numerically necessary and algebraically natural
(it converts multiplicative probability into additive score).**

---

## 6. Synthesis: what IS genuinely new?

After analyzing all five directions, here is an honest assessment:

### What the algebra tells us about optimization that is TRUE and IMPORTANT:

1. **Score decomposes additively** (score-split). This is the algebraic
   foundation of:
   - Minibatch SGD (unbiased gradient estimation from subsets)
   - Data-parallel training (gradients sum across segments)
   - Per-position independence of the optimal predictor

2. **The Bellman equation collapses** for additive score. Because
   `score = sum of per-position terms`, the globally optimal predictor is
   locally optimal at each position independently. This is WHY supervised
   cross-entropy training is sufficient and RL methods are unnecessary
   (for the log-likelihood objective). This is theoretically clean
   and not widely appreciated in its algebraic form.

3. **The Fisher metric is canonical** from the Kleisli structure. The
   natural gradient is the "right" optimizer from the categorical
   perspective. For softmax outputs, standard cross-entropy SGD is
   already natural gradient at the output layer.

4. **The FIM decomposes over corpus segments** (second-order version of
   score-split), providing algebraic justification for per-batch Fisher
   computation in approximate natural gradient methods.

### What is NOT new despite being algebraically grounded:

- Data-parallel training (everyone does this)
- Natural gradient / K-FAC (Amari, Martens, etc.)
- The KL divergence / cross-entropy equivalence (textbook)
- Value function baselines (standard RL)
- Mirror descent / Bregman divergences (standard optimization theory)

### The one potentially novel observation:

**The additive decomposition as a SEPARABILITY THEOREM for optimization.**

Here is the cleanest statement. Define the "per-position optimization problem"
as: for each history h, find the distribution p(h, -) that maximizes
`E_{c ~ D(h)} [log p(h, c)]`. The Gibbs inequality (TrueSpec.agda) says
the solution is `p(h, -) = D(h, -)`.

The score-split theorem says the total score is the sum of per-position
scores. Therefore:

```
argmax_p total_score(p) = argmax_p sum_i score_at_position_i(p)
```

Because the score is a SUM and the optimal choice at each position is
INDEPENDENT of other positions (each position's contribution depends on
the predictor only through `p(h_i, -)`), the total optimization SEPARATES
into independent per-position problems.

**This means:** If the architecture can represent arbitrary conditional
distributions at each history (i.e., it's not capacity-limited), the global
optimum of gradient ascent is the per-position MLE. The ARCHITECTURE
determines how the per-position problems are coupled (through shared
parameters), and this coupling is the sole source of optimization difficulty.

For a bigram model: all positions with the same last character share
parameters. The per-position problems are coupled through these shared
parameters. The coupling is mild enough that MLE has a closed-form solution
(count and normalize).

For a transformer: all positions share the same network parameters. The
per-position problems are coupled through the shared weights. The coupling
is severe (nonlinear, high-dimensional), making gradient descent the only
tractable approach.

**The algebraic insight for optimization design:** The DIFFICULTY of
optimization is determined by the ARCHITECTURE's parameter sharing pattern,
not by the objective function. The objective is separable; the architecture
couples the variables. This suggests:

- For architectures with "local" parameter sharing (like bigrams, n-grams),
  closed-form solutions exist.
- For architectures with "global" parameter sharing (like transformers),
  gradient methods are necessary but the objective landscape is still a
  SUM of convex-in-output per-position terms (for exponential family outputs).
- Intermediate architectures might admit specialized optimization algorithms
  that exploit their specific coupling structure.

This is not a specific new algorithm, but it IS a clear framework for
understanding WHY certain architectures are easy to optimize and others are
hard. The algebra tells us the OBJECTIVE is always nice (separable, convex
in the output); the ARCHITECTURE is what makes optimization hard.

### The most concrete potential contribution:

**Decomposed second-order methods.** Since both the gradient AND the Fisher
information decompose over corpus positions:

```
F_total = sum_i F_i
grad_total = sum_i grad_i
```

We could use position-level curvature information for local preconditioning:

```
update = sum_i F_i^{-1} * grad_i
```

This is a "locally-preconditioned" gradient where each position's gradient
is corrected by its own curvature, then summed. This differs from the
standard natural gradient `F_total^{-1} * grad_total` in that it ignores
cross-position curvature interactions.

For parameter-sharing architectures, F_i is a rank-1 (or low-rank) update
to the FIM per position. The sum F_total has rank at most n (corpus length),
but inverting it exactly is expensive. The locally-preconditioned version
avoids the inversion by using per-position Fisher information.

**Is this actually novel?** It's similar to LARS (Layer-wise Adaptive Rate
Scaling) and LAMB (Layer-wise Adaptive Moments for Batch training), which
use per-layer (rather than per-position) local curvature. The per-position
version is finer-grained but may be noisy for stochastic optimization.
Adagrad and Adam approximate diagonal Fisher information accumulated over
positions, which is a diagonal version of this idea.

**Honest assessment: this is an incremental variation on existing adaptive
methods, not a fundamentally new algorithm. The algebra provides a clean
justification but the practical difference from Adam is likely small.**

---

## 7. Summary table

| Direction | What algebra says | Novel? | Practical value |
|-----------|-------------------|--------|-----------------|
| Score decomposition -> parallel training | Gradients sum over segments | No (standard data-parallel) | Already universally used |
| Bellman equation -> RL methods | Bellman equation collapses (separability) | Somewhat (clean theoretical statement) | Explains why RL is unnecessary for CE |
| Kleisli -> Fisher metric -> natural gradient | Fisher metric is canonical | No (Amari 1985) | Explains why SGD+softmax is natural gradient at output |
| Sufficient statistics from spec | Must be updatable, minimal | No (restates architecture choice) | No new statistics emerge |
| Primal/dual optimization | KL = -score (same problem) | No (textbook) | Mirror descent = natural gradient |
| Separability of the optimization | Objective separable; architecture couples | Somewhat (clear framework) | Explains why different archs have different optimization difficulty |
| Per-position Fisher preconditioning | FIM decomposes over positions | Incremental (variation on Adam/K-FAC) | Possibly useful but likely small improvement |

---

## 8. Conclusion

The algebraic structure of text prediction (score decomposition, Kleisli
category, Fisher metric) provides a rigorous and unified framework for
understanding WHY standard optimization methods work. But it does not derive
a genuinely novel optimization algorithm in the way that Conal's AD work
derived reverse-mode from continuations.

The fundamental reason is the same as for architecture (EXPLORATION.md):
**the algebra describes the STRUCTURE of the problem, not its CONTENT.**
The score-split theorem tells us the objective decomposes additively. This
is powerful (it implies separability, justifies minibatching, etc.), but it's
a STRUCTURAL fact that standard methods already implicitly exploit.

A genuine algorithmic surprise from the algebra would require finding a
structural property that practitioners have MISSED -- a decomposition, a
symmetry, or a duality that suggests a different algorithmic approach. The
closest we found is the separability theorem (the objective is nice; the
architecture makes optimization hard), which is a clean theoretical statement
but does not immediately yield a new algorithm.

**The most promising direction for future work:** Instead of looking for
novel optimization of a fixed architecture, look for architectures whose
parameter-sharing structure makes optimization PROVABLY easier. The algebra
tells us the objective is always separable at the output level. The question
becomes: which parameter-sharing patterns preserve enough of this separability
to allow specialized (faster than SGD) optimization? Bigrams are one extreme
(fully separable -> closed-form MLE). Transformers are the other (fully
coupled -> only SGD works). Is there something useful in between?
