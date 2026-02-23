# Denotational LLM — Project Guide

## Conal's challenge (the north star)

Every decision in this project should be evaluated against Conal Elliott's guidance. Here are his exact words:

> I don't think you'll get deep insight about microGPT by exploring only at the level of computation (implementation). As George Polya put it, "It is foolish to answer a question that you do not understand." The Curry-Howard correspondence contains a profound insight: computation is epiphenomenal — a shadow of a shadow.

> Start with "next fragment of text completion" as specification. Then check whether it is
> - **precise** (expressible in Agda),
> - **satisfiable** (solvable/computable),
> - **adequate** (cannot be gamed).
>
> Likely at least one of these crucial properties fails. Reflect, and retry.

> My paper is about how to *compute* AD correctly and efficiently. I'm suggesting that you back up from computation (inessential answer) to specification (essential question). For AD, the specification is differentiability.

**Apply this check constantly.** When you're about to write code, ask: am I working on the specification or on computation? If computation, am I confident the spec is right first?

## Where we stand on Conal's three properties

### Precise? Partially.
The spec in `Spec.agda` defines `Predictor = List Char → Char → ℝ` scored by log-likelihood. This is precise and expressible in Agda. But it may not be the *right* spec — it only measures how well a predictor scores on a given corpus, not whether it generalizes or learns.

### Satisfiable? Yes, trivially.
Any function `List Char → Char → ℝ` is a predictor. The uniform distribution satisfies it. A lookup table satisfies it perfectly on any finite corpus. This is a warning sign — too easy to satisfy often means the spec isn't adequate.

### Adequate? No — this is the open problem.
The current spec can be gamed: a predictor that memorizes the training corpus scores perfectly. This is exactly the generalization problem in ML, and we haven't addressed it in the spec. Possible directions:
- Require the predictor to be *smaller* than the corpus (compression = understanding)
- Hold back test data (but this is operational, not specification-level)
- Require the predictor to be a *computable function* of bounded complexity
- Something else the algebra might suggest

**This is probably where Conal would tell us to focus next.** Not more computation, not more proofs about score decomposition — fix the spec.

## Honest assessment of what we've built

**What's real:**
- A precise type for predictors and score (`Spec.agda`)
- Score decomposition: `score(xs++ys) = score(xs) + scoreFrom(h++xs, ys)` — this is a monoid homomorphism from (List Char, ++) to (ℝ, +). Proven in Agda.
- Kleisli category structure for predictors (`Kleisli.agda`)
- Architecture hierarchy with embeddings (`Architectures.agda`)
- Forward-mode AD via dual numbers (`AD.agda`)
- Executable bigrams matching Karpathy's numbers

**Where we're overclaiming or loose:**
- "Score is a functor" — it's a monoid homomorphism, not a functor in the precise categorical sense. Score maps (predictor, corpus) → ℝ. In Conal's AD work, D is genuinely a functor between categories (smooth functions → linear maps). We should be more careful with this language.
- The architecture "analogy" is weaker than the AD analogy. In AD, forward-mode and reverse-mode compute the *same derivative* — they're genuinely different representations of the same thing. In text prediction, bigram and attention compute *different functions* with different expressive power. That's not a representation choice in Conal's sense.
- `gradient-improves` is proven, but it relies on `gradient-ascent-lemma` which postulates exactly the hard part. The "proof" is essentially: if we assume gradient ascent works, then gradient ascent works.
- Most theorems verify known facts (log of a product = sum of logs, etc.) rather than deriving anything new.

**What we haven't done that matters:**
- We haven't *derived* anything from the spec that we didn't already know. In Conal's AD work, reverse-mode via continuations was a genuine surprise. We haven't had that moment.
- We haven't addressed adequacy. The spec can be gamed by memorization.
- The executable modules (`Bigram.agda`, `BigramCount.agda`) use Float and "follow the structure" of the proofs but aren't formally connected to them.

## Goals

Follow Karpathy's [makemore](https://github.com/karpathy/makemore) progression (bigram → MLP → RNN → GPT) as a concrete target task, but derive it from specification using Conal's methodology rather than building it up from neural network primitives. The dream: the algebra reveals something about text prediction that we didn't already know — a novel architecture, optimization strategy, or insight that *falls out* of the spec the way reverse-mode AD fell out of the algebra of linear maps.

## Next steps (ordered by what Conal would prioritize)

### 1. Fix the spec (adequacy)
The most important unsolved problem. Our spec rewards memorization. How do we express "this predictor *generalizes*" in the type system? Ideas:
- Parameterized complexity: the predictor must be expressible with ≤ k parameters (compression)
- Information-theoretic: the predictor's description length must be less than the corpus
- Held-out scoring: spec includes both train and test corpora, score only counts test
- Can Agda's type system capture any of these naturally?

This is where Conal would focus. Everything else is computation built on a possibly-wrong spec.

### 2. Scale the executable (Karpathy benchmarks)
- **BigramCount to 32k names** — Read from `names.txt` instead of hardcoded `corpus50`. Need `readFiniteFile` and `lines`. Target: match Karpathy's NLL = 2.454.
- **Executable MLP (makemore part 2)** — Fixed context window, character embeddings, hidden layer with tanh, softmax. This is an n-gram architecture with learned embeddings.

### 3. Connect AD to training
- `Bigram.agda` uses numerical perturbation (slow, approximate). Wire `AD.agda`'s dual numbers to get exact forward-mode gradients.
- Then reverse-mode AD (continuations = Conal's key AD insight) for training anything with >100 parameters.

### 4. Close postulate gaps
- `log-prob-is-score`: threading positivity proofs (tedious but straightforward)
- `attn-subsumes-rnn`: auxiliary lemmas about `enumerate`
- `gradient-ascent-lemma`: the big one — this postulates the punchline. Can we at least narrow what's assumed?

### 5. Derive something new (the frontier)
The real test of whether this project succeeds in Conal's sense. Don't just classify known architectures as representation choices — find one the algebra *forces*. Or find an optimization insight. Or discover that the spec, once made adequate, constrains the solution space in a surprising way.

## Module structure

**Foundation:** `Real.agda` → `Probability.agda` → `Spec.agda`

**Theory (all build on Spec):**
- `Properties.agda` — monotonicity, convex combinations, Jensen
- `Kleisli.agda` — categorical structure, score as indexed monoid homomorphism
- `Architectures.agda` — bigram ⊂ n-gram ⊂ RNN ⊂ attention ⊂ full predictor
- `AD.agda` — forward-mode AD via dual numbers
- `Parameterize.agda` — parameter search, gradient ascent validity

**Executable (use Float, not postulated ℝ):**
- `Bigram.agda` — small gradient-descent bigram (10 names)
- `BigramCount.agda` — count-based MLE bigram (50 names, needs scaling to 32k)

## Conventions

- Proof modules use postulated ℝ from `Real.agda`; executable modules use `Float`
- `{-# OPTIONS --guardedness #-}` is required for any module using `IO`
- The `.agda-lib` file references `standard-library-2.3`
- `names.txt` has 32,032 names (one per line) from Karpathy's makemore

## Agda gotchas

- `where` blocks are NOT allowed inside `postulate` blocks
- `let` is NOT allowed inside `do` blocks — use `let ... in` outside or `where`
- Use `++-identityʳ` and `++-assoc` from `Data.List.Properties`
- Use `readFiniteFile` from `IO` for file reading
- Use `lines` from `Data.String.Base` for string splitting
- Agda's termination checker can be finicky with mutual recursion

## Type-checking

```bash
# All proof modules (order matters for dependencies)
agda Spec.agda && agda Properties.agda && agda Kleisli.agda && \
agda Architectures.agda && agda AD.agda && agda Parameterize.agda
```

Compilation of `BigramCount.agda` reads `names.txt` at runtime and takes ~30s to compile.
