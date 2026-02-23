# Denotational LLM — Project Guide

## What this is

An Agda formalization of text prediction following Conal Elliott's denotational design methodology. The central analogy: just as derivatives form a functor from smooth functions to linear maps, score forms an indexed homomorphism from predictors (Kleisli morphisms) to (ℝ, +). Different representations of the morphism yield different architectures.

## Goals

The dream: derive a text prediction system from pure algebraic specification the way Conal derived AD — where new algorithms *fall out* of the algebra rather than being layered on top.

**Accomplished so far:**
- Full specification of what "better predictor" means (log-likelihood score)
- Kleisli category structure with score as indexed functor (the core insight)
- Architecture hierarchy (bigram ⊂ n-gram ⊂ RNN ⊂ attention) as representation choices
- Proof that gradient ascent on any parameterized family is valid
- Executable bigrams matching Karpathy's makemore numbers

**The open frontier:**
- We have an elegant explanation of *why* existing architectures work, but haven't yet derived a *novel* architecture from the algebra. In Conal's AD work, reverse-mode via continuations was a genuine surprise — the algebra revealed something new. We want the same: a representation of the Kleisli morphism that nobody has tried, or an optimization strategy suggested by the algebraic structure.
- Beating Karpathy's performance with a more elegant, provably-correct representation would be the real win.
- The analogy is strong for optimization (score decomposition = chain rule) but has an honest gap for architecture choice: in AD, different reps of linear maps compute the *same* derivative; in text prediction, different reps of predictors have genuinely *different* expressive power.

## Module structure

**Foundation:** `Real.agda` → `Probability.agda` → `Spec.agda`

**Theory (all build on Spec):**
- `Properties.agda` — monotonicity, convex combinations, Jensen
- `Kleisli.agda` — categorical structure, score-as-functor (the key insight)
- `Architectures.agda` — bigram ⊂ n-gram ⊂ RNN ⊂ attention ⊂ full predictor
- `AD.agda` — forward-mode AD via dual numbers
- `Parameterize.agda` — parameter search, gradient ascent validity

**Executable (use Float, not postulated ℝ):**
- `Bigram.agda` — small gradient-descent bigram (10 names)
- `BigramCount.agda` — count-based MLE bigram (32k names)

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
