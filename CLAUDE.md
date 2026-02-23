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

## Next steps (detailed)

### Immediate
1. **Scale BigramCount to full 32k names** — Read from `names.txt` instead of the hardcoded `corpus50`. Need `readFiniteFile` and `lines` from the standard library (see Agda gotchas below). Target: match Karpathy's NLL = 2.454 on 32k names. BigramScaled.agda used to do this but was deleted for being too slow to compile — the count-based approach should be fast.

2. **Executable MLP (makemore part 2)** — Fixed context window (e.g. 3 chars), character embeddings into a vector space, one hidden layer with tanh, softmax output. This is Karpathy's Bengio et al. 2003 reimplementation. In our framework it's an n-gram architecture (from `Architectures.agda`) with learned embeddings. Needs gradient-based training, so depends on wiring up AD.

3. **Wire AD into executable training** — `Bigram.agda` currently uses numerical perturbation (perturb each parameter, measure score change). `AD.agda` proves forward-mode dual numbers compute correct derivatives. Connect them: lift the score computation to dual numbers, get exact gradients. This replaces O(n) perturbation passes with one forward pass per parameter.

### Medium-term
4. **Reverse-mode AD** — Forward-mode is O(params) cost per gradient. For MLP (hundreds of params) we need reverse-mode. Conal's insight: reverse-mode AD = use continuations as your representation of linear maps. Adding this to `AD.agda` would:
   - Be practically necessary for anything bigger than bigram
   - Demonstrate the core Conal pattern (representation choice → algorithm)
   - Be the exact analogy: forward-mode/reverse-mode AD ↔ ???/??? in text prediction

5. **Close postulate gaps** — Some postulates are described as "straightforward but tedious":
   - `log-prob-is-score`: needs threading positivity proofs through the score computation
   - `attn-subsumes-rnn`: needs auxiliary lemmas about `enumerate`
   - Closing these would strengthen the formalization without new ideas

### Exploratory (the frontier)
6. **Novel Kleisli morphism representations** — The open question. Known architectures map to known representations:
   - Bigram = last char only
   - n-gram = last n chars
   - RNN = fixed-dimensional state with recurrent update
   - Attention = look at all positions

   What others exist? Possibilities to explore:
   - What does the algebraic structure of score decomposition *force*? The history-shift in the indexed homomorphism means any representation must handle "context accumulation." Are there representations where this accumulation is particularly natural?
   - Sparse/hashed history (bloom-filter-like states)?
   - Hierarchical decomposition (score splits over concatenation — what if the architecture mirrors this recursive split)?
   - Can we characterize which representations are "optimal" in some algebraic sense?

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
