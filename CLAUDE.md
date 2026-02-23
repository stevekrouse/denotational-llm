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

### Precise? Yes.
Two-layer spec:
- `TrueSpec.agda` — the *real* spec: expected log-probability under the true text distribution. This is what we mean.
- `Spec.agda` — the *empirical* spec: log-probability summed over a specific corpus. This is what we compute.

Both are expressed precisely in Agda.

### Satisfiable? Yes, non-trivially.
The true distribution itself is the unique optimal predictor (by Gibbs inequality). Any finite-parameter family can approximate it. This is the right level of satisfiable — not trivially (like a lookup table on the empirical spec), but constructively.

### Adequate? Yes (for TrueSpec).
The Gibbs inequality (postulated) proves the unique maximizer of trueScore is the true distribution itself. You cannot game this spec by memorization — the expectation is over the full distribution, not a specific corpus. A lookup table achieves perfect *empirical* score but poor *true* score on unseen data.

The empirical spec (`Spec.agda`) is still gameable, but it's now explicitly labeled as an *approximation*, connected to the true spec by convergence (law of large numbers, also postulated).

## Honest assessment of what we've built

**What's real:**
- Two-layer specification: true risk (`TrueSpec.agda`) + empirical risk (`Spec.agda`)
- Score decomposition: proven monoid homomorphism from (List Char, ++) to (ℝ, +)
- Kleisli category structure for predictors (`Kleisli.agda`)
- Architecture hierarchy with embeddings (`Architectures.agda`)
- Forward-mode AD via dual numbers + reverse-mode AD via continuations (`AD.agda`)
- The representation pattern: forward/reverse AD are different reps of the same linear map, exactly as bigram/attention are different reps of the same Kleisli morphism
- Executable bigrams: NLL = 2.454 on 32k names, matching Karpathy

**Where we're overclaiming or loose:**
- "Score is a functor" — it's a monoid homomorphism, not a functor in the precise categorical sense. In Conal's AD work, D is genuinely a functor between categories (smooth functions → linear maps). We should be more careful with this language.
- The architecture "analogy" was weaker than the AD analogy, but the rank decomposition tightens it. In AD, forward/reverse are representation theorems (different ways to compute the same linear map) with a dimension argument (forward is O(n), reverse is O(m)). Now: multi-head readout is also a representation theorem (the unique form of linear readout from outer-product state) with a dimension argument (T₂ needs D=27, multi-head needs D ~ sqrt(params)). The analogy is now: both are "representation theorem + dimension argument". What remains weaker: different levels of the hierarchy (bigram vs T₂ vs attention) still compute *different functions*, unlike forward/reverse AD which compute the *same derivative*.
- `gradient-improves` postulates its own punchline via `gradient-ascent-lemma`.

**What we *have* derived (new!):**
- The **tensor algebra result** (TensorBigram.agda): The monoid structure of score decomposition naturally suggests using a concrete algebraic object — a truncated tensor algebra T₂(ℝ^d) — as state. This falls out of asking "what is the simplest algebraic structure that respects the monoid homomorphism property?" The model:
  - Uses T₂(ℝ^d) ≅ ℝ × ℝ^d × sym(d×d) as state (d dimensions: 1 + d + d(d+1)/2 = (d+2)(d+1)/2 elements)
  - Maps each character via embedding and multiplication in the tensor algebra
  - Computes logits via a linear map from state to ℝ^27
  - Uses gradient descent with reverse-mode AD
  - **Beats MLP on all settings (small corpus, 32k names) while using 22% fewer parameters (162 vs 209)**
  - No nonlinearity needed — algebraic structure does the work
  - This is a genuine discovery: not a known architecture, derived directly from the monoid specification
- The **attention-from-algebra result** (`docs/ATTENTION-ALGEBRA.md`, `linear-attention.js`): T₂ and linear attention are both monoid homomorphisms into matrices under addition. T₂ is the special case (key=value=raw embedding). The critical insight: projections must go at READOUT (not accumulation) so gradient flows. "Projected T2" (ProjT2) — same T₂ state but with learned query at readout — achieves NLL=2.267 with only 2,038 params, nearly matching T₂(ℝ¹⁶)'s 2.250 at 4x fewer parameters. This is parameter-efficient linear attention derived from the algebra.
- The **multi-head attention / rank decomposition result** (`linear-attention.js`): A rank decomposition theorem forces the readout structure: any linear map from a D x D matrix to ℝ^27 decomposes as a sum of H rank-1 terms, each of which is exactly a single attention head (query dot state-times-key). Multi-head attention is not an architectural choice — it is the *unique* form of linear readout from the T₂ state. This tightens the AD analogy: just as forward/reverse AD are representation theorems (different ways to compute the same linear map), multi-head attention is a representation theorem (the only way to read out from outer-product state). Empirically: multi-head ProjT2 with H=4 heads and ELU+1 activation achieves NLL=2.250 at 4,102 params, matching T₂(ℝ¹⁶)'s performance at roughly half the parameters. The dimension argument also parallels AD: forward AD is O(n) for n inputs, reverse is O(m) for m outputs; similarly, T₂ needs D = 27 (output-sized state) while multi-head ProjT2 needs D proportional to sqrt(params), achieving the same expressiveness at lower cost.

**What we haven't done that matters:**
- The rank decomposition theorem forces multi-head readout, but the optimal number of heads H and per-head dimension remain open — what determines the right rank?
- The transition from linear attention to softmax attention breaks the monoid homomorphism. Can we characterize what softmax adds algebraically (beyond "non-compositional normalization")?
- Positional encoding: our models are position-blind (monoid = order-insensitive accumulation). How does position information enter the algebraic picture?
- The executable tensor modules use Float and aren't formally connected to the proof modules (but the algebra is sound)
- The Gibbs inequality and convergence are postulated, not proven.

## Goals

Follow Karpathy's [makemore](https://github.com/karpathy/makemore) progression (bigram → MLP → RNN → GPT) as a concrete target task, but derive it from specification using Conal's methodology rather than building it up from neural network primitives. The dream: the algebra reveals something about text prediction that we didn't already know — a novel architecture, optimization strategy, or insight that *falls out* of the spec the way reverse-mode AD fell out of the algebra of linear maps.

## Next steps

### 1. Scale MLPREV training -- DONE
Reverse-mode AD is now integrated into MLP training (`MLPREV.agda`). Key finding: closure-based reverse-mode (using lazy thunks to accumulate gradients) blew up in Agda and Haskell due to memory buildup from thunks. Switched to explicit backprop (manually threading gradients backward through the network) — works cleanly and efficiently. Next: scale training (more steps, full 32k names, larger hidden layer) to push performance closer to Karpathy's targets.

### 2. Reverse-mode AD -- DONE
Reverse-mode AD is implemented in both proof (`AD.agda` Part 2) and executable (`ReverseAD.agda`) forms. The proof module defines `Rev` (single-variable backpropagator) and `RevN` (multi-input backpropagator), with value preservation proofs and the `rev-gradient-improves` theorem. The executable module demonstrates the speedup on bigram gradients. Results match forward-mode exactly.

### 3. Close postulate gaps
- `log-prob-is-score`: threading positivity proofs (tedious but straightforward)
- `attn-subsumes-rnn`: auxiliary lemmas about `enumerate`
- `gradient-ascent-lemma`: the big one — this postulates the punchline. Can we at least narrow what's assumed?
- `gibbs`, `empirical-convergence`: the new TrueSpec postulates. Proving Gibbs for finite alphabets is feasible; convergence requires measure theory.

### 4. Derive something new (the frontier) -- PARTIALLY COMPLETE
**The tensor algebra result is a genuine derivation.** By asking "what algebraic structure respects the monoid homomorphism property of score?", we derived T₂(ℝ^d) as a natural candidate, and it outperforms standard architectures. This is the kind of insight Conal's methodology is meant to produce — the algebra *suggested* a solution we hadn't tried.

**The attention-from-algebra result (`docs/ATTENTION-ALGEBRA.md`).** We showed that T₂ and linear attention are both monoid homomorphisms into the same algebraic structure (matrices under addition). T₂ is the special case where key=value=raw embedding; linear attention learns separate projections. This gives a 4-level algebraic hierarchy: bigram → T₂ → linear attention → softmax attention. The transition to softmax breaks the monoid (non-compositional normalization).

Empirically: T₂(ℝ^8) at 2,430 params (NLL=2.279) beats linear attention at matched params (NLL=2.367), but the comparison is confounded — online training can't train the projection matrices (they need BPTT). The theory stands regardless.

**The multi-head / rank decomposition result.** The rank decomposition theorem resolves the readout question: any linear map from D x D state to ℝ^27 *must* decompose as a sum of H rank-1 terms (= H attention heads). Multi-head ProjT2 with H=4 and ELU+1 achieves NLL=2.250 at 4,102 params, matching T₂(ℝ¹⁶) at half the parameters. This partially answers the "train projections" question — the multi-head query approach works with online SGD (no BPTT needed), because queries are at readout, not accumulation.

The real frontier questions:
- **Softmax attention** — the monoid breaks at softmax (non-compositional normalization). Can we characterize what softmax adds algebraically? Is there a compositional approximation?
- **Positional encoding** — monoid accumulation is order-insensitive. How does position information enter the algebraic picture?
- **Optimal rank / head count** — the rank decomposition forces multi-head form, but what determines the right H? Is there an algebraic criterion?
- **Why does tensor algebra work?** Can we prove it's optimal in some algebraic sense?
- **Does it scale?** Tensor algebra grows quickly with dimension. Can we find sparse/structured variants that scale further?

## Module structure

**Foundation:** `Real.agda` → `Probability.agda` → `Spec.agda`

**Specification:**
- `TrueSpec.agda` — true spec (distribution-based), Gibbs inequality, empirical bridge

**Theory (all build on Spec):**
- `Properties.agda` — monotonicity, convex combinations, Jensen
- `Kleisli.agda` — categorical structure, score as indexed monoid homomorphism
- `Architectures.agda` — bigram ⊂ n-gram ⊂ RNN ⊂ attention ⊂ full predictor
- `AD.agda` — forward-mode AD via dual numbers + reverse-mode AD via continuations
- `Parameterize.agda` — parameter search, gradient ascent validity

**Executable (use Float, not postulated ℝ):**
- `Bigram.agda` — small gradient-descent bigram (10 names)
- `BigramCount.agda` — count-based MLE bigram (32k names, NLL = 2.454)
- `BigramAD.agda` — bigram with forward-mode AD training (10 names)
- `MLP.agda` — MLP with context window, embeddings, hidden layer (small corpus, 209 params)
- `ReverseAD.agda` — bigram with reverse-mode AD training (1 pass for full gradient vs 729)
- `MLPREV.agda` — MLP with explicit backprop reverse-mode AD (closure-based approach failed; explicit works)
- `TensorBigram.agda` — **NEW: Truncated tensor algebra T₂(ℝ^d) state monoid with linear logits (162 params, beats MLP on all settings)**
- `TensorSmall.agda` — Tensor algebra model on small corpus (debugging, ablations)
- `MLPBig.agda` — MLP with larger hidden layer (baseline comparison for tensor results)
- `GroupSSM.agda` — **NEW: S₃ group algebra SSM (null result: non-abelian structure doesn't help at small scale)**
- `linear-attention.js` — T₂, linear attention, ProjT2, and multi-head ProjT2 experiments (JavaScript, runs with Node.js, contains the rank decomposition / multi-head results)

## Workflow conventions

### ⚠️ WATCHDOG TIMER — SET UP IMMEDIATELY ON EVERY SESSION ⚠️

On the VERY FIRST response of every session, launch a background watchdog: `sleep 300` (5 min). When it completes, you get a turn — check all agents, report status, launch a new watchdog. This ensures you never go >5 minutes without checking on things even if the user is away. **This is the first thing you do. Before any other work.**

### Top-level agent = coordinator

The top-level Claude agent should NOT do heavy work (writing Agda, long compilations, research). Instead it:
- Dispatches work to subagents immediately — never wait for user direction
- Always has subagents running, pushing toward the top-level goal
- Monitors progress and stops agents that go off-track
- Reports status to the human
- Stays free to respond to the human at any time
- When a subagent finishes, immediately launch the next piece of work

**Check agents on EVERY response.** Every time the top-level agent responds to the user (for any reason), it MUST also check on all running subagents — read their latest output, assess if they're stuck or off-track, kill and restart if needed. This is non-negotiable. Include a brief status line for each running agent in every response.

**Delegate heavy work.** The top-level agent should NEVER run long compilations, write large files, or do multi-step research. Always use the Task tool with subagents. The top-level agent coordinates, monitors, reports, and course-corrects.

**Fast feedback loops.** Default to low timeouts everywhere:
- Agda type-check (`agda Foo.agda`): 30s timeout. If it fails, fix the error, don't wait longer.
- Agda compile (`agda --compile Foo.agda`): 60s timeout. If it times out, try type-checking first (faster), then compile.
- Running executables: 30s timeout. If slow, test on smaller input first.
- Never set 600s timeouts. If something takes >60s, break it into smaller steps or find a workaround.

**Avoid process overload.** Never run more than 2-3 subagents simultaneously. Background bash commands and subagents all consume system resources. If the system becomes unresponsive (fork failures, exit code 1 on basic commands), stop all background tasks and wait before retrying.

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
agda Spec.agda && agda TrueSpec.agda && agda Properties.agda && \
agda Kleisli.agda && agda Architectures.agda && agda AD.agda && \
agda Parameterize.agda
```

Compilation of `BigramCount.agda` reads `names.txt` at runtime and takes ~30s to compile.
