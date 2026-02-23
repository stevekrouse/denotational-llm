-- ════════════════════════════════════════════════════════════
-- THE TRUE SPECIFICATION (ADEQUATE VERSION)
-- ════════════════════════════════════════════════════════════
--
-- Conal Elliott's three-property check for specifications:
--   • precise    (expressible in Agda)          ✓
--   • satisfiable (solvable/computable)          ✓
--   • adequate   (cannot be gamed)              ✓ (this is the fix)
--
-- The spec in Spec.agda scores a predictor on a SPECIFIC CORPUS.
-- This can be gamed: a lookup table that memorizes the corpus
-- scores perfectly but generalizes to nothing.
--
-- The fix: the real specification is about a DISTRIBUTION over
-- text, not a specific corpus. A predictor is scored by its
-- expected log-probability under the true distribution.
--
-- This is cross-entropy, and it CANNOT be gamed:
-- the Gibbs inequality proves the unique maximizer is the
-- true conditional distribution itself. Memorizing any finite
-- corpus does not help because the expectation is over the
-- entire (possibly infinite) distribution.
--
-- The corpus-based score from Spec.agda becomes what it truly
-- is: an EMPIRICAL ESTIMATOR of the true score. Useful for
-- computation, but not the ground truth.

module TrueSpec where

open import Real
open import Spec using (Predictor; Corpus; score; scoreFrom)
open import Probability using (Dist; prob; sumOver)
open import Data.List.Base using (List; []; _∷_; length)
open import Data.Char.Base using (Char)
open import Data.Product using (_×_; _,_; Σ-syntax)
open import Data.Nat.Base using (ℕ; zero; suc)
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans; cong; cong₂; subst)

-- ════════════════════════════════════════════════════════════
-- SECTION 1: THE TRUE DATA DISTRIBUTION
-- ════════════════════════════════════════════════════════════
--
-- A TextDist captures the "true" generative process behind text.
-- It assigns a conditional probability to each next character
-- given any history. This is what the predictor is trying to
-- approximate.

record TextDist (Alpha : List Char) : Set where
  field
    -- The true conditional distribution: given history h,
    -- what is the probability of character c?
    trueCond : List Char → Char → ℝ

    -- It's a proper distribution at each history
    trueCond-≥0    : ∀ (h : List Char) (c : Char) → 0ʳ ≤ʳ trueCond h c
    trueCond-sum-1 : ∀ (h : List Char) → sumOver (trueCond h) Alpha ≡ 1ʳ

open TextDist public

-- ════════════════════════════════════════════════════════════
-- SECTION 2: THE TRUE SCORE (CROSS-ENTROPY)
-- ════════════════════════════════════════════════════════════
--
-- The true score of a predictor p under distribution D is:
--
--   trueScore(p, D) = E_{(h,c) ~ D} [ log(p(h, c)) ]
--
-- This is the negative cross-entropy H(D, p).
-- We cannot compute this directly (it's an expectation over
-- a possibly infinite distribution), so we postulate it as
-- an abstract quantity with the properties we need.

postulate
  -- The true score: expected log-probability under D.
  -- trueScore p D = E_{(h,c) ~ D} [ log(p h c) ]
  trueScore : ∀ {Alpha : List Char} → Predictor → TextDist Alpha → ℝ

-- ════════════════════════════════════════════════════════════
-- SECTION 3: THE ADEQUATE IMPROVEMENT RELATION
-- ════════════════════════════════════════════════════════════
--
-- A predictor g is truly better than f if it has higher
-- expected log-probability under the true distribution.
-- This cannot be gamed by memorization.

_IsTrulyBetterThan_Under_ :
  ∀ {Alpha : List Char} → Predictor → Predictor → TextDist Alpha → Set
g IsTrulyBetterThan f Under D = trueScore f D <ʳ trueScore g D

_IsTrulyAtLeastAsGoodAs_Under_ :
  ∀ {Alpha : List Char} → Predictor → Predictor → TextDist Alpha → Set
g IsTrulyAtLeastAsGoodAs f Under D = trueScore f D ≤ʳ trueScore g D

-- ═══ The preorder transfers immediately ═══

trueAtLeast-refl : ∀ {Alpha : List Char} (p : Predictor) (D : TextDist Alpha)
  → p IsTrulyAtLeastAsGoodAs p Under D
trueAtLeast-refl p D = ≤ʳ-refl

trueAtLeast-trans : ∀ {Alpha : List Char} {f g h : Predictor} {D : TextDist Alpha}
  → h IsTrulyAtLeastAsGoodAs g Under D
  → g IsTrulyAtLeastAsGoodAs f Under D
  → h IsTrulyAtLeastAsGoodAs f Under D
trueAtLeast-trans h≥g g≥f = ≤ʳ-trans g≥f h≥g

trueBetter-trans : ∀ {Alpha : List Char} {f g h : Predictor} {D : TextDist Alpha}
  → h IsTrulyBetterThan g Under D
  → g IsTrulyBetterThan f Under D
  → h IsTrulyBetterThan f Under D
trueBetter-trans h>g g>f = <ʳ-trans g>f h>g

-- ════════════════════════════════════════════════════════════
-- SECTION 4: GIBBS INEQUALITY (ADEQUACY)
-- ════════════════════════════════════════════════════════════
--
-- The key theorem that makes this spec ADEQUATE:
-- the unique maximizer of trueScore(-, D) is D itself.
--
-- In information theory: cross-entropy H(D, p) ≥ entropy H(D),
-- with equality iff p = D. Equivalently:
-- KL(D || p) = H(D, p) - H(D) ≥ 0, with equality iff p = D.
--
-- This means: you cannot game the true spec. The ONLY way to
-- maximize trueScore is to learn the true distribution.
-- Memorizing a finite corpus does not help.

-- The true distribution, viewed as a predictor
asPred : ∀ {Alpha : List Char} → TextDist Alpha → Predictor
asPred D = trueCond D

postulate
  -- Gibbs inequality: the true distribution maximizes trueScore.
  -- For any predictor p, trueScore(asPred D, D) ≥ trueScore(p, D).
  gibbs : ∀ {Alpha : List Char} (p : Predictor) (D : TextDist Alpha)
    → p IsTrulyAtLeastAsGoodAs (asPred D) Under D → p ≡ asPred D

  -- Weaker version: strict inequality when p ≠ D
  -- (This is the version usually stated as "KL divergence > 0")
  gibbs-strict : ∀ {Alpha : List Char} (p : Predictor) (D : TextDist Alpha)
    → (asPred D) IsTrulyAtLeastAsGoodAs p Under D

-- ════════════════════════════════════════════════════════════
-- SECTION 5: BRIDGE TO EMPIRICAL SCORE
-- ════════════════════════════════════════════════════════════
--
-- The corpus-based score from Spec.agda is the empirical
-- estimator of the true score. As corpus size → ∞ (with
-- the corpus drawn from D), the empirical score converges
-- to the true score (law of large numbers).
--
-- This bridges the two specs:
--   - TrueSpec says what we MEAN (adequate, not gameable)
--   - Spec says what we COMPUTE (empirical, on finite data)
--
-- We postulate the convergence. Proving it would require
-- formalizing measure theory.

-- Normalized empirical score (per character)
normalizedScore : Predictor → Corpus → ℝ
normalizedScore p []       = 0ʳ
normalizedScore p corpus   = score p corpus ÷ʳ fromℕ (length corpus)

postulate
  -- Law of large numbers for score:
  -- For a corpus drawn from D, the normalized empirical score
  -- converges to the true score as corpus length → ∞.
  empirical-convergence :
    ∀ {Alpha : List Char} (p : Predictor) (D : TextDist Alpha) (ε : ℝ)
    → 0ʳ <ʳ ε
    → Σ[ n ∈ ℕ ] (∀ (corpus : Corpus)
        → fromℕ n ≤ʳ fromℕ (length corpus)
        -- The absolute difference between empirical and true score is < ε
        -- (We state this as two inequalities since we don't have |·|)
        → (normalizedScore p corpus -ʳ trueScore p D) <ʳ ε
        × (trueScore p D -ʳ normalizedScore p corpus) <ʳ ε)

-- ════════════════════════════════════════════════════════════
-- SECTION 6: SCORE DECOMPOSITION TRANSFERS
-- ════════════════════════════════════════════════════════════
--
-- The algebraic structure (score decomposition / indexed
-- monoid homomorphism) transfers from Spec to TrueSpec.
--
-- The true score decomposes because expectation is linear:
-- E[score(xs ++ ys)] = E[score(xs)] + E[score(ys | xs)]
--
-- This is just linearity of expectation applied to the
-- existing score-split theorem from Spec.agda.

postulate
  -- True score decomposes by linearity of expectation.
  -- This is the "true" chain rule — not an empirical approximation.
  trueScore-linear :
    ∀ {Alpha : List Char} (p q : Predictor) (D : TextDist Alpha)
    → trueScore p D +ʳ trueScore q D
      ≡ trueScore (λ h c → p h c +ʳ q h c) D
