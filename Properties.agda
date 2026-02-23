-- ════════════════════════════════════════════════════════════
-- STRUCTURAL PROPERTIES OF TEXT PREDICTION
-- ════════════════════════════════════════════════════════════
--
-- Theorems about the algebraic structure of predictors and scores:
--   - Score monotonicity (pointwise ≥ → global ≥)
--   - Score of empty/single corpus
--   - Convex combinations of predictors
--   - Jensen's inequality for predictor mixtures

module Properties where

open import Real
open import Spec
open import Data.List.Base using (List; []; _∷_)
  renaming (_++_ to _L++_)
open import Data.Char.Base using (Char)
open import Data.Product using (_×_; _,_)
open import Data.Unit using (⊤; tt)
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans; cong; cong₂; subst)

-- ═══ SCORE OF EMPTY CORPUS ═══

score-[] : ∀ (p : Predictor) → score p [] ≡ 0ʳ
score-[] p = refl

-- ═══ SCORE OF SINGLE CHARACTER ═══

score-one : ∀ (p : Predictor) (c : Char)
  → score p (c ∷ []) ≡ logʳ (p [] c) +ʳ 0ʳ
score-one p c = refl

-- ═══ SCORE MONOTONICITY ═══
-- If g assigns higher log-probability than f at every (history, char),
-- then g scores at least as well as f on any corpus.

scoreFrom-mono : ∀ (f g : Predictor) (h : List Char) (cs : List Char)
  → (∀ (h' : List Char) (c : Char) → logʳ (f h' c) ≤ʳ logʳ (g h' c))
  → scoreFrom f h cs ≤ʳ scoreFrom g h cs
scoreFrom-mono f g h [] _ = ≤ʳ-refl
scoreFrom-mono f g h (c ∷ cs) dom =
  ≤ʳ-+-mono (dom h c) (scoreFrom-mono f g (h L++ (c ∷ [])) cs dom)

score-mono : ∀ (f g : Predictor) (corpus : List Char)
  → (∀ (h : List Char) (c : Char) → logʳ (f h c) ≤ʳ logʳ (g h c))
  → score f corpus ≤ʳ score g corpus
score-mono f g corpus dom = scoreFrom-mono f g [] corpus dom

-- ═══ COROLLARY: Pointwise dominance → IsAtLeastAsGoodAs ═══

pointwise→atLeast : ∀ (f g : Predictor) (corpus : List Char)
  → (∀ (h : List Char) (c : Char) → logʳ (f h c) ≤ʳ logʳ (g h c))
  → g IsAtLeastAsGoodAs f On corpus
pointwise→atLeast f g corpus dom = score-mono f g corpus dom

-- ═══ CONVEX COMBINATIONS ═══

-- Mix two predictors: λ·p + (1-λ)·q
mix : ℝ → Predictor → Predictor → Predictor
mix λ' p q = λ h c → (λ' *ʳ p h c) +ʳ ((1ʳ -ʳ λ') *ʳ q h c)

-- ═══ JENSEN'S INEQUALITY FOR LOG ═══
-- log(λ·a + (1-λ)·b) ≥ λ·log(a) + (1-λ)·log(b)
-- (log is concave, so Jensen gives this direction)
-- Postulated — proving requires calculus.

postulate
  jensen-log : ∀ (λ' a b : ℝ)
    → 0ʳ ≤ʳ λ' → λ' ≤ʳ 1ʳ → 0ʳ <ʳ a → 0ʳ <ʳ b
    → (λ' *ʳ logʳ a) +ʳ ((1ʳ -ʳ λ') *ʳ logʳ b)
      ≤ʳ logʳ ((λ' *ʳ a) +ʳ ((1ʳ -ʳ λ') *ʳ b))

-- ═══ MIXTURE SCORE BOUND ═══
-- The score of a mixture is at least the weighted average of
-- the individual scores. This means mixing a better predictor
-- with a worse one never produces something worse than the worse.
-- Postulated — full proof would apply Jensen at each corpus position.

postulate
  score-mix-bound : ∀ (λ' : ℝ) (p q : Predictor) (corpus : List Char)
    → 0ʳ ≤ʳ λ' → λ' ≤ʳ 1ʳ
    → (∀ (h : List Char) (c : Char) → 0ʳ <ʳ p h c)
    → (∀ (h : List Char) (c : Char) → 0ʳ <ʳ q h c)
    → (λ' *ʳ score p corpus) +ʳ ((1ʳ -ʳ λ') *ʳ score q corpus)
      ≤ʳ score (mix λ' p q) corpus

-- ═══ COROLLARY: Mixing with a better predictor improves things ═══
-- If score(p) ≥ score(q), then score(mix λ p q) ≥ score(q).
-- Proof: By score-mix-bound,
--   score(mix λ p q) ≥ λ·score(p) + (1-λ)·score(q)
--                    ≥ λ·score(q) + (1-λ)·score(q)  (since score(p) ≥ score(q))
--                    = score(q)

-- We need: λ·a + (1-λ)·b ≥ b when a ≥ b (weighted average ≥ minimum)
postulate
  weighted-avg-≥-min : ∀ (λ' a b : ℝ)
    → 0ʳ ≤ʳ λ' → λ' ≤ʳ 1ʳ → b ≤ʳ a
    → b ≤ʳ (λ' *ʳ a) +ʳ ((1ʳ -ʳ λ') *ʳ b)

mix-with-better : ∀ (λ' : ℝ) (p q : Predictor) (corpus : List Char)
  → 0ʳ ≤ʳ λ' → λ' ≤ʳ 1ʳ
  → (∀ (h : List Char) (c : Char) → 0ʳ <ʳ p h c)
  → (∀ (h : List Char) (c : Char) → 0ʳ <ʳ q h c)
  → p IsAtLeastAsGoodAs q On corpus
  → (mix λ' p q) IsAtLeastAsGoodAs q On corpus
mix-with-better λ' p q corpus 0≤λ λ≤1 p>0 q>0 p≥q =
  ≤ʳ-trans
    (weighted-avg-≥-min λ' (score p corpus) (score q corpus) 0≤λ λ≤1 p≥q)
    (score-mix-bound λ' p q corpus 0≤λ λ≤1 p>0 q>0)

-- ═══ INFORMATION-THEORETIC BOUND ═══
-- The optimal predictor (the true distribution) achieves the
-- maximum score (minimum cross-entropy = entropy).
-- Any other predictor scores strictly worse.
-- This is the Gibbs inequality.
-- We state it as a type, connecting our spec to information theory.

-- The cross-entropy of predictor p with respect to true distribution q:
-- H(q, p) = -Σ q(c) log p(c)
-- Our score is related: score ≈ -H(true, p) when corpus ~ true distribution.

-- Gibbs inequality: H(q, p) ≥ H(q, q), with equality iff p = q.
-- Equivalently: score(true) ≥ score(p) for any p.
-- We don't formalize this fully (it requires a notion of "true distribution")
-- but note that it means our spec has a well-defined optimum.
