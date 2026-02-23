-- ════════════════════════════════════════════════════════════
-- THE SPECIFICATION
-- ════════════════════════════════════════════════════════════
--
-- This module defines what text prediction MEANS, independent
-- of any implementation. It provides:
--   - Predictor: the type of text predictors
--   - score: how to evaluate a predictor on a corpus
--   - _IsBetterThan_On_: the improvement relation
--   - Proofs that the improvement relation is a preorder
--   - The Score Decomposition theorem (our "chain rule")

module Spec where

open import Real
open import Data.List.Base using (List; []; _∷_; length; map)
  renaming (_++_ to _L++_)
open import Data.Char.Base using (Char)
open import Data.Nat.Base using (ℕ; zero; suc)
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans; cong; cong₂; subst)
open import Data.List.Properties using (++-identityʳ; ++-assoc)

-- ═══ Core Types ═══

Predictor : Set
Predictor = List Char → Char → ℝ

Corpus : Set
Corpus = List Char

-- ═══ Score ═══
-- Walk left to right. At each position, the predictor sees the history
-- (all prior characters) and is scored on log P(actual next char).

-- scoreFrom p h cs: score the predictor p on corpus cs,
-- given that h is the history seen so far.
scoreFrom : Predictor → List Char → Corpus → ℝ
scoreFrom p history []       = 0ʳ
scoreFrom p history (c ∷ cs) =
  logʳ (p history c) +ʳ scoreFrom p (history L++ (c ∷ [])) cs

-- Top-level score starts with empty history.
score : Predictor → Corpus → ℝ
score p corpus = scoreFrom p [] corpus

-- ═══ The Improvement Relation ═══

_IsBetterThan_On_ : Predictor → Predictor → Corpus → Set
g IsBetterThan f On corpus = score f corpus <ʳ score g corpus

_IsAtLeastAsGoodAs_On_ : Predictor → Predictor → Corpus → Set
g IsAtLeastAsGoodAs f On corpus = score f corpus ≤ʳ score g corpus

-- ═══ Preorder Proofs ═══

atLeastAsGood-refl : ∀ (p : Predictor) (c : Corpus)
  → p IsAtLeastAsGoodAs p On c
atLeastAsGood-refl p c = ≤ʳ-refl

atLeastAsGood-trans : ∀ {f g h : Predictor} {c : Corpus}
  → h IsAtLeastAsGoodAs g On c
  → g IsAtLeastAsGoodAs f On c
  → h IsAtLeastAsGoodAs f On c
atLeastAsGood-trans h≥g g≥f = ≤ʳ-trans g≥f h≥g

isBetter-trans : ∀ {f g h : Predictor} {c : Corpus}
  → h IsBetterThan g On c
  → g IsBetterThan f On c
  → h IsBetterThan f On c
isBetter-trans h>g g>f = <ʳ-trans g>f h>g

isBetter→atLeast : ∀ {f g : Predictor} {c : Corpus}
  → g IsBetterThan f On c
  → g IsAtLeastAsGoodAs f On c
isBetter→atLeast g>f = <ʳ→≤ʳ g>f

-- ═══ Score Decomposition Theorem ═══
-- "Our chain rule": the score decomposes as a sum over positions.
-- More precisely: score on a concatenation splits into the sum of
-- scores on the parts (with appropriate history threading).

-- Theorem: score of a single character
score-single : ∀ (p : Predictor) (h : List Char) (c : Char)
  → scoreFrom p h (c ∷ []) ≡ logʳ (p h c) +ʳ 0ʳ
score-single p h c = cong (logʳ (p h c) +ʳ_) refl

-- Theorem: score decomposes over a split.
-- scoreFrom p h (xs ++ ys) = scoreFrom p h xs + scoreFrom p (h ++ xs) ys
score-split : ∀ (p : Predictor) (h : List Char) (xs ys : Corpus)
  → scoreFrom p h (xs L++ ys)
    ≡ scoreFrom p h xs +ʳ scoreFrom p (h L++ xs) ys
score-split p h [] ys
  rewrite ++-identityʳ h = sym (+ʳ-identityˡ (scoreFrom p h ys))
score-split p h (x ∷ xs) ys =
  let
    -- Key list identity: (h ++ [x]) ++ xs ≡ h ++ (x ∷ xs)
    h++x∷xs≡ : (h L++ (x ∷ [])) L++ xs ≡ h L++ (x ∷ xs)
    h++x∷xs≡ = ++-assoc h (x ∷ []) xs

    -- Inductive hypothesis
    ih : scoreFrom p (h L++ (x ∷ [])) (xs L++ ys)
       ≡ scoreFrom p (h L++ (x ∷ [])) xs
         +ʳ scoreFrom p ((h L++ (x ∷ [])) L++ xs) ys
    ih = score-split p (h L++ (x ∷ [])) xs ys

    -- Transport the history in the last scoreFrom term
    transport : scoreFrom p ((h L++ (x ∷ [])) L++ xs) ys
              ≡ scoreFrom p (h L++ (x ∷ xs)) ys
    transport = cong (λ hist → scoreFrom p hist ys) h++x∷xs≡

    -- Combine: rearrange using +ʳ-assoc
    step1 : logʳ (p h x) +ʳ scoreFrom p (h L++ (x ∷ [])) (xs L++ ys)
          ≡ logʳ (p h x)
            +ʳ (scoreFrom p (h L++ (x ∷ [])) xs
                +ʳ scoreFrom p ((h L++ (x ∷ [])) L++ xs) ys)
    step1 = cong (logʳ (p h x) +ʳ_) ih

    step2 : logʳ (p h x)
            +ʳ (scoreFrom p (h L++ (x ∷ [])) xs
                +ʳ scoreFrom p ((h L++ (x ∷ [])) L++ xs) ys)
          ≡ (logʳ (p h x) +ʳ scoreFrom p (h L++ (x ∷ [])) xs)
            +ʳ scoreFrom p ((h L++ (x ∷ [])) L++ xs) ys
    step2 = sym (+ʳ-assoc (logʳ (p h x))
                          (scoreFrom p (h L++ (x ∷ [])) xs)
                          (scoreFrom p ((h L++ (x ∷ [])) L++ xs) ys))

    step3 : (logʳ (p h x) +ʳ scoreFrom p (h L++ (x ∷ [])) xs)
            +ʳ scoreFrom p ((h L++ (x ∷ [])) L++ xs) ys
          ≡ (logʳ (p h x) +ʳ scoreFrom p (h L++ (x ∷ [])) xs)
            +ʳ scoreFrom p (h L++ (x ∷ xs)) ys
    step3 = cong ((logʳ (p h x) +ʳ scoreFrom p (h L++ (x ∷ [])) xs) +ʳ_)
                 transport
  in trans step1 (trans step2 step3)
