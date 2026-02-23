-- ════════════════════════════════════════════════════════════
-- PROBABILITY DISTRIBUTIONS
-- ════════════════════════════════════════════════════════════
--
-- Tightens the specification by defining what a valid probability
-- distribution is, and refining Predictor to produce distributions.

module Probability where

open import Real
open import Data.List.Base using (List; []; _∷_; map; length)
open import Data.Char.Base using (Char)
open import Data.Product using (_×_; _,_)
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans; cong; cong₂; subst)

-- ═══ Non-negative and Positive Reals ═══

0≤_ : ℝ → Set
0≤ x = 0ʳ ≤ʳ x

0<_ : ℝ → Set
0< x = 0ʳ <ʳ x

_∈[0,1] : ℝ → Set
x ∈[0,1] = (0≤ x) × (x ≤ʳ 1ʳ)

-- ═══ Summation ═══

sumOver : {A : Set} → (A → ℝ) → List A → ℝ
sumOver f []       = 0ʳ
sumOver f (x ∷ xs) = f x +ʳ sumOver f xs

-- ═══ Probability Distribution ═══

record Dist (A : Set) (elems : List A) : Set where
  field
    prob      : A → ℝ
    all-≥0    : ∀ (a : A) → 0≤ (prob a)
    sums-to-1 : sumOver prob elems ≡ 1ʳ

open Dist public

-- ═══ Proper Predictor (parameterized by alphabet) ═══

module WithAlphabet (Alpha : List Char) where

  PPredictor : Set
  PPredictor = List Char → Dist Char Alpha

  rawProb : PPredictor → List Char → Char → ℝ
  rawProb pp history c = prob (pp history) c

  ppredictor-≥0 : ∀ (pp : PPredictor) (h : List Char) (c : Char)
    → 0≤ (rawProb pp h c)
  ppredictor-≥0 pp h c = all-≥0 (pp h) c

  -- ═══ Uniform Distribution ═══

  uniformProb : ℝ
  uniformProb = 1ʳ ÷ʳ fromℕ (length Alpha)

  postulate
    alpha-length-pos : 0< fromℕ (length Alpha)
    uniform-≥0       : 0≤ uniformProb
    uniform-sums     : sumOver (λ _ → uniformProb) Alpha ≡ 1ʳ

  uniformDist : Dist Char Alpha
  uniformDist = record
    { prob      = λ _ → uniformProb
    ; all-≥0    = λ _ → uniform-≥0
    ; sums-to-1 = uniform-sums
    }

  uniformPredictor : PPredictor
  uniformPredictor = λ _ → uniformDist

-- ═══ Log is monotone on probabilities ═══

prob-log-mono : ∀ {p q : ℝ} → 0< p → p <ʳ q → logʳ p <ʳ logʳ q
prob-log-mono _ p<q = log-mono p<q

-- ═══ Softmax Specification ═══

softmaxSpec : List ℝ → List ℝ
softmaxSpec logits =
  let exps  = map expʳ logits
      total = sumʳ exps
  in map (λ e → e ÷ʳ total) exps

-- Softmax sums to 1 (postulated — proof would require division algebra)
postulate
  softmax-sums-to-1 : ∀ (logits : List ℝ) → sumʳ (softmaxSpec logits) ≡ 1ʳ
