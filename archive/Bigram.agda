{-# OPTIONS --guardedness #-}
-- ════════════════════════════════════════════════════════════
-- BIGRAM: CONCRETE EXECUTABLE INSTANCE
-- ════════════════════════════════════════════════════════════
--
-- This module provides:
--   1. A concrete bigram predictor family (à la Karpathy's makemore)
--   2. Numerical gradient descent (executable with Float)
--   3. A main function that trains and generates
--
-- This is the "implementation phase" — it uses Float for execution
-- but follows the same structure as the proven spec.
-- The spec modules (Real.agda, Spec.agda, Parameterize.agda)
-- prove the correctness properties using postulated ℝ.

module Bigram where

open import IO
open import Data.String.Base as String using (String; toList; fromList; _++_)
open import Data.List.Base using (List; []; _∷_; length; map; take; drop; zipWith)
  renaming (_++_ to _L++_)
open import Data.Char.Base using (Char; toℕ; fromℕ)
open import Data.Float.Base as Float using (Float; log; show; e^_; _<ᵇ_)
open import Data.Nat.Base using (ℕ; suc; zero; _≡ᵇ_; _≤ᵇ_; _∸_; _*_; _+_)
open import Data.Nat.Show as ℕShow using ()
open import Data.Bool.Base using (Bool; true; false; if_then_else_)

-- ═══ CONCRETE SCORE (mirrors Spec.scoreFrom with Float) ═══

scoreFrom : (List Char → Char → Float) → List Char → List Char → Float
scoreFrom p history []       = 0.0
scoreFrom p history (c ∷ cs) =
  log (p history c) Float.+ scoreFrom p (history L++ (c ∷ [])) cs

score : (List Char → Char → Float) → List Char → Float
score p corpus = scoreFrom p [] corpus

avgScore : (List Char → Char → Float) → List Char → Float
avgScore p corpus = score p corpus Float.÷ Float.fromℕ (length corpus)

-- ═══ BIGRAM PREDICTOR FAMILY ═══
-- (Params → Predictor) — the Family from Parameterize.agda

alphaSize : ℕ
alphaSize = 27

charToIdx : Char → ℕ
charToIdx c =
  let n = toℕ c in
  if n ≡ᵇ 46 then 0
  else if 97 ≤ᵇ n then (n ∸ 96)
  else 0

idxToChar : ℕ → Char
idxToChar zero    = '.'
idxToChar (suc n) = fromℕ (96 + suc n)

-- Softmax
sumFloats : List Float → Float
sumFloats []       = 0.0
sumFloats (x ∷ xs) = x Float.+ sumFloats xs

softmax : List Float → List Float
softmax logits =
  let exps  = map (e^_) logits
      total = sumFloats exps
  in map (Float._÷ total) exps

lookupF : List Float → ℕ → Float → Float
lookupF []       _       def = def
lookupF (x ∷ _)  zero    _   = x
lookupF (_ ∷ xs) (suc n) def = lookupF xs n def

-- The family: params → predictor
bigramPredictor : List Float → List Char → Char → Float
bigramPredictor θ history c =
  let prevIdx = lastCharIdx history
      row     = take alphaSize (drop (prevIdx * alphaSize) θ)
      probs   = softmax row
  in lookupF probs (charToIdx c) (1.0 Float.÷ Float.fromℕ alphaSize)
  where
    lastCharIdx : List Char → ℕ
    lastCharIdx []       = 0
    lastCharIdx (x ∷ []) = charToIdx x
    lastCharIdx (_ ∷ xs) = lastCharIdx xs

-- ═══ GRADIENT DESCENT (mirrors Parameterize.gradStep with Float) ═══

ε : Float
ε = 0.001

learningRate : Float
learningRate = 0.1

initParams : ℕ → List Float
initParams zero    = []
initParams (suc n) = 0.0 ∷ initParams n

replaceAt : ℕ → Float → List Float → List Float
replaceAt _       _ []       = []
replaceAt zero    v (_ ∷ xs) = v ∷ xs
replaceAt (suc n) v (x ∷ xs) = x ∷ replaceAt n v xs

perturbAt : ℕ → List Float → List Float
perturbAt i θ = replaceAt i (lookupF θ i 0.0 Float.+ ε) θ

-- Gradient: for each param, perturb and measure score change
gradient : List Float → List Char → List Float
gradient θ corpus = go 0 θ
  where
    baseScore = score (bigramPredictor θ) corpus
    go : ℕ → List Float → List Float
    go _ []       = []
    go i (_ ∷ xs) =
      let gi = (score (bigramPredictor (perturbAt i θ)) corpus Float.- baseScore) Float.÷ ε
      in gi ∷ go (suc i) xs

-- One gradient step: θ' = θ + η * ∇S(θ)
step : List Float → List Char → List Float
step θ corpus =
  let grad = gradient θ corpus
  in zipWith (λ ti gi → ti Float.+ (learningRate Float.* gi)) θ grad

-- Train for n steps
train : ℕ → List Float → List Char → List Float
train zero    θ _      = θ
train (suc n) θ corpus = train n (step θ corpus) corpus

-- ═══ GENERATION ═══

alpha : List Char
alpha = '.' ∷ map (λ n → fromℕ (97 + n)) (upTo 26)
  where open import Data.List.Base using (upTo)

argmax : (List Char → Char → Float) → List Char → List Char → Char → Float → Char
argmax _ _       []       best _     = best
argmax p history (c ∷ cs) best bestP =
  let prob = p history c in
  if bestP <ᵇ prob
    then argmax p history cs c    prob
    else argmax p history cs best bestP

generateName : (List Char → Char → Float) → ℕ → List Char → List Char
generateName _ zero    _       = []
generateName p (suc n) history =
  let next = argmax p history alpha '.' 0.0
  in if toℕ next ≡ᵇ 46 then []
     else next ∷ generateName p n (history L++ (next ∷ []))

-- ═══ MAIN ═══

showBool : Bool → String
showBool true  = "true"
showBool false = "false"

nameCorpus : List Char
nameCorpus = toList ".emma.olivia.ava.sophia.isabella.mia.charlotte.amelia.harper.evelyn."

main : Main
main = run do
  putStrLn "=== Denotational LLM: Name Generator ==="
  putStrLn "(Derived from spec in Agda — see Real.agda, Spec.agda, Parameterize.agda)"
  putStrLn ""

  -- The spec
  putStrLn "SPEC (proven in Spec.agda):"
  putStrLn "  Predictor = List Char → Char → ℝ"
  putStrLn "  score p corpus = Σᵢ log P(charᵢ | historyᵢ)"
  putStrLn "  g IsBetterThan f ⟺ score f < score g"
  putStrLn ""

  putStrLn "THEOREMS PROVEN:"
  putStrLn "  1. IsBetterThan is a preorder (Spec.agda)"
  putStrLn "  2. Score Decomposition: score splits over corpus (Spec.agda)"
  putStrLn "  3. Parameterized Improvement: S(θ') > S(θ) → f(θ') better (Parameterize.agda)"
  putStrLn "  4. Gradient Ascent Validity: ∇-step improves predictor (Parameterize.agda)"
  putStrLn ""

  -- Concrete execution
  putStrLn ("Corpus: " String.++ fromList nameCorpus)
  putStrLn ("  (" String.++ ℕShow.show (length nameCorpus) String.++ " chars)")
  putStrLn ""

  let uniform = λ (_ : List Char) (_ : Char) → 1.0 Float.÷ 27.0
  putStrLn ("Baseline (uniform): " String.++ Float.show (avgScore uniform nameCorpus) String.++ " avg log-prob/char")
  putStrLn ""

  let nP = alphaSize * alphaSize
  let θ₀ = initParams nP
  putStrLn "Training bigram (729 params, numerical gradient)..."
  let θ₁₀ = train 10 θ₀ nameCorpus
  putStrLn ("  step 10: " String.++ Float.show (avgScore (bigramPredictor θ₁₀) nameCorpus) String.++ " avg/char")
  let θ₃₀ = train 20 θ₁₀ nameCorpus
  putStrLn ("  step 30: " String.++ Float.show (avgScore (bigramPredictor θ₃₀) nameCorpus) String.++ " avg/char")
  let θ₅₀ = train 20 θ₃₀ nameCorpus
  putStrLn ("  step 50: " String.++ Float.show (avgScore (bigramPredictor θ₅₀) nameCorpus) String.++ " avg/char")
  putStrLn ""

  let p = bigramPredictor θ₅₀
  putStrLn "Learned P(next | prev):"
  putStrLn ("  after '.': a=" String.++ Float.show (p (toList ".") 'a')
    String.++ "  e=" String.++ Float.show (p (toList ".") 'e')
    String.++ "  m=" String.++ Float.show (p (toList ".") 'm'))
  putStrLn ""

  putStrLn "Generated names (greedy decoding):"
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".")))
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".e")))
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".s")))
  putStrLn ""

  putStrLn ("Trained beats uniform? " String.++ showBool (score uniform nameCorpus <ᵇ score (bigramPredictor θ₅₀) nameCorpus))
