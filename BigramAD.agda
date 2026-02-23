{-# OPTIONS --guardedness #-}
-- ════════════════════════════════════════════════════════════
-- BIGRAM WITH FORWARD-MODE AD (DUAL NUMBERS)
-- ════════════════════════════════════════════════════════════
--
-- This module replaces the numerical perturbation gradient
-- from Bigram.agda with exact forward-mode AD using dual numbers.
--
-- Key differences from Bigram.agda:
--   - DualF record: Float-based dual numbers (val, der)
--   - Lifted operations: +ᴅ, *ᴅ, ÷ᴅ, logᴅ, expᴅ
--   - softmaxᴅ: softmax on List DualF → List DualF
--   - Scoring produces DualF: val = score, der = ∂score/∂θᵢ
--   - For each param i, seed θᵢ as variable (der=1), run score,
--     read off der → exact ∂S/∂θᵢ in one forward pass.
--
-- Same asymptotic cost as perturbation (729 forward passes) but
-- exact gradients instead of O(ε) approximations.
--
-- This is the executable counterpart of AD.agda, which proves
-- the correctness of dual-number AD using postulated ℝ.

module BigramAD where

open import IO
open import Data.String.Base as String using (String; toList; fromList; _++_)
open import Data.List.Base using (List; []; _∷_; length; map; take; drop; zipWith)
  renaming (_++_ to _L++_)
open import Data.Char.Base using (Char; toℕ; fromℕ)
open import Data.Float.Base as Float using (Float; log; show; e^_; _<ᵇ_)
open import Data.Nat.Base using (ℕ; suc; zero; _≡ᵇ_; _≤ᵇ_; _∸_; _*_; _+_)
open import Data.Nat.Show as ℕShow using ()
open import Data.Bool.Base using (Bool; true; false; if_then_else_)

-- ═══ FLOAT-BASED DUAL NUMBERS ═══
-- Mirrors AD.agda's Dual but uses Float for execution.
-- val = primal value, der = tangent (derivative w.r.t. seed).

record DualF : Set where
  constructor dualF
  field
    val : Float
    der : Float

open DualF public

-- Embed a constant (derivative is 0)
constᴅ : Float → DualF
constᴅ x = dualF x 0.0

-- Embed the seed variable (derivative is 1)
varᴅ : Float → DualF
varᴅ x = dualF x 1.0

-- ═══ LIFTED ARITHMETIC ═══
-- Each operation follows the same derivative rules as AD.agda.

-- Addition: d/dx(a + b) = a' + b'
_+ᴅ_ : DualF → DualF → DualF
(dualF a a') +ᴅ (dualF b b') = dualF (a Float.+ b) (a' Float.+ b')
infixl 6 _+ᴅ_

-- Subtraction: d/dx(a - b) = a' - b'
_-ᴅ_ : DualF → DualF → DualF
(dualF a a') -ᴅ (dualF b b') = dualF (a Float.- b) (a' Float.- b')
infixl 6 _-ᴅ_

-- Multiplication (product rule): d/dx(a * b) = a'*b + a*b'
_*ᴅ_ : DualF → DualF → DualF
(dualF a a') *ᴅ (dualF b b') = dualF (a Float.* b) ((a' Float.* b) Float.+ (a Float.* b'))
infixl 7 _*ᴅ_

-- Division (quotient rule): d/dx(a / b) = (a'*b - a*b') / (b*b)
_÷ᴅ_ : DualF → DualF → DualF
(dualF a a') ÷ᴅ (dualF b b') =
  dualF (a Float.÷ b) (((a' Float.* b) Float.- (a Float.* b')) Float.÷ (b Float.* b))
infixl 7 _÷ᴅ_

-- Logarithm (chain rule): d/dx(log a) = a' / a
logᴅ : DualF → DualF
logᴅ (dualF a a') = dualF (log a) (a' Float.÷ a)

-- Exponential (chain rule): d/dx(exp a) = a' * exp(a)
expᴅ : DualF → DualF
expᴅ (dualF a a') = dualF (e^ a) (a' Float.* (e^ a))

-- ═══ DUAL CONSTANTS ═══

0ᴅ : DualF
0ᴅ = constᴅ 0.0

-- ═══ SUMMATION OVER DUALS ═══

sumᴅ : List DualF → DualF
sumᴅ []       = 0ᴅ
sumᴅ (x ∷ xs) = x +ᴅ sumᴅ xs

-- ═══ ALPHABET ═══

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

-- ═══ DUAL SOFTMAX ═══
-- softmax on a list of DualF values.
-- exp each, divide each by sum of exps — all with dual arithmetic.

softmaxᴅ : List DualF → List DualF
softmaxᴅ logits = map (λ ex → ex ÷ᴅ total) exps
  where
    exps  = map expᴅ logits
    total = sumᴅ exps

-- ═══ DUAL LOOKUP ═══

lookupᴅ : List DualF → ℕ → DualF → DualF
lookupᴅ []       _       def = def
lookupᴅ (x ∷ _)  zero    _   = x
lookupᴅ (_ ∷ xs) (suc n) def = lookupᴅ xs n def

-- ═══ BIGRAM PREDICTOR FAMILY (DUAL) ═══
-- Takes List DualF params and returns DualF probabilities.
-- This is the same structure as Bigram.agda's bigramPredictor
-- but operating in dual-number arithmetic.

bigramPredictorᴅ : List DualF → List Char → Char → DualF
bigramPredictorᴅ θ history c =
  let prevIdx = lastCharIdx history
      row     = take alphaSize (drop (prevIdx * alphaSize) θ)
      probs   = softmaxᴅ row
  in lookupᴅ probs (charToIdx c) (constᴅ (1.0 Float.÷ Float.fromℕ alphaSize))
  where
    lastCharIdx : List Char → ℕ
    lastCharIdx []       = 0
    lastCharIdx (x ∷ []) = charToIdx x
    lastCharIdx (_ ∷ xs) = lastCharIdx xs

-- ═══ DUAL SCORE ═══
-- scoreFrom but in dual arithmetic — val gives the score,
-- der gives ∂score/∂θᵢ (for whichever param was seeded).

scoreFromᴅ : (List Char → Char → DualF) → List Char → List Char → DualF
scoreFromᴅ p history []       = 0ᴅ
scoreFromᴅ p history (c ∷ cs) =
  logᴅ (p history c) +ᴅ scoreFromᴅ p (history L++ (c ∷ [])) cs

scoreᴅ : (List Char → Char → DualF) → List Char → DualF
scoreᴅ p corpus = scoreFromᴅ p [] corpus

-- ═══ SEEDING PARAMETERS ═══
-- To compute ∂S/∂θᵢ: seed θᵢ with der=1, all others with der=0,
-- then evaluate score in dual arithmetic and read off der.

seedParamsᴅ : List Float → ℕ → List DualF
seedParamsᴅ []       _       = []
seedParamsᴅ (θ ∷ θs) zero    = varᴅ θ ∷ map constᴅ θs
seedParamsᴅ (θ ∷ θs) (suc i) = constᴅ θ ∷ seedParamsᴅ θs i

-- ═══ GRADIENT COMPUTATION ═══
-- For each parameter i: seed i, run dual-number score, read off der.

gradient : List Float → List Char → List Float
gradient θ corpus = go 0 θ
  where
    go : ℕ → List Float → List Float
    go _ []       = []
    go i (_ ∷ xs) =
      let seeded = seedParamsᴅ θ i
          dualScore = scoreᴅ (bigramPredictorᴅ seeded) corpus
      in der dualScore ∷ go (suc i) xs

-- ═══ GRADIENT ASCENT ═══

learningRate : Float
learningRate = 0.1

initParams : ℕ → List Float
initParams zero    = []
initParams (suc n) = 0.0 ∷ initParams n

-- One gradient step: θ' = θ + η * ∇S(θ)
step : List Float → List Char → List Float
step θ corpus =
  let grad = gradient θ corpus
  in zipWith (λ ti gi → ti Float.+ (learningRate Float.* gi)) θ grad

-- Train for n steps
train : ℕ → List Float → List Char → List Float
train zero    θ _      = θ
train (suc n) θ corpus = train n (step θ corpus) corpus

-- ═══ PLAIN-FLOAT SCORING (for reporting) ═══

lookupF : List Float → ℕ → Float → Float
lookupF []       _       def = def
lookupF (x ∷ _)  zero    _   = x
lookupF (_ ∷ xs) (suc n) def = lookupF xs n def

sumFloats : List Float → Float
sumFloats []       = 0.0
sumFloats (x ∷ xs) = x Float.+ sumFloats xs

softmax : List Float → List Float
softmax logits =
  let exps  = map (e^_) logits
      total = sumFloats exps
  in map (Float._÷ total) exps

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

scoreFrom : (List Char → Char → Float) → List Char → List Char → Float
scoreFrom p history []       = 0.0
scoreFrom p history (c ∷ cs) =
  log (p history c) Float.+ scoreFrom p (history L++ (c ∷ [])) cs

score : (List Char → Char → Float) → List Char → Float
score p corpus = scoreFrom p [] corpus

avgScore : (List Char → Char → Float) → List Char → Float
avgScore p corpus = score p corpus Float.÷ Float.fromℕ (length corpus)

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
  putStrLn "=== Denotational LLM: Bigram with Forward-Mode AD ==="
  putStrLn "(Exact gradients via dual numbers — see AD.agda for proofs)"
  putStrLn ""

  putStrLn "METHOD:"
  putStrLn "  Bigram.agda:   numerical perturbation (approximate, O(ε) error)"
  putStrLn "  BigramAD.agda: forward-mode AD / dual numbers (exact gradients)"
  putStrLn "  Same 729 forward passes, but each gives EXACT ∂S/∂θᵢ."
  putStrLn ""

  putStrLn ("Corpus: " String.++ fromList nameCorpus)
  putStrLn ("  (" String.++ ℕShow.show (length nameCorpus) String.++ " chars)")
  putStrLn ""

  let uniform = λ (_ : List Char) (_ : Char) → 1.0 Float.÷ 27.0
  putStrLn ("Baseline (uniform): " String.++ Float.show (avgScore uniform nameCorpus) String.++ " avg log-prob/char")
  putStrLn ""

  let nP = alphaSize * alphaSize
  let θ₀ = initParams nP
  putStrLn "Training bigram (729 params, AD gradient)..."
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
