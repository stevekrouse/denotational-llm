{-# OPTIONS --guardedness #-}
-- ════════════════════════════════════════════════════════════
-- REVERSE-MODE AD: EXECUTABLE (Float)
-- ════════════════════════════════════════════════════════════
--
-- This module implements reverse-mode automatic differentiation
-- using Float, mirroring the proof-level Rev and RevN types
-- from AD.agda but made executable.
--
-- THE KEY INSIGHT (Conal Elliott):
--   Reverse-mode AD = represent linear maps as continuations.
--   Instead of (value, tangent) as in forward-mode, we use
--   (value, backpropagator) where the backpropagator maps
--   an output adjoint to a LIST of input adjoints.
--
-- PERFORMANCE:
--   Forward-mode (BigramAD.agda): 729 forward passes for 729 params
--   Reverse-mode (this module):   1 forward + 1 backward pass
--   Speedup: 729x for bigram, 209x for MLP
--
-- STRUCTURE:
--   We define RevNF: a Float value paired with a backpropagator
--   (Float → List Float) that maps an output adjoint to the
--   gradient w.r.t. all tracked inputs. Each arithmetic operation
--   composes backpropagators using the chain rule in reverse.
--
-- This is the executable counterpart of AD.agda's RevN type.
-- The proof module guarantees the algebraic properties hold;
-- this module provides the concrete Float computation.

module ReverseAD where

open import IO
open import Data.String.Base as String using (String; toList; fromList; _++_)
open import Data.List.Base as List using (List; []; _∷_; length; map; take;
  drop; zipWith; replicate)
  renaming (_++_ to _L++_)
open import Data.Char.Base using (Char; toℕ; fromℕ)
open import Data.Float.Base as Float using (Float; log; show; e^_; _<ᵇ_)
open import Data.Nat.Base using (ℕ; suc; zero; _≡ᵇ_; _≤ᵇ_; _∸_; _*_; _+_)
open import Data.Nat.Show as ℕShow using ()
open import Data.Bool.Base using (Bool; true; false; if_then_else_)


-- ═══════════════════════════════════════════════════════════
-- SECTION 1: THE RevNF TYPE
-- ═══════════════════════════════════════════════════════════
--
-- A RevNF value carries:
--   val: the primal (forward) value
--   back: a backpropagator that, given an output adjoint δ,
--         returns the list of adjoints for ALL tracked inputs
--
-- This is the multi-input continuation representation of the
-- derivative. One backward pass computes the entire gradient.

record RevNF : Set where
  constructor revNF
  field
    val  : Float
    back : Float → List Float

open RevNF public


-- ═══════════════════════════════════════════════════════════
-- SECTION 2: ADJOINT LIST UTILITIES
-- ═══════════════════════════════════════════════════════════

-- Pointwise addition of adjoint lists
addAdj : List Float → List Float → List Float
addAdj []       ys       = ys
addAdj xs       []       = xs
addAdj (x ∷ xs) (y ∷ ys) = (x Float.+ y) ∷ addAdj xs ys

-- Zero adjoint list of length n
zerosF : ℕ → List Float
zerosF zero    = []
zerosF (suc n) = 0.0 ∷ zerosF n

-- One-hot adjoint list: δ in position i, 0 elsewhere
oneHotF : ℕ → ℕ → Float → List Float
oneHotF zero    _       _ = []
oneHotF (suc m) zero    δ = δ ∷ zerosF m
oneHotF (suc m) (suc j) δ = 0.0 ∷ oneHotF m j δ


-- ═══════════════════════════════════════════════════════════
-- SECTION 3: EMBEDDING INPUTS
-- ═══════════════════════════════════════════════════════════
--
-- Constants: backpropagator returns all zeros (no gradient)
-- Variables: backpropagator returns one-hot adjoint list

-- Embed a constant (not tracked)
constNF : ℕ → Float → RevNF
constNF n x = revNF x (λ _ → zerosF n)

-- Embed the i-th input variable (tracked)
varNF : ℕ → ℕ → Float → RevNF
varNF n i x = revNF x (λ δ → oneHotF n i δ)

-- Tag ALL parameters as tracked variables
-- This is the setup for reverse-mode: every input gets a
-- backpropagator that returns its adjoint in the right slot.
tagParamsF : List Float → List RevNF
tagParamsF θ = go (length θ) zero θ
  where
    go : ℕ → ℕ → List Float → List RevNF
    go n _ []       = []
    go n i (x ∷ xs) = varNF n i x ∷ go n (suc i) xs


-- ═══════════════════════════════════════════════════════════
-- SECTION 4: LIFTED ARITHMETIC (REVERSE-MODE)
-- ═══════════════════════════════════════════════════════════
--
-- Each operation:
--   1. Computes the value (same as plain Float arithmetic)
--   2. Builds a backpropagator that distributes the incoming
--      adjoint to both operands according to the chain rule
--
-- The magic: backpropagators COMPOSE automatically through
-- the chain rule. No explicit "tape" or "graph" needed.

-- Addition: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
-- Adjoint δ flows unchanged to both inputs
_+ᴺᶠ_ : RevNF → RevNF → RevNF
(revNF a ka) +ᴺᶠ (revNF b kb) =
  revNF (a Float.+ b) (λ δ → addAdj (ka δ) (kb δ))

infixl 6 _+ᴺᶠ_

-- Subtraction: ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
_-ᴺᶠ_ : RevNF → RevNF → RevNF
(revNF a ka) -ᴺᶠ (revNF b kb) =
  revNF (a Float.- b) (λ δ → addAdj (ka δ) (kb (Float.- 0.0 Float.- δ)))

infixl 6 _-ᴺᶠ_

-- Multiplication: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
-- Adjoint δ → (δ*b to a, δ*a to b)
_*ᴺᶠ_ : RevNF → RevNF → RevNF
(revNF a ka) *ᴺᶠ (revNF b kb) =
  revNF (a Float.* b) (λ δ → addAdj (ka (δ Float.* b)) (kb (δ Float.* a)))

infixl 7 _*ᴺᶠ_

-- Division: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
_÷ᴺᶠ_ : RevNF → RevNF → RevNF
(revNF a ka) ÷ᴺᶠ (revNF b kb) =
  revNF (a Float.÷ b)
        (λ δ → addAdj (ka (δ Float.÷ b))
                       (kb ((Float.- 0.0 Float.- (δ Float.* a))
                            Float.÷ (b Float.* b))))

infixl 7 _÷ᴺᶠ_

-- Negation: ∂(-a)/∂a = -1
negNF : RevNF → RevNF
negNF (revNF a ka) = revNF (Float.- 0.0 Float.- a) (λ δ → ka (Float.- 0.0 Float.- δ))

-- Logarithm: ∂(log a)/∂a = 1/a
logNF : RevNF → RevNF
logNF (revNF a ka) = revNF (log a) (λ δ → ka (δ Float.÷ a))

-- Exponential: ∂(exp a)/∂a = exp(a)
expNF : RevNF → RevNF
expNF (revNF a ka) = revNF (e^ a) (λ δ → ka (δ Float.* (e^ a)))


-- ═══════════════════════════════════════════════════════════
-- SECTION 5: SUMMATION AND LOOKUP
-- ═══════════════════════════════════════════════════════════

sumNF : List RevNF → RevNF
sumNF []       = revNF 0.0 (λ _ → [])
sumNF (x ∷ xs) = x +ᴺᶠ sumNF xs

lookupNF : List RevNF → ℕ → RevNF → RevNF
lookupNF []       _       def = def
lookupNF (x ∷ _)  zero    _   = x
lookupNF (_ ∷ xs) (suc n) def = lookupNF xs n def


-- ═══════════════════════════════════════════════════════════
-- SECTION 6: SOFTMAX (REVERSE-MODE)
-- ═══════════════════════════════════════════════════════════
-- softmax on a list of RevNF values.
-- exp each, divide each by sum of exps.

softmaxNF : List RevNF → List RevNF
softmaxNF logits = map (λ ex → ex ÷ᴺᶠ total) exps
  where
    exps  = map expNF logits
    total = sumNF exps


-- ═══════════════════════════════════════════════════════════
-- SECTION 7: ALPHABET
-- ═══════════════════════════════════════════════════════════

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


-- ═══════════════════════════════════════════════════════════
-- SECTION 8: BIGRAM PREDICTOR (REVERSE-MODE)
-- ═══════════════════════════════════════════════════════════
-- Same structure as BigramAD.agda's predictor, but using RevNF
-- arithmetic. The backpropagators compose automatically.

bigramPredictorNF : List RevNF → List Char → Char → RevNF
bigramPredictorNF θ history c =
  let prevIdx = lastCharIdx history
      row     = take alphaSize (drop (prevIdx * alphaSize) θ)
      probs   = softmaxNF row
  in lookupNF probs (charToIdx c)
              (constNF (length θ) (1.0 Float.÷ Float.fromℕ alphaSize))
  where
    lastCharIdx : List Char → ℕ
    lastCharIdx []       = 0
    lastCharIdx (x ∷ []) = charToIdx x
    lastCharIdx (_ ∷ xs) = lastCharIdx xs


-- ═══════════════════════════════════════════════════════════
-- SECTION 9: SCORING (REVERSE-MODE)
-- ═══════════════════════════════════════════════════════════
-- scoreFrom in reverse-mode arithmetic.
-- val gives the score, back gives the ENTIRE gradient.

scoreFromNF : (List Char → Char → RevNF) → List Char → List Char → RevNF
scoreFromNF p history []       = revNF 0.0 (λ _ → [])
scoreFromNF p history (c ∷ cs) =
  logNF (p history c) +ᴺᶠ scoreFromNF p (history L++ (c ∷ [])) cs

scoreNF : (List Char → Char → RevNF) → List Char → RevNF
scoreNF p corpus = scoreFromNF p [] corpus


-- ═══════════════════════════════════════════════════════════
-- SECTION 10: REVERSE-MODE GRADIENT (THE PAYOFF)
-- ═══════════════════════════════════════════════════════════
--
-- THIS IS THE KEY FUNCTION.
--
-- Forward-mode gradient (BigramAD.agda):
--   For each of n parameters:
--     seed parameter i, run forward pass → ∂S/∂θᵢ
--   Total: n forward passes
--
-- Reverse-mode gradient (this module):
--   1. Tag ALL parameters as tracked (one tagParamsF call)
--   2. Run ONE forward pass (computing scoreNF)
--   3. Call backpropagator with adjoint 1.0
--   Result: [∂S/∂θ₁, ∂S/∂θ₂, ..., ∂S/∂θₙ] — the ENTIRE gradient
--   Total: 1 forward pass + 1 backward pass = O(1) passes
--
-- For bigram (729 params): 729x speedup
-- For MLP (209 params): 209x speedup

gradientRev : List Float → List Char → List Float
gradientRev θ corpus =
  let θNF    = tagParamsF θ
      result = scoreNF (bigramPredictorNF θNF) corpus
  in back result 1.0


-- ═══════════════════════════════════════════════════════════
-- SECTION 11: GRADIENT ASCENT
-- ═══════════════════════════════════════════════════════════

learningRate : Float
learningRate = 0.1

initParams : ℕ → List Float
initParams zero    = []
initParams (suc n) = 0.0 ∷ initParams n

-- One gradient step: θ' = θ + η * ∇S(θ)
step : List Float → List Char → List Float
step θ corpus =
  let grad = gradientRev θ corpus
  in zipWith (λ ti gi → ti Float.+ (learningRate Float.* gi)) θ grad

-- Train for n steps
train : ℕ → List Float → List Char → List Float
train zero    θ _      = θ
train (suc n) θ corpus = train n (step θ corpus) corpus


-- ═══════════════════════════════════════════════════════════
-- SECTION 12: PLAIN-FLOAT SCORING (for reporting)
-- ═══════════════════════════════════════════════════════════

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


-- ═══════════════════════════════════════════════════════════
-- SECTION 13: GENERATION (same as BigramAD.agda)
-- ═══════════════════════════════════════════════════════════

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


-- ═══════════════════════════════════════════════════════════
-- SECTION 14: MAIN
-- ═══════════════════════════════════════════════════════════

showBool : Bool → String
showBool true  = "true"
showBool false = "false"

nameCorpus : List Char
nameCorpus = toList ".emma.olivia.ava.sophia.isabella.mia.charlotte.amelia.harper.evelyn."

main : Main
main = run do
  putStrLn "=== Denotational LLM: Bigram with REVERSE-MODE AD ==="
  putStrLn "(Continuations as representation of linear maps — Conal Elliott's insight)"
  putStrLn ""

  putStrLn "METHOD COMPARISON:"
  putStrLn "  Bigram.agda:     numerical perturbation (729 perturbations)"
  putStrLn "  BigramAD.agda:   forward-mode AD (729 forward passes)"
  putStrLn "  ReverseAD.agda:  REVERSE-mode AD (1 forward + 1 backward pass)"
  putStrLn ""
  putStrLn "  Forward-mode:  seed one param → one partial derivative"
  putStrLn "  Reverse-mode:  seed all params → ENTIRE gradient at once"
  putStrLn "  Speedup:       729x fewer passes for bigram's 729 params"
  putStrLn ""
  putStrLn "  Both are correct by construction (see AD.agda proofs)."
  putStrLn "  They are DIFFERENT REPRESENTATIONS of the SAME linear map."
  putStrLn ""

  putStrLn ("Corpus: " String.++ fromList nameCorpus)
  putStrLn ("  (" String.++ ℕShow.show (length nameCorpus) String.++ " chars)")
  putStrLn ""

  let uniform = λ (_ : List Char) (_ : Char) → 1.0 Float.÷ 27.0
  putStrLn ("Baseline (uniform): " String.++ Float.show (avgScore uniform nameCorpus) String.++ " avg log-prob/char")
  putStrLn ""

  let nP = alphaSize * alphaSize
  let θ₀ = initParams nP
  putStrLn "Training bigram (729 params, REVERSE-MODE AD gradient)..."

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
  putStrLn ""

  putStrLn "=== THE REPRESENTATION PATTERN ==="
  putStrLn "  AD.agda proves: forward-mode (Dual) and reverse-mode (Rev/RevN)"
  putStrLn "  are DIFFERENT REPRESENTATIONS of the SAME derivative."
  putStrLn "  Dual = (value, tangent)         — forward pass propagates tangent"
  putStrLn "  Rev  = (value, backpropagator)  — backward pass propagates adjoint"
  putStrLn "  Same meaning, different algorithm, different cost."
  putStrLn ""
  putStrLn "  This mirrors the text prediction story:"
  putStrLn "  Bigram, n-gram, RNN, attention are different representations"
  putStrLn "  of the SAME Kleisli morphism (predictor)."
  putStrLn "  Same meaning, different architecture, different expressiveness."
