{-# OPTIONS --guardedness #-}
-- ════════════════════════════════════════════════════════════
-- MLP: EXECUTABLE MULTI-LAYER PERCEPTRON (makemore part 2)
-- ════════════════════════════════════════════════════════════
--
-- Character-level name generation using a Bengio et al. 2003
-- style MLP, following Karpathy's makemore progression.
--
-- Architecture:
--   1. Context window: last 2 characters (bigram context)
--   2. Character embeddings: 27 chars → 2-dimensional vectors
--   3. Concatenate embeddings → 4-dimensional input
--   4. Hidden layer: 4 → 4 units with tanh activation
--   5. Output layer: 4 → 27 logits, softmax → probabilities
--
-- In our framework (Architectures.agda), this is an n-gram
-- architecture with n=2, where the representation is
-- parameterized by learned embeddings + MLP weights.
-- The key difference from a plain bigram: the MLP can learn
-- nonlinear combinations of the embedding dimensions.
--
-- Training uses forward-mode AD via dual numbers (DualF).
-- With 209 parameters and forward-mode, we need 209 forward
-- passes per gradient step. Slow but correct.
-- (Reverse-mode AD would reduce this to 1 pass — see AD.agda.)
--
-- This is the executable (Float) counterpart — the proof
-- modules (Spec.agda, AD.agda) guarantee the algebraic
-- properties hold for any Predictor, including this one.

module MLP where

open import IO
open import Data.String.Base as String using (String; toList; fromList; _++_)
open import Data.List.Base as List using (List; []; _∷_; length; map; take;
  drop; zipWith; foldr; concatMap; upTo; replicate)
  renaming (_++_ to _L++_)
open import Data.Char.Base using (Char; toℕ; fromℕ)
open import Data.Float.Base as Float
  using (Float; log; show; e^_; tanh; _<ᵇ_)
open import Data.Nat.Base as Nat using (ℕ; suc; zero; _≡ᵇ_; _≤ᵇ_; _∸_; _+_)
  renaming (_*_ to _ℕ*_)
open import Data.Nat.Show as ℕShow using ()
open import Data.Bool.Base using (Bool; true; false; if_then_else_)
open import Level using (0ℓ)
open import Data.Unit.Polymorphic.Base using (⊤)

-- ═══════════════════════════════════════════════════════════
-- SECTION 1: ALPHABET AND CHARACTER ENCODING
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
-- SECTION 2: HYPERPARAMETERS
-- ═══════════════════════════════════════════════════════════

-- Small dimensions to keep runtime reasonable with forward-mode AD.
-- In Karpathy's full version: embedDim=10, hiddenSize=100, context=3.
-- Forward-mode AD requires nParams forward passes per gradient step,
-- so we minimize dimensions aggressively.

contextLen : ℕ
contextLen = 2

embedDim : ℕ
embedDim = 2

hiddenSize : ℕ
hiddenSize = 4

-- Derived sizes
inputDim : ℕ
inputDim = contextLen ℕ* embedDim   -- 2*2 = 4

-- Parameter counts:
--   C:  27 * 2  = 54   (embedding matrix)
--   W1: 4 * 4   = 16   (hidden weights)
--   b1: 4       = 4    (hidden biases)
--   W2: 4 * 27  = 108  (output weights)
--   b2: 27      = 27   (output biases)
--   Total:        209
nEmbedParams : ℕ
nEmbedParams = alphaSize ℕ* embedDim          -- 54

nW1Params : ℕ
nW1Params = inputDim ℕ* hiddenSize             -- 16

nB1Params : ℕ
nB1Params = hiddenSize                          -- 4

nW2Params : ℕ
nW2Params = hiddenSize ℕ* alphaSize            -- 108

nB2Params : ℕ
nB2Params = alphaSize                           -- 27

totalParams : ℕ
totalParams = nEmbedParams + nW1Params + nB1Params + nW2Params + nB2Params

-- ═══════════════════════════════════════════════════════════
-- SECTION 3: DUAL NUMBERS (Float-based forward-mode AD)
-- ═══════════════════════════════════════════════════════════
-- Same approach as AD.agda but using Float instead of
-- postulated ℝ. Each DualF carries (value, derivative).

record DualF : Set where
  constructor dualF
  field
    val : Float
    der : Float

open DualF public

-- Constants and variables
constDF : Float → DualF
constDF x = dualF x 0.0

varDF : Float → DualF
varDF x = dualF x 1.0

-- Lifted arithmetic
_+D_ : DualF → DualF → DualF
(dualF a a') +D (dualF b b') = dualF (a Float.+ b) (a' Float.+ b')

_-D_ : DualF → DualF → DualF
(dualF a a') -D (dualF b b') = dualF (a Float.- b) (a' Float.- b')

_*D_ : DualF → DualF → DualF
(dualF a a') *D (dualF b b') = dualF (a Float.* b) ((a' Float.* b) Float.+ (a Float.* b'))

_÷D_ : DualF → DualF → DualF
(dualF a a') ÷D (dualF b b') =
  dualF (a Float.÷ b) (((a' Float.* b) Float.- (a Float.* b')) Float.÷ (b Float.* b))

infixl 6 _+D_ _-D_
infixl 7 _*D_ _÷D_

-- Logarithm: d/dx log(a) = a'/a
logD : DualF → DualF
logD (dualF a a') = dualF (log a) (a' Float.÷ a)

-- Exponential: d/dx exp(a) = a' * exp(a)
expD : DualF → DualF
expD (dualF a a') = dualF (e^ a) (a' Float.* (e^ a))

-- Tanh: d/dx tanh(a) = a' * (1 - tanh(a)^2)
tanhD : DualF → DualF
tanhD (dualF a a') =
  let t = tanh a
  in dualF t (a' Float.* (1.0 Float.- (t Float.* t)))

-- ═══════════════════════════════════════════════════════════
-- SECTION 4: GENERIC LIST UTILITIES
-- ═══════════════════════════════════════════════════════════

-- Lookup in a flat list
lookupF : List Float → ℕ → Float
lookupF []       _       = 0.0
lookupF (x ∷ _)  zero    = x
lookupF (_ ∷ xs) (suc n) = lookupF xs n

lookupD : List DualF → ℕ → DualF
lookupD []       _       = constDF 0.0
lookupD (x ∷ _)  zero    = x
lookupD (_ ∷ xs) (suc n) = lookupD xs n

-- Sum a list of Floats
sumF : List Float → Float
sumF []       = 0.0
sumF (x ∷ xs) = x Float.+ sumF xs

-- Sum a list of DualFs
sumD : List DualF → DualF
sumD []       = constDF 0.0
sumD (x ∷ xs) = x +D sumD xs

-- Dot product of two Float lists
dotF : List Float → List Float → Float
dotF [] _              = 0.0
dotF _ []              = 0.0
dotF (a ∷ as) (b ∷ bs) = (a Float.* b) Float.+ dotF as bs

-- Dot product of two DualF lists
dotD : List DualF → List DualF → DualF
dotD [] _              = constDF 0.0
dotD _ []              = constDF 0.0
dotD (a ∷ as) (b ∷ bs) = (a *D b) +D dotD as bs

-- Generate a list of indices [0, 1, ..., n-1]
range : ℕ → List ℕ
range = upTo

-- ═══════════════════════════════════════════════════════════
-- SECTION 5: PARAMETER LAYOUT
-- ═══════════════════════════════════════════════════════════
-- All parameters are stored in one flat List Float.
-- Layout: [C | W1 | b1 | W2 | b2]
-- We use offsets to extract each matrix/vector.

-- Offset into the flat parameter list
offC  : ℕ
offC  = 0

offW1 : ℕ
offW1 = nEmbedParams

offB1 : ℕ
offB1 = offW1 + nW1Params

offW2 : ℕ
offW2 = offB1 + nB1Params

offB2 : ℕ
offB2 = offW2 + nW2Params

-- Slice a sublist [off .. off+len)
sliceF : List Float → ℕ → ℕ → List Float
sliceF xs off len = take len (drop off xs)

sliceD : List DualF → ℕ → ℕ → List DualF
sliceD xs off len = take len (drop off xs)

-- Get embedding for character index i: C[i*embedDim .. (i+1)*embedDim)
getEmbedF : List Float → ℕ → List Float
getEmbedF θ i = sliceF θ (offC + i ℕ* embedDim) embedDim

getEmbedD : List DualF → ℕ → List DualF
getEmbedD θ i = sliceD θ (offC + i ℕ* embedDim) embedDim

-- Get row j of W1: W1[j*inputDim .. (j+1)*inputDim)
getW1RowF : List Float → ℕ → List Float
getW1RowF θ j = sliceF θ (offW1 + j ℕ* inputDim) inputDim

getW1RowD : List DualF → ℕ → List DualF
getW1RowD θ j = sliceD θ (offW1 + j ℕ* inputDim) inputDim

-- Get b1[j]
getB1F : List Float → ℕ → Float
getB1F θ j = lookupF (drop offB1 θ) j

getB1D : List DualF → ℕ → DualF
getB1D θ j = lookupD (drop offB1 θ) j

-- Get row k of W2: W2[k*hiddenSize .. (k+1)*hiddenSize)
-- W2 is hiddenSize × alphaSize stored in row-major: W2[h][alpha]
-- We need column k (for output index k): W2[0][k], W2[1][k], ..., W2[h-1][k]
-- Alternatively, store as W2[alpha][hidden] = alphaSize rows of hiddenSize
-- Let's use: row k of W2 is the weights from all hidden units to output k
getW2RowF : List Float → ℕ → List Float
getW2RowF θ k = sliceF θ (offW2 + k ℕ* hiddenSize) hiddenSize

getW2RowD : List DualF → ℕ → List DualF
getW2RowD θ k = sliceD θ (offW2 + k ℕ* hiddenSize) hiddenSize

-- Get b2[k]
getB2F : List Float → ℕ → Float
getB2F θ k = lookupF (drop offB2 θ) k

getB2D : List DualF → ℕ → DualF
getB2D θ k = lookupD (drop offB2 θ) k

-- ═══════════════════════════════════════════════════════════
-- SECTION 6: MLP FORWARD PASS (Float version)
-- ═══════════════════════════════════════════════════════════
-- Given parameters θ and a context of contextLen character
-- indices, compute the output probability distribution.

-- Extract last contextLen characters from history as indices
-- Pads with 0 (= '.') if history is shorter than contextLen
-- Simple approach: take the last contextLen chars, pad with '.'
lastNChars : List Char → List Char
lastNChars cs =
  let n   = length cs
      raw = drop (n ∸ contextLen) cs
      pad = replicate (contextLen ∸ length raw) '.'
  in pad L++ raw

lastNIdx : List Char → List ℕ
lastNIdx cs = map charToIdx (lastNChars cs)

-- Concatenate embeddings for the context
concatEmbedF : List Float → List ℕ → List Float
concatEmbedF θ []       = []
concatEmbedF θ (i ∷ is) = getEmbedF θ i L++ concatEmbedF θ is

concatEmbedD : List DualF → List ℕ → List DualF
concatEmbedD θ []       = []
concatEmbedD θ (i ∷ is) = getEmbedD θ i L++ concatEmbedD θ is

-- Hidden layer: for each hidden unit j, compute tanh(W1[j] · input + b1[j])
hiddenLayerF : List Float → List Float → List Float
hiddenLayerF θ input = map computeUnit (range hiddenSize)
  where
    computeUnit : ℕ → Float
    computeUnit j = tanh (dotF (getW1RowF θ j) input Float.+ getB1F θ j)

hiddenLayerD : List DualF → List DualF → List DualF
hiddenLayerD θ input = map computeUnit (range hiddenSize)
  where
    computeUnit : ℕ → DualF
    computeUnit j = tanhD (dotD (getW1RowD θ j) input +D getB1D θ j)

-- Output logits: for each output k, compute W2[k] · hidden + b2[k]
logitsF : List Float → List Float → List Float
logitsF θ hidden = map computeLogit (range alphaSize)
  where
    computeLogit : ℕ → Float
    computeLogit k = dotF (getW2RowF θ k) hidden Float.+ getB2F θ k

logitsD : List DualF → List DualF → List DualF
logitsD θ hidden = map computeLogit (range alphaSize)
  where
    computeLogit : ℕ → DualF
    computeLogit k = dotD (getW2RowD θ k) hidden +D getB2D θ k

-- Softmax (Float version)
softmaxF : List Float → List Float
softmaxF logits =
  let exps  = map (e^_) logits
      total = sumF exps
  in map (Float._÷ total) exps

-- Softmax (DualF version)
softmaxD : List DualF → List DualF
softmaxD logits =
  let exps  = map expD logits
      total = sumD exps
  in map (_÷D total) exps

-- Full forward pass: context → probability distribution
forwardF : List Float → List ℕ → List Float
forwardF θ ctx =
  let emb    = concatEmbedF θ ctx
      hidden = hiddenLayerF θ emb
      lgts   = logitsF θ hidden
  in softmaxF lgts

forwardD : List DualF → List ℕ → List DualF
forwardD θ ctx =
  let emb    = concatEmbedD θ ctx
      hidden = hiddenLayerD θ emb
      lgts   = logitsD θ hidden
  in softmaxD lgts

-- ═══════════════════════════════════════════════════════════
-- SECTION 7: PREDICTOR
-- ═══════════════════════════════════════════════════════════
-- The MLP as a Predictor: List Char → Char → Float

mlpPredictor : List Float → List Char → Char → Float
mlpPredictor θ history c =
  let ctx   = lastNIdx history
      probs = forwardF θ ctx
  in lookupF probs (charToIdx c)

mlpPredictorD : List DualF → List Char → Char → DualF
mlpPredictorD θ history c =
  let ctx   = lastNIdx history
      probs = forwardD θ ctx
  in lookupD probs (charToIdx c)

-- ═══════════════════════════════════════════════════════════
-- SECTION 8: SCORING
-- ═══════════════════════════════════════════════════════════

scoreFromF : (List Char → Char → Float) → List Char → List Char → Float
scoreFromF p h []       = 0.0
scoreFromF p h (c ∷ cs) =
  log (p h c) Float.+ scoreFromF p (h L++ (c ∷ [])) cs

scoreF : (List Char → Char → Float) → List Char → Float
scoreF p corpus = scoreFromF p [] corpus

avgScoreF : (List Char → Char → Float) → List Char → Float
avgScoreF p corpus = scoreF p corpus Float.÷ Float.fromℕ (length corpus)

-- Dual-valued score
scoreFromD : (List Char → Char → DualF) → List Char → List Char → DualF
scoreFromD p h []       = constDF 0.0
scoreFromD p h (c ∷ cs) =
  logD (p h c) +D scoreFromD p (h L++ (c ∷ [])) cs

scoreD : (List Char → Char → DualF) → List Char → DualF
scoreD p corpus = scoreFromD p [] corpus

-- ═══════════════════════════════════════════════════════════
-- SECTION 9: FORWARD-MODE AD GRADIENT
-- ═══════════════════════════════════════════════════════════
-- Seed parameter i as variable, all others as constants.
-- Run forward pass + score. Read der component → ∂S/∂θᵢ.

seedParams : List Float → ℕ → List DualF
seedParams []       _       = []
seedParams (θ ∷ θs) zero    = varDF θ ∷ map constDF θs
seedParams (θ ∷ θs) (suc i) = constDF θ ∷ seedParams θs i

-- Compute one partial derivative
partialI : List Float → List Char → ℕ → Float
partialI θ corpus i =
  let θD = seedParams θ i
  in der (scoreD (mlpPredictorD θD) corpus)

-- Compute full gradient (one forward pass per parameter)
gradient : List Float → List Char → List Float
gradient θ corpus = go 0 θ
  where
    go : ℕ → List Float → List Float
    go _ []       = []
    go i (_ ∷ xs) = partialI θ corpus i ∷ go (suc i) xs

-- ═══════════════════════════════════════════════════════════
-- SECTION 10: GRADIENT ASCENT
-- ═══════════════════════════════════════════════════════════

learningRate : Float
learningRate = 0.1

-- One gradient step: θ' = θ + η * ∇S(θ)
gradStep : List Float → List Char → List Float
gradStep θ corpus =
  let grad = gradient θ corpus
  in zipWith (λ t g → t Float.+ (learningRate Float.* g)) θ grad

-- Train for n steps, returning updated parameters
train : ℕ → List Float → List Char → List Float
train zero    θ _      = θ
train (suc n) θ corpus = train n (gradStep θ corpus) corpus

-- ═══════════════════════════════════════════════════════════
-- SECTION 11: INITIALIZATION
-- ═══════════════════════════════════════════════════════════
-- Initialize parameters with small values.
-- We use a simple deterministic scheme (no random init).

-- Small deterministic values spread around 0
-- Range: [-0.1, 0.1] — keeps initial predictions near uniform
initSmall : ℕ → List Float
initSmall n = map mkVal (range n)
  where
    mkVal : ℕ → Float
    mkVal i =
      let fi  = Float.fromℕ i
          fn  = Float.fromℕ n
      in ((fi Float.÷ fn) Float.- 0.5) Float.* 0.2

-- ═══════════════════════════════════════════════════════════
-- SECTION 12: NAME GENERATION (greedy decoding)
-- ═══════════════════════════════════════════════════════════

alpha : List Char
alpha = '.' ∷ map (λ n → fromℕ (97 + n)) (upTo 26)

argmaxF : (List Char → Char → Float) → List Char → List Char → Char → Float → Char
argmaxF _ _  []       best _     = best
argmaxF p h  (c ∷ cs) best bestP =
  let prob = p h c in
  if bestP <ᵇ prob
    then argmaxF p h cs c    prob
    else argmaxF p h cs best bestP

generateName : (List Char → Char → Float) → ℕ → List Char → List Char
generateName _ zero    _  = []
generateName p (suc n) h  =
  let next = argmaxF p h alpha '.' 0.0
  in if toℕ next ≡ᵇ 46 then []
     else next ∷ generateName p n (h L++ (next ∷ []))

-- ═══════════════════════════════════════════════════════════
-- SECTION 13: CORPUS
-- ═══════════════════════════════════════════════════════════
-- Small hardcoded corpus for fast compilation.

-- 10 names: small enough for forward-mode AD to finish in minutes
nameCorpus : List Char
nameCorpus = toList ".emma.olivia.ava.sophia.isabella.mia.charlotte.amelia.harper.evelyn."

-- ═══════════════════════════════════════════════════════════
-- SECTION 14: MAIN
-- ═══════════════════════════════════════════════════════════

showBool : Bool → String
showBool true  = "true"
showBool false = "false"

main : Main
main = run do
  putStrLn "=== Denotational LLM: MLP Name Generator ==="
  putStrLn "(Bengio et al. 2003 / Karpathy makemore part 2)"
  putStrLn ""

  putStrLn "Architecture:"
  putStrLn ("  Context window: " String.++ ℕShow.show contextLen String.++ " chars")
  putStrLn ("  Embedding dim:  " String.++ ℕShow.show embedDim)
  putStrLn ("  Hidden units:   " String.++ ℕShow.show hiddenSize)
  putStrLn ("  Total params:   " String.++ ℕShow.show totalParams)
  putStrLn ""

  -- Baseline
  let uniform = λ (_ : List Char) (_ : Char) → 1.0 Float.÷ Float.fromℕ alphaSize
  let uScore  = avgScoreF uniform nameCorpus
  putStrLn ("Baseline (uniform): " String.++ Float.show uScore String.++ " avg log-prob/char")
  putStrLn ("  (NLL = " String.++ Float.show (0.0 Float.- uScore) String.++ ")")
  putStrLn ""

  -- Initialize
  let θ₀ = initSmall totalParams
  let s₀ = avgScoreF (mlpPredictor θ₀) nameCorpus
  putStrLn ("Initial MLP:        " String.++ Float.show s₀ String.++ " avg log-prob/char")
  putStrLn ""

  putStrLn "Training (forward-mode AD, 209 passes per step)..."

  -- Train 10 steps
  let θ₁₀  = train 10 θ₀ nameCorpus
  let s₁₀  = avgScoreF (mlpPredictor θ₁₀) nameCorpus
  putStrLn ("  step 10: " String.++ Float.show s₁₀ String.++ " avg/char")

  -- Train 10 more steps (20 total)
  let θ₂₀ = train 10 θ₁₀ nameCorpus
  let s₂₀ = avgScoreF (mlpPredictor θ₂₀) nameCorpus
  putStrLn ("  step 20: " String.++ Float.show s₂₀ String.++ " avg/char")

  -- Train 30 more steps (50 total)
  let θ₅₀ = train 30 θ₂₀ nameCorpus
  let s₅₀ = avgScoreF (mlpPredictor θ₅₀) nameCorpus
  putStrLn ("  step 50: " String.++ Float.show s₅₀ String.++ " avg/char")
  putStrLn ""

  -- Report
  putStrLn ("NLL after training: " String.++ Float.show (0.0 Float.- s₅₀))
  putStrLn ("Improved over uniform? " String.++ showBool (uScore <ᵇ s₅₀))
  putStrLn ""

  -- Generate
  let p = mlpPredictor θ₅₀
  putStrLn "Generated names (greedy decoding):"
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".")))
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".a")))
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".s")))
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".m")))
  putStrLn ""
  putStrLn "Done."
