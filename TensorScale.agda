{-# OPTIONS --guardedness #-}
-- ════════════════════════════════════════════════════════════
-- TENSOR vs MLP SCALE TEST: Tensor version
-- ════════════════════════════════════════════════════════════
-- Train on first N names, eval on NEXT N names (held out).
-- Stripped down for speed: just train + eval, no intermediates.

module TensorScale where

open import IO
open import Data.String.Base as String using (String; toList; fromList; _++_; lines)
open import Data.List.Base as List using (List; []; _∷_; length; map; take;
  drop; zipWith; foldr; concatMap; upTo; replicate)
  renaming (_++_ to _L++_)
open import Data.Char.Base using (Char; toℕ; fromℕ)
open import Data.Float.Base as Float
  using (Float; log; show; e^_; _<ᵇ_)
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

-- ═══════════════════════════════════════════════════════════
-- SECTION 2: HYPERPARAMETERS
-- ═══════════════════════════════════════════════════════════

embedDim : ℕ
embedDim = 2

order0Dim : ℕ
order0Dim = 1

order1Dim : ℕ
order1Dim = embedDim

order2Dim : ℕ
order2Dim = embedDim ℕ* embedDim

stateDim : ℕ
stateDim = order0Dim + order1Dim + order2Dim

nEmbed : ℕ
nEmbed = alphaSize ℕ* embedDim

nWstate : ℕ
nWstate = alphaSize ℕ* stateDim

nWprev : ℕ
nWprev = alphaSize ℕ* embedDim

nBias : ℕ
nBias = alphaSize

totalParams : ℕ
totalParams = nEmbed + nWstate + nWprev + nBias   -- 324

-- ═══════════════════════════════════════════════════════════
-- SECTION 3: LIST UTILITIES
-- ═══════════════════════════════════════════════════════════

lookupF : List Float → ℕ → Float
lookupF []       _       = 0.0
lookupF (x ∷ _)  zero    = x
lookupF (_ ∷ xs) (suc n) = lookupF xs n

sumF : List Float → Float
sumF []       = 0.0
sumF (x ∷ xs) = x Float.+ sumF xs

dotF : List Float → List Float → Float
dotF [] _              = 0.0
dotF _ []              = 0.0
dotF (a ∷ as) (b ∷ bs) = (a Float.* b) Float.+ dotF as bs

range : ℕ → List ℕ
range = upTo

addList : List Float → List Float → List Float
addList []       ys       = ys
addList xs       []       = xs
addList (x ∷ xs) (y ∷ ys) = (x Float.+ y) ∷ addList xs ys

zerosL : ℕ → List Float
zerosL zero    = []
zerosL (suc n) = 0.0 ∷ zerosL n

scaleList : Float → List Float → List Float
scaleList _ []       = []
scaleList s (x ∷ xs) = (s Float.* x) ∷ scaleList s xs

flattenL : List (List Float) → List Float
flattenL []         = []
flattenL (xs ∷ xss) = xs L++ flattenL xss

sliceF : List Float → ℕ → ℕ → List Float
sliceF xs off len = take len (drop off xs)

-- ═══════════════════════════════════════════════════════════
-- SECTION 4: PARAMETER LAYOUT
-- ═══════════════════════════════════════════════════════════

offE : ℕ
offE = 0

offWs : ℕ
offWs = nEmbed

offWp : ℕ
offWp = offWs + nWstate

offB : ℕ
offB = offWp + nWprev

getEmbed : List Float → ℕ → List Float
getEmbed θ i = sliceF θ (offE + i ℕ* embedDim) embedDim

getWsRow : List Float → ℕ → List Float
getWsRow θ k = sliceF θ (offWs + k ℕ* stateDim) stateDim

getWpRow : List Float → ℕ → List Float
getWpRow θ k = sliceF θ (offWp + k ℕ* embedDim) embedDim

getBias : List Float → ℕ → Float
getBias θ k = lookupF (drop offB θ) k

-- ═══════════════════════════════════════════════════════════
-- SECTION 5: TENSOR ALGEBRA STATE
-- ═══════════════════════════════════════════════════════════

initState : List Float
initState = zerosL stateDim

initPrevEmb : List Float
initPrevEmb = zerosL embedDim

outerProduct : List Float → List Float → List Float
outerProduct []       _ = []
outerProduct (u ∷ us) v = scaleList u v L++ outerProduct us v

updateState : List Float → List Float → List Float → List Float
updateState prevEmb curEmb state =
  let s0 = sliceF state 0 order0Dim
      s1 = sliceF state order0Dim order1Dim
      s2 = sliceF state (order0Dim + order1Dim) order2Dim
      newS0 = addList s0 (1.0 ∷ [])
      newS1 = addList s1 curEmb
      newS2 = addList s2 (outerProduct prevEmb curEmb)
  in newS0 L++ newS1 L++ newS2

-- ═══════════════════════════════════════════════════════════
-- SECTION 6: FORWARD PASS
-- ═══════════════════════════════════════════════════════════

softmaxF : List Float → List Float
softmaxF logits =
  let exps  = map (e^_) logits
      total = sumF exps
  in map (Float._÷ total) exps

normalizeState : List Float → List Float
normalizeState state =
  let count = lookupF state 0
      s1    = sliceF state order0Dim order1Dim
      s2    = sliceF state (order0Dim + order1Dim) order2Dim
      safe  = if count <ᵇ 0.5 then 1.0 else count
      norm1 = scaleList (1.0 Float.÷ safe) s1
      norm2 = scaleList (1.0 Float.÷ safe) s2
  in (1.0 ∷ []) L++ norm1 L++ norm2

computeLogits : List Float → List Float → List Float → List Float
computeLogits θ state prevEmb =
  let nstate = normalizeState state
  in map (λ k → dotF (getWsRow θ k) nstate
              Float.+ dotF (getWpRow θ k) prevEmb
              Float.+ getBias θ k)
      (range alphaSize)

outputProbs : List Float → List Float → List Float → List Float
outputProbs θ state prevEmb = softmaxF (computeLogits θ state prevEmb)

-- ═══════════════════════════════════════════════════════════
-- SECTION 7: EFFICIENT SCORING
-- ═══════════════════════════════════════════════════════════

tensorScoreFrom : List Float → List Float → List Float → List Char → Float
tensorScoreFrom θ prevEmb state []       = 0.0
tensorScoreFrom θ prevEmb state (c ∷ cs) =
  let probs    = outputProbs θ state prevEmb
      p        = lookupF probs (charToIdx c)
      logp     = log p
      curEmb   = getEmbed θ (charToIdx c)
      newState = updateState prevEmb curEmb state
  in logp Float.+ tensorScoreFrom θ curEmb newState cs

tensorScore : List Float → List Char → Float
tensorScore θ corpus = tensorScoreFrom θ initPrevEmb initState corpus

tensorAvgScore : List Float → List Char → Float
tensorAvgScore θ corpus = tensorScore θ corpus Float.÷ Float.fromℕ (length corpus)

-- ═══════════════════════════════════════════════════════════
-- SECTION 8: EXPLICIT BACKPROPAGATION
-- ═══════════════════════════════════════════════════════════

embedGradForChar : ℕ → ℕ → List Float → List Float
embedGradForChar ci targetCi dEmb =
  if ci ≡ᵇ targetCi then dEmb else zerosL embedDim

charGrad : List Float → List Float → List Float → ℕ → ℕ → List Float
charGrad θ prevEmb state prevCharIdx targetIdx =
  let
    nstate = normalizeState state
    logits = computeLogits θ state prevEmb
    probs  = softmaxF logits
    dLogits = map (λ k →
      (if k ≡ᵇ targetIdx then 1.0 else 0.0) Float.- lookupF probs k)
      (range alphaSize)
    dPrevEmb = map (λ j →
      sumF (zipWith (λ dk wprow → dk Float.* lookupF wprow j)
                    dLogits
                    (map (getWpRow θ) (range alphaSize))))
      (range embedDim)
    dE = flattenL (map (λ ci → embedGradForChar ci prevCharIdx dPrevEmb) (range alphaSize))
    dWs = flattenL (map (λ dl → scaleList dl nstate) dLogits)
    dWp = flattenL (map (λ dl → scaleList dl prevEmb) dLogits)
    db = dLogits
  in dE L++ dWs L++ dWp L++ db

gradAccum : List Float → List Float → List Float → ℕ → List Char → List Float → List Float
gradAccum θ prevEmb state prevCIdx []       acc = acc
gradAccum θ prevEmb state prevCIdx (c ∷ cs) acc =
  let cIdx     = charToIdx c
      g        = charGrad θ prevEmb state prevCIdx cIdx
      newAcc   = addList acc g
      curEmb   = getEmbed θ cIdx
      newState = updateState prevEmb curEmb state
  in gradAccum θ curEmb newState cIdx cs newAcc

gradientRev : List Float → List Char → List Float
gradientRev θ corpus = gradAccum θ initPrevEmb initState 0 corpus (zerosL totalParams)

-- ═══════════════════════════════════════════════════════════
-- SECTION 9: GRADIENT ASCENT
-- ═══════════════════════════════════════════════════════════

learningRate : Float
learningRate = 5.0

gradStep : List Float → List Char → List Float
gradStep θ corpus =
  let grad = gradientRev θ corpus
      n    = Float.fromℕ (length corpus)
  in zipWith (λ t g → t Float.+ ((learningRate Float.÷ n) Float.* g)) θ grad

train : ℕ → List Float → List Char → List Float
train zero    θ _      = θ
train (suc n) θ corpus = train n (gradStep θ corpus) corpus

-- ═══════════════════════════════════════════════════════════
-- SECTION 10: INITIALIZATION
-- ═══════════════════════════════════════════════════════════

initSmall : ℕ → List Float
initSmall n = map mkVal (range n)
  where
    mkVal : ℕ → Float
    mkVal i =
      let fi  = Float.fromℕ i
          fn  = Float.fromℕ n
      in ((fi Float.÷ fn) Float.- 0.5) Float.* 0.2

-- ═══════════════════════════════════════════════════════════
-- SECTION 11: CORPUS BUILDING
-- ═══════════════════════════════════════════════════════════

filterBool : (String → Bool) → List String → List String
filterBool _ []       = []
filterBool f (x ∷ xs) = if f x then x ∷ filterBool f xs else filterBool f xs

isNonEmpty : String → Bool
isNonEmpty s with toList s
... | []    = false
... | _ ∷ _ = true

buildCorpus : List String → List Char
buildCorpus names = toList (foldr (λ name acc → "." String.++ name String.++ acc) "." names)

nTrain : ℕ
nTrain = 200

nEval : ℕ
nEval = 200

nSteps : ℕ
nSteps = 25

-- ═══════════════════════════════════════════════════════════
-- SECTION 12: MAIN
-- ═══════════════════════════════════════════════════════════

main : Main
main = run do
  putStrLn "=== TENSOR SCALE TEST ==="
  putStrLn ("  Params: " String.++ ℕShow.show totalParams)
  putStrLn ("  Train:  " String.++ ℕShow.show nTrain String.++ " names")
  putStrLn ("  Eval:   " String.++ ℕShow.show nEval String.++ " names (held out)")
  putStrLn ("  Steps:  " String.++ ℕShow.show nSteps)
  putStrLn ""

  contents ← readFiniteFile "names.txt"
  let allNames = filterBool isNonEmpty (lines contents)

  -- DISJOINT train/eval split
  let trainNameList = take nTrain allNames
  let evalNameList  = take nEval (drop nTrain allNames)
  let trainCorpus   = buildCorpus trainNameList
  let evalCorpus    = buildCorpus evalNameList
  putStrLn ("  Train corpus: " String.++ ℕShow.show (length trainCorpus) String.++ " chars")
  putStrLn ("  Eval corpus:  " String.++ ℕShow.show (length evalCorpus) String.++ " chars")
  putStrLn ""

  -- Initialize and train
  let θ₀ = initSmall totalParams
  putStrLn "Training..."
  let θ = train nSteps θ₀ trainCorpus

  -- Evaluate on BOTH sets
  let trainNLL = 0.0 Float.- tensorAvgScore θ trainCorpus
  let evalNLL  = 0.0 Float.- tensorAvgScore θ evalCorpus
  putStrLn ""
  putStrLn "=== RESULTS ==="
  putStrLn ("  Train NLL: " String.++ Float.show trainNLL)
  putStrLn ("  Eval NLL:  " String.++ Float.show evalNLL)
  putStrLn ""
  putStrLn "Done."
