{-# OPTIONS --guardedness #-}
-- ════════════════════════════════════════════════════════════
-- TENSOR vs MLP SCALE TEST: MLP version
-- ════════════════════════════════════════════════════════════
-- Train on first N names, eval on NEXT N names (held out).
-- Stripped down for speed: just train + eval, no intermediates.

module MLPScale where

open import IO
open import Data.String.Base as String using (String; toList; fromList; _++_; lines)
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

-- ═══════════════════════════════════════════════════════════
-- SECTION 2: HYPERPARAMETERS
-- ═══════════════════════════════════════════════════════════

contextLen : ℕ
contextLen = 2

embedDim : ℕ
embedDim = 2

hiddenSize : ℕ
hiddenSize = 4

inputDim : ℕ
inputDim = contextLen ℕ* embedDim   -- 4

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

getEmbedF : List Float → ℕ → List Float
getEmbedF θ i = sliceF θ (offC + i ℕ* embedDim) embedDim

getW1RowF : List Float → ℕ → List Float
getW1RowF θ j = sliceF θ (offW1 + j ℕ* inputDim) inputDim

getB1F : List Float → ℕ → Float
getB1F θ j = lookupF (drop offB1 θ) j

getW2RowF : List Float → ℕ → List Float
getW2RowF θ k = sliceF θ (offW2 + k ℕ* hiddenSize) hiddenSize

getB2F : List Float → ℕ → Float
getB2F θ k = lookupF (drop offB2 θ) k

-- ═══════════════════════════════════════════════════════════
-- SECTION 5: MLP FORWARD PASS
-- ═══════════════════════════════════════════════════════════

lastNChars : List Char → List Char
lastNChars cs =
  let n   = length cs
      raw = drop (n ∸ contextLen) cs
      pad = replicate (contextLen ∸ length raw) '.'
  in pad L++ raw

lastNIdx : List Char → List ℕ
lastNIdx cs = map charToIdx (lastNChars cs)

concatEmbedF : List Float → List ℕ → List Float
concatEmbedF θ []       = []
concatEmbedF θ (i ∷ is) = getEmbedF θ i L++ concatEmbedF θ is

softmaxF : List Float → List Float
softmaxF logits =
  let exps  = map (e^_) logits
      total = sumF exps
  in map (Float._÷ total) exps

forwardF : List Float → List ℕ → List Float
forwardF θ ctx =
  let emb    = concatEmbedF θ ctx
      hidden = map (λ j → tanh (dotF (getW1RowF θ j) emb Float.+ getB1F θ j)) (range hiddenSize)
      lgts   = map (λ k → dotF (getW2RowF θ k) hidden Float.+ getB2F θ k) (range alphaSize)
  in softmaxF lgts

-- ═══════════════════════════════════════════════════════════
-- SECTION 6: PREDICTOR AND SCORING
-- ═══════════════════════════════════════════════════════════

mlpPredictor : List Float → List Char → Char → Float
mlpPredictor θ history c =
  let ctx   = lastNIdx history
      probs = forwardF θ ctx
  in lookupF probs (charToIdx c)

keepLast : List Char → Char → List Char
keepLast ctx c =
  let extended = ctx L++ (c ∷ [])
      n = length extended
  in drop (n ∸ contextLen) extended

mlpScoreFrom : List Float → List Char → List Char → Float
mlpScoreFrom θ ctx []       = 0.0
mlpScoreFrom θ ctx (c ∷ cs) =
  log (mlpPredictor θ ctx c) Float.+ mlpScoreFrom θ (keepLast ctx c) cs

mlpScore : List Float → List Char → Float
mlpScore θ corpus = mlpScoreFrom θ [] corpus

mlpAvgScore : List Float → List Char → Float
mlpAvgScore θ corpus = mlpScore θ corpus Float.÷ Float.fromℕ (length corpus)

-- ═══════════════════════════════════════════════════════════
-- SECTION 7: EXPLICIT BACKPROPAGATION
-- ═══════════════════════════════════════════════════════════

embedGradForChar : ℕ → List ℕ → List Float → List Float
embedGradForChar ci ctx dEmb = go ctx dEmb zero (zerosL embedDim)
  where
    go : List ℕ → List Float → ℕ → List Float → List Float
    go []       _  _ acc = acc
    go (i ∷ is) dE p acc =
      if i ≡ᵇ ci
      then go is dE (suc p) (addList acc (sliceF dE (p ℕ* embedDim) embedDim))
      else go is dE (suc p) acc

embedGradSection : List ℕ → List Float → List Float
embedGradSection ctx dEmb = flattenL (map (λ ci → embedGradForChar ci ctx dEmb) (range alphaSize))

charBackprop : List Float → List ℕ → ℕ → List Float
charBackprop θ ctx targetIdx =
  let
    emb    = concatEmbedF θ ctx
    preact = map (λ j → dotF (getW1RowF θ j) emb Float.+ getB1F θ j) (range hiddenSize)
    hidden = map tanh preact
    lgts   = map (λ k → dotF (getW2RowF θ k) hidden Float.+ getB2F θ k) (range alphaSize)
    probs  = softmaxF lgts
    dLogits = map (λ k →
      (if k ≡ᵇ targetIdx then 1.0 else 0.0) Float.- lookupF probs k)
      (range alphaSize)
    dHidden = map (λ j →
      sumF (zipWith (λ dk w2row → dk Float.* lookupF w2row j)
                    dLogits
                    (map (getW2RowF θ) (range alphaSize))))
      (range hiddenSize)
    dPreact = zipWith (λ dh pa →
      let t = tanh pa
      in dh Float.* (1.0 Float.- (t Float.* t)))
      dHidden preact
    dEmb = map (λ i →
      sumF (zipWith (λ dp w1row → dp Float.* lookupF w1row i)
                    dPreact
                    (map (getW1RowF θ) (range hiddenSize))))
      (range inputDim)
    dC  = embedGradSection ctx dEmb
    dW1 = flattenL (map (λ dp → scaleList dp emb) dPreact)
    db1 = dPreact
    dW2 = flattenL (map (λ dl → scaleList dl hidden) dLogits)
    db2 = dLogits
  in dC L++ dW1 L++ db1 L++ dW2 L++ db2

-- ═══════════════════════════════════════════════════════════
-- SECTION 8: GRADIENT ACCUMULATION
-- ═══════════════════════════════════════════════════════════

gradAccum : List Float → List Char → List Char → List Float → List Float
gradAccum θ ctx []       acc = acc
gradAccum θ ctx (c ∷ cs) acc =
  let cIdx   = charToIdx c
      ctxIdx = lastNIdx ctx
      g      = charBackprop θ ctxIdx cIdx
      newAcc = addList acc g
      newCtx = keepLast ctx c
  in gradAccum θ newCtx cs newAcc

gradientRev : List Float → List Char → List Float
gradientRev θ corpus = gradAccum θ [] corpus (zerosL totalParams)

-- ═══════════════════════════════════════════════════════════
-- SECTION 9: GRADIENT ASCENT
-- ═══════════════════════════════════════════════════════════

learningRate : Float
learningRate = 10.0

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
  putStrLn "=== MLP SCALE TEST ==="
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
  let trainNLL = 0.0 Float.- mlpAvgScore θ trainCorpus
  let evalNLL  = 0.0 Float.- mlpAvgScore θ evalCorpus
  putStrLn ""
  putStrLn "=== RESULTS ==="
  putStrLn ("  Train NLL: " String.++ Float.show trainNLL)
  putStrLn ("  Eval NLL:  " String.++ Float.show evalNLL)
  putStrLn ""
  putStrLn "Done."
