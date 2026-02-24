{-# OPTIONS --guardedness #-}
-- ════════════════════════════════════════════════════════════
-- MLP WITH REVERSE-MODE AD (EXPLICIT BACKPROPAGATION)
-- ════════════════════════════════════════════════════════════
--
-- Same MLP architecture as MLP.agda (embeddings, hidden layer,
-- tanh, softmax, 209 parameters), but trained with REVERSE-MODE
-- AD via explicit backpropagation instead of forward-mode.
--
-- THE KEY DIFFERENCE:
--   MLP.agda (forward-mode):  209 forward passes per character
--   MLPREV.agda (reverse-mode): 1 forward + 1 backward per character
--   Speedup: ~209x per character position
--
-- THE REPRESENTATION PATTERN (Conal Elliott):
--   Forward-mode = (value, tangent)         = Dual numbers
--   Reverse-mode = (value, backpropagator)  = Continuations
--   Explicit backprop = hand-unrolled reverse-mode
--   All compute the SAME derivative.

module MLPBig where

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
hiddenSize = 8

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

sliceF : List Float → ℕ → ℕ → List Float
sliceF xs off len = take len (drop off xs)

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
--
-- For log P(target | context), the gradient w.r.t. logits is:
--   ∂(log P_k)/∂z_j = δ_{jk} - P_j
--
-- The gradient vector is built by concatenating sections:
--   [dC | dW1 | db1 | dW2 | db2]
--
-- Embedding gradient: scatter dEmb to the correct char positions.
-- Rather than scanning all 27 chars, we build the section directly:
--   For char index ci, gradient = sum of dEmb slices where ctx[p] == ci

-- Build embedding gradient directly for char index ci
-- given context indices and dEmb (length = inputDim)
embedGradForChar : ℕ → List ℕ → List Float → List Float
embedGradForChar ci ctx dEmb = go ctx dEmb zero (zerosL embedDim)
  where
    go : List ℕ → List Float → ℕ → List Float → List Float
    go []       _  _ acc = acc
    go (i ∷ is) dE p acc =
      if i ≡ᵇ ci
      then go is dE (suc p) (addList acc (sliceF dE (p ℕ* embedDim) embedDim))
      else go is dE (suc p) acc

-- Full embedding gradient section (54 floats)
embedGradSection : List ℕ → List Float → List Float
embedGradSection ctx dEmb = flattenL (map (λ ci → embedGradForChar ci ctx dEmb) (range alphaSize))

-- Compute gradient for one character prediction
charBackprop : List Float → List ℕ → ℕ → List Float
charBackprop θ ctx targetIdx =
  let
    -- ═══ FORWARD ═══
    emb    = concatEmbedF θ ctx
    preact = map (λ j → dotF (getW1RowF θ j) emb Float.+ getB1F θ j) (range hiddenSize)
    hidden = map tanh preact
    lgts   = map (λ k → dotF (getW2RowF θ k) hidden Float.+ getB2F θ k) (range alphaSize)
    probs  = softmaxF lgts

    -- ═══ BACKWARD ═══
    -- dLogits[k] = δ_{k,target} - P_k
    dLogits = map (λ k →
      (if k ≡ᵇ targetIdx then 1.0 else 0.0) Float.- lookupF probs k)
      (range alphaSize)

    -- dHidden[j] = Σ_k dLogits[k] * W2[k][j]
    dHidden = map (λ j →
      sumF (zipWith (λ dk w2row → dk Float.* lookupF w2row j)
                    dLogits
                    (map (getW2RowF θ) (range alphaSize))))
      (range hiddenSize)

    -- dPreact[j] = dHidden[j] * (1 - tanh²(preact[j]))
    dPreact = zipWith (λ dh pa →
      let t = tanh pa
      in dh Float.* (1.0 Float.- (t Float.* t)))
      dHidden preact

    -- dEmb[i] = Σ_j dPreact[j] * W1[j][i]
    dEmb = map (λ i →
      sumF (zipWith (λ dp w1row → dp Float.* lookupF w1row i)
                    dPreact
                    (map (getW1RowF θ) (range hiddenSize))))
      (range inputDim)

    -- ═══ BUILD GRADIENT BY SECTION ═══
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
learningRate = 2.0

-- Gradient step with learning rate normalized by corpus length
-- This prevents the gradient from scaling with corpus size,
-- making training stable across different corpus sizes.
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
-- SECTION 11: NAME GENERATION
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
-- SECTION 12: CORPUS BUILDING
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

trainNames : ℕ
trainNames = 50

evalNames : ℕ
evalNames = 50

-- ═══════════════════════════════════════════════════════════
-- SECTION 13: MAIN
-- ═══════════════════════════════════════════════════════════

main : Main
main = run do
  putStrLn "=== MLP (BIG: hidden=8) with REVERSE-MODE AD ==="
  putStrLn "(Explicit backpropagation — reverse-mode AD unrolled)"
  putStrLn ""

  putStrLn "Architecture:"
  putStrLn ("  Context window: " String.++ ℕShow.show contextLen String.++ " chars")
  putStrLn ("  Embedding dim:  " String.++ ℕShow.show embedDim)
  putStrLn ("  Hidden units:   " String.++ ℕShow.show hiddenSize)
  putStrLn ("  Total params:   " String.++ ℕShow.show totalParams)
  putStrLn ""

  -- Read corpus
  putStrLn "Reading names.txt..."
  contents ← readFiniteFile "names.txt"
  let allNames   = filterBool isNonEmpty (lines contents)
  let nAllNames  = length allNames

  -- Training subset
  let trainNameList = take trainNames allNames
  let nTrainNames   = length trainNameList
  let trainCorpus   = buildCorpus trainNameList
  let nTrainChars   = length trainCorpus
  putStrLn ("Full corpus:  " String.++ ℕShow.show nAllNames String.++ " names")
  putStrLn ("Train subset: " String.++ ℕShow.show nTrainNames String.++ " names, "
    String.++ ℕShow.show nTrainChars String.++ " chars")

  -- Eval subset
  let evalNameList = take evalNames allNames
  let nEvalNames   = length evalNameList
  let evalCorpus   = buildCorpus evalNameList
  let nEvalChars   = length evalCorpus
  putStrLn ("Eval subset:  " String.++ ℕShow.show nEvalNames String.++ " names, "
    String.++ ℕShow.show nEvalChars String.++ " chars")
  putStrLn ""

  -- Build a small 20-name corpus for cross-comparison
  let smallNameList = take 20 allNames
  let smallCorpus   = buildCorpus smallNameList

  -- Initialize
  let θ₀ = initSmall totalParams

  -- === COMPARISON POINT 1: 50 names, 10 steps ===
  putStrLn "Training with reverse-mode AD..."
  let θ₁₀ = train 10 θ₀ trainCorpus
  let s₁₀ = mlpAvgScore θ₁₀ trainCorpus
  putStrLn ("  step 10: train NLL (50 names) = " String.++ Float.show (0.0 Float.- s₁₀))

  -- === COMPARISON POINT 2: 50 names, 25 steps ===
  let θfinal = train 15 θ₁₀ trainCorpus
  let sFinal = mlpAvgScore θfinal trainCorpus
  putStrLn ("  step 25: train NLL (50 names) = " String.++ Float.show (0.0 Float.- sFinal))
  putStrLn ""

  -- Also report on 20-name corpus for cross-comparison
  let s₂₅small = mlpAvgScore θfinal smallCorpus
  putStrLn ("  step 25: train NLL (20 names) = " String.++ Float.show (0.0 Float.- s₂₅small))
  putStrLn ""

  -- Evaluate on larger corpus
  putStrLn ("Evaluating on " String.++ ℕShow.show nEvalNames String.++ " names...")
  let sEval = mlpAvgScore θfinal evalCorpus
  putStrLn ""

  -- Report
  putStrLn "=== COMPARISON RESULTS ==="
  putStrLn ("  [10 steps, 50 names]  Train NLL = " String.++ Float.show (0.0 Float.- s₁₀))
  putStrLn ("  [25 steps, 50 names]  Train NLL = " String.++ Float.show (0.0 Float.- sFinal))
  putStrLn ("  [25 steps, 20 names]  Train NLL = " String.++ Float.show (0.0 Float.- s₂₅small))
  putStrLn ("  Eval NLL (" String.++ ℕShow.show nEvalNames String.++ " names):   "
    String.++ Float.show (0.0 Float.- sEval))
  putStrLn ""
  putStrLn "  Benchmarks (Karpathy, 32k names, full training):"
  putStrLn "    Bigram (count-based):         2.454"
  putStrLn "    MLP (Bengio et al.):          ~2.3"
  putStrLn "    Uniform random:               3.296"
  putStrLn ""

  -- Generate
  let p = mlpPredictor θfinal
  putStrLn "Generated names (greedy decoding):"
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".")))
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".a")))
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".s")))
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".m")))
  putStrLn ""
  putStrLn "Done."
