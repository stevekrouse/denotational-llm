{-# OPTIONS --guardedness #-}
-- ════════════════════════════════════════════════════════════
-- TENSOR ALGEBRA PREDICTOR: A Novel Architecture from Algebra
-- ════════════════════════════════════════════════════════════
--
-- MOTIVATION (from the denotational design framework):
--
-- In Architectures.agda we showed that different architectures
-- are different REPRESENTATIONS of the Kleisli morphism
-- (Predictor = List Char → Char → ℝ). The state must form
-- a monoid to handle "context accumulation" (the indexed
-- homomorphism property from Kleisli.agda).
--
-- THE NEW IDEA: Use the TRUNCATED TENSOR ALGEBRA as the state
-- monoid. The tensor algebra T(V) = ℝ ⊕ V ⊕ (V⊗V) ⊕ ...
-- is the universal construction for the free associative algebra.
-- Truncated at order k, it captures k-gram correlations in a
-- compressed embedding space (d^k dims instead of |Σ|^k).
--
-- Concretely, with embedding dim d and truncation at order 2:
--   State = (scalar, vector ∈ ℝ^d, matrix ∈ ℝ^{d×d})
--   State update on seeing char c with embedding e:
--     order 0: s₀ += 1               (character count)
--     order 1: s₁ += e               (sum of embeddings)
--     order 2: s₂ += prev_emb ⊗ e    (bigram outer products)
--
-- Output combines TWO pathways:
--   1. W_state · state  — long-range features from tensor algebra
--   2. W_prev · prevEmb — immediate bigram context (last char embedding)
-- This gives the model both accumulated statistics and direct
-- access to the most recent character for sharp bigram predictions.
--
-- WHY THIS IS INTERESTING:
--   1. Derived from algebra (tensor algebra = universal construction)
--   2. Captures bigram correlations in d² dims instead of |Σ|² = 729
--   3. State update is a monoid homomorphism by construction
--   4. A genuine new point in the architecture design space

module TensorSmall where

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
embedDim = 1

-- State dimensions (truncated tensor algebra at order 2):
order0Dim : ℕ
order0Dim = 1

order1Dim : ℕ
order1Dim = embedDim                -- 1

order2Dim : ℕ
order2Dim = embedDim ℕ* embedDim    -- 1

stateDim : ℕ
stateDim = order0Dim + order1Dim + order2Dim   -- 3

-- Parameter counts (d=1):
--   E:      alphaSize * embedDim        = 27   (embedding matrix)
--   W_state: alphaSize * stateDim       = 81   (state → logits)
--   W_prev: alphaSize * embedDim        = 27   (prevEmb → logits)
--   b:      alphaSize                   = 27   (output bias)
--   Total:                                162

nEmbed : ℕ
nEmbed = alphaSize ℕ* embedDim         -- 54

nWstate : ℕ
nWstate = alphaSize ℕ* stateDim        -- 189

nWprev : ℕ
nWprev = alphaSize ℕ* embedDim         -- 54

nBias : ℕ
nBias = alphaSize                       -- 27

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
-- Layout: [E | W_state | W_prev | b]

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
-- State is a flat list of stateDim floats:
--   [s₀ | s₁₀..s₁₂ | s₂₀₀..s₂₂₂]

initState : List Float
initState = zerosL stateDim

initPrevEmb : List Float
initPrevEmb = zerosL embedDim

-- Outer product: u ⊗ v as flat list
outerProduct : List Float → List Float → List Float
outerProduct []       _ = []
outerProduct (u ∷ us) v = scaleList u v L++ outerProduct us v

-- State update:
--   s₀' = s₀ + 1, s₁' = s₁ + curEmb, s₂' = s₂ + outer(prevEmb, curEmb)
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

-- Accumulate state over history
accumState : List Float → List Float → List Float → List Char → List Float
accumState θ prevEmb state []       = state
accumState θ prevEmb state (c ∷ cs) =
  let curEmb   = getEmbed θ (charToIdx c)
      newState = updateState prevEmb curEmb state
  in accumState θ curEmb newState cs

-- Get last embedding from history
getLastEmb : List Float → List Char → List Float
getLastEmb θ []       = initPrevEmb
getLastEmb θ (c ∷ []) = getEmbed θ (charToIdx c)
getLastEmb θ (_ ∷ cs) = getLastEmb θ cs

-- Softmax
softmaxF : List Float → List Float
softmaxF logits =
  let exps  = map (e^_) logits
      total = sumF exps
  in map (Float._÷ total) exps

-- Normalize state: divide order1 and order2 by count to keep bounded.
-- Order 0 (count) is kept as-is; order1 → average embedding;
-- order2 → average outer product. If count = 0, return zeros.
normalizeState : List Float → List Float
normalizeState state =
  let count = lookupF state 0
      s1    = sliceF state order0Dim order1Dim
      s2    = sliceF state (order0Dim + order1Dim) order2Dim
      safe  = if count <ᵇ 0.5 then 1.0 else count
      norm1 = scaleList (1.0 Float.÷ safe) s1
      norm2 = scaleList (1.0 Float.÷ safe) s2
  in (1.0 ∷ []) L++ norm1 L++ norm2

-- Output: logits = W_state · normalizedState + W_prev · prevEmb + b
-- Then softmax for probabilities.
-- "prevEmb" here is the embedding of the last character seen.
computeLogits : List Float → List Float → List Float → List Float
computeLogits θ state prevEmb =
  let nstate = normalizeState state
  in map (λ k → dotF (getWsRow θ k) nstate
              Float.+ dotF (getWpRow θ k) prevEmb
              Float.+ getBias θ k)
      (range alphaSize)

outputProbs : List Float → List Float → List Float → List Float
outputProbs θ state prevEmb = softmaxF (computeLogits θ state prevEmb)

-- Full predictor: history → char → probability
tensorPredictor : List Float → List Char → Char → Float
tensorPredictor θ history c =
  let state   = accumState θ initPrevEmb initState history
      prevEmb = getLastEmb θ history
      probs   = outputProbs θ state prevEmb
  in lookupF probs (charToIdx c)

-- ═══════════════════════════════════════════════════════════
-- SECTION 7: EFFICIENT SCORING
-- ═══════════════════════════════════════════════════════════
-- P(c | history) is computed from (state, prevEmb) BEFORE seeing c.
-- Then state is updated to include c for the next prediction.

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
--
-- At each position, the forward computation is:
--   logits_k = W_state[k] · state + W_prev[k] · prevEmb + b[k]
--   probs = softmax(logits)
--   loss = log probs[target]
--
-- The gradient ∂(log P_target)/∂logits_k = δ_{k,target} - P_k
--
-- Parameters with gradients at this position:
--   dW_state[k] = dLogits[k] * state    (outer product)
--   dW_prev[k]  = dLogits[k] * prevEmb  (outer product)
--   db[k]       = dLogits[k]
--   dPrevEmb    = Σ_k dLogits[k] * W_prev[k]  (for embedding gradient)
--
-- The prevEmb is the embedding of the PREVIOUS character, so dPrevEmb
-- contributes to the embedding gradient of that previous char.

-- Scatter an embedding gradient to the right character slot
embedGradForChar : ℕ → ℕ → List Float → List Float
embedGradForChar ci targetCi dEmb =
  if ci ≡ᵇ targetCi then dEmb else zerosL embedDim

-- Compute gradient for one character position
-- prevCharIdx: char index of the previous character (whose embedding = prevEmb)
-- targetIdx: char index of the character being predicted
charGrad : List Float → List Float → List Float → ℕ → ℕ → List Float
charGrad θ prevEmb state prevCharIdx targetIdx =
  let
    -- ═══ FORWARD ═══
    nstate = normalizeState state
    logits = computeLogits θ state prevEmb
    probs  = softmaxF logits

    -- ═══ BACKWARD ═══
    dLogits = map (λ k →
      (if k ≡ᵇ targetIdx then 1.0 else 0.0) Float.- lookupF probs k)
      (range alphaSize)

    -- dPrevEmb = Σ_k dLogits[k] * W_prev[k]
    dPrevEmb = map (λ j →
      sumF (zipWith (λ dk wprow → dk Float.* lookupF wprow j)
                    dLogits
                    (map (getWpRow θ) (range alphaSize))))
      (range embedDim)

    -- ═══ BUILD GRADIENT ═══
    -- dE: scatter dPrevEmb to the previous character's embedding slot
    dE = flattenL (map (λ ci → embedGradForChar ci prevCharIdx dPrevEmb) (range alphaSize))

    -- dW_state[k] = dLogits[k] * normalizedState  (matches forward pass)
    dWs = flattenL (map (λ dl → scaleList dl nstate) dLogits)

    -- dW_prev[k] = dLogits[k] * prevEmb
    dWp = flattenL (map (λ dl → scaleList dl prevEmb) dLogits)

    -- db[k] = dLogits[k]
    db = dLogits

  in dE L++ dWs L++ dWp L++ db

-- Accumulate gradient over the corpus
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
  putStrLn "=== Tensor Algebra Predictor (SMALL: d=1) ==="
  putStrLn "(Novel architecture from truncated tensor algebra T₂(ℝ^1))"
  putStrLn ""

  putStrLn "Architecture:"
  putStrLn ("  Embedding dim d:   " String.++ ℕShow.show embedDim)
  putStrLn ("  State dim:         " String.++ ℕShow.show stateDim
    String.++ " (1 + " String.++ ℕShow.show order1Dim
    String.++ " + " String.++ ℕShow.show order2Dim String.++ ")")
  putStrLn ("  Total params:      " String.++ ℕShow.show totalParams)
  putStrLn ""
  putStrLn "State = truncated tensor algebra T₂(ℝ^d):"
  putStrLn "  Order 0: scalar (char count)"
  putStrLn "  Order 1: ℝ^d (sum of embeddings — unigram features)"
  putStrLn "  Order 2: ℝ^{d×d} (sum of outer products — bigram correlations)"
  putStrLn ""
  putStrLn "Output: W_state · state + W_prev · prevEmb + b → softmax"
  putStrLn "  (state captures accumulated statistics;"
  putStrLn "   prevEmb gives direct bigram context)"
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
  let s₀ = tensorAvgScore θ₀ trainCorpus
  putStrLn ("Initial NLL (50-name train): " String.++ Float.show (0.0 Float.- s₀))
  putStrLn ""

  -- === COMPARISON POINT 1: 50 names, 10 steps ===
  putStrLn "Training on 50 names..."
  let θ₁₀ = train 10 θ₀ trainCorpus
  let s₁₀ = tensorAvgScore θ₁₀ trainCorpus
  putStrLn ("  step 10: train NLL (50 names) = " String.++ Float.show (0.0 Float.- s₁₀))

  -- === COMPARISON POINT 2: 50 names, 25 steps ===
  let θ₂₅ = train 15 θ₁₀ trainCorpus
  let s₂₅ = tensorAvgScore θ₂₅ trainCorpus
  putStrLn ("  step 25: train NLL (50 names) = " String.++ Float.show (0.0 Float.- s₂₅))
  putStrLn ""

  -- Also report on 20-name corpus for cross-comparison
  let s₂₅small = tensorAvgScore θ₂₅ smallCorpus
  putStrLn ("  step 25: train NLL (20 names) = " String.++ Float.show (0.0 Float.- s₂₅small))
  putStrLn ""

  -- Evaluate
  putStrLn ("Evaluating on " String.++ ℕShow.show nEvalNames String.++ " names...")
  let sEval = tensorAvgScore θ₂₅ evalCorpus
  putStrLn ""

  -- Report
  putStrLn "=== COMPARISON RESULTS ==="
  putStrLn ("  [10 steps, 50 names]  Train NLL = " String.++ Float.show (0.0 Float.- s₁₀))
  putStrLn ("  [25 steps, 50 names]  Train NLL = " String.++ Float.show (0.0 Float.- s₂₅))
  putStrLn ("  [25 steps, 20 names]  Train NLL = " String.++ Float.show (0.0 Float.- s₂₅small))
  putStrLn ("  Eval NLL (" String.++ ℕShow.show nEvalNames String.++ " names):  "
    String.++ Float.show (0.0 Float.- sEval))
  putStrLn ""
  putStrLn "  Benchmarks:"
  putStrLn "    Karpathy bigram (32k):        2.454"
  putStrLn "    Karpathy MLP (32k, part 2):   ~2.3"
  putStrLn "    Uniform random:               3.296"
  putStrLn ""

  -- Generate
  let p = tensorPredictor θ₂₅
  putStrLn "Generated names (greedy decoding):"
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".")))
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".a")))
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".s")))
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".m")))
  putStrLn ""
  putStrLn "Done."
