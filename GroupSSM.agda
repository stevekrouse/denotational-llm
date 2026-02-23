{-# OPTIONS --guardedness #-}
-- ════════════════════════════════════════════════════════════
-- GROUP ALGEBRA SSM: Non-Abelian State Space Model via R[S₃]
-- ════════════════════════════════════════════════════════════
--
-- THE IDEA: Use the group algebra R[S₃] as the state monoid.
-- S₃ (symmetric group on 3 elements) has order 6.
-- By Wedderburn's theorem, R[S₃] ≅ R × R × M₂(R),
-- so the 6-dimensional group algebra decomposes into:
--   - Two 1-dimensional blocks (trivial and sign representations)
--   - One 2×2 block (standard representation)
--
-- The 2×2 block is the key: it provides non-abelian structure
-- that diagonal SSMs cannot express.
--
-- ARCHITECTURE:
--   1. Each character c maps to a 6-dim vector (element of R[S₃])
--   2. State is 6-dim, initialized to identity element [1,0,0,0,0,0]
--   3. State update: state' = state * embed(c) via group algebra mult
--   4. Output: W · state + b → softmax over 27 chars
--   5. Train with explicit backprop
--
-- Also includes a DIAGONAL BASELINE of the same dimension (6-dim
-- componentwise multiplication) to test whether non-abelian structure
-- actually helps.

module GroupSSM where

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

-- |S₃| = 6, so state dimension is 6
groupDim : ℕ
groupDim = 6

-- Parameters:
--   Character embeddings: 27 × 6 = 162
--   Output weights:       27 × 6 = 162
--   Output bias:          27
--   Total:                351
nEmbed : ℕ
nEmbed = alphaSize ℕ* groupDim      -- 162

nWout : ℕ
nWout = alphaSize ℕ* groupDim       -- 162

nBias : ℕ
nBias = alphaSize                    -- 27

totalParams : ℕ
totalParams = nEmbed + nWout + nBias -- 351

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
-- Layout: [Embed | W_out | bias]

offE : ℕ
offE = 0

offW : ℕ
offW = nEmbed

offB : ℕ
offB = offW + nWout

getEmbed : List Float → ℕ → List Float
getEmbed θ i = sliceF θ (offE + i ℕ* groupDim) groupDim

getWoutRow : List Float → ℕ → List Float
getWoutRow θ k = sliceF θ (offW + k ℕ* groupDim) groupDim

getBias : List Float → ℕ → Float
getBias θ k = lookupF (drop offB θ) k

-- ═══════════════════════════════════════════════════════════
-- SECTION 5: S₃ GROUP ALGEBRA MULTIPLICATION
-- ═══════════════════════════════════════════════════════════
--
-- S₃ elements indexed 0..5:
--   0 = e       (identity)
--   1 = (12)    (swap 1,2)
--   2 = (13)    (swap 1,3)
--   3 = (23)    (swap 2,3)
--   4 = (123)   (cycle 1→2→3→1)
--   5 = (132)   (cycle 1→3→2→1)
--
-- Multiplication table (row * col):
--       e    (12)  (13)  (23)  (123) (132)
-- e     e    (12)  (13)  (23)  (123) (132)
-- (12)  (12)  e    (123) (132) (13)  (23)
-- (13)  (13)  (132) e    (123) (23)  (12)
-- (23)  (23)  (123) (132) e    (12)  (13)
-- (123) (123) (23)  (12)  (13)  (132) e
-- (132) (132) (13)  (23)  (12)  e    (123)
--
-- Stored as: mulTable[i][j] = index of g_i * g_j

-- S₃ multiplication table: mulT i j = index of (element i * element j)
mulT : ℕ → ℕ → ℕ
-- Row 0: e * x = x
mulT 0 j = j
-- Row 1: (12) * ...
mulT 1 0 = 1
mulT 1 1 = 0
mulT 1 2 = 4
mulT 1 3 = 5
mulT 1 4 = 2
mulT 1 5 = 3
-- Row 2: (13) * ...
mulT 2 0 = 2
mulT 2 1 = 5
mulT 2 2 = 0
mulT 2 3 = 4
mulT 2 4 = 3
mulT 2 5 = 1
-- Row 3: (23) * ...
mulT 3 0 = 3
mulT 3 1 = 4
mulT 3 2 = 5
mulT 3 3 = 0
mulT 3 4 = 1
mulT 3 5 = 2
-- Row 4: (123) * ...
mulT 4 0 = 4
mulT 4 1 = 2
mulT 4 2 = 3
mulT 4 3 = 1
mulT 4 4 = 5
mulT 4 5 = 0
-- Row 5: (132) * ...
mulT 5 0 = 5
mulT 5 1 = 3
mulT 5 2 = 1
mulT 5 3 = 2
mulT 5 4 = 0
mulT 5 5 = 4
-- Fallback (should never reach)
mulT _ _ = 0

-- Group algebra multiplication:
-- (a * b)_k = Σ_{i,j : mulT(i,j) = k} a_i * b_j
--
-- For each output position k, we sum over all pairs (i,j) that
-- multiply to give element k.  Since there are only 36 pairs total
-- and the table is fixed, we compute each component directly.
--
-- Helper: a_i * b_j
_·_ : Float → Float → Float
x · y = x Float.* y

accumProd : List Float → List Float → List Float
accumProd a b =
  let a0 = lookupF a 0
      a1 = lookupF a 1
      a2 = lookupF a 2
      a3 = lookupF a 3
      a4 = lookupF a 4
      a5 = lookupF a 5
      b0 = lookupF b 0
      b1 = lookupF b 1
      b2 = lookupF b 2
      b3 = lookupF b 3
      b4 = lookupF b 4
      b5 = lookupF b 5
      -- result_k = Σ over (i,j) with mulT(i,j) = k of a_i * b_j
      -- k=0: (0,0),(1,1),(2,2),(3,3),(4,5),(5,4)
      r0 = (a0 · b0) Float.+ (a1 · b1) Float.+ (a2 · b2)
           Float.+ (a3 · b3) Float.+ (a4 · b5) Float.+ (a5 · b4)
      -- k=1: (0,1),(1,0),(2,5),(3,4),(4,3),(5,2)
      r1 = (a0 · b1) Float.+ (a1 · b0) Float.+ (a2 · b5)
           Float.+ (a3 · b4) Float.+ (a4 · b3) Float.+ (a5 · b2)
      -- k=2: (0,2),(1,4),(2,0),(3,5),(4,1),(5,3)
      r2 = (a0 · b2) Float.+ (a1 · b4) Float.+ (a2 · b0)
           Float.+ (a3 · b5) Float.+ (a4 · b1) Float.+ (a5 · b3)
      -- k=3: (0,3),(1,5),(2,4),(3,0),(4,2),(5,1)
      r3 = (a0 · b3) Float.+ (a1 · b5) Float.+ (a2 · b4)
           Float.+ (a3 · b0) Float.+ (a4 · b2) Float.+ (a5 · b1)
      -- k=4: (0,4),(1,2),(2,3),(3,1),(4,0),(5,5)
      r4 = (a0 · b4) Float.+ (a1 · b2) Float.+ (a2 · b3)
           Float.+ (a3 · b1) Float.+ (a4 · b0) Float.+ (a5 · b5)
      -- k=5: (0,5),(1,3),(2,1),(3,2),(4,4),(5,0)
      r5 = (a0 · b5) Float.+ (a1 · b3) Float.+ (a2 · b1)
           Float.+ (a3 · b2) Float.+ (a4 · b4) Float.+ (a5 · b0)
  in r0 ∷ r1 ∷ r2 ∷ r3 ∷ r4 ∷ r5 ∷ []

-- Identity element in R[S₃]: e = [1, 0, 0, 0, 0, 0]
identityS3 : List Float
identityS3 = 1.0 ∷ 0.0 ∷ 0.0 ∷ 0.0 ∷ 0.0 ∷ 0.0 ∷ []

-- ═══════════════════════════════════════════════════════════
-- SECTION 6: DIAGONAL BASELINE
-- ═══════════════════════════════════════════════════════════
-- For comparison: 6-dim componentwise multiplication (abelian monoid).
-- This is equivalent to Z/2Z × Z/3Z or any abelian group of order 6.

diagMul : List Float → List Float → List Float
diagMul [] _ = []
diagMul _ [] = []
diagMul (a ∷ as) (b ∷ bs) = (a Float.* b) ∷ diagMul as bs

identityDiag : List Float
identityDiag = 1.0 ∷ 1.0 ∷ 1.0 ∷ 1.0 ∷ 1.0 ∷ 1.0 ∷ []

-- ═══════════════════════════════════════════════════════════
-- SECTION 7: SOFTMAX AND FORWARD PASS
-- ═══════════════════════════════════════════════════════════

softmaxF : List Float → List Float
softmaxF logits =
  let exps  = map (e^_) logits
      total = sumF exps
  in map (Float._÷ total) exps

-- Forward: compute logits from state
computeLogits : List Float → List Float → List Float
computeLogits θ state =
  map (λ k → dotF (getWoutRow θ k) state Float.+ getBias θ k) (range alphaSize)

outputProbs : List Float → List Float → List Float
outputProbs θ state = softmaxF (computeLogits θ state)

-- ═══════════════════════════════════════════════════════════
-- SECTION 8: GROUP SSM SCORING (efficient sequential)
-- ═══════════════════════════════════════════════════════════

groupScoreFrom : List Float → List Float → List Char → Float
groupScoreFrom θ state []       = 0.0
groupScoreFrom θ state (c ∷ cs) =
  let probs    = outputProbs θ state
      p        = lookupF probs (charToIdx c)
      logp     = log p
      curEmb   = getEmbed θ (charToIdx c)
      newState = accumProd state curEmb
  in logp Float.+ groupScoreFrom θ newState cs

groupScore : List Float → List Char → Float
groupScore θ corpus = groupScoreFrom θ identityS3 corpus

groupAvgScore : List Float → List Char → Float
groupAvgScore θ corpus = groupScore θ corpus Float.÷ Float.fromℕ (length corpus)

-- Diagonal SSM scoring (same structure, componentwise multiply)
diagScoreFrom : List Float → List Float → List Char → Float
diagScoreFrom θ state []       = 0.0
diagScoreFrom θ state (c ∷ cs) =
  let probs    = outputProbs θ state
      p        = lookupF probs (charToIdx c)
      logp     = log p
      curEmb   = getEmbed θ (charToIdx c)
      newState = diagMul state curEmb
  in logp Float.+ diagScoreFrom θ newState cs

diagScore : List Float → List Char → Float
diagScore θ corpus = diagScoreFrom θ identityDiag corpus

diagAvgScore : List Float → List Char → Float
diagAvgScore θ corpus = diagScore θ corpus Float.÷ Float.fromℕ (length corpus)

-- ═══════════════════════════════════════════════════════════
-- SECTION 9: BACKPROPAGATION — GROUP SSM
-- ═══════════════════════════════════════════════════════════
--
-- At each position t:
--   logits_k = W[k] · state + b[k]
--   probs = softmax(logits)
--   loss += log probs[target]
--
-- Gradient of log P(target) w.r.t. logits:
--   dLogits[k] = delta(k, target) - P[k]
--
-- Gradient w.r.t. parameters at this position:
--   dW[k]  = dLogits[k] * state      (outer product, contributes to W section)
--   db[k]  = dLogits[k]               (contributes to bias section)
--   dState = Σ_k dLogits[k] * W[k]   (for backprop through state)
--
-- For the embedding of char c at this position:
--   dEmbed(c) += (∂state'/∂embed(c))ᵀ · dState_next
--   But tracking dState through the group algebra multiply is complex.
--   We simplify: only backprop through the OUTPUT, not through the
--   state recurrence. This is "truncated BPTT at depth 1" — each
--   character's embedding only gets gradient from the NEXT position.
--
-- Simpler approach: at each position, the embedding of the current char c
-- contributes via state' = state * embed(c). The gradient of the NEXT
-- position's log-prob w.r.t. embed(c) flows through the group algebra
-- multiplication.
--
-- EVEN SIMPLER: For a first pass, we backprop each position independently.
-- dEmbed(c) gets gradient only from how embed(c) affects state' and thus
-- the next prediction. This is exact for the last layer (W, b) and
-- approximate for embeddings (missing long-range dependencies through state).

-- Embed gradient for char: scatter dEmb to the right char slot
embedGradForChar : ℕ → ℕ → List Float → List Float
embedGradForChar ci targetCi dEmb =
  if ci ≡ᵇ targetCi then dEmb else zerosL groupDim

-- Gradient of group algebra multiplication w.r.t. second argument:
-- result_k = Σ_{i,j: g_i*g_j = g_k} a_i * b_j
-- ∂result_k / ∂b_j = Σ_{i: g_i*g_j = g_k} a_i
-- So dB_j = Σ_k dResult_k * (Σ_{i: mulT(i,j) = k} a_i)
--
-- For each j, we need: Σ_k dR_k * Σ_{i: mulT(i,j)=k} a_i
--                     = Σ_i a_i * dR_{mulT(i,j)}
groupMulGradB : List Float → List Float → List Float
groupMulGradB a dResult =
  map (λ j →
    sumF (map (λ i → lookupF a i Float.* lookupF dResult (mulT i j)) (range groupDim))
  ) (range groupDim)

-- Full per-position gradient for GROUP SSM
groupCharGrad : List Float → List Float → ℕ → ℕ → List Float
groupCharGrad θ state prevCharIdx targetIdx =
  let
    -- Forward
    logits = computeLogits θ state
    probs  = softmaxF logits

    -- dLogits
    dLogits = map (λ k →
      (if k ≡ᵇ targetIdx then 1.0 else 0.0) Float.- lookupF probs k)
      (range alphaSize)

    -- dState = Σ_k dLogits[k] * W[k]
    dState = map (λ j →
      sumF (zipWith (λ dk wrow → dk Float.* lookupF wrow j)
                    dLogits
                    (map (getWoutRow θ) (range alphaSize))))
      (range groupDim)

    -- dEmb for previous char: gradient through group mul
    -- state' = state * embed(prevChar), so dEmbed = groupMulGradB(state, dState)
    dEmbPrev = groupMulGradB state dState

    -- Scatter to embedding section
    dE = flattenL (map (λ ci → embedGradForChar ci prevCharIdx dEmbPrev) (range alphaSize))

    -- dW[k] = dLogits[k] * state
    dW = flattenL (map (λ dl → scaleList dl state) dLogits)

    -- db = dLogits
    db = dLogits

  in dE L++ dW L++ db

-- Accumulate gradient over corpus
groupGradAccum : List Float → List Float → ℕ → List Char → List Float → List Float
groupGradAccum θ state prevCIdx []       acc = acc
groupGradAccum θ state prevCIdx (c ∷ cs) acc =
  let cIdx     = charToIdx c
      g        = groupCharGrad θ state prevCIdx cIdx
      newAcc   = addList acc g
      curEmb   = getEmbed θ cIdx
      newState = accumProd state curEmb
  in groupGradAccum θ newState cIdx cs newAcc

groupGradient : List Float → List Char → List Float
groupGradient θ corpus = groupGradAccum θ identityS3 0 corpus (zerosL totalParams)

-- ═══════════════════════════════════════════════════════════
-- SECTION 10: BACKPROPAGATION — DIAGONAL BASELINE
-- ═══════════════════════════════════════════════════════════
-- For diagonal: state' = state ⊙ embed(c)
-- ∂state'_j / ∂embed(c)_j = state_j
-- So dEmb_j = dState_j * state_j (componentwise)

diagCharGrad : List Float → List Float → ℕ → ℕ → List Float
diagCharGrad θ state prevCharIdx targetIdx =
  let
    logits = computeLogits θ state
    probs  = softmaxF logits

    dLogits = map (λ k →
      (if k ≡ᵇ targetIdx then 1.0 else 0.0) Float.- lookupF probs k)
      (range alphaSize)

    dState = map (λ j →
      sumF (zipWith (λ dk wrow → dk Float.* lookupF wrow j)
                    dLogits
                    (map (getWoutRow θ) (range alphaSize))))
      (range groupDim)

    -- For diagonal: dEmb_j = dState_j * state_j
    dEmbPrev = zipWith Float._*_ dState state

    dE = flattenL (map (λ ci → embedGradForChar ci prevCharIdx dEmbPrev) (range alphaSize))
    dW = flattenL (map (λ dl → scaleList dl state) dLogits)
    db = dLogits

  in dE L++ dW L++ db

diagGradAccum : List Float → List Float → ℕ → List Char → List Float → List Float
diagGradAccum θ state prevCIdx []       acc = acc
diagGradAccum θ state prevCIdx (c ∷ cs) acc =
  let cIdx     = charToIdx c
      g        = diagCharGrad θ state prevCIdx cIdx
      newAcc   = addList acc g
      curEmb   = getEmbed θ cIdx
      newState = diagMul state curEmb
  in diagGradAccum θ newState cIdx cs newAcc

diagGradient : List Float → List Char → List Float
diagGradient θ corpus = diagGradAccum θ identityDiag 0 corpus (zerosL totalParams)

-- ═══════════════════════════════════════════════════════════
-- SECTION 11: GRADIENT ASCENT
-- ═══════════════════════════════════════════════════════════

learningRate : Float
learningRate = 5.0

groupGradStep : List Float → List Char → List Float
groupGradStep θ corpus =
  let grad = groupGradient θ corpus
      n    = Float.fromℕ (length corpus)
  in zipWith (λ t g → t Float.+ ((learningRate Float.÷ n) Float.* g)) θ grad

groupTrain : ℕ → List Float → List Char → List Float
groupTrain zero    θ _      = θ
groupTrain (suc n) θ corpus = groupTrain n (groupGradStep θ corpus) corpus

diagGradStep : List Float → List Char → List Float
diagGradStep θ corpus =
  let grad = diagGradient θ corpus
      n    = Float.fromℕ (length corpus)
  in zipWith (λ t g → t Float.+ ((learningRate Float.÷ n) Float.* g)) θ grad

diagTrain : ℕ → List Float → List Char → List Float
diagTrain zero    θ _      = θ
diagTrain (suc n) θ corpus = diagTrain n (diagGradStep θ corpus) corpus

-- ═══════════════════════════════════════════════════════════
-- SECTION 12: INITIALIZATION
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
-- SECTION 13: GENERATION
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

-- Full predictor wrappers (for generation — these recompute state from scratch)
groupPredictor : List Float → List Char → Char → Float
groupPredictor θ history c =
  let state = List.foldl (λ s ch → accumProd s (getEmbed θ (charToIdx ch))) identityS3 history
      probs = outputProbs θ state
  in lookupF probs (charToIdx c)

-- ═══════════════════════════════════════════════════════════
-- SECTION 14: CORPUS BUILDING
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

-- ═══════════════════════════════════════════════════════════
-- SECTION 15: MAIN
-- ═══════════════════════════════════════════════════════════

trainNames : ℕ
trainNames = 30

nSteps : ℕ
nSteps = 25

main : Main
main = run do
  putStrLn "=== Group Algebra SSM: R[S₃] vs Diagonal ==="
  putStrLn ""
  putStrLn "Architecture: Non-abelian group algebra state space model"
  putStrLn "  State monoid: R[S₃] (group algebra of symmetric group S₃)"
  putStrLn "  |S₃| = 6, so state dim = 6"
  putStrLn "  Wedderburn decomposition: R[S₃] ≅ R × R × M₂(R)"
  putStrLn "  The 2×2 block captures non-abelian structure"
  putStrLn ("  Total params: " String.++ ℕShow.show totalParams)
  putStrLn ""
  putStrLn "Baseline: Diagonal SSM (componentwise multiply, same 6 dims)"
  putStrLn "  This is an abelian monoid — equivalent to diagonal state matrix"
  putStrLn ""

  -- Read corpus
  contents ← readFiniteFile "names.txt"
  let allNames   = filterBool isNonEmpty (lines contents)
  let trainNameList = take trainNames allNames
  let trainCorpus   = buildCorpus trainNameList
  let nChars        = length trainCorpus
  putStrLn ("Training: " String.++ ℕShow.show trainNames String.++ " names, "
    String.++ ℕShow.show nChars String.++ " chars")
  putStrLn ("Steps:    " String.++ ℕShow.show nSteps)
  putStrLn ""

  -- Initialize both models with same params
  let θ₀ = initSmall totalParams

  -- Initial scores
  let gS₀ = groupAvgScore θ₀ trainCorpus
  let dS₀ = diagAvgScore θ₀ trainCorpus
  putStrLn ("Initial NLL — Group: " String.++ Float.show (0.0 Float.- gS₀)
    String.++ "  Diag: " String.++ Float.show (0.0 Float.- dS₀))
  putStrLn ""

  -- Train Group SSM
  putStrLn "Training Group SSM (R[S₃])..."
  let θG₁₀ = groupTrain 10 θ₀ trainCorpus
  let gS₁₀ = groupAvgScore θG₁₀ trainCorpus
  putStrLn ("  step 10: NLL = " String.++ Float.show (0.0 Float.- gS₁₀))

  let θG₂₅ = groupTrain 15 θG₁₀ trainCorpus
  let gS₂₅ = groupAvgScore θG₂₅ trainCorpus
  putStrLn ("  step 25: NLL = " String.++ Float.show (0.0 Float.- gS₂₅))
  putStrLn ""

  -- Train Diagonal SSM
  putStrLn "Training Diagonal SSM (componentwise)..."
  let θD₁₀ = diagTrain 10 θ₀ trainCorpus
  let dS₁₀ = diagAvgScore θD₁₀ trainCorpus
  putStrLn ("  step 10: NLL = " String.++ Float.show (0.0 Float.- dS₁₀))

  let θD₂₅ = diagTrain 15 θD₁₀ trainCorpus
  let dS₂₅ = diagAvgScore θD₂₅ trainCorpus
  putStrLn ("  step 25: NLL = " String.++ Float.show (0.0 Float.- dS₂₅))
  putStrLn ""

  -- Summary
  putStrLn "=== RESULTS ==="
  putStrLn ("  Group R[S₃]:  NLL = " String.++ Float.show (0.0 Float.- gS₂₅))
  putStrLn ("  Diagonal:     NLL = " String.++ Float.show (0.0 Float.- dS₂₅))
  putStrLn ("  Gap (diag - group): " String.++ Float.show ((0.0 Float.- dS₂₅) Float.- (0.0 Float.- gS₂₅)))
  putStrLn ""
  putStrLn "  Benchmarks:"
  putStrLn "    Uniform random:               3.296"
  putStrLn "    Karpathy bigram (32k):        2.454"
  putStrLn ""
  putStrLn "KEY QUESTION: Does non-abelian structure (the 2x2 block"
  putStrLn "from the standard representation of S₃) help compared"
  putStrLn "to a diagonal SSM of the same dimension?"
  putStrLn ""
  putStrLn "INTERPRETATION:"
  putStrLn "  Negative gap = group wins, positive gap = diagonal wins"
  putStrLn ""

  -- Generate from group model
  let p = groupPredictor θG₂₅
  putStrLn "Generated names (Group SSM, greedy):"
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".")))
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".a")))
  putStrLn ("  " String.++ fromList (generateName p 20 (toList ".s")))
  putStrLn ""
  putStrLn "Done."
