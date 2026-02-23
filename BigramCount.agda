{-# OPTIONS --guardedness #-}
-- ════════════════════════════════════════════════════════════
-- COUNT-BASED BIGRAM: Fast, Correct, Incrementally Scalable
-- ════════════════════════════════════════════════════════════
--
-- A minimal count-based bigram that:
--   1. Compiles fast (< 30 seconds)
--   2. Uses MLE (the closed-form optimum for the bigram family)
--   3. Matches Karpathy's makemore approach exactly
--   4. Reports scores in the same format for direct comparison
--
-- Target: Karpathy's bigram achieves NLL = 2.454 on 32k names.
-- Our score = -NLL, so we target avgScore ≈ -2.454.

module BigramCount where

open import IO
open import Data.String.Base as String using (String; toList; fromList; _++_)
open import Data.List.Base as List using (List; []; _∷_; length; map; take; drop;
  foldr; concatMap; upTo)
  renaming (_++_ to _L++_)
open import Data.Char.Base using (Char; toℕ; fromℕ)
open import Data.Float.Base as Float using (Float; log; show; _<ᵇ_)
open import Data.Nat.Base as Nat using (ℕ; suc; zero; _≡ᵇ_; _≤ᵇ_; _∸_; _*_; _+_)
open import Data.Nat.Show as ℕShow using ()
open import Data.Bool.Base using (Bool; true; false; if_then_else_)

-- ═══ ALPHABET: 27 chars (. + a-z) ═══

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

-- ═══ FLAT TABLE OPERATIONS ═══

lookupF : List Float → ℕ → Float
lookupF []       _       = 0.0
lookupF (x ∷ _)  zero    = x
lookupF (_ ∷ xs) (suc n) = lookupF xs n

replaceAt : ℕ → Float → List Float → List Float
replaceAt _       _ []       = []
replaceAt zero    v (_ ∷ xs) = v ∷ xs
replaceAt (suc n) v (x ∷ xs) = x ∷ replaceAt n v xs

initTable : Float → List Float
initTable α = List.replicate (alphaSize * alphaSize) α

-- ═══ COUNT BIGRAMS ═══

incAt : ℕ → ℕ → List Float → List Float
incAt prev next tbl =
  let idx = prev * alphaSize + next
  in replaceAt idx (lookupF tbl idx Float.+ 1.0) tbl

countBigrams : List Char → List Float → List Float
countBigrams []           tbl = tbl
countBigrams (_ ∷ [])     tbl = tbl
countBigrams (a ∷ b ∷ cs) tbl =
  countBigrams (b ∷ cs) (incAt (charToIdx a) (charToIdx b) tbl)

-- ═══ NORMALIZE ═══

sumFloats : List Float → Float
sumFloats []       = 0.0
sumFloats (x ∷ xs) = x Float.+ sumFloats xs

normalize : List Float → List Float
normalize tbl = concatMap normalizeRow (upTo alphaSize)
  where
    getRow : ℕ → List Float
    getRow i = take alphaSize (drop (i * alphaSize) tbl)
    normalizeRow : ℕ → List Float
    normalizeRow i =
      let row = getRow i
          total = sumFloats row
      in map (Float._÷ total) row

-- ═══ PREDICTOR FROM TABLE ═══

lastCharIdx : List Char → ℕ
lastCharIdx []       = 0
lastCharIdx (x ∷ []) = charToIdx x
lastCharIdx (_ ∷ xs) = lastCharIdx xs

predict : List Float → List Char → Char → Float
predict probs history c =
  lookupF probs (lastCharIdx history * alphaSize + charToIdx c)

-- ═══ SCORING (matches Spec.agda definition exactly) ═══

scoreFrom : (List Char → Char → Float) → List Char → List Char → Float
scoreFrom p h []       = 0.0
scoreFrom p h (c ∷ cs) =
  log (p h c) Float.+ scoreFrom p (h L++ (c ∷ [])) cs

score : (List Char → Char → Float) → List Char → Float
score p corpus = scoreFrom p [] corpus

avgScore : (List Char → Char → Float) → List Char → Float
avgScore p corpus = score p corpus Float.÷ Float.fromℕ (length corpus)

-- ═══ NAME GENERATION ═══

alpha : List Char
alpha = '.' ∷ map (λ n → fromℕ (97 + n)) (upTo 26)

argmax : (List Char → Char → Float) → List Char → List Char → Char → Float → Char
argmax _ _  []       best _     = best
argmax p h  (c ∷ cs) best bestP =
  let prob = p h c in
  if bestP <ᵇ prob
    then argmax p h cs c    prob
    else argmax p h cs best bestP

generate : (List Char → Char → Float) → ℕ → List Char → List Char
generate _ zero    _  = []
generate p (suc n) h  =
  let next = argmax p h alpha '.' 0.0
  in if toℕ next ≡ᵇ 46 then []
     else next ∷ generate p n (h L++ (next ∷ []))

-- ═══ BUILD MODEL ═══

buildBigram : List Char → List Float
buildBigram corpus = normalize (countBigrams corpus (initTable 1.0))

-- ═══ CORPUS ═══
-- Start with 50 names, increase later

corpus50 : List Char
corpus50 = toList ".emma.olivia.ava.isabella.sophia.charlotte.mia.amelia.harper.evelyn.abigail.emily.elizabeth.mila.ella.avery.sofia.camila.aria.scarlett.victoria.madison.luna.grace.chloe.penelope.layla.riley.zoey.nora.lily.eleanor.hannah.lillian.addison.aubrey.ellie.stella.natalie.zoe.leah.hazel.violet.aurora.savannah.audrey.brooklyn.bella.claire.skylar."

showBool : Bool → String
showBool true  = "true"
showBool false = "false"

-- ═══ MAIN ═══

main : Main
main = run do
  putStrLn "=== Count-Based Bigram (MLE) ==="
  putStrLn ""

  -- Karpathy's target
  putStrLn "Karpathy's bigram on 32k names: NLL = 2.454"
  putStrLn "Uniform random baseline:        NLL = 3.296 (log 27)"
  putStrLn ""

  let corpus = corpus50
  putStrLn ("Corpus: 50 names, " String.++ ℕShow.show (length corpus) String.++ " chars")

  -- Uniform baseline
  let uniform = λ (_ : List Char) (_ : Char) → 1.0 Float.÷ Float.fromℕ alphaSize
  let uScore = avgScore uniform corpus
  putStrLn ("Uniform:  " String.++ Float.show uScore String.++ " avg log-prob/char"
    String.++ " (NLL = " String.++ Float.show (0.0 Float.- uScore) String.++ ")")

  -- Count-based bigram
  let probs = buildBigram corpus
  let p = predict probs
  let bScore = avgScore p corpus
  putStrLn ("Bigram:   " String.++ Float.show bScore String.++ " avg log-prob/char"
    String.++ " (NLL = " String.++ Float.show (0.0 Float.- bScore) String.++ ")")
  putStrLn ("Beats uniform: " String.++ showBool (uScore <ᵇ bScore))
  putStrLn ""

  -- Some learned probabilities
  putStrLn "Learned P(next | prev):"
  putStrLn ("  P(a|.) = " String.++ Float.show (p ('.' ∷ []) 'a')
    String.++ "  P(e|.) = " String.++ Float.show (p ('.' ∷ []) 'e')
    String.++ "  P(s|.) = " String.++ Float.show (p ('.' ∷ []) 's'))
  putStrLn ("  P(.|a) = " String.++ Float.show (p ('a' ∷ []) '.')
    String.++ "  P(n|a) = " String.++ Float.show (p ('a' ∷ []) 'n')
    String.++ "  P(r|a) = " String.++ Float.show (p ('a' ∷ []) 'r'))
  putStrLn ""

  -- Generated names
  putStrLn "Generated (greedy):"
  putStrLn ("  " String.++ fromList (generate p 20 ('.' ∷ [])))
  putStrLn ("  " String.++ fromList (generate p 20 ('.' ∷ 'a' ∷ [])))
  putStrLn ("  " String.++ fromList (generate p 20 ('.' ∷ 's' ∷ [])))
  putStrLn ("  " String.++ fromList (generate p 20 ('.' ∷ 'k' ∷ [])))
