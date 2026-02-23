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
open import Data.String.Base as String using (String; toList; fromList; _++_; lines)
open import Data.List.Base as List using (List; []; _∷_; length; map; take; drop;
  foldr; concatMap; upTo)
  renaming (_++_ to _L++_)
open import Data.Char.Base using (Char; toℕ; fromℕ)
open import Data.Float.Base as Float using (Float; log; show; _<ᵇ_)
open import Data.Nat.Base as Nat using (ℕ; suc; zero; _≡ᵇ_; _≤ᵇ_; _∸_; _*_; _+_)
open import Data.Nat.Show as ℕShow using ()
open import Data.Bool.Base using (Bool; true; false; if_then_else_; not)
open import Level using (0ℓ)
open import Data.Unit.Polymorphic.Base using (⊤)

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

-- ═══ SCORING ═══

-- General scoring (matches Spec.agda definition exactly)
scoreFrom : (List Char → Char → Float) → List Char → List Char → Float
scoreFrom p h []       = 0.0
scoreFrom p h (c ∷ cs) =
  log (p h c) Float.+ scoreFrom p (h L++ (c ∷ [])) cs

score : (List Char → Char → Float) → List Char → Float
score p corpus = scoreFrom p [] corpus

avgScore : (List Char → Char → Float) → List Char → Float
avgScore p corpus = score p corpus Float.÷ Float.fromℕ (length corpus)

-- Optimized bigram scoring: only track previous character (O(n) instead of O(n²))
-- Equivalent to scoreFrom (predict probs) for bigrams, but avoids O(n²) history.
bigramScoreFrom : List Float → Char → List Char → Float
bigramScoreFrom _     _ []       = 0.0
bigramScoreFrom probs prev (c ∷ cs) =
  log (lookupF probs (charToIdx prev * alphaSize + charToIdx c))
  Float.+ bigramScoreFrom probs c cs

-- Start with '.' as initial context (matching lastCharIdx [] = 0 = charToIdx '.')
bigramScore : List Float → List Char → Float
bigramScore probs corpus = bigramScoreFrom probs '.' corpus

bigramAvgScore : List Float → List Char → Float
bigramAvgScore probs corpus = bigramScore probs corpus Float.÷ Float.fromℕ (length corpus)

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

-- ═══ CORPUS BUILDING ═══

-- Filter a list by a boolean predicate
filterBool : (String → Bool) → List String → List String
filterBool _ []       = []
filterBool f (x ∷ xs) = if f x then x ∷ filterBool f xs else filterBool f xs

-- Check if a string is non-empty
isNonEmpty : String → Bool
isNonEmpty s with toList s
... | []    = false
... | _ ∷ _ = true

-- Build ".name1.name2.name3." corpus from a list of names
-- Each name is delimited by '.' on both sides (matching Karpathy's format)
buildCorpus : List String → List Char
buildCorpus names = toList (foldr (λ name acc → "." String.++ name String.++ acc) "." names)

-- Hardcoded test corpus (50 names) — kept as fallback/test
corpus50 : List Char
corpus50 = toList ".emma.olivia.ava.isabella.sophia.charlotte.mia.amelia.harper.evelyn.abigail.emily.elizabeth.mila.ella.avery.sofia.camila.aria.scarlett.victoria.madison.luna.grace.chloe.penelope.layla.riley.zoey.nora.lily.eleanor.hannah.lillian.addison.aubrey.ellie.stella.natalie.zoe.leah.hazel.violet.aurora.savannah.audrey.brooklyn.bella.claire.skylar."

showBool : Bool → String
showBool true  = "true"
showBool false = "false"

-- ═══ REPORTING ═══

-- Run scoring and print results for a given corpus
reportModel : List Float → List Char → IO {0ℓ} ⊤
reportModel probs corpus = do
  let p = predict probs
  let uNLL = 0.0 Float.- (log (1.0 Float.÷ Float.fromℕ alphaSize))
  let bScore = bigramAvgScore probs corpus
  putStrLn ("Uniform:  NLL = " String.++ Float.show uNLL
    String.++ " (log 27)")
  putStrLn ("Bigram:   " String.++ Float.show bScore String.++ " avg log-prob/char"
    String.++ " (NLL = " String.++ Float.show (0.0 Float.- bScore) String.++ ")")
  putStrLn ""
  putStrLn "Learned P(next | prev):"
  putStrLn ("  P(a|.) = " String.++ Float.show (p ('.' ∷ []) 'a')
    String.++ "  P(e|.) = " String.++ Float.show (p ('.' ∷ []) 'e')
    String.++ "  P(s|.) = " String.++ Float.show (p ('.' ∷ []) 's'))
  putStrLn ("  P(.|a) = " String.++ Float.show (p ('a' ∷ []) '.')
    String.++ "  P(n|a) = " String.++ Float.show (p ('a' ∷ []) 'n')
    String.++ "  P(r|a) = " String.++ Float.show (p ('a' ∷ []) 'r'))
  putStrLn ""
  putStrLn "Generated (greedy):"
  putStrLn ("  " String.++ fromList (generate p 20 ('.' ∷ [])))
  putStrLn ("  " String.++ fromList (generate p 20 ('.' ∷ 'a' ∷ [])))
  putStrLn ("  " String.++ fromList (generate p 20 ('.' ∷ 's' ∷ [])))
  putStrLn ("  " String.++ fromList (generate p 20 ('.' ∷ 'k' ∷ [])))

-- ═══ MAIN ═══

main : Main
main = run do
  putStrLn "=== Count-Based Bigram (MLE) ==="
  putStrLn ""

  -- Karpathy's target
  putStrLn "Karpathy's bigram on 32k names: NLL = 2.454"
  putStrLn "Uniform random baseline:        NLL = 3.296 (log 27)"
  putStrLn ""

  -- Read full corpus from names.txt
  putStrLn "Reading names.txt..."
  contents ← readFiniteFile "names.txt"
  let nameList = filterBool isNonEmpty (lines contents)
  let nNames = length nameList
  let corpus = buildCorpus nameList
  let nChars = length corpus
  putStrLn ("Corpus: " String.++ ℕShow.show nNames String.++ " names, "
    String.++ ℕShow.show nChars String.++ " chars")
  putStrLn ""

  let probs = buildBigram corpus
  reportModel probs corpus
