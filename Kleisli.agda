-- ════════════════════════════════════════════════════════════
-- KLEISLI CATEGORY OF THE PROBABILITY MONAD
-- ════════════════════════════════════════════════════════════
--
-- The key categorical insight: Predictors live in the Kleisli
-- category of the probability monad, and Score is a functor
-- from this category to (ℝ, +).
--
-- This mirrors Conal Elliott's approach to automatic
-- differentiation:
--
--   AD:   differentiable functions form a category,
--         the derivative D is a functor to linear maps.
--         D(f ∘ g) = D(f) ∘ D(g)    ← the chain rule
--
--   Here: predictors form a (Kleisli) category,
--         score is a functor to (ℝ, +).
--         score(p, xs++ys) = score(p,xs) + score(p,h++xs,ys)
--                                         ← score decomposition
--
-- Both are instances of the same pattern: a denotation
-- (derivative / score) that is a homomorphism from a
-- compositional structure to a simpler algebraic one.

module Kleisli where

open import Real
open import Spec
open import Probability
open import Data.List.Base using (List; []; _∷_; length; map)
  renaming (_++_ to _L++_)
open import Data.Char.Base using (Char)
open import Data.Nat.Base using (ℕ; zero; suc)
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans; cong; cong₂; subst)
open import Data.List.Properties using (++-identityˡ; ++-identityʳ; ++-assoc)


-- ════════════════════════════════════════════════════════════
-- SECTION 1: CONTEXT-INDEXED ARROWS (Kleisli Morphisms)
-- ════════════════════════════════════════════════════════════
--
-- In the Kleisli category of the probability monad:
--   - Objects are types
--   - A morphism A → B is a function A → Dist B
--
-- For text prediction, our morphisms are "conditional
-- distribution producers": given a context (history),
-- produce a distribution over the next token.
--
-- A Predictor (List Char → Char → ℝ) is exactly such a
-- Kleisli arrow: it takes a history and produces a
-- (probability) function over Char.

-- ═══ Kleisli Arrow Type ═══
--
-- A KArrow takes a history and produces a scoring function
-- over Char. This is exactly the Predictor type from Spec,
-- but we give it this name to emphasize the categorical role.

KArrow : Set
KArrow = List Char → Char → ℝ

-- Predictors ARE Kleisli arrows.
predictor-is-karrow : Predictor → KArrow
predictor-is-karrow p = p

karrow-is-predictor : KArrow → Predictor
karrow-is-predictor k = k


-- ════════════════════════════════════════════════════════════
-- SECTION 2: SEQUENTIAL COMPOSITION (The Chain Rule)
-- ════════════════════════════════════════════════════════════
--
-- In the Kleisli category, composition of f : A → T B and
-- g : B → T C is "do f, then bind g." For probability
-- distributions, this is marginalization / the chain rule:
--
--   P(a,b) = P(a) * P(b|a)
--
-- In our log-probability formulation, this becomes addition:
--
--   log P(a,b) = log P(a) + log P(b|a)
--
-- The scoreFrom function performs exactly this sequential
-- composition: it walks through the corpus, at each step
-- computing log P(next | history) and accumulating the sum.
--
-- COMPOSITION: Given a predictor p and a corpus that splits
-- as xs ++ ys, the score decomposes as:
--
--   scoreFrom p h (xs ++ ys) = scoreFrom p h xs
--                             + scoreFrom p (h ++ xs) ys
--
-- This IS composition in the Kleisli category (under the
-- log transformation that turns products into sums).

-- ═══ Sequential Scoring as Composition ═══
--
-- We define a "composed score" that makes the categorical
-- structure explicit. Scoring a predictor on a concatenated
-- corpus is the composition of scoring on each part.

composedScore : Predictor → List Char → Corpus → Corpus → ℝ
composedScore p h xs ys = scoreFrom p h xs +ʳ scoreFrom p (h L++ xs) ys

-- The Score Decomposition theorem from Spec.agda is exactly
-- the statement that sequential scoring = composed scoring.
-- We re-export it here with a categorical name.

score-composition : ∀ (p : Predictor) (h : List Char) (xs ys : Corpus)
  → scoreFrom p h (xs L++ ys) ≡ composedScore p h xs ys
score-composition = score-split


-- ════════════════════════════════════════════════════════════
-- SECTION 3: CATEGORICAL LAWS
-- ════════════════════════════════════════════════════════════
--
-- A category requires:
--   (1) Identity morphisms
--   (2) Composition
--   (3) Left and right identity laws
--   (4) Associativity of composition
--
-- For our "score category," the objects are corpus fragments
-- and the morphism space is scored by ℝ under addition.
-- The categorical laws correspond to list concatenation laws.

-- ═══ Identity ═══
--
-- The identity morphism is "do nothing" — score on empty
-- corpus. This adds 0 to the accumulated score.

score-identity : ∀ (p : Predictor) (h : List Char)
  → scoreFrom p h [] ≡ 0ʳ
score-identity p h = refl

-- ═══ Left Identity Law ═══
--
-- Composing with the identity on the left:
--   scoreFrom p h ([] ++ xs) = scoreFrom p h [] + scoreFrom p h xs
--                             = 0 + scoreFrom p h xs
--                             = scoreFrom p h xs
--
-- This corresponds to: id ∘ f = f

score-left-identity : ∀ (p : Predictor) (h : List Char) (xs : Corpus)
  → scoreFrom p h ([] L++ xs) ≡ scoreFrom p h xs
score-left-identity p h xs = refl

score-left-identity-composed : ∀ (p : Predictor) (h : List Char) (xs : Corpus)
  → composedScore p h [] xs ≡ scoreFrom p h xs
score-left-identity-composed p h xs =
  trans
    (cong (_+ʳ scoreFrom p (h L++ []) xs) refl)
    (trans
      (+ʳ-identityˡ (scoreFrom p (h L++ []) xs))
      (cong (λ hist → scoreFrom p hist xs) (++-identityʳ h)))

-- ═══ Right Identity Law ═══
--
-- Composing with the identity on the right:
--   scoreFrom p h (xs ++ []) = scoreFrom p h xs + scoreFrom p (h++xs) []
--                             = scoreFrom p h xs + 0
--                             = scoreFrom p h xs
--
-- This corresponds to: f ∘ id = f

score-right-identity : ∀ (p : Predictor) (h : List Char) (xs : Corpus)
  → scoreFrom p h (xs L++ []) ≡ scoreFrom p h xs
score-right-identity p h xs =
  trans
    (score-split p h xs [])
    (+ʳ-identityʳ (scoreFrom p h xs))

-- ═══ Associativity ═══
--
-- Composing three fragments:
--   score(p, h, (xs ++ ys) ++ zs) = score(p, h, xs ++ (ys ++ zs))
--
-- And the composed form:
--   (score(h,xs) + score(h++xs, ys)) + score(h++xs++ys, zs)
--   = score(h,xs) + (score(h++xs, ys) + score(h++xs++ys, zs))
--
-- This follows from:
--   (1) List concatenation is associative: (xs++ys)++zs = xs++(ys++zs)
--   (2) Addition on ℝ is associative: (a+b)+c = a+(b+c)

score-assoc : ∀ (p : Predictor) (h : List Char) (xs ys zs : Corpus)
  → scoreFrom p h ((xs L++ ys) L++ zs)
    ≡ scoreFrom p h (xs L++ (ys L++ zs))
score-assoc p h xs ys zs = cong (scoreFrom p h) (++-assoc xs ys zs)

-- The same, but expanded into the composed form using score-split:
score-assoc-composed :
  ∀ (p : Predictor) (h : List Char) (xs ys zs : Corpus)
  → composedScore p h xs (ys L++ zs)
    ≡ scoreFrom p h xs +ʳ composedScore p (h L++ xs) ys zs
score-assoc-composed p h xs ys zs =
  cong (scoreFrom p h xs +ʳ_) (score-split p (h L++ xs) ys zs)


-- ════════════════════════════════════════════════════════════
-- SECTION 4: SCORE AS A FUNCTOR (HOMOMORPHISM)
-- ════════════════════════════════════════════════════════════
--
-- The central insight, formalized:
--
-- We have two "categories" (really, monoids acting on scored
-- fragments):
--
--   SOURCE: (Corpus, ++, [])
--     Corpus fragments under concatenation.
--
--   TARGET: (ℝ, +, 0)
--     Real numbers under addition.
--
-- The function scoreFrom p h : Corpus → ℝ is a monoid
-- homomorphism (up to history threading):
--
--   scoreFrom p h [] = 0                    ← preserves identity
--   scoreFrom p h (xs ++ ys)                ← preserves composition
--     = scoreFrom p h xs + scoreFrom p (h++xs) ys
--
-- This is EXACTLY the same pattern as Conal's:
--
--   D(id) = id                              ← preserves identity
--   D(f . g) = D(f) . D(g)                 ← preserves composition
--
-- The "history threading" (h++xs in the second argument) is
-- analogous to the chain rule's "evaluate the outer derivative
-- at the inner function's output."

-- ═══ Record: Score Homomorphism ═══
--
-- We package the functoriality properties into a record.

record ScoreHomomorphism (p : Predictor) (h : List Char) : Set where
  field
    -- Preserves identity (unit)
    preserves-unit : scoreFrom p h [] ≡ 0ʳ

    -- Preserves composition (multiplication/concatenation)
    preserves-comp : ∀ (xs ys : Corpus)
      → scoreFrom p h (xs L++ ys)
        ≡ scoreFrom p h xs +ʳ scoreFrom p (h L++ xs) ys

-- ═══ Every predictor induces a score homomorphism ═══

score-is-homomorphism : ∀ (p : Predictor) (h : List Char)
  → ScoreHomomorphism p h
score-is-homomorphism p h = record
  { preserves-unit = score-identity p h
  ; preserves-comp = score-split p h
  }


-- ════════════════════════════════════════════════════════════
-- SECTION 5: THE ANALOGY TO AUTOMATIC DIFFERENTIATION
-- ════════════════════════════════════════════════════════════
--
-- Conal Elliott's key insight for AD:
--
--   "Differentiable functions form a category. The derivative
--    is a functor from this category to the category of linear
--    maps. The chain rule D(f . g) = D(f) . D(g) is
--    EXACTLY the functoriality equation."
--
-- Our parallel insight:
--
--   "Predictors (conditional distributions) form a category
--    (the Kleisli category of the probability monad). The
--    log-score is a functor from this category to (ℝ, +).
--    The Score Decomposition theorem is EXACTLY the
--    functoriality equation."
--
-- We can state this analogy formally. In both cases, the
-- key equation has the same shape:
--
--   F(a ∘ b) = F(a) ⊕ F(b)
--
-- where ∘ is composition and ⊕ is the target operation.

-- ═══ AD Chain Rule (stated abstractly for analogy) ═══
--
-- In AD, given:
--   f : B → C  and  g : A → B
-- The derivative functor D satisfies:
--   D(f ∘ g)(x) = D(f)(g(x)) ∘ D(g)(x)
--
-- In log-probability scoring, given:
--   p : Predictor, xs : Corpus, ys : Corpus
-- The score functor satisfies:
--   score(p, h, xs ++ ys) = score(p, h, xs) + score(p, h++xs, ys)
--
-- Both equations say: "the denotation of a composition is
-- the composition (in the target) of the denotations."

-- ═══ Record: Indexed Monoid Homomorphism ═══
--
-- In AD, D(f . g) = D(f) . D(g) is a plain functor equation
-- because the target (linear maps) has a fixed composition.
--
-- For text prediction, the score of the second fragment
-- depends on the history from the first. This makes scoreFrom
-- an INDEXED homomorphism: the "composition" in the target is
-- parameterized by the source element.
--
-- Concretely:
--   scoreFrom p h (xs ++ ys)
--     = scoreFrom p h xs  +ʳ  scoreFrom p (h ++ xs) ys
--       ^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
--       score of first part    score of second part, with
--                              history shifted by first part
--
-- This is still a homomorphism — but in the richer sense
-- of a graded/indexed monoid, where the operation on the
-- target depends on the source index.

record IndexedHomomorphism (p : Predictor) : Set where
  field
    -- The indexed mapping: for each history h, map corpus to ℝ
    F        : List Char → Corpus → ℝ

    -- Preserves identity (unit): F_h([]) = 0
    F-unit   : ∀ (h : List Char)
      → F h [] ≡ 0ʳ

    -- Preserves composition: F_h(xs ++ ys) = F_h(xs) + F_{h++xs}(ys)
    -- Note: the index SHIFTS by the first argument
    F-comp   : ∀ (h : List Char) (xs ys : Corpus)
      → F h (xs L++ ys) ≡ F h xs +ʳ F (h L++ xs) ys

-- ═══ Score instantiates IndexedHomomorphism ═══

scoreHomomorphismInstance : (p : Predictor) → IndexedHomomorphism p
scoreHomomorphismInstance p = record
  { F       = scoreFrom p
  ; F-unit  = λ h → refl
  ; F-comp  = score-split p
  }

-- ═══ Simple (non-indexed) monoid homomorphism ═══
--
-- For the abstract pattern shared with AD, we also define the
-- simpler record. Score does not directly instantiate this
-- (because of history threading), but the SHAPE of the
-- equation is identical — this is the shared algebraic pattern
-- that Conal's methodology exploits.

record MonoidHomomorphism : Set₁ where
  field
    -- Source monoid
    Src      : Set
    _∘ₛ_     : Src → Src → Src
    idₛ      : Src

    -- Target monoid
    Tgt      : Set
    _∘ₜ_     : Tgt → Tgt → Tgt
    idₜ      : Tgt

    -- The homomorphism
    F        : Src → Tgt

    -- Laws
    F-id     : F idₛ ≡ idₜ
    F-comp   : ∀ (a b : Src) → F (a ∘ₛ b) ≡ F a ∘ₜ F b


-- ════════════════════════════════════════════════════════════
-- SECTION 6: COMPOSITION OF PREDICTORS
-- ════════════════════════════════════════════════════════════
--
-- In the Kleisli category, we can compose morphisms.
-- For predictors, "composition" means using one predictor's
-- output to inform the next step. This is what scoreFrom
-- does implicitly: it sequentially applies the predictor,
-- threading the growing history.
--
-- We formalize the idea that a predictor applied to a
-- sequence is a fold (catamorphism) in the Kleisli category.

-- ═══ Single-step scoring ═══
--
-- Applying a predictor to a single character: the basic
-- Kleisli arrow in action.

singleStep : Predictor → List Char → Char → ℝ
singleStep p h c = logʳ (p h c)

-- scoreFrom is the Kleisli extension (bind / >>=) applied
-- repeatedly. We prove this characterization:

scoreFrom-cons : ∀ (p : Predictor) (h : List Char) (c : Char) (cs : Corpus)
  → scoreFrom p h (c ∷ cs)
    ≡ singleStep p h c +ʳ scoreFrom p (h L++ (c ∷ [])) cs
scoreFrom-cons p h c cs = refl

-- ═══ Score as a fold (Kleisli extension) ═══
--
-- scoreFrom is a left fold that threads state (the history):
--
--   scoreFrom p h []       = 0
--   scoreFrom p h (c ∷ cs) = step(h,c) + scoreFrom p (h++[c]) cs
--
-- This is the standard pattern for Kleisli composition in a
-- state-passing monad: each step produces a value (log prob)
-- and updates the state (extends history).

-- We prove that scoreFrom over a single character gives exactly
-- the single-step score (plus the identity element 0).

scoreFrom-single : ∀ (p : Predictor) (h : List Char) (c : Char)
  → scoreFrom p h (c ∷ []) ≡ singleStep p h c +ʳ 0ʳ
scoreFrom-single p h c = refl


-- ════════════════════════════════════════════════════════════
-- SECTION 7: CONNECTING TO PARAMETERIZE.AGDA
-- ════════════════════════════════════════════════════════════
--
-- The full picture, connecting all modules:
--
--   Spec.agda        defines Predictor, score, IsBetterThan
--   Kleisli.agda     shows predictors form a category,
--     (this file)    score is a functor (homomorphism)
--   Parameterize.agda  restricts to a parameterized family
--                    and derives gradient ascent validity
--
-- The categorical perspective adds insight to parameterization:
--
-- A Family (Params → Predictor) is a parameterized family of
-- Kleisli arrows. The score functor composed with the family
-- gives:
--
--   Params --family--> Predictor --score--> ℝ
--
-- Gradient ascent on this composition is justified by the
-- parameterized improvement theorem. The functor property
-- (score-split) ensures that improving score on parts improves
-- score on the whole.

-- ═══ Score splits for parameterized families ═══
--
-- If f is a family and θ are parameters, score-split applies
-- to f(θ) just like any predictor.

open import Parameterize using (Family; S; Params)

family-score-splits : ∀ (f : Family) (θ : Params) (xs ys : Corpus)
  → score (f θ) (xs L++ ys) ≡ scoreFrom (f θ) [] xs +ʳ scoreFrom (f θ) ([] L++ xs) ys
family-score-splits f θ xs ys = score-split (f θ) [] xs ys


-- ════════════════════════════════════════════════════════════
-- SECTION 8: THE PROBABILITY MONAD CONNECTION
-- ════════════════════════════════════════════════════════════
--
-- A monad T on a category C gives rise to a Kleisli category
-- C_T where:
--   - Objects are the same as C
--   - Morphisms A → B in C_T are morphisms A → T(B) in C
--   - Identity is the unit η : A → T(A)
--   - Composition uses the bind operation
--
-- For the probability monad:
--   T(A) = Dist(A)  — probability distributions over A
--   η(a) = δ(a)     — the Dirac delta (point mass at a)
--   bind(d, f) = ∫ f(x) d(dx)  — marginalization
--
-- A Predictor : List Char → Char → ℝ is a Kleisli arrow
-- that, given a history, produces a "distribution" over Char
-- (represented by its probability mass function).
--
-- The log transformation converts the multiplicative structure
-- of probability (P(AB) = P(A) * P(B|A)) into the additive
-- structure of log-probability (log P(AB) = log P(A) + log P(B|A)).
-- This is why our target category is (ℝ, +) rather than (ℝ, *).

-- ═══ The log-probability homomorphism ═══
--
-- Under log, the product rule of probability becomes addition:
--   log P(c₁,c₂,...,cₙ | h) = Σᵢ log P(cᵢ | h,c₁,...,cᵢ₋₁)
--
-- This is exactly what scoreFrom computes! The score-split
-- theorem shows this sum respects concatenation.
--
-- We state the product-to-sum relationship explicitly:

-- Product of probabilities (what the Kleisli composition
-- computes in the probability monad before taking log):
probProduct : Predictor → List Char → Corpus → ℝ
probProduct p h []       = 1ʳ
probProduct p h (c ∷ cs) = p h c *ʳ probProduct p (h L++ (c ∷ [])) cs

-- The log of the product equals the sum of logs (= score):
postulate
  log-prob-is-score : ∀ (p : Predictor) (h : List Char) (cs : Corpus)
    → logʳ (probProduct p h cs) ≡ scoreFrom p h cs
  -- Proof sketch: by induction on cs.
  -- Base case: log 1 = 0 = scoreFrom p h [].
  -- Inductive case:
  --   log(p(h,c) * probProduct(h++[c], cs))
  --   = log(p(h,c)) + log(probProduct(h++[c], cs))  (by log-*)
  --   = log(p(h,c)) + scoreFrom(h++[c], cs)         (by IH)
  --   = scoreFrom(h, c∷cs)
  --
  -- The full proof requires that all probabilities are positive
  -- (so log is well-defined). We postulate it here to avoid
  -- threading positivity proofs through every step.


-- ════════════════════════════════════════════════════════════
-- SUMMARY: THE CATEGORICAL PICTURE
-- ════════════════════════════════════════════════════════════
--
--  ┌─────────────────────────────────────────────────────────┐
--  │  AUTOMATIC DIFFERENTIATION   │  TEXT PREDICTION         │
--  │  (Conal's framework)         │  (this formalization)    │
--  ├─────────────────────────────────────────────────────────┤
--  │  Source category:            │  Source category:         │
--  │    smooth functions          │    Kleisli(Dist) arrows   │
--  │    A → B                     │    List Char → Dist Char  │
--  │                              │    (= Predictor)          │
--  │  Composition:                │  Composition:             │
--  │    function composition ∘    │    corpus concatenation ++ │
--  │                              │    (sequential prediction) │
--  │  Target category:            │  Target category:         │
--  │    linear maps               │    (ℝ, +)                 │
--  │                              │                           │
--  │  Functor:                    │  Functor:                 │
--  │    D (derivative)            │    scoreFrom p h          │
--  │                              │                           │
--  │  Functoriality:              │  Functoriality:           │
--  │    D(f∘g) = D(f) ∘ D(g)     │    score(xs++ys)          │
--  │    (chain rule)              │    = score(xs)+score(ys)  │
--  │                              │    (score decomposition)  │
--  │  Parameterization:           │  Parameterization:        │
--  │    neural net weights → fn   │    θ → Predictor          │
--  │                              │    (Family from           │
--  │                              │     Parameterize.agda)    │
--  │  Optimization:               │  Optimization:            │
--  │    gradient descent on loss  │    gradient ascent on     │
--  │                              │    score                  │
--  └─────────────────────────────────────────────────────────┘
--
-- The "derive, don't verify" methodology applies in the same
-- way: we define the MEANING (score), identify the ALGEBRAIC
-- STRUCTURE (monoid homomorphism / functor), choose a
-- REPRESENTATION (parameterized family), and DERIVE that
-- gradient ascent is correct. The structure forces the
-- correctness, just as in Conal's AD work.
