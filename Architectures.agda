-- ════════════════════════════════════════════════════════════
-- ARCHITECTURES AS REPRESENTATION CHOICES
-- ════════════════════════════════════════════════════════════
--
-- Following Conal Elliott's methodology: different
-- REPRESENTATIONS of the same abstract type yield different
-- correct-by-construction implementations.
--
-- In AD: different representations of linear maps give
--   forward-mode, reverse-mode, etc.
--
-- Here: different representations of the Kleisli morphism
--   (List Char → Char → ℝ) give different architectures:
--
--   • Bigram:    remember only the last character
--   • n-gram:    remember the last n characters
--   • RNN/MLP:   maintain a fixed-dimensional state
--   • Attention: look at all positions
--
-- Each representation comes with an EMBEDDING into the full
-- Predictor type. The embedding is structure-preserving, so
-- ALL theorems (score decomposition, gradient validity, etc.)
-- transfer automatically to the architecture.
--
-- This is the key contribution: architectures aren't
-- arbitrary design choices — they're representation choices
-- in the sense of Conal's methodology, and correctness
-- follows from the algebra.

module Architectures where

open import Real
open import Spec
open import Data.List.Base using (List; []; _∷_; length; map; take; drop; foldl)
  renaming (_++_ to _L++_)
open import Data.Char.Base using (Char)
open import Data.Nat.Base using (ℕ; zero; suc; _+_)
open import Data.Product using (_×_; _,_; proj₁; proj₂; ∃-syntax)
open import Relation.Binary.PropositionalEquality
  using (_≡_; refl; sym; trans; cong; cong₂; subst)


-- ════════════════════════════════════════════════════════════
-- SECTION 1: THE REPRESENTATION PRINCIPLE
-- ════════════════════════════════════════════════════════════
--
-- A Predictor is a function (List Char → Char → ℝ).
-- This is the FULL type — it can look at arbitrary history.
--
-- An architecture RESTRICTS how the history is used.
-- Formally, an architecture is:
--   1. A representation type R
--   2. An embedding: R → Predictor
--
-- The embedding injects the restricted predictor into the
-- full predictor space. All spec theorems apply to the
-- image of the embedding.

-- ═══ Architecture Record ═══
--
-- An Architecture packages a representation type with its
-- embedding into Predictor.

record Architecture : Set₁ where
  field
    -- The representation type (what the architecture stores/computes)
    Rep : Set

    -- How to embed a representation into a full predictor
    embed : Rep → Predictor

-- ═══ Architecture Correctness ═══
--
-- An architecture is "correct" in the sense that:
--   - Every theorem about Predictors applies to embed(r) for any r
--   - This is automatic — embed(r) IS a Predictor, so all proofs work
--
-- The interesting question is: which architectures are EXPRESSIVE
-- enough to capture good predictors? This is empirical, but the
-- CORRECTNESS is algebraic.


-- ════════════════════════════════════════════════════════════
-- SECTION 2: BIGRAM ARCHITECTURE
-- ════════════════════════════════════════════════════════════
--
-- The simplest architecture: look only at the last character.
--
-- Rep = Char → Char → ℝ
--   A function from (previous char) to a distribution over
--   (next char). This is a 27×27 matrix.
--
-- Embedding: given history, extract the last character and
-- apply the bigram table.

-- ═══ Last Character Extraction ═══

lastChar : List Char → Char → Char
lastChar []       def = def
lastChar (x ∷ []) _   = x
lastChar (_ ∷ xs) def = lastChar xs def

-- ═══ Bigram Representation ═══

BigramRep : Set
BigramRep = Char → Char → ℝ

bigramEmbed : BigramRep → Predictor
bigramEmbed table history c = table (lastChar history '.') c

bigramArch : Architecture
bigramArch = record
  { Rep   = BigramRep
  ; embed = bigramEmbed
  }

-- ═══ Bigram Properties ═══
--
-- A bigram predictor's score depends only on consecutive
-- character pairs. This means the predictor is "Markovian" —
-- history beyond the last character is irrelevant.

bigram-markov : ∀ (b : BigramRep) (h₁ h₂ : List Char) (c₀ c : Char)
  → lastChar h₁ '.' ≡ lastChar h₂ '.'
  → bigramEmbed b h₁ c ≡ bigramEmbed b h₂ c
bigram-markov b h₁ h₂ c₀ c eq = cong (λ prev → b prev c) eq


-- ════════════════════════════════════════════════════════════
-- SECTION 3: N-GRAM ARCHITECTURE
-- ════════════════════════════════════════════════════════════
--
-- Generalization: look at the last n characters.
--
-- Rep = List Char → Char → ℝ (where the list has length ≤ n)
--   A function from (context of length ≤ n) to distribution.
--
-- Bigram is the special case n = 1.

-- ═══ Context Extraction ═══

-- Take the last n characters from a list
lastN : ℕ → List Char → List Char
lastN n xs with length xs
... | len = drop (len ∸ n) xs
  where open import Data.Nat.Base using (_∸_)

-- ═══ N-gram Representation ═══

NgramRep : ℕ → Set
NgramRep n = List Char → Char → ℝ

ngramEmbed : (n : ℕ) → NgramRep n → Predictor
ngramEmbed n table history c = table (lastN n history) c

ngramArch : ℕ → Architecture
ngramArch n = record
  { Rep   = NgramRep n
  ; embed = ngramEmbed n
  }

-- ═══ Bigram is 1-gram ═══
--
-- Note: The bigram architecture is essentially the n-gram
-- architecture with n=1. (Not definitionally equal due to
-- the different representations of "last 1" vs "last char",
-- but morally the same.)


-- ════════════════════════════════════════════════════════════
-- SECTION 4: RNN / STATEFUL ARCHITECTURE
-- ════════════════════════════════════════════════════════════
--
-- Key idea: instead of looking at raw history, maintain a
-- FIXED-DIMENSIONAL STATE that summarizes the history.
--
-- Rep = State × (State → Char → State) × (State → Char → ℝ)
--   - An initial state s₀
--   - A transition function: state × char → state
--   - An output function: state → distribution over next char
--
-- This is exactly a recurrent neural network (RNN).
-- The state compresses the history into a fixed-size vector.
--
-- Embedding: fold the transition function over the history,
-- starting from s₀, then apply the output function.

-- ═══ RNN Representation ═══

record RNNRep (State : Set) : Set where
  field
    init   : State                    -- initial state s₀
    step   : State → Char → State     -- transition function
    output : State → Char → ℝ         -- output distribution

open RNNRep public

-- Fold the transition function over a history to get the final state
runRNN : {S : Set} → RNNRep S → List Char → S
runRNN rnn history = foldl (step rnn) (init rnn) history

-- ═══ RNN Embedding ═══

rnnEmbed : {S : Set} → RNNRep S → Predictor
rnnEmbed rnn history c = output rnn (runRNN rnn history) c

rnnArch : (S : Set) → Architecture
rnnArch S = record
  { Rep   = RNNRep S
  ; embed = rnnEmbed
  }

-- ═══ RNN is a Lossy Compression of History ═══
--
-- The RNN embedding factors through the state space:
--
--   List Char --runRNN--> State --output--> (Char → ℝ)
--
-- This means the RNN can only distinguish histories that
-- produce different states. Two histories with the same
-- state get the same prediction.

rnn-state-determines-prediction :
  ∀ {S : Set} (rnn : RNNRep S) (h₁ h₂ : List Char) (c : Char)
  → runRNN rnn h₁ ≡ runRNN rnn h₂
  → rnnEmbed rnn h₁ c ≡ rnnEmbed rnn h₂ c
rnn-state-determines-prediction rnn h₁ h₂ c eq =
  cong (λ s → output rnn s c) eq


-- ════════════════════════════════════════════════════════════
-- SECTION 5: ATTENTION ARCHITECTURE
-- ════════════════════════════════════════════════════════════
--
-- Key idea: instead of compressing history into a fixed state,
-- ATTEND to all positions in the history.
--
-- Rep = (List (ℕ × Char) → Char → ℝ)
--   A function from (positioned history) to distribution.
--   The positioned history is the history with position
--   indices attached: [(0,c₁), (1,c₂), ..., (n-1,cₙ)].
--
-- Unlike the RNN, there is NO information bottleneck.
-- The attention mechanism can look at any position.
--
-- This is what makes transformers powerful: they don't
-- compress history through a fixed-size state.

-- ═══ Positioned History ═══

-- Attach position indices to a list
enumerate : List Char → List (ℕ × Char)
enumerate = go 0
  where
    go : ℕ → List Char → List (ℕ × Char)
    go _ []       = []
    go n (c ∷ cs) = (n , c) ∷ go (suc n) cs

-- ═══ Attention Representation ═══

AttnRep : Set
AttnRep = List (ℕ × Char) → Char → ℝ

attnEmbed : AttnRep → Predictor
attnEmbed attn history c = attn (enumerate history) c

attnArch : Architecture
attnArch = record
  { Rep   = AttnRep
  ; embed = attnEmbed
  }

-- ═══ Attention Subsumes RNN ═══
--
-- Any RNN predictor can be simulated by an attention predictor
-- (by attending to all positions and folding). The converse is
-- not true in general — attention can capture patterns that
-- require the full history, which a finite-state RNN cannot.
--
-- This is why transformers are more expressive than RNNs.

postulate
  attn-subsumes-rnn : ∀ {S : Set} (rnn : RNNRep S)
    → ∃[ attn ] (∀ (h : List Char) (c : Char)
        → attnEmbed attn h c ≡ rnnEmbed rnn h c)
  -- Proof sketch: define attn = λ positioned c →
  --   output rnn (foldl step init (map proj₂ positioned)) c
  -- Then attnEmbed attn h c
  --   = attn (enumerate h) c
  --   = output rnn (foldl step init (map proj₂ (enumerate h))) c
  --   = output rnn (foldl step init h) c    (since map proj₂ ∘ enumerate = id)
  --   = rnnEmbed rnn h c


-- ════════════════════════════════════════════════════════════
-- SECTION 6: REPRESENTATION HIERARCHY
-- ════════════════════════════════════════════════════════════
--
-- The architectures form a hierarchy of expressiveness:
--
--   Bigram ⊂ n-gram ⊂ RNN ⊂ Attention ⊂ Full Predictor
--
-- More precisely, each architecture's image under embedding
-- is a subset of the next:
--
--   { bigramEmbed(b) | b : BigramRep }
--   ⊆ { ngramEmbed(2, t) | t : NgramRep 2 }
--   ⊆ ...
--   ⊆ { ngramEmbed(n, t) | t : NgramRep n }
--   ⊆ { rnnEmbed(r) | r : RNNRep S, for large enough S }
--   ⊆ { attnEmbed(a) | a : AttnRep }
--   ⊆ Predictor
--
-- Each step gains expressiveness at the cost of more parameters.
-- The spec doesn't tell you which level to use — that depends
-- on the data. But the spec guarantees that optimization works
-- correctly at every level.

-- ═══ Architecture Subsumption ═══

-- One architecture subsumes another if every predictor
-- expressible in the first is also expressible in the second.

_Subsumes_ : Architecture → Architecture → Set
a₂ Subsumes a₁ = ∀ (r₁ : Architecture.Rep a₁) →
  ∃[ r₂ ] (∀ (h : List Char) (c : Char)
    → Architecture.embed a₂ r₂ h c ≡ Architecture.embed a₁ r₁ h c)


-- ════════════════════════════════════════════════════════════
-- SECTION 7: CONNECTION TO CONAL'S METHODOLOGY
-- ════════════════════════════════════════════════════════════
--
-- In Conal's AD work:
--   - The abstract type: differentiable functions
--   - Different representations of LINEAR MAPS give different
--     AD algorithms
--   - Matrices → standard Jacobian
--   - Functions → forward-mode AD
--   - Continuations → reverse-mode AD
--
-- In our text prediction work:
--   - The abstract type: Predictor (Kleisli morphism)
--   - Different representations of PREDICTORS give different
--     architectures
--   - Last char → Bigram
--   - Fixed state + fold → RNN
--   - All positions → Attention
--
-- The EXACT SAME principle operates:
--   "The meaning of the representation is the representation
--    of the meaning."
--
-- For any architecture with embedding e : Rep → Predictor:
--   score (e r) corpus = scoreFrom (e r) [] corpus
--
-- Score decomposition, monotonicity, gradient validity —
-- all transfer automatically because e(r) IS a Predictor.

-- ═══ Automatic Theorem Transfer ═══
--
-- Any theorem about Predictors immediately applies to any
-- architecture via the embedding. We demonstrate this for
-- the most important theorems.

-- Score decomposition for any architecture:
arch-score-split : ∀ (arch : Architecture) (r : Architecture.Rep arch)
  (h : List Char) (xs ys : List Char)
  → scoreFrom (Architecture.embed arch r) h (xs L++ ys)
    ≡ scoreFrom (Architecture.embed arch r) h xs
      +ʳ scoreFrom (Architecture.embed arch r) (h L++ xs) ys
arch-score-split arch r h xs ys = score-split (Architecture.embed arch r) h xs ys

-- Score preorder for any architecture:
arch-atLeastAsGood-refl : ∀ (arch : Architecture) (r : Architecture.Rep arch)
  (c : List Char)
  → (Architecture.embed arch r) IsAtLeastAsGoodAs (Architecture.embed arch r) On c
arch-atLeastAsGood-refl arch r c = atLeastAsGood-refl (Architecture.embed arch r) c


-- ════════════════════════════════════════════════════════════
-- SECTION 8: THE EXPRESSIVENESS-EFFICIENCY TRADEOFF
-- ════════════════════════════════════════════════════════════
--
-- The representation principle reveals a fundamental tradeoff:
--
-- MORE RESTRICTIVE representation (e.g., bigram):
--   + Fewer parameters → faster training, less overfitting
--   + Closed-form solutions may exist (MLE for bigrams)
--   - Less expressive → can't capture long-range dependencies
--
-- LESS RESTRICTIVE representation (e.g., attention):
--   + More expressive → can capture complex patterns
--   - More parameters → slower training, risk of overfitting
--   - No closed-form solution → must use gradient methods
--
-- The spec is SILENT about this tradeoff. It only guarantees:
--   (1) score measures quality (by definition)
--   (2) gradient ascent improves quality (by theorem)
--   (3) score decomposes compositionally (by theorem)
--
-- Which architecture to choose is an EMPIRICAL question.
-- The algebra tells you HOW to optimize; the data tells you
-- WHAT to optimize.
--
-- This is exactly parallel to AD:
--   - The algebra tells you HOW to differentiate (chain rule)
--   - The problem tells you WHAT to differentiate (loss function)
--   - The choice of forward vs reverse mode depends on the
--     shape of the computation (empirical consideration)
