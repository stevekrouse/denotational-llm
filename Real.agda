-- ════════════════════════════════════════════════════════════
-- Postulated Real Numbers
-- ════════════════════════════════════════════════════════════
--
-- We postulate ℝ as an ordered field with log/exp.
-- This gives us the algebraic properties needed for proofs
-- without committing to a concrete representation.
-- For execution, we instantiate with Float.

module Real where

open import Relation.Binary.PropositionalEquality using (_≡_; refl; sym; trans; cong; cong₂; subst)
open import Data.Nat.Base using (ℕ; zero; suc)
open import Data.Empty using (⊥)
open import Data.Sum.Base using (_⊎_)
open import Data.List.Base using (List; []; _∷_)

-- ═══ The Type ═══

postulate
  ℝ : Set

-- ═══ Field Operations ═══

postulate
  _+ʳ_  : ℝ → ℝ → ℝ
  _*ʳ_  : ℝ → ℝ → ℝ
  _-ʳ_  : ℝ → ℝ → ℝ
  _÷ʳ_  : ℝ → ℝ → ℝ
  negʳ   : ℝ → ℝ
  0ʳ     : ℝ
  1ʳ     : ℝ

infixl 6 _+ʳ_
infixl 6 _-ʳ_
infixl 7 _*ʳ_
infixl 7 _÷ʳ_

-- ═══ Order ═══

postulate
  _<ʳ_  : ℝ → ℝ → Set
  _≤ʳ_  : ℝ → ℝ → Set

infix 4 _<ʳ_
infix 4 _≤ʳ_

-- ═══ Transcendental Functions ═══

postulate
  logʳ  : ℝ → ℝ
  expʳ  : ℝ → ℝ

-- ═══ Embedding from ℕ ═══

postulate
  fromℕ : ℕ → ℝ

-- ═══ Field Axioms ═══

postulate
  +ʳ-comm      : ∀ (a b : ℝ) → a +ʳ b ≡ b +ʳ a
  +ʳ-assoc     : ∀ (a b c : ℝ) → (a +ʳ b) +ʳ c ≡ a +ʳ (b +ʳ c)
  +ʳ-identityˡ : ∀ (a : ℝ) → 0ʳ +ʳ a ≡ a
  +ʳ-identityʳ : ∀ (a : ℝ) → a +ʳ 0ʳ ≡ a
  +ʳ-inverseʳ  : ∀ (a : ℝ) → a +ʳ negʳ a ≡ 0ʳ
  *ʳ-comm      : ∀ (a b : ℝ) → a *ʳ b ≡ b *ʳ a
  *ʳ-assoc     : ∀ (a b c : ℝ) → (a *ʳ b) *ʳ c ≡ a *ʳ (b *ʳ c)
  *ʳ-identityˡ : ∀ (a : ℝ) → 1ʳ *ʳ a ≡ a
  *ʳ-identityʳ : ∀ (a : ℝ) → a *ʳ 1ʳ ≡ a
  *ʳ-distribˡ  : ∀ (a b c : ℝ) → a *ʳ (b +ʳ c) ≡ (a *ʳ b) +ʳ (a *ʳ c)
  -ʳ-def       : ∀ (a b : ℝ) → a -ʳ b ≡ a +ʳ negʳ b

-- ═══ Order Axioms ═══

postulate
  <ʳ-irrefl    : ∀ (a : ℝ) → a <ʳ a → ⊥
  <ʳ-trans     : ∀ {a b c : ℝ} → a <ʳ b → b <ʳ c → a <ʳ c
  <ʳ-+-mono    : ∀ {a b : ℝ} (c : ℝ) → a <ʳ b → (a +ʳ c) <ʳ (b +ʳ c)
  <ʳ-*-mono    : ∀ {a b : ℝ} {c : ℝ} → 0ʳ <ʳ c → a <ʳ b → (a *ʳ c) <ʳ (b *ʳ c)
  ≤ʳ-refl      : ∀ {a : ℝ} → a ≤ʳ a
  ≤ʳ-trans     : ∀ {a b c : ℝ} → a ≤ʳ b → b ≤ʳ c → a ≤ʳ c
  <ʳ→≤ʳ        : ∀ {a b : ℝ} → a <ʳ b → a ≤ʳ b
  ≡→≤ʳ         : ∀ {a b : ℝ} → a ≡ b → a ≤ʳ b
  ≤ʳ-+-mono    : ∀ {a b c d : ℝ} → a ≤ʳ b → c ≤ʳ d → (a +ʳ c) ≤ʳ (b +ʳ d)

-- ═══ Log/Exp Axioms ═══

postulate
  log-*    : ∀ (a b : ℝ) → logʳ (a *ʳ b) ≡ logʳ a +ʳ logʳ b
  exp-log  : ∀ (a : ℝ) → expʳ (logʳ a) ≡ a
  log-exp  : ∀ (a : ℝ) → logʳ (expʳ a) ≡ a
  log-mono : ∀ {a b : ℝ} → a <ʳ b → logʳ a <ʳ logʳ b
  log-1    : logʳ 1ʳ ≡ 0ʳ

-- ═══ Summation ═══

sumʳ : List ℝ → ℝ
sumʳ []       = 0ʳ
sumʳ (x ∷ xs) = x +ʳ sumʳ xs
