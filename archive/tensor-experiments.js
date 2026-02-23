// tensor-experiments.js — Systematic sweep over tensor algebra configurations
// Tests T₂(R^d) for d=2,4,8,16 and T₃(R^d) for d=2,4
// Pure Node.js, Float64Array, mini-batch SGD with explicit backprop
//
// ═══════════════════════════════════════════════════════════════════
// TENSOR ALGEBRA ARCHITECTURES
// ═══════════════════════════════════════════════════════════════════
//
// T₂(R^d): state = (scalar, R^d, R^{d×d})
//   dims = 1 + d + d²
//   Update: s₀ += 1, s₁ += curEmb, s₂ += prevEmb⊗curEmb
//
// T₃(R^d): state = (scalar, R^d, R^{d×d}, R^{d×d×d})
//   dims = 1 + d + d² + d³
//   Update: s₀ += 1, s₁ += curEmb, s₂ += prevEmb⊗curEmb,
//           s₃ += prevPrevEmb⊗prevEmb⊗curEmb
//   Captures trigram correlations in embedding space
//
// Output: W_s · normalize(state) + W_p · prevEmb + bias → softmax
// ═══════════════════════════════════════════════════════════════════

const fs = require('fs');

const names = fs.readFileSync('/Users/stevekrouse/Desktop/denotational-llm/names.txt', 'utf-8')
  .trim().split('\n').map(s => s.trim().toLowerCase());

console.log(`Loaded ${names.length} names`);

const VOCAB = 27;
function charToIdx(c) { return c === '.' ? 0 : c.charCodeAt(0) - 96; }

// Precompute training data
let totalExamples = 0;
for (let i = 0; i < names.length; i++) totalExamples += names[i].length + 1;
console.log(`Training examples: ${totalExamples}`);

const prevArr = new Uint8Array(totalExamples);
const targetArr = new Uint8Array(totalExamples);

let eIdx = 0;
for (let ni = 0; ni < names.length; ni++) {
  const name = names[ni];
  let p = 0;
  for (let ci = 0; ci <= name.length; ci++) {
    const t = ci < name.length ? charToIdx(name[ci]) : 0;
    prevArr[eIdx] = p; targetArr[eIdx] = t;
    eIdx++; p = t;
  }
}

const nameOffsets = new Int32Array(names.length + 1);
let noff = 0;
for (let ni = 0; ni < names.length; ni++) { nameOffsets[ni] = noff; noff += names[ni].length + 1; }
nameOffsets[names.length] = noff;

// Pre-encode names as char index sequences
const nameChars = [];
for (let ni = 0; ni < names.length; ni++) {
  const name = names[ni];
  const arr = new Uint8Array(name.length + 2);
  arr[0] = 0;
  for (let ci = 0; ci < name.length; ci++) arr[ci + 1] = charToIdx(name[ci]);
  arr[name.length + 1] = 0;
  nameChars.push(arr);
}

const nameIdx = new Int32Array(names.length);
for (let i = 0; i < names.length; i++) nameIdx[i] = i;

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    const t = arr[i]; arr[i] = arr[j]; arr[j] = t;
  }
}

// ═══════════════════════════════════════════════════════════════════
// Count-based bigram baseline
// ═══════════════════════════════════════════════════════════════════
function runCountBigram() {
  const counts = new Float64Array(VOCAB * VOCAB);
  for (let i = 0; i < totalExamples; i++) counts[prevArr[i] * VOCAB + targetArr[i]] += 1;
  const rowProbs = new Float64Array(VOCAB * VOCAB);
  for (let r = 0; r < VOCAB; r++) {
    let s = 0;
    for (let c = 0; c < VOCAB; c++) s += counts[r * VOCAB + c] + 1;
    for (let c = 0; c < VOCAB; c++) rowProbs[r * VOCAB + c] = (counts[r * VOCAB + c] + 1) / s;
  }
  let ll = 0;
  for (let i = 0; i < totalExamples; i++) ll += Math.log(rowProbs[prevArr[i] * VOCAB + targetArr[i]]);
  return -ll / totalExamples;
}

// ═══════════════════════════════════════════════════════════════════
// Unified tensor trainer (handles both T2 and T3)
// ═══════════════════════════════════════════════════════════════════
function runTensor(order, D, lr, STEPS, BATCH, EVAL_NAMES) {
  const D2 = D * D;
  const D3 = order >= 3 ? D * D * D : 0;
  const SD = 1 + D + D2 + D3;
  const stateDynDim = D + D2 + D3; // non-scalar state dims
  const nParams = VOCAB * D + VOCAB * SD + VOCAB * D + VOCAB;

  // Parameters
  const E  = new Float64Array(VOCAB * D);
  const Ws = new Float64Array(VOCAB * SD);
  const Wp = new Float64Array(VOCAB * D);
  const b  = new Float64Array(VOCAB);

  // Smaller init for larger models
  const eScale = Math.min(0.5, 2.0 / Math.sqrt(D));
  const wsScale = Math.min(0.2, 1.0 / Math.sqrt(SD));
  for (let i = 0; i < E.length; i++)  E[i] = (Math.random() - 0.5) * eScale;
  for (let i = 0; i < Ws.length; i++) Ws[i] = (Math.random() - 0.5) * wsScale;
  for (let i = 0; i < Wp.length; i++) Wp[i] = (Math.random() - 0.5) * eScale;

  // Gradient buffers
  const gE  = new Float64Array(VOCAB * D);
  const gWs = new Float64Array(VOCAB * SD);
  const gWp = new Float64Array(VOCAB * D);
  const gB  = new Float64Array(VOCAB);
  const logits = new Float64Array(VOCAB);
  const probs  = new Float64Array(VOCAB);
  const state = new Float64Array(SD);

  const outerPrevCur = order >= 3 ? new Float64Array(D2) : null;

  const reportSteps = new Set([0, 50, 100, 200]);
  const results = [];
  const t0 = Date.now();

  // Evaluate NLL on evalCount names
  function evalNLL() {
    let fullLL = 0, fullN = 0;
    const evalCount = Math.min(EVAL_NAMES, names.length);
    for (let ni = 0; ni < evalCount; ni++) {
      const chars = nameChars[ni];
      const seqLen = chars.length - 1;
      fullN += seqLen;

      state[0] = 1;
      for (let d = 0; d < D; d++) state[1 + d] = E[d];
      for (let d = 0; d < D2 + D3; d++) state[1 + D + d] = 0;

      let prevI = 0, prevPrevI = 0;

      for (let j = 0; j < seqLen; j++) {
        const tgt = chars[j + 1];
        const inv = 1.0 / state[0];

        let maxL = -1e9;
        for (let k = 0; k < VOCAB; k++) {
          let v = b[k] + Ws[k * SD];
          const wsBase = k * SD + 1;
          for (let d = 0; d < stateDynDim; d++) v += Ws[wsBase + d] * state[1 + d] * inv;
          const wpOff = k * D, peOff = prevI * D;
          for (let d = 0; d < D; d++) v += Wp[wpOff + d] * E[peOff + d];
          logits[k] = v;
          if (v > maxL) maxL = v;
        }

        let se = 0;
        for (let k = 0; k < VOCAB; k++) { probs[k] = Math.exp(logits[k] - maxL); se += probs[k]; }
        for (let k = 0; k < VOCAB; k++) probs[k] /= se;
        fullLL += Math.log(probs[tgt] + 1e-30);

        // Update state
        const ceOff = tgt * D, peOff = prevI * D;
        state[0] += 1;
        for (let d = 0; d < D; d++) state[1 + d] += E[ceOff + d];
        for (let di = 0; di < D; di++) {
          const ei = E[peOff + di], off = 1 + D + di * D;
          for (let dj = 0; dj < D; dj++) state[off + dj] += ei * E[ceOff + dj];
        }
        if (order >= 3) {
          const ppOff = prevPrevI * D;
          for (let di = 0; di < D; di++)
            for (let dj = 0; dj < D; dj++)
              outerPrevCur[di * D + dj] = E[peOff + di] * E[ceOff + dj];
          for (let di = 0; di < D; di++) {
            const ppi = E[ppOff + di], base = 1 + D + D2 + di * D2;
            for (let d = 0; d < D2; d++) state[base + d] += ppi * outerPrevCur[d];
          }
        }
        prevPrevI = prevI;
        prevI = tgt;
      }
    }
    return -fullLL / fullN;
  }

  // Initial eval
  results.push({ step: 0, nll: evalNLL() });

  for (let step = 1; step <= STEPS; step++) {
    shuffle(nameIdx);
    gE.fill(0); gWs.fill(0); gWp.fill(0); gB.fill(0);
    let batchN = 0;

    const bn = Math.min(BATCH, names.length);
    for (let bi = 0; bi < bn; bi++) {
      const ni = nameIdx[bi];
      const chars = nameChars[ni];
      const seqLen = chars.length - 1;
      batchN += seqLen;

      state[0] = 1;
      for (let d = 0; d < D; d++) state[1 + d] = E[d];
      for (let d = 0; d < D2 + D3; d++) state[1 + D + d] = 0;

      let prevI = 0, prevPrevI = 0;

      for (let j = 0; j < seqLen; j++) {
        const tgt = chars[j + 1];
        const inv = 1.0 / state[0];

        // Forward
        let maxL = -1e9;
        for (let k = 0; k < VOCAB; k++) {
          let v = b[k] + Ws[k * SD];
          const wsBase = k * SD + 1;
          for (let d = 0; d < stateDynDim; d++) v += Ws[wsBase + d] * state[1 + d] * inv;
          const wpOff = k * D, peOff = prevI * D;
          for (let d = 0; d < D; d++) v += Wp[wpOff + d] * E[peOff + d];
          logits[k] = v;
          if (v > maxL) maxL = v;
        }

        let se = 0;
        for (let k = 0; k < VOCAB; k++) { probs[k] = Math.exp(logits[k] - maxL); se += probs[k]; }
        for (let k = 0; k < VOCAB; k++) probs[k] /= se;

        // Backward
        for (let k = 0; k < VOCAB; k++) {
          const dl = (k === tgt ? 1 : 0) - probs[k];
          gWs[k * SD] += dl;
          const wsBase = k * SD + 1;
          for (let d = 0; d < stateDynDim; d++) gWs[wsBase + d] += dl * state[1 + d] * inv;
          const wpOff = k * D, peOff = prevI * D;
          for (let d = 0; d < D; d++) {
            gWp[wpOff + d] += dl * E[peOff + d];
            gE[peOff + d] += dl * Wp[wpOff + d];
          }
          gB[k] += dl;
        }

        // Update state
        const ceOff = tgt * D, peOff = prevI * D;
        state[0] += 1;
        for (let d = 0; d < D; d++) state[1 + d] += E[ceOff + d];
        for (let di = 0; di < D; di++) {
          const ei = E[peOff + di], off = 1 + D + di * D;
          for (let dj = 0; dj < D; dj++) state[off + dj] += ei * E[ceOff + dj];
        }
        if (order >= 3) {
          const ppOff = prevPrevI * D;
          for (let di = 0; di < D; di++)
            for (let dj = 0; dj < D; dj++)
              outerPrevCur[di * D + dj] = E[peOff + di] * E[ceOff + dj];
          for (let di = 0; di < D; di++) {
            const ppi = E[ppOff + di], base = 1 + D + D2 + di * D2;
            for (let d = 0; d < D2; d++) state[base + d] += ppi * outerPrevCur[d];
          }
        }
        prevPrevI = prevI;
        prevI = tgt;
      }
    }

    // Update params
    const sc = lr / batchN;
    for (let i = 0; i < E.length; i++) E[i] += sc * gE[i];
    for (let i = 0; i < Ws.length; i++) Ws[i] += sc * gWs[i];
    for (let i = 0; i < Wp.length; i++) Wp[i] += sc * gWp[i];
    for (let i = 0; i < b.length; i++) b[i] += sc * gB[i];

    if (isNaN(E[0])) return { nParams, SD, results, time: Date.now() - t0, diverged: true };

    if (reportSteps.has(step)) {
      results.push({ step, nll: evalNLL() });
    }
  }

  return { nParams, SD, results, time: Date.now() - t0, diverged: false };
}


// ═══════════════════════════════════════════════════════════════════
// Main: systematic sweep
// ═══════════════════════════════════════════════════════════════════
const totalStart = Date.now();

console.log('\n============================================================');
console.log('Count-Based Bigram (MLE baseline)');
console.log('============================================================');
const countNLL = runCountBigram();
console.log(`  NLL = ${countNLL.toFixed(4)} (Karpathy target: ~2.454)`);

const STEPS = 200;

// Compute budget: keep each experiment roughly equal time
// Base: d=2 T2 at batch=4000 takes ~9s, so target ~8-10s each
// Cost per step ~ batch * avgNameLen * VOCAB * SD
// We want cost ~ constant across experiments
const BASE_COST = 4000 * 7 * 27 * 7; // d=2 T2 baseline cost per step

const experiments = [
  // { name, order, D, batch (training names per step), evalNames (for NLL eval) }
  { name: 'T2(R^2)',   order: 2, D: 2,  batch: 4000, evalNames: 32033 },
  { name: 'T2(R^4)',   order: 2, D: 4,  batch: 2000, evalNames: 32033 },
  { name: 'T2(R^8)',   order: 2, D: 8,  batch: 500,  evalNames: 16000 },
  { name: 'T2(R^16)',  order: 2, D: 16, batch: 300,  evalNames: 8000  },
  { name: 'T3(R^2)',   order: 3, D: 2,  batch: 2000, evalNames: 32033 },
  { name: 'T3(R^4)',   order: 3, D: 4,  batch: 400,  evalNames: 16000 },
];

const allResults = [];

for (const exp of experiments) {
  const D = exp.D;
  const D2 = D * D;
  const D3 = exp.order >= 3 ? D * D * D : 0;
  const SD = 1 + D + D2 + D3;
  const nParams = VOCAB * D + VOCAB * SD + VOCAB * D + VOCAB;

  console.log(`\n============================================================`);
  console.log(`${exp.name}  |  d=${D}  state_dim=${SD}  params=${nParams}  batch=${exp.batch}  eval=${exp.evalNames}`);
  console.log(`============================================================`);

  // Try lr=5 first, fall back to lr=2, then lr=1
  let result = null;
  for (const tryLr of [5, 2, 1]) {
    result = runTensor(exp.order, D, tryLr, STEPS, exp.batch, exp.evalNames);
    if (!result.diverged) break;
    console.log(`  lr=${tryLr} diverged (NaN), retrying...`);
  }

  if (result.diverged) {
    console.log(`  All learning rates diverged. Skipping.`);
    allResults.push({ ...exp, SD, nParams, bestNLL: NaN, results: [], time: 0 });
    continue;
  }

  for (const r of result.results) {
    console.log(`  Step ${String(r.step).padStart(3)}: NLL = ${r.nll.toFixed(4)}`);
  }
  console.log(`  Time: ${(result.time / 1000).toFixed(1)}s`);

  const bestNLL = Math.min(...result.results.map(r => r.nll));
  allResults.push({ ...exp, SD, nParams, bestNLL, results: result.results, time: result.time });
}

// ═══════════════════════════════════════════════════════════════════
// Summary table
// ═══════════════════════════════════════════════════════════════════
console.log('\n\n================================================================');
console.log('SUMMARY');
console.log('================================================================');
console.log(`Count-based bigram:  NLL = ${countNLL.toFixed(4)} (optimal MLE)`);
console.log('');
console.log('Config       | d  | State | Params |  NLL@0 | NLL@50 | NLL@100 | NLL@200 | Best   | Time');
console.log('-------------|----| ------|--------|--------|--------|---------|---------|--------|------');

for (const r of allResults) {
  const nllAt = (step) => {
    const found = r.results.find(x => x.step === step);
    return found ? found.nll.toFixed(4) : '  N/A ';
  };
  const bestStr = isNaN(r.bestNLL) ? '  N/A ' : r.bestNLL.toFixed(4);
  console.log(
    `${r.name.padEnd(12)} | ${String(r.D).padStart(2)} | ${String(r.SD).padStart(5)} | ${String(r.nParams).padStart(6)} | ${nllAt(0)} | ${nllAt(50)} |  ${nllAt(100)} |  ${nllAt(200)} | ${bestStr} | ${(r.time / 1000).toFixed(1).padStart(5)}s`
  );
}

console.log('');
console.log('Key:');
console.log('  State = state dimension (1+d+d^2 for T2, 1+d+d^2+d^3 for T3)');
console.log('  Params = total trainable parameters');
console.log('  NLL@N = negative log-likelihood at training step N');
console.log('  T2 = truncated tensor algebra of order 2 (unigram + bigram in R^d)');
console.log('  T3 = truncated tensor algebra of order 3 (+ trigram outer products)');

const validResults = allResults.filter(r => !isNaN(r.bestNLL));
if (validResults.length > 0) {
  const best = validResults.reduce((a, b) => b.bestNLL < a.bestNLL ? b : a);
  console.log(`\nBest configuration: ${best.name} with NLL = ${best.bestNLL.toFixed(4)}`);
  const gap = best.bestNLL - countNLL;
  console.log(`Gap to count-based optimal: ${gap > 0 ? '+' : ''}${gap.toFixed(4)}`);
  if (gap < 0) {
    console.log(`** Beats the count-based bigram! The tensor algebra learns richer representations. **`);
  }
}

console.log(`\nTotal sweep time: ${((Date.now() - totalStart) / 1000).toFixed(1)}s`);
