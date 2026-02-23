// tensor-scaling.js — Push tensor algebra NLL as low as possible
// Explores 4 scaling axes:
//   Axis 1: Larger d in T₂ (d=32, d=64)
//   Axis 2: More training steps (500, 1000 steps for T₂(R¹⁶))
//   Axis 3: Windowed tensor — only last W chars for state (W=3, W=5)
//   Axis 4: Learning rate schedule (cosine decay)
//
// Karpathy reference points:
//   Count-based bigram: 2.454
//   MLP (basic):        ~2.3
//   Our T₂(R¹⁶):       2.210  (from tensor-experiments.js)
//   Karpathy RNN/GRU:   ~2.0
//   Karpathy Transformer:~1.9
//
// Can we break NLL = 2.0 with algebraic methods?

const fs = require('fs');

const names = fs.readFileSync('/Users/stevekrouse/Desktop/denotational-llm/names.txt', 'utf-8')
  .trim().split('\n').map(s => s.trim().toLowerCase());

console.log(`Loaded ${names.length} names`);

const VOCAB = 27;
function charToIdx(c) { return c === '.' ? 0 : c.charCodeAt(0) - 96; }

// Pre-encode names as char index sequences: ['.', c1, c2, ..., cn, '.']
const nameChars = [];
let totalExamples = 0;
for (let ni = 0; ni < names.length; ni++) {
  const name = names[ni];
  const arr = new Uint8Array(name.length + 2);
  arr[0] = 0;
  for (let ci = 0; ci < name.length; ci++) arr[ci + 1] = charToIdx(name[ci]);
  arr[name.length + 1] = 0;
  nameChars.push(arr);
  totalExamples += name.length + 1;
}
console.log(`Training examples: ${totalExamples}`);

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
  for (let ni = 0; ni < names.length; ni++) {
    const chars = nameChars[ni];
    for (let j = 0; j < chars.length - 1; j++) {
      counts[chars[j] * VOCAB + chars[j + 1]] += 1;
    }
  }
  const rowProbs = new Float64Array(VOCAB * VOCAB);
  for (let r = 0; r < VOCAB; r++) {
    let s = 0;
    for (let c = 0; c < VOCAB; c++) s += counts[r * VOCAB + c] + 1;
    for (let c = 0; c < VOCAB; c++) rowProbs[r * VOCAB + c] = (counts[r * VOCAB + c] + 1) / s;
  }
  let ll = 0;
  for (let ni = 0; ni < names.length; ni++) {
    const chars = nameChars[ni];
    for (let j = 0; j < chars.length - 1; j++) {
      ll += Math.log(rowProbs[chars[j] * VOCAB + chars[j + 1]]);
    }
  }
  return -ll / totalExamples;
}

// ═══════════════════════════════════════════════════════════════════
// Unified T₂ trainer
//
// Architecture: T₂(R^d)
//   state = (count, sum_emb, sum_outer) where dims = 1 + d + d²
//   Output: W_s · normalize(state) + W_p · prevEmb + bias → softmax
//
// Options:
//   window: null (full history) or integer W (only last W chars)
//   lrSchedule: 'constant' or 'cosine'
// ═══════════════════════════════════════════════════════════════════

function runT2(config) {
  const {
    D, lr: lrConst, lrStart, lrEnd, lrSchedule,
    steps, batch, evalNames,
    window: W,
    reportSteps,
  } = config;

  const D2 = D * D;
  const SD = 1 + D + D2;
  const stateDynDim = D + D2;
  const nParams = VOCAB * D + VOCAB * SD + VOCAB * D + VOCAB;

  // Parameters
  const E  = new Float64Array(VOCAB * D);
  const Ws = new Float64Array(VOCAB * SD);
  const Wp = new Float64Array(VOCAB * D);
  const b  = new Float64Array(VOCAB);

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

  const results = [];
  const t0 = Date.now();
  const reportSet = new Set(reportSteps);
  const evalN = evalNames > 0 ? Math.min(evalNames, names.length) : names.length;

  function getLr(step) {
    if (lrSchedule === 'cosine') {
      const t = step / steps;
      return lrEnd + 0.5 * (lrStart - lrEnd) * (1 + Math.cos(Math.PI * t));
    }
    return lrConst;
  }

  // Build state from a window of characters ending at position j
  // chars[start..j] contribute to state
  function buildWindowState(chars, j, stateOut) {
    const start = W != null ? Math.max(0, j - W + 1) : 0;
    const count = j - start + 1;

    stateOut[0] = count;
    for (let d = 0; d < D; d++) stateOut[1 + d] = 0;
    for (let p = start; p <= j; p++) {
      const eOff = chars[p] * D;
      for (let d = 0; d < D; d++) stateOut[1 + d] += E[eOff + d];
    }
    for (let d = 0; d < D2; d++) stateOut[1 + D + d] = 0;
    for (let p = start; p < j; p++) {
      const prevOff = chars[p] * D;
      const curOff = chars[p + 1] * D;
      for (let di = 0; di < D; di++) {
        const ei = E[prevOff + di];
        const off = 1 + D + di * D;
        for (let dj = 0; dj < D; dj++) {
          stateOut[off + dj] += ei * E[curOff + dj];
        }
      }
    }
  }

  function updateStateIncremental(prevI, curI) {
    const ceOff = curI * D;
    const peOff = prevI * D;
    state[0] += 1;
    for (let d = 0; d < D; d++) state[1 + d] += E[ceOff + d];
    for (let di = 0; di < D; di++) {
      const ei = E[peOff + di];
      const off = 1 + D + di * D;
      for (let dj = 0; dj < D; dj++) state[off + dj] += ei * E[ceOff + dj];
    }
  }

  function forward(prevI) {
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
  }

  function backward(prevI, tgt) {
    const inv = 1.0 / state[0];
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
  }

  // Evaluate NLL
  function evalNLL() {
    let fullLL = 0, fullN = 0;
    for (let ni = 0; ni < evalN; ni++) {
      const chars = nameChars[ni];
      const seqLen = chars.length - 1;
      fullN += seqLen;

      if (W != null) {
        for (let j = 0; j < seqLen; j++) {
          buildWindowState(chars, j, state);
          forward(chars[j]);
          fullLL += Math.log(probs[chars[j + 1]] + 1e-30);
        }
      } else {
        state[0] = 1;
        for (let d = 0; d < D; d++) state[1 + d] = E[d];
        for (let d = 0; d < D2; d++) state[1 + D + d] = 0;

        for (let j = 0; j < seqLen; j++) {
          forward(chars[j]);
          fullLL += Math.log(probs[chars[j + 1]] + 1e-30);
          updateStateIncremental(chars[j], chars[j + 1]);
        }
      }
    }
    return -fullLL / fullN;
  }

  results.push({ step: 0, nll: evalNLL() });

  for (let step = 1; step <= steps; step++) {
    const currentLr = getLr(step);
    shuffle(nameIdx);
    gE.fill(0); gWs.fill(0); gWp.fill(0); gB.fill(0);
    let batchN = 0;
    const bn = Math.min(batch, names.length);

    for (let bi = 0; bi < bn; bi++) {
      const ni = nameIdx[bi];
      const chars = nameChars[ni];
      const seqLen = chars.length - 1;
      batchN += seqLen;

      if (W != null) {
        for (let j = 0; j < seqLen; j++) {
          buildWindowState(chars, j, state);
          forward(chars[j]);
          backward(chars[j], chars[j + 1]);
        }
      } else {
        state[0] = 1;
        for (let d = 0; d < D; d++) state[1 + d] = E[d];
        for (let d = 0; d < D2; d++) state[1 + D + d] = 0;

        for (let j = 0; j < seqLen; j++) {
          forward(chars[j]);
          backward(chars[j], chars[j + 1]);
          updateStateIncremental(chars[j], chars[j + 1]);
        }
      }
    }

    const sc = currentLr / batchN;
    for (let i = 0; i < E.length; i++) E[i] += sc * gE[i];
    for (let i = 0; i < Ws.length; i++) Ws[i] += sc * gWs[i];
    for (let i = 0; i < Wp.length; i++) Wp[i] += sc * gWp[i];
    for (let i = 0; i < b.length; i++) b[i] += sc * gB[i];

    if (isNaN(E[0])) return { nParams, SD, results, time: Date.now() - t0, diverged: true };

    if (reportSet.has(step)) {
      results.push({ step, nll: evalNLL() });
    }
  }

  return { nParams, SD, results, time: Date.now() - t0, diverged: false };
}


// ═══════════════════════════════════════════════════════════════════
// EXPERIMENTS
// ═══════════════════════════════════════════════════════════════════

const totalStart = Date.now();

console.log('\n============================================================');
console.log('Count-Based Bigram (MLE baseline)');
console.log('============================================================');
const countNLL = runCountBigram();
console.log(`  NLL = ${countNLL.toFixed(4)} (Karpathy target: ~2.454)`);

const allResults = [];

function runExperiment(label, config) {
  const D = config.D;
  const D2 = D * D;
  const SD = 1 + D + D2;
  const nParams = VOCAB * D + VOCAB * SD + VOCAB * D + VOCAB;

  console.log(`\n============================================================`);
  console.log(`${label}`);
  console.log(`  d=${D}  state_dim=${SD}  params=${nParams}  batch=${config.batch}` +
    `  steps=${config.steps}  window=${config.window || 'full'}` +
    `  lr=${config.lrSchedule === 'cosine' ? `cosine(${config.lrStart}->${config.lrEnd})` : config.lr}`);
  console.log(`============================================================`);

  let result = null;
  const lrTries = config.lrSchedule === 'cosine'
    ? [{ lrStart: config.lrStart, lrEnd: config.lrEnd }]
    : [{ lr: config.lr }, { lr: config.lr * 0.4 }, { lr: config.lr * 0.1 }];

  for (const lrOpts of lrTries) {
    const cfg = { ...config, ...lrOpts };
    result = runT2(cfg);
    if (!result.diverged) break;
    console.log(`  lr=${lrOpts.lr || lrOpts.lrStart} diverged, retrying...`);
  }

  if (result.diverged) {
    console.log(`  All learning rates diverged. Skipping.`);
    allResults.push({ label, D, SD, nParams, bestNLL: NaN, results: [], time: 0, config });
    return;
  }

  for (const r of result.results) {
    console.log(`  Step ${String(r.step).padStart(4)}: NLL = ${r.nll.toFixed(4)}`);
  }
  console.log(`  Time: ${(result.time / 1000).toFixed(1)}s`);

  const bestNLL = Math.min(...result.results.map(r => r.nll));
  allResults.push({ label, D, SD, nParams, bestNLL, results: result.results, time: result.time, config });
}

// ─────────────────────────────────────────────────────────────────
// AXIS 1: Larger d (reduced batch/eval to keep time manageable)
// ─────────────────────────────────────────────────────────────────
console.log('\n\n################################################################');
console.log('AXIS 1: LARGER EMBEDDING DIMENSION');
console.log('################################################################');

runExperiment('T2(R^32) full-history', {
  D: 32, lr: 5.0, lrSchedule: 'constant',
  steps: 200, batch: 500, evalNames: 4000,
  window: null,
  reportSteps: [0, 50, 100, 200],
});

runExperiment('T2(R^64) full-history', {
  D: 64, lr: 5.0, lrSchedule: 'constant',
  steps: 100, batch: 200, evalNames: 2000,
  window: null,
  reportSteps: [0, 50, 100],
});

// ─────────────────────────────────────────────────────────────────
// AXIS 2: More training steps for T₂(R¹⁶)
// ─────────────────────────────────────────────────────────────────
console.log('\n\n################################################################');
console.log('AXIS 2: MORE TRAINING STEPS (T2 R^16)');
console.log('################################################################');

runExperiment('T2(R^16) 500 steps', {
  D: 16, lr: 5.0, lrSchedule: 'constant',
  steps: 500, batch: 2000, evalNames: 8000,
  window: null,
  reportSteps: [0, 100, 200, 300, 400, 500],
});

runExperiment('T2(R^16) 1000 steps', {
  D: 16, lr: 5.0, lrSchedule: 'constant',
  steps: 1000, batch: 2000, evalNames: 8000,
  window: null,
  reportSteps: [0, 100, 200, 500, 750, 1000],
});

// ─────────────────────────────────────────────────────────────────
// AXIS 3: Windowed tensor
// ─────────────────────────────────────────────────────────────────
console.log('\n\n################################################################');
console.log('AXIS 3: WINDOWED TENSOR (LOCAL CONTEXT)');
console.log('################################################################');

runExperiment('T2(R^8) window=3', {
  D: 8, lr: 5.0, lrSchedule: 'constant',
  steps: 200, batch: 4000, evalNames: 8000,
  window: 3,
  reportSteps: [0, 50, 100, 200],
});

runExperiment('T2(R^8) window=5', {
  D: 8, lr: 5.0, lrSchedule: 'constant',
  steps: 200, batch: 4000, evalNames: 8000,
  window: 5,
  reportSteps: [0, 50, 100, 200],
});

runExperiment('T2(R^16) window=3', {
  D: 16, lr: 5.0, lrSchedule: 'constant',
  steps: 200, batch: 2000, evalNames: 8000,
  window: 3,
  reportSteps: [0, 50, 100, 200],
});

runExperiment('T2(R^16) window=5', {
  D: 16, lr: 5.0, lrSchedule: 'constant',
  steps: 200, batch: 2000, evalNames: 8000,
  window: 5,
  reportSteps: [0, 50, 100, 200],
});

runExperiment('T2(R^16) window=5, 500 steps', {
  D: 16, lr: 5.0, lrSchedule: 'constant',
  steps: 500, batch: 2000, evalNames: 8000,
  window: 5,
  reportSteps: [0, 100, 200, 300, 400, 500],
});

// ─────────────────────────────────────────────────────────────────
// AXIS 4: Cosine LR schedule
// ─────────────────────────────────────────────────────────────────
console.log('\n\n################################################################');
console.log('AXIS 4: COSINE LR SCHEDULE');
console.log('################################################################');

runExperiment('T2(R^16) cosine 5->0.5, 500 steps', {
  D: 16, lrSchedule: 'cosine', lrStart: 5.0, lrEnd: 0.5,
  steps: 500, batch: 2000, evalNames: 8000,
  window: null,
  reportSteps: [0, 100, 200, 300, 400, 500],
});

runExperiment('T2(R^16) window=5, cosine 5->0.5, 500 steps', {
  D: 16, lrSchedule: 'cosine', lrStart: 5.0, lrEnd: 0.5,
  steps: 500, batch: 2000, evalNames: 8000,
  window: 5,
  reportSteps: [0, 100, 200, 300, 400, 500],
});

// ─────────────────────────────────────────────────────────────────
// BONUS: Best combinations
// ─────────────────────────────────────────────────────────────────
console.log('\n\n################################################################');
console.log('BONUS: BEST COMBINATION ATTEMPTS');
console.log('################################################################');

runExperiment('T2(R^32) window=5, cosine 5->0.5, 300 steps', {
  D: 32, lrSchedule: 'cosine', lrStart: 5.0, lrEnd: 0.5,
  steps: 300, batch: 1000, evalNames: 4000,
  window: 5,
  reportSteps: [0, 100, 200, 300],
});

runExperiment('T2(R^16) window=7, cosine 5->0.5, 500 steps', {
  D: 16, lrSchedule: 'cosine', lrStart: 5.0, lrEnd: 0.5,
  steps: 500, batch: 2000, evalNames: 8000,
  window: 7,
  reportSteps: [0, 100, 200, 300, 400, 500],
});


// ═══════════════════════════════════════════════════════════════════
// GRAND SUMMARY
// ═══════════════════════════════════════════════════════════════════
console.log('\n\n================================================================');
console.log('GRAND SUMMARY');
console.log('================================================================');
console.log(`Count-based bigram:  NLL = ${countNLL.toFixed(4)}`);
console.log(`Previous best T2(R^16): NLL ~ 2.210 (from tensor-experiments.js)\n`);

console.log('Experiment                                         |  d | Steps | Window | LR Sched |  Best NLL |  Time');
console.log('---------------------------------------------------|----| ------|--------|----------|-----------|------');

for (const r of allResults) {
  const wStr = r.config.window ? String(r.config.window).padStart(6) : '  full';
  const lrStr = r.config.lrSchedule === 'cosine' ? '  cosine' : 'constant';
  const bestStr = isNaN(r.bestNLL) ? '     N/A' : `   ${r.bestNLL.toFixed(4)}`;
  console.log(
    `${r.label.padEnd(50)} | ${String(r.D).padStart(2)} | ${String(r.config.steps).padStart(5)} | ${wStr} | ${lrStr} | ${bestStr} | ${(r.time / 1000).toFixed(1).padStart(5)}s`
  );
}

// Detailed progression
console.log('\n\nDETAILED PROGRESSION:');
for (const r of allResults) {
  if (r.results.length === 0) continue;
  const vals = r.results.map(x => `${x.step}:${x.nll.toFixed(4)}`).join('  ');
  console.log(`  ${r.label}: ${vals}`);
}

// Best overall
const valid = allResults.filter(r => !isNaN(r.bestNLL));
if (valid.length > 0) {
  const best = valid.reduce((a, b) => b.bestNLL < a.bestNLL ? b : a);
  console.log(`\n*** BEST OVERALL: ${best.label} with NLL = ${best.bestNLL.toFixed(4)} ***`);
  console.log(`    Gap to count bigram (2.454): ${(best.bestNLL - countNLL).toFixed(4)}`);
  console.log(`    Gap to Karpathy MLP (~2.3):  ${(best.bestNLL - 2.3).toFixed(4)}`);
  console.log(`    Gap to Karpathy RNN (~2.0):  ${(best.bestNLL - 2.0).toFixed(4)}`);
  if (best.bestNLL < 2.0) {
    console.log(`    *** BROKE THE 2.0 BARRIER! ***`);
  } else {
    console.log(`    Distance to 2.0 barrier: ${(best.bestNLL - 2.0).toFixed(4)}`);
  }
}

console.log(`\nTotal time: ${((Date.now() - totalStart) / 1000).toFixed(1)}s`);
