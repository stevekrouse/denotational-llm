// linear-attention.js — Linear attention derived from tensor algebra
//
// The T2(R^d) model accumulates (1, e[c], e[prev] (x) e[c]) as state.
// This uses the RAW embedding for both "what to store" and "what to retrieve."
//
// Linear attention generalizes this by learning SEPARATE projections:
//   key   k = W_k * e[c]       (what to index by)
//   value v = W_v * e[c]       (what to store)
//   query q = W_q * e[c]       (what to retrieve)
//
// State update (still a monoid homomorphism!):
//   count  += 1
//   v_sum  += v                 (degree-1: accumulated values)
//   KV_mat += k (x) v           (degree-2: key-value outer product)
//
// Retrieval for current character c:
//   q = W_q * e[c]
//   retrieved = KV_mat^T * q    (linear attention: no softmax over keys)
//   logits = W_out * [count, v_sum/count, retrieved/count] + W_p * e[prev] + bias
//
// This is exactly linear attention (Katharopoulos et al., 2020) but derived
// from the tensor algebra perspective: we're asking "what if the monoid state
// uses LEARNED projections instead of raw embeddings?"
//
// Key insight: both T2 and linear attention are monoid homomorphisms from
// (List Char, ++) to (State, +). The difference is what gets accumulated.
// T2 accumulates e (x) e (symmetric, d^2 params in state).
// Linear attention accumulates k (x) v (asymmetric, d_k * d_v params in state).
// The query projection lets the model learn WHAT to retrieve from the state.
//
// =====================================================================
// EXPERIMENT PLAN
// =====================================================================
//
// 1. T2(R^16) baseline: raw embeddings, no projections  (~2.21 NLL)
// 2. Linear attention (matched params): d_embed=16, d_k=d_v=4, d_q=4
// 3. Linear attention (more params): d_embed=16, d_k=d_v=8, d_q=8
// 4. Linear attention (large): d_embed=16, d_k=d_v=12, d_q=12
//
// All use: SGD, cosine LR schedule, 200 steps, batch=2000, eval on 8000 names
// =====================================================================

const fs = require('fs');

// =====================================================================
// DATA LOADING
// =====================================================================

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

// =====================================================================
// COUNT-BASED BIGRAM BASELINE
// =====================================================================

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

// =====================================================================
// T2(R^d) BASELINE — raw embeddings, no learned projections
// =====================================================================
//
// State = (count, sum_emb, sum_outer) where sum_outer = sum of e[prev] (x) e[cur]
// State dim = 1 + D + D^2
// Output: W_s * normalize(state) + W_p * E[prev] + bias -> softmax
//
// Parameters:
//   E:  VOCAB * D           (embeddings)
//   Ws: VOCAB * (1+D+D^2)   (state-to-logits)
//   Wp: VOCAB * D           (prev-char-to-logits)
//   b:  VOCAB               (bias)

function runT2Baseline(config) {
  const { D, steps, batch, evalNames, lrStart, lrEnd, reportEvery, label } = config;

  const D2 = D * D;
  const SD = 1 + D + D2;
  const nParams = VOCAB * D + VOCAB * SD + VOCAB * D + VOCAB;

  // Initialize parameters
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
  const state  = new Float64Array(SD);

  const stateDynDim = D + D2;

  const results = [];
  const t0 = Date.now();
  const evalN = Math.min(evalNames, names.length);

  function getLr(step) {
    const t = step / steps;
    return lrEnd + 0.5 * (lrStart - lrEnd) * (1 + Math.cos(Math.PI * t));
  }

  function evalNLL() {
    let fullLL = 0, fullN = 0;
    for (let ni = 0; ni < evalN; ni++) {
      const chars = nameChars[ni];
      const seqLen = chars.length - 1;
      fullN += seqLen;

      // Init state: count=1, vec=E['.'], mat=zeros
      state[0] = 1;
      for (let d = 0; d < D; d++) state[1 + d] = E[d]; // E['.'] = E[0*D..]
      for (let d = 0; d < D2; d++) state[1 + D + d] = 0;

      for (let j = 0; j < seqLen; j++) {
        const prevI = chars[j];
        const tgt = chars[j + 1];
        const inv = 1.0 / state[0];

        // Forward: logits = Ws * norm(state) + Wp * E[prev] + b
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

        // Update state: count += 1, vec += E[tgt], mat += E[prev] (x) E[tgt]
        const ceOff = tgt * D, peOff = prevI * D;
        state[0] += 1;
        for (let d = 0; d < D; d++) state[1 + d] += E[ceOff + d];
        for (let di = 0; di < D; di++) {
          const ei = E[peOff + di], off = 1 + D + di * D;
          for (let dj = 0; dj < D; dj++) state[off + dj] += ei * E[ceOff + dj];
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

      state[0] = 1;
      for (let d = 0; d < D; d++) state[1 + d] = E[d];
      for (let d = 0; d < D2; d++) state[1 + D + d] = 0;

      for (let j = 0; j < seqLen; j++) {
        const prevI = chars[j];
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

        // Backward: gradient of cross-entropy loss
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
      }
    }

    // Parameter update
    const sc = currentLr / batchN;
    for (let i = 0; i < E.length; i++) E[i] += sc * gE[i];
    for (let i = 0; i < Ws.length; i++) Ws[i] += sc * gWs[i];
    for (let i = 0; i < Wp.length; i++) Wp[i] += sc * gWp[i];
    for (let i = 0; i < b.length; i++) b[i] += sc * gB[i];

    if (isNaN(E[0])) return { nParams, label, results, time: Date.now() - t0, diverged: true };

    if (step % reportEvery === 0 || step === steps) {
      results.push({ step, nll: evalNLL() });
    }
  }

  return { nParams, label, results, time: Date.now() - t0, diverged: false };
}

// =====================================================================
// LINEAR ATTENTION — learned key/value/query projections
// =====================================================================
//
// Architecture:
//   Embed each char c -> e[c] in R^D_e
//   Key:   k = W_k * e[c]  in R^D_k
//   Value: v = W_v * e[c]  in R^D_v
//   Query: q = W_q * e[c]  in R^D_k  (query lives in key-space for dot product)
//
// State = (count, v_sum in R^D_v, KV_mat in R^{D_k x D_v})
// State dim = 1 + D_v + D_k * D_v
//
// State update (monoid homomorphism):
//   count  += 1
//   v_sum  += v_cur                  (degree-1 value accumulation)
//   KV_mat += k_prev (x) v_cur      (degree-2 key-value outer product)
//
// Note: we use k_prev (x) v_cur (key of previous char, value of current char)
// to match T2's e[prev] (x) e[cur] structure. This stores "after seeing key k,
// we observed value v" — the same transition structure as T2.
//
// Retrieval:
//   q = W_q * e[prev_char]           (query based on previous char)
//   retrieved = KV_mat * q           (D_v-dimensional retrieval, but KV is D_k x D_v
//                                     so we compute q^T * KV = D_v vector)
//   Actually: retrieved_j = sum_i q_i * KV[i][j] for j in D_v
//
// Output:
//   logits = W_out * [count, v_sum/count, retrieved/count] + W_p * e[prev] + bias
//
// Parameters:
//   E:     VOCAB * D_e          (embeddings)
//   W_k:   D_k * D_e            (key projection)
//   W_v:   D_v * D_e            (value projection)
//   W_q:   D_k * D_e            (query projection)
//   W_out: VOCAB * (1+D_v+D_v)  (state-to-logits: count + v_sum + retrieved)
//   W_p:   VOCAB * D_e          (prev-char direct connection)
//   bias:  VOCAB

function runLinearAttention(config) {
  const { De, Dk, Dv, steps, batch, evalNames, lrStart, lrEnd, reportEvery, label } = config;

  // State dimensions
  const KV_SIZE = Dk * Dv;         // key-value matrix size
  const SD = 1 + Dv + Dv;         // output features: count + v_sum + retrieved (both D_v)
  // Actual state stored: count + v_sum(D_v) + KV_mat(D_k * D_v)
  const STATE_SIZE = 1 + Dv + KV_SIZE;

  // Parameter counts
  const nE   = VOCAB * De;         // embeddings
  const nWk  = Dk * De;            // key projection
  const nWv  = Dv * De;            // value projection
  const nWq  = Dk * De;            // query projection
  const nWo  = VOCAB * SD;         // output: count + v_sum + retrieved
  const nWp  = VOCAB * De;         // prev-char direct
  const nB   = VOCAB;              // bias
  const nParams = nE + nWk + nWv + nWq + nWo + nWp + nB;

  // Allocate parameters
  const E   = new Float64Array(nE);
  const Wk  = new Float64Array(nWk);
  const Wv  = new Float64Array(nWv);
  const Wq  = new Float64Array(nWq);
  const Wo  = new Float64Array(nWo);
  const Wp  = new Float64Array(nWp);
  const b   = new Float64Array(nB);

  // Initialize
  const eScale  = Math.min(0.5, 2.0 / Math.sqrt(De));
  const projScale = 1.0 / Math.sqrt(De);
  const woScale = Math.min(0.2, 1.0 / Math.sqrt(SD));
  for (let i = 0; i < E.length; i++)  E[i]  = (Math.random() - 0.5) * eScale;
  for (let i = 0; i < Wk.length; i++) Wk[i] = (Math.random() - 0.5) * projScale;
  for (let i = 0; i < Wv.length; i++) Wv[i] = (Math.random() - 0.5) * projScale;
  for (let i = 0; i < Wq.length; i++) Wq[i] = (Math.random() - 0.5) * projScale;
  for (let i = 0; i < Wo.length; i++) Wo[i] = (Math.random() - 0.5) * woScale;
  for (let i = 0; i < Wp.length; i++) Wp[i] = (Math.random() - 0.5) * eScale;

  // Gradient buffers
  const gE  = new Float64Array(nE);
  const gWk = new Float64Array(nWk);
  const gWv = new Float64Array(nWv);
  const gWq = new Float64Array(nWq);
  const gWo = new Float64Array(nWo);
  const gWp = new Float64Array(nWp);
  const gB  = new Float64Array(nB);

  // Working buffers
  const logits    = new Float64Array(VOCAB);
  const probs     = new Float64Array(VOCAB);
  const vSum      = new Float64Array(Dv);            // accumulated values
  const kvMat     = new Float64Array(KV_SIZE);       // key-value matrix (row-major: Dk x Dv)
  const query     = new Float64Array(Dk);            // query vector
  const retrieved = new Float64Array(Dv);            // retrieved vector
  const key       = new Float64Array(Dk);            // key vector (for current char)
  const val       = new Float64Array(Dv);            // value vector (for current char)
  const prevKey   = new Float64Array(Dk);            // key of previous char

  const results = [];
  const t0 = Date.now();
  const evalN = Math.min(evalNames, names.length);

  function getLr(step) {
    const t = step / steps;
    return lrEnd + 0.5 * (lrStart - lrEnd) * (1 + Math.cos(Math.PI * t));
  }

  // Compute key for character index ci: key = Wk * E[ci]
  function computeKey(ci, out) {
    const eOff = ci * De;
    for (let i = 0; i < Dk; i++) {
      let s = 0;
      const wOff = i * De;
      for (let j = 0; j < De; j++) s += Wk[wOff + j] * E[eOff + j];
      out[i] = s;
    }
  }

  // Compute value for character index ci: val = Wv * E[ci]
  function computeVal(ci, out) {
    const eOff = ci * De;
    for (let i = 0; i < Dv; i++) {
      let s = 0;
      const wOff = i * De;
      for (let j = 0; j < De; j++) s += Wv[wOff + j] * E[eOff + j];
      out[i] = s;
    }
  }

  // Compute query for character index ci: query = Wq * E[ci]
  function computeQuery(ci, out) {
    const eOff = ci * De;
    for (let i = 0; i < Dk; i++) {
      let s = 0;
      const wOff = i * De;
      for (let j = 0; j < De; j++) s += Wq[wOff + j] * E[eOff + j];
      out[i] = s;
    }
  }

  // Retrieve from KV matrix: retrieved = KV^T * q
  // KV is Dk x Dv (row-major), so retrieved[j] = sum_i q[i] * KV[i*Dv + j]
  function computeRetrieval(q, kvM, out) {
    for (let j = 0; j < Dv; j++) {
      let s = 0;
      for (let i = 0; i < Dk; i++) s += q[i] * kvM[i * Dv + j];
      out[j] = s;
    }
  }

  // Forward pass: compute logits from state + prev char
  // features = [1, vSum/count, retrieved/count]  (dim = 1 + Dv + Dv = SD)
  // logits[k] = Wo[k] * features + Wp[k] * E[prev] + b[k]
  function forward(prevI, count) {
    const inv = 1.0 / count;

    // Compute query from previous character
    computeQuery(prevI, query);

    // Retrieve: retrieved = KV^T * query
    computeRetrieval(query, kvMat, retrieved);

    let maxL = -1e9;
    for (let k = 0; k < VOCAB; k++) {
      let v = b[k];
      const woBase = k * SD;
      // count feature (normalized by count -- just 1.0)
      v += Wo[woBase];
      // v_sum / count features
      for (let d = 0; d < Dv; d++) v += Wo[woBase + 1 + d] * vSum[d] * inv;
      // retrieved / count features
      for (let d = 0; d < Dv; d++) v += Wo[woBase + 1 + Dv + d] * retrieved[d] * inv;
      // prev-char direct connection
      const wpOff = k * De, peOff = prevI * De;
      for (let d = 0; d < De; d++) v += Wp[wpOff + d] * E[peOff + d];
      logits[k] = v;
      if (v > maxL) maxL = v;
    }
    let se = 0;
    for (let k = 0; k < VOCAB; k++) { probs[k] = Math.exp(logits[k] - maxL); se += probs[k]; }
    for (let k = 0; k < VOCAB; k++) probs[k] /= se;
  }

  // Backward pass: compute gradients for one prediction
  // dl[k] = (k == tgt ? 1 : 0) - probs[k]  (gradient of log-softmax cross-entropy)
  function backward(prevI, tgt, count) {
    const inv = 1.0 / count;

    // dl[k] for each output class
    // Gradient w.r.t. Wo, Wp, b
    // Also need gradient w.r.t. query (for Wq, E backprop)
    // and w.r.t. vSum (for Wv, E backprop)

    // Accumulate gradient w.r.t. retrieved and vSum
    const dRetrieved = new Float64Array(Dv);
    const dVsum = new Float64Array(Dv);
    const dPrevEmb = new Float64Array(De);

    for (let k = 0; k < VOCAB; k++) {
      const dl = (k === tgt ? 1 : 0) - probs[k];
      const woBase = k * SD;

      // d/d bias
      gB[k] += dl;

      // d/d Wo[k][0] (count feature)
      gWo[woBase] += dl;

      // d/d Wo[k][1..Dv] (v_sum features)
      for (let d = 0; d < Dv; d++) {
        gWo[woBase + 1 + d] += dl * vSum[d] * inv;
        dVsum[d] += dl * Wo[woBase + 1 + d] * inv;
      }

      // d/d Wo[k][1+Dv..1+2*Dv] (retrieved features)
      for (let d = 0; d < Dv; d++) {
        gWo[woBase + 1 + Dv + d] += dl * retrieved[d] * inv;
        dRetrieved[d] += dl * Wo[woBase + 1 + Dv + d] * inv;
      }

      // d/d Wp and d/d E[prev] via Wp
      const wpOff = k * De, peOff = prevI * De;
      for (let d = 0; d < De; d++) {
        gWp[wpOff + d] += dl * E[peOff + d];
        dPrevEmb[d] += dl * Wp[wpOff + d];
      }
    }

    // Backprop through E[prev] (from Wp path)
    const peOff = prevI * De;
    for (let d = 0; d < De; d++) gE[peOff + d] += dPrevEmb[d];

    // Backprop through retrieval: retrieved = KV^T * query
    // d/d query: dQuery[i] = sum_j dRetrieved[j] * KV[i*Dv + j]
    // d/d KV[i][j]: sum over uses... but we don't backprop through KV state
    // (treating accumulated state as detached, like the T2 baseline does)
    const dQuery = new Float64Array(Dk);
    for (let i = 0; i < Dk; i++) {
      let s = 0;
      for (let j = 0; j < Dv; j++) s += dRetrieved[j] * kvMat[i * Dv + j];
      dQuery[i] = s;
    }

    // Backprop query: q = Wq * E[prev]
    // d/d Wq[i][j] += dQuery[i] * E[prev][j]
    // d/d E[prev][j] += sum_i dQuery[i] * Wq[i][j]
    for (let i = 0; i < Dk; i++) {
      const wOff = i * De;
      for (let j = 0; j < De; j++) {
        gWq[wOff + j] += dQuery[i] * E[peOff + j];
        gE[peOff + j] += dQuery[i] * Wq[wOff + j];
      }
    }

    // Note: We do NOT backprop through the accumulated state (vSum, kvMat)
    // to past embeddings/projections. This matches the T2 baseline behavior:
    // gradients flow through the output layer and the current-step projections,
    // but the state accumulation is treated as a forward-only running sum.
    // This is standard for online/streaming models and keeps training O(1) per step.
  }

  // Evaluate NLL on first evalN names
  function evalNLL() {
    let fullLL = 0, fullN = 0;
    for (let ni = 0; ni < evalN; ni++) {
      const chars = nameChars[ni];
      const seqLen = chars.length - 1;
      fullN += seqLen;

      // Reset state
      let count = 1;
      computeVal(0, val);  // val for '.'
      for (let d = 0; d < Dv; d++) vSum[d] = val[d];
      kvMat.fill(0);
      computeKey(0, prevKey);  // key for '.'

      for (let j = 0; j < seqLen; j++) {
        const prevI = chars[j];
        const tgt = chars[j + 1];

        forward(prevI, count);
        fullLL += Math.log(probs[tgt] + 1e-30);

        // Update state with new character tgt
        computeVal(tgt, val);
        computeKey(tgt, key);

        count += 1;
        for (let d = 0; d < Dv; d++) vSum[d] += val[d];
        // KV update: kvMat += prevKey (x) val
        for (let i = 0; i < Dk; i++) {
          const off = i * Dv;
          const ki = prevKey[i];
          for (let j2 = 0; j2 < Dv; j2++) kvMat[off + j2] += ki * val[j2];
        }

        // prevKey = key for next iteration
        for (let d = 0; d < Dk; d++) prevKey[d] = key[d];
      }
    }
    return -fullLL / fullN;
  }

  results.push({ step: 0, nll: evalNLL() });

  for (let step = 1; step <= steps; step++) {
    const currentLr = getLr(step);
    shuffle(nameIdx);
    gE.fill(0); gWk.fill(0); gWv.fill(0); gWq.fill(0);
    gWo.fill(0); gWp.fill(0); gB.fill(0);
    let batchN = 0;
    const bn = Math.min(batch, names.length);

    for (let bi = 0; bi < bn; bi++) {
      const ni = nameIdx[bi];
      const chars = nameChars[ni];
      const seqLen = chars.length - 1;
      batchN += seqLen;

      // Reset state
      let count = 1;
      computeVal(0, val);
      for (let d = 0; d < Dv; d++) vSum[d] = val[d];
      kvMat.fill(0);
      computeKey(0, prevKey);

      for (let j = 0; j < seqLen; j++) {
        const prevI = chars[j];
        const tgt = chars[j + 1];

        forward(prevI, count);
        backward(prevI, tgt, count);

        // Update state
        computeVal(tgt, val);
        computeKey(tgt, key);

        count += 1;
        for (let d = 0; d < Dv; d++) vSum[d] += val[d];
        for (let i = 0; i < Dk; i++) {
          const off = i * Dv;
          const ki = prevKey[i];
          for (let j2 = 0; j2 < Dv; j2++) kvMat[off + j2] += ki * val[j2];
        }
        for (let d = 0; d < Dk; d++) prevKey[d] = key[d];
      }
    }

    // Parameter update
    const sc = currentLr / batchN;
    for (let i = 0; i < E.length; i++)  E[i]  += sc * gE[i];
    for (let i = 0; i < Wk.length; i++) Wk[i] += sc * gWk[i];
    for (let i = 0; i < Wv.length; i++) Wv[i] += sc * gWv[i];
    for (let i = 0; i < Wq.length; i++) Wq[i] += sc * gWq[i];
    for (let i = 0; i < Wo.length; i++) Wo[i] += sc * gWo[i];
    for (let i = 0; i < Wp.length; i++) Wp[i] += sc * gWp[i];
    for (let i = 0; i < b.length; i++)  b[i]  += sc * gB[i];

    if (isNaN(E[0])) return { nParams, label, results, time: Date.now() - t0, diverged: true };

    if (step % reportEvery === 0 || step === steps) {
      results.push({ step, nll: evalNLL() });
    }
  }

  return { nParams, label, results, time: Date.now() - t0, diverged: false };
}


// =====================================================================
// PROJECTED T2 — T2 state with learned query readout
// =====================================================================
//
// Key insight: Instead of projecting BEFORE accumulation (linear attention,
// where Wk/Wv get no gradient without BPTT), project AFTER accumulation
// at readout time, where gradient flows through the output layer.
//
// Architecture:
//   State is EXACTLY T2: (count, sum_emb in R^D, outer_mat in R^{D x D})
//   where sum_emb = sum of e[cur], outer_mat = sum of e[prev] (x) e[cur]
//
//   At readout time:
//     q = Wq * E[prev]                  (D-dim query, gradient flows through Wq and E)
//     retrieved = outer_mat * q          (D-dim: "how much did past prev-chars match q"
//                                         weighted sum of corresponding cur-embeddings)
//     features = [1, sum_emb/count, retrieved/count]   (1 + 2*D dims)
//     logits = Wo * features + Wp * E[prev] + bias
//
// This computes: retrieved = sum_t (e_prev_t . q) * e_cur_t
//   = "weighted sum of past current-embeddings, weighted by similarity of
//      past prev-embeddings to the query"
//
// This IS linear attention, but with the projection at readout instead of
// accumulation! The outer_mat stores raw e[prev] (x) e[cur], and the
// query projection decides what to retrieve at output time.
//
// Gradient flow:
//   - Wq gets gradient: loss -> Wo -> retrieved -> outer_mat^T * dRetrieved -> dQ -> dWq
//   - E gets gradient from Wq path AND from Wp path AND from being used in state
//   - Wo, Wp, b get gradient as usual
//   - ALL parameters receive gradient (unlike linear attention where Wk/Wv are dead)
//
// Parameters:
//   E:  VOCAB * D           (embeddings, shared for state and readout)
//   Wq: D * D               (query projection at readout)
//   Wo: VOCAB * (1 + 2*D)   (output weights for [count, sum_emb/count, retrieved/count])
//   Wp: VOCAB * D           (prev-char direct connection)
//   b:  VOCAB               (bias)
//
// For D=16: 432 + 256 + 27*33=891 + 432 + 27 = 2038 params
//   (comparable to LinAttn's 2550, but ALL params get gradient!)

function runProjectedT2(config) {
  const { De, steps, batch, evalNames, lrStart, lrEnd, reportEvery, label } = config;

  const D = De;  // alias for clarity
  const D2 = D * D;
  const STATE_SIZE = 1 + D + D2;      // actual state (same as T2)
  const SD_out = 1 + 2 * D;           // features fed to output layer

  // Parameter counts
  const nE   = VOCAB * D;             // embeddings
  const nWq  = D * D;                 // query projection (D x D)
  const nWo  = VOCAB * SD_out;        // output: [count, sum_emb/count, retrieved/count]
  const nWp  = VOCAB * D;             // prev-char direct
  const nB   = VOCAB;                 // bias
  const nParams = nE + nWq + nWo + nWp + nB;

  // Allocate parameters
  const E   = new Float64Array(nE);
  const Wq  = new Float64Array(nWq);
  const Wo  = new Float64Array(nWo);
  const Wp  = new Float64Array(nWp);
  const b   = new Float64Array(nB);

  // Initialize
  const eScale  = Math.min(0.5, 2.0 / Math.sqrt(D));
  const wqScale = 1.0 / Math.sqrt(D);
  const woScale = Math.min(0.2, 1.0 / Math.sqrt(SD_out));
  for (let i = 0; i < E.length; i++)  E[i]  = (Math.random() - 0.5) * eScale;
  for (let i = 0; i < Wq.length; i++) Wq[i] = (Math.random() - 0.5) * wqScale;
  for (let i = 0; i < Wo.length; i++) Wo[i] = (Math.random() - 0.5) * woScale;
  for (let i = 0; i < Wp.length; i++) Wp[i] = (Math.random() - 0.5) * eScale;

  // Gradient buffers
  const gE  = new Float64Array(nE);
  const gWq = new Float64Array(nWq);
  const gWo = new Float64Array(nWo);
  const gWp = new Float64Array(nWp);
  const gB  = new Float64Array(nB);

  // Working buffers
  const logits    = new Float64Array(VOCAB);
  const probs     = new Float64Array(VOCAB);
  const state     = new Float64Array(STATE_SIZE);  // [count, sum_emb(D), outer_mat(D*D)]
  const query     = new Float64Array(D);           // query vector
  const retrieved = new Float64Array(D);           // retrieved vector

  const results = [];
  const t0 = Date.now();
  const evalN = Math.min(evalNames, names.length);

  function getLr(step) {
    const t = step / steps;
    return lrEnd + 0.5 * (lrStart - lrEnd) * (1 + Math.cos(Math.PI * t));
  }

  // Evaluate NLL on first evalN names
  function evalNLL() {
    let fullLL = 0, fullN = 0;
    for (let ni = 0; ni < evalN; ni++) {
      const chars = nameChars[ni];
      const seqLen = chars.length - 1;
      fullN += seqLen;

      // Init state: count=1, sum_emb=E['.'], outer_mat=zeros
      state[0] = 1;
      for (let d = 0; d < D; d++) state[1 + d] = E[d]; // E['.'] = E[0*D..]
      for (let d = 0; d < D2; d++) state[1 + D + d] = 0;

      for (let j = 0; j < seqLen; j++) {
        const prevI = chars[j];
        const tgt = chars[j + 1];
        const inv = 1.0 / state[0];

        // Compute query: q = Wq * E[prev]
        const peOff = prevI * D;
        for (let i = 0; i < D; i++) {
          let s = 0;
          const wOff = i * D;
          for (let j2 = 0; j2 < D; j2++) s += Wq[wOff + j2] * E[peOff + j2];
          query[i] = s;
        }

        // Retrieve: retrieved = outer_mat * q
        // outer_mat is D x D (row-major at state[1+D..]), retrieved[j] = sum_i outer_mat[j*D+i] * q[i]
        for (let j2 = 0; j2 < D; j2++) {
          let s = 0;
          const matOff = 1 + D + j2 * D;
          for (let i = 0; i < D; i++) s += state[matOff + i] * query[i];
          retrieved[j2] = s;
        }

        // Forward: logits = Wo * [1, sum_emb/count, retrieved/count] + Wp * E[prev] + b
        let maxL = -1e9;
        for (let k = 0; k < VOCAB; k++) {
          let v = b[k];
          const woBase = k * SD_out;
          // count feature (normalized = 1.0)
          v += Wo[woBase];
          // sum_emb / count
          for (let d = 0; d < D; d++) v += Wo[woBase + 1 + d] * state[1 + d] * inv;
          // retrieved / count
          for (let d = 0; d < D; d++) v += Wo[woBase + 1 + D + d] * retrieved[d] * inv;
          // prev-char direct
          const wpOff = k * D;
          for (let d = 0; d < D; d++) v += Wp[wpOff + d] * E[peOff + d];
          logits[k] = v;
          if (v > maxL) maxL = v;
        }
        let se = 0;
        for (let k = 0; k < VOCAB; k++) { probs[k] = Math.exp(logits[k] - maxL); se += probs[k]; }
        for (let k = 0; k < VOCAB; k++) probs[k] /= se;
        fullLL += Math.log(probs[tgt] + 1e-30);

        // Update state: count += 1, sum_emb += E[tgt], outer_mat += E[prev] (x) E[tgt]
        const ceOff = tgt * D;
        state[0] += 1;
        for (let d = 0; d < D; d++) state[1 + d] += E[ceOff + d];
        for (let di = 0; di < D; di++) {
          const ei = E[peOff + di], off = 1 + D + di * D;
          for (let dj = 0; dj < D; dj++) state[off + dj] += ei * E[ceOff + dj];
        }
      }
    }
    return -fullLL / fullN;
  }

  results.push({ step: 0, nll: evalNLL() });

  for (let step = 1; step <= steps; step++) {
    const currentLr = getLr(step);
    shuffle(nameIdx);
    gE.fill(0); gWq.fill(0); gWo.fill(0); gWp.fill(0); gB.fill(0);
    let batchN = 0;
    const bn = Math.min(batch, names.length);

    for (let bi = 0; bi < bn; bi++) {
      const ni = nameIdx[bi];
      const chars = nameChars[ni];
      const seqLen = chars.length - 1;
      batchN += seqLen;

      // Init state
      state[0] = 1;
      for (let d = 0; d < D; d++) state[1 + d] = E[d];
      for (let d = 0; d < D2; d++) state[1 + D + d] = 0;

      for (let j = 0; j < seqLen; j++) {
        const prevI = chars[j];
        const tgt = chars[j + 1];
        const inv = 1.0 / state[0];

        // Forward: compute query
        const peOff = prevI * D;
        for (let i = 0; i < D; i++) {
          let s = 0;
          const wOff = i * D;
          for (let j2 = 0; j2 < D; j2++) s += Wq[wOff + j2] * E[peOff + j2];
          query[i] = s;
        }

        // Forward: retrieve from outer_mat
        for (let j2 = 0; j2 < D; j2++) {
          let s = 0;
          const matOff = 1 + D + j2 * D;
          for (let i = 0; i < D; i++) s += state[matOff + i] * query[i];
          retrieved[j2] = s;
        }

        // Forward: compute logits and softmax
        let maxL = -1e9;
        for (let k = 0; k < VOCAB; k++) {
          let v = b[k];
          const woBase = k * SD_out;
          v += Wo[woBase];
          for (let d = 0; d < D; d++) v += Wo[woBase + 1 + d] * state[1 + d] * inv;
          for (let d = 0; d < D; d++) v += Wo[woBase + 1 + D + d] * retrieved[d] * inv;
          const wpOff = k * D;
          for (let d = 0; d < D; d++) v += Wp[wpOff + d] * E[peOff + d];
          logits[k] = v;
          if (v > maxL) maxL = v;
        }
        let se = 0;
        for (let k = 0; k < VOCAB; k++) { probs[k] = Math.exp(logits[k] - maxL); se += probs[k]; }
        for (let k = 0; k < VOCAB; k++) probs[k] /= se;

        // Backward: compute gradients
        // dl[k] = (k == tgt ? 1 : 0) - probs[k]
        const dRetrieved = new Float64Array(D);
        const dSumEmb = new Float64Array(D);
        const dPrevEmb = new Float64Array(D);

        for (let k = 0; k < VOCAB; k++) {
          const dl = (k === tgt ? 1 : 0) - probs[k];
          const woBase = k * SD_out;

          // d/d bias
          gB[k] += dl;

          // d/d Wo[k][0] (count feature)
          gWo[woBase] += dl;

          // d/d Wo[k][1..D] (sum_emb features) and backprop to sum_emb
          for (let d = 0; d < D; d++) {
            gWo[woBase + 1 + d] += dl * state[1 + d] * inv;
            dSumEmb[d] += dl * Wo[woBase + 1 + d] * inv;
          }

          // d/d Wo[k][1+D..1+2D] (retrieved features) and backprop to retrieved
          for (let d = 0; d < D; d++) {
            gWo[woBase + 1 + D + d] += dl * retrieved[d] * inv;
            dRetrieved[d] += dl * Wo[woBase + 1 + D + d] * inv;
          }

          // d/d Wp and d/d E[prev] via Wp path
          const wpOff = k * D;
          for (let d = 0; d < D; d++) {
            gWp[wpOff + d] += dl * E[peOff + d];
            dPrevEmb[d] += dl * Wp[wpOff + d];
          }
        }

        // Backprop through retrieval: retrieved[j] = sum_i outer_mat[j*D+i] * query[i]
        // d/d query[i] = sum_j dRetrieved[j] * outer_mat[j*D+i]
        //              = sum_j dRetrieved[j] * state[1+D+j*D+i]
        // (this is outer_mat^T * dRetrieved)
        const dQuery = new Float64Array(D);
        for (let i = 0; i < D; i++) {
          let s = 0;
          for (let j2 = 0; j2 < D; j2++) s += dRetrieved[j2] * state[1 + D + j2 * D + i];
          dQuery[i] = s;
        }

        // Backprop query: q = Wq * E[prev]
        // d/d Wq[i][j] += dQuery[i] * E[prev][j]
        // d/d E[prev][j] += sum_i dQuery[i] * Wq[i*D+j]
        for (let i = 0; i < D; i++) {
          const wOff = i * D;
          for (let j2 = 0; j2 < D; j2++) {
            gWq[wOff + j2] += dQuery[i] * E[peOff + j2];
            dPrevEmb[j2] += dQuery[i] * Wq[wOff + j2];
          }
        }

        // Apply accumulated E[prev] gradient
        for (let d = 0; d < D; d++) gE[peOff + d] += dPrevEmb[d];

        // Note: dSumEmb gives gradient w.r.t. the accumulated sum_emb in state.
        // We do NOT backprop through the state to past embeddings (same as T2).
        // But E still gets gradient from: (1) Wp path, (2) Wq path, and
        // (3) being the same E used in state accumulation (shared parameter).

        // Update state: count += 1, sum_emb += E[tgt], outer_mat += E[prev] (x) E[tgt]
        const ceOff = tgt * D;
        state[0] += 1;
        for (let d = 0; d < D; d++) state[1 + d] += E[ceOff + d];
        for (let di = 0; di < D; di++) {
          const ei = E[peOff + di], off = 1 + D + di * D;
          for (let dj = 0; dj < D; dj++) state[off + dj] += ei * E[ceOff + dj];
        }
      }
    }

    // Parameter update
    const sc = currentLr / batchN;
    for (let i = 0; i < E.length; i++)  E[i]  += sc * gE[i];
    for (let i = 0; i < Wq.length; i++) Wq[i] += sc * gWq[i];
    for (let i = 0; i < Wo.length; i++) Wo[i] += sc * gWo[i];
    for (let i = 0; i < Wp.length; i++) Wp[i] += sc * gWp[i];
    for (let i = 0; i < b.length; i++)  b[i]  += sc * gB[i];

    if (isNaN(E[0])) return { nParams, label, results, time: Date.now() - t0, diverged: true };

    if (step % reportEvery === 0 || step === steps) {
      results.push({ step, nll: evalNLL() });
    }
  }

  return { nParams, label, results, time: Date.now() - t0, diverged: false };
}


// =====================================================================
// MULTI-HEAD PROJECTED T2 — rank decomposition of readout
// =====================================================================
//
// Key insight: the D x D outer_mat state contains rich information, and
// ANY linear readout of it can be decomposed into rank-1 components.
// Each rank-1 component IS one attention head:
//   head_h: query_h = Wq_h * E[prev], retrieved_h = outer_mat * query_h
//
// Multi-head ProjT2 is this rank decomposition made explicit.
// H=1 recovers single-head ProjT2. H=D recovers full-rank readout.
//
// Architecture:
//   State: exactly T2 — (count, sum_emb in R^D, outer_mat in R^{D x D})
//   H query matrices: Wq_1..Wq_H, each D x D
//   Each head h: query_h = Wq_h * E[prev], retrieved_h = outer_mat * query_h  (D-dim)
//   Output features: [count, sum_emb/count, retrieved_1/count, ..., retrieved_H/count]
//     dimension = 1 + D + H*D = 1 + (1+H)*D
//   Logits: Wo * features + Wp * E[prev] + b
//
// Optional: positiveFeatureMap flag applies ELU+1 to query before retrieval,
// bridging toward softmax attention (positive feature maps).
//
// Parameters:
//   E:  VOCAB * D                       (embeddings)
//   Wq: H * D * D                       (H query matrices)
//   Wo: VOCAB * (1 + D + H*D)           (output)
//   Wp: VOCAB * D                       (prev-char direct)
//   b:  VOCAB                           (bias)
//
// At D=16:
//   H=1: E=432, Wq=256,  Wo=27*33=891,   Wp=432, b=27 -> 2,038
//   H=2: E=432, Wq=512,  Wo=27*49=1323,  Wp=432, b=27 -> 2,726
//   H=4: E=432, Wq=1024, Wo=27*81=2187,  Wp=432, b=27 -> 4,102
//   H=8: E=432, Wq=2048, Wo=27*145=3915, Wp=432, b=27 -> 6,854

function runMultiHeadProjT2(config) {
  const { De, H, steps, batch, evalNames, lrStart, lrEnd, reportEvery, label,
          positiveFeatureMap } = config;
  const usePFM = positiveFeatureMap || false;

  const D = De;
  const D2 = D * D;
  const STATE_SIZE = 1 + D + D2;              // actual state (same as T2)
  const SD_out = 1 + D + H * D;              // features: count + sum_emb + H*retrieved

  // Parameter counts
  const nE   = VOCAB * D;                     // embeddings
  const nWq  = H * D * D;                     // H query matrices (each D x D)
  const nWo  = VOCAB * SD_out;                // output weights
  const nWp  = VOCAB * D;                     // prev-char direct
  const nB   = VOCAB;                         // bias
  const nParams = nE + nWq + nWo + nWp + nB;

  // Allocate parameters
  const E   = new Float64Array(nE);
  const Wq  = new Float64Array(nWq);          // laid out as H blocks of D*D
  const Wo  = new Float64Array(nWo);
  const Wp  = new Float64Array(nWp);
  const b   = new Float64Array(nB);

  // Initialize
  const eScale  = Math.min(0.5, 2.0 / Math.sqrt(D));
  const wqScale = 1.0 / Math.sqrt(D);
  const woScale = Math.min(0.2, 1.0 / Math.sqrt(SD_out));
  for (let i = 0; i < E.length; i++)  E[i]  = (Math.random() - 0.5) * eScale;
  for (let i = 0; i < Wq.length; i++) Wq[i] = (Math.random() - 0.5) * wqScale;
  for (let i = 0; i < Wo.length; i++) Wo[i] = (Math.random() - 0.5) * woScale;
  for (let i = 0; i < Wp.length; i++) Wp[i] = (Math.random() - 0.5) * eScale;

  // Gradient buffers
  const gE  = new Float64Array(nE);
  const gWq = new Float64Array(nWq);
  const gWo = new Float64Array(nWo);
  const gWp = new Float64Array(nWp);
  const gB  = new Float64Array(nB);

  // Working buffers
  const logits    = new Float64Array(VOCAB);
  const probs     = new Float64Array(VOCAB);
  const state     = new Float64Array(STATE_SIZE);  // [count, sum_emb(D), outer_mat(D*D)]
  const queries   = new Float64Array(H * D);       // H query vectors, each D-dim
  const retrieveds = new Float64Array(H * D);      // H retrieved vectors, each D-dim

  const results = [];
  const t0 = Date.now();
  const evalN = Math.min(evalNames, names.length);

  function getLr(step) {
    const t = step / steps;
    return lrEnd + 0.5 * (lrStart - lrEnd) * (1 + Math.cos(Math.PI * t));
  }

  // ELU(x) + 1 positive feature map
  function eluPlus1(x) {
    return x >= 0 ? x + 1.0 : Math.exp(x);
  }
  // Derivative of ELU(x) + 1
  function eluPlus1Deriv(x) {
    return x >= 0 ? 1.0 : Math.exp(x);
  }

  // Compute queries for all H heads: queries[h*D..h*D+D] = Wq_h * E[prev]
  // If positiveFeatureMap, apply ELU+1 element-wise after linear projection
  function computeQueries(prevI) {
    const peOff = prevI * D;
    for (let h = 0; h < H; h++) {
      const wqBase = h * D2;
      const qBase = h * D;
      for (let i = 0; i < D; i++) {
        let s = 0;
        const wOff = wqBase + i * D;
        for (let j = 0; j < D; j++) s += Wq[wOff + j] * E[peOff + j];
        if (usePFM) {
          queries[qBase + i] = eluPlus1(s);
        } else {
          queries[qBase + i] = s;
        }
      }
    }
  }

  // Compute retrievals for all H heads: retrieved_h = outer_mat * query_h
  function computeRetrievals() {
    for (let h = 0; h < H; h++) {
      const qBase = h * D;
      const rBase = h * D;
      for (let j = 0; j < D; j++) {
        let s = 0;
        const matOff = 1 + D + j * D;
        for (let i = 0; i < D; i++) s += state[matOff + i] * queries[qBase + i];
        retrieveds[rBase + j] = s;
      }
    }
  }

  // Evaluate NLL on first evalN names
  function evalNLL() {
    let fullLL = 0, fullN = 0;
    for (let ni = 0; ni < evalN; ni++) {
      const chars = nameChars[ni];
      const seqLen = chars.length - 1;
      fullN += seqLen;

      // Init state: count=1, sum_emb=E['.'], outer_mat=zeros
      state[0] = 1;
      for (let d = 0; d < D; d++) state[1 + d] = E[d];
      for (let d = 0; d < D2; d++) state[1 + D + d] = 0;

      for (let j = 0; j < seqLen; j++) {
        const prevI = chars[j];
        const tgt = chars[j + 1];
        const inv = 1.0 / state[0];
        const peOff = prevI * D;

        // Compute queries and retrievals
        computeQueries(prevI);
        computeRetrievals();

        // Forward: logits = Wo * features + Wp * E[prev] + b
        // features = [1, sum_emb/count, retrieved_1/count, ..., retrieved_H/count]
        let maxL = -1e9;
        for (let k = 0; k < VOCAB; k++) {
          let v = b[k];
          const woBase = k * SD_out;
          // count feature (normalized = 1.0)
          v += Wo[woBase];
          // sum_emb / count
          for (let d = 0; d < D; d++) v += Wo[woBase + 1 + d] * state[1 + d] * inv;
          // retrieved_h / count for each head
          for (let h = 0; h < H; h++) {
            const rBase = h * D;
            const woOff = woBase + 1 + D + h * D;
            for (let d = 0; d < D; d++) v += Wo[woOff + d] * retrieveds[rBase + d] * inv;
          }
          // prev-char direct
          const wpOff = k * D;
          for (let d = 0; d < D; d++) v += Wp[wpOff + d] * E[peOff + d];
          logits[k] = v;
          if (v > maxL) maxL = v;
        }
        let se = 0;
        for (let k = 0; k < VOCAB; k++) { probs[k] = Math.exp(logits[k] - maxL); se += probs[k]; }
        for (let k = 0; k < VOCAB; k++) probs[k] /= se;
        fullLL += Math.log(probs[tgt] + 1e-30);

        // Update state
        const ceOff = tgt * D;
        state[0] += 1;
        for (let d = 0; d < D; d++) state[1 + d] += E[ceOff + d];
        for (let di = 0; di < D; di++) {
          const ei = E[peOff + di], off = 1 + D + di * D;
          for (let dj = 0; dj < D; dj++) state[off + dj] += ei * E[ceOff + dj];
        }
      }
    }
    return -fullLL / fullN;
  }

  results.push({ step: 0, nll: evalNLL() });

  for (let step = 1; step <= steps; step++) {
    const currentLr = getLr(step);
    shuffle(nameIdx);
    gE.fill(0); gWq.fill(0); gWo.fill(0); gWp.fill(0); gB.fill(0);
    let batchN = 0;
    const bn = Math.min(batch, names.length);

    for (let bi = 0; bi < bn; bi++) {
      const ni = nameIdx[bi];
      const chars = nameChars[ni];
      const seqLen = chars.length - 1;
      batchN += seqLen;

      // Init state
      state[0] = 1;
      for (let d = 0; d < D; d++) state[1 + d] = E[d];
      for (let d = 0; d < D2; d++) state[1 + D + d] = 0;

      for (let j = 0; j < seqLen; j++) {
        const prevI = chars[j];
        const tgt = chars[j + 1];
        const inv = 1.0 / state[0];
        const peOff = prevI * D;

        // Forward: compute queries and retrievals
        computeQueries(prevI);
        computeRetrievals();

        // Forward: compute logits and softmax
        let maxL = -1e9;
        for (let k = 0; k < VOCAB; k++) {
          let v = b[k];
          const woBase = k * SD_out;
          v += Wo[woBase];
          for (let d = 0; d < D; d++) v += Wo[woBase + 1 + d] * state[1 + d] * inv;
          for (let h = 0; h < H; h++) {
            const rBase = h * D;
            const woOff = woBase + 1 + D + h * D;
            for (let d = 0; d < D; d++) v += Wo[woOff + d] * retrieveds[rBase + d] * inv;
          }
          const wpOff = k * D;
          for (let d = 0; d < D; d++) v += Wp[wpOff + d] * E[peOff + d];
          logits[k] = v;
          if (v > maxL) maxL = v;
        }
        let se = 0;
        for (let k = 0; k < VOCAB; k++) { probs[k] = Math.exp(logits[k] - maxL); se += probs[k]; }
        for (let k = 0; k < VOCAB; k++) probs[k] /= se;

        // Backward: compute gradients
        const dRetrieveds = new Float64Array(H * D);  // gradient for each head's retrieved
        const dPrevEmb = new Float64Array(D);

        for (let k = 0; k < VOCAB; k++) {
          const dl = (k === tgt ? 1 : 0) - probs[k];
          const woBase = k * SD_out;

          // d/d bias
          gB[k] += dl;

          // d/d Wo[k][0] (count feature)
          gWo[woBase] += dl;

          // d/d Wo[k][1..D] (sum_emb features)
          for (let d = 0; d < D; d++) {
            gWo[woBase + 1 + d] += dl * state[1 + d] * inv;
          }

          // d/d Wo for each head's retrieved features, and backprop to dRetrieveds
          for (let h = 0; h < H; h++) {
            const rBase = h * D;
            const woOff = woBase + 1 + D + h * D;
            for (let d = 0; d < D; d++) {
              gWo[woOff + d] += dl * retrieveds[rBase + d] * inv;
              dRetrieveds[rBase + d] += dl * Wo[woOff + d] * inv;
            }
          }

          // d/d Wp and d/d E[prev] via Wp path
          const wpOff = k * D;
          for (let d = 0; d < D; d++) {
            gWp[wpOff + d] += dl * E[peOff + d];
            dPrevEmb[d] += dl * Wp[wpOff + d];
          }
        }

        // Backprop through each head's retrieval and query
        for (let h = 0; h < H; h++) {
          const rBase = h * D;
          const qBase = h * D;
          const wqBase = h * D2;

          // Backprop retrieval: retrieved_h[j] = sum_i outer_mat[j*D+i] * query_h[i]
          // d/d query_h[i] = sum_j dRetrieved_h[j] * outer_mat[j*D+i]
          const dQuery = new Float64Array(D);
          for (let i = 0; i < D; i++) {
            let s = 0;
            for (let j2 = 0; j2 < D; j2++) s += dRetrieveds[rBase + j2] * state[1 + D + j2 * D + i];
            dQuery[i] = s;
          }

          // If positive feature map, backprop through ELU+1
          // query_h[i] = eluPlus1(linear_h[i]), so dLinear_h[i] = dQuery[i] * eluPlus1Deriv(linear_h[i])
          // We need the pre-activation value. Recompute the linear part.
          if (usePFM) {
            for (let i = 0; i < D; i++) {
              // Recompute linear value: linear = Wq_h[i] . E[prev]
              let lin = 0;
              const wOff = wqBase + i * D;
              for (let j2 = 0; j2 < D; j2++) lin += Wq[wOff + j2] * E[peOff + j2];
              dQuery[i] *= eluPlus1Deriv(lin);
            }
          }

          // Backprop query: q_h = Wq_h * E[prev] (or ELU+1 applied after)
          // d/d Wq_h[i][j] += dQuery[i] * E[prev][j]
          // d/d E[prev][j] += sum_i dQuery[i] * Wq_h[i*D+j]
          for (let i = 0; i < D; i++) {
            const wOff = wqBase + i * D;
            for (let j2 = 0; j2 < D; j2++) {
              gWq[wOff + j2] += dQuery[i] * E[peOff + j2];
              dPrevEmb[j2] += dQuery[i] * Wq[wOff + j2];
            }
          }
        }

        // Apply accumulated E[prev] gradient
        for (let d = 0; d < D; d++) gE[peOff + d] += dPrevEmb[d];

        // Update state: count += 1, sum_emb += E[tgt], outer_mat += E[prev] (x) E[tgt]
        const ceOff = tgt * D;
        state[0] += 1;
        for (let d = 0; d < D; d++) state[1 + d] += E[ceOff + d];
        for (let di = 0; di < D; di++) {
          const ei = E[peOff + di], off = 1 + D + di * D;
          for (let dj = 0; dj < D; dj++) state[off + dj] += ei * E[ceOff + dj];
        }
      }
    }

    // Parameter update
    const sc = currentLr / batchN;
    for (let i = 0; i < E.length; i++)  E[i]  += sc * gE[i];
    for (let i = 0; i < Wq.length; i++) Wq[i] += sc * gWq[i];
    for (let i = 0; i < Wo.length; i++) Wo[i] += sc * gWo[i];
    for (let i = 0; i < Wp.length; i++) Wp[i] += sc * gWp[i];
    for (let i = 0; i < b.length; i++)  b[i]  += sc * gB[i];

    if (isNaN(E[0])) return { nParams, label, results, time: Date.now() - t0, diverged: true };

    if (step % reportEvery === 0 || step === steps) {
      results.push({ step, nll: evalNLL() });
    }
  }

  return { nParams, label, results, time: Date.now() - t0, diverged: false };
}


// =====================================================================
// EXPERIMENTS
// =====================================================================

const totalStart = Date.now();

console.log('\n============================================================');
console.log('Count-Based Bigram (MLE baseline)');
console.log('============================================================');
const countNLL = runCountBigram();
console.log(`  NLL = ${countNLL.toFixed(4)} (Karpathy target: ~2.454)`);

const allResults = [];

function runAndReport(runFn, config) {
  const label = config.label;
  console.log(`\n============================================================`);
  console.log(`${label}`);
  console.log(`  params will be computed inside...`);
  console.log(`============================================================`);

  // Try with configured LR, then fall back to lower LR if diverges
  let result = null;
  const lrTries = [
    { lrStart: config.lrStart, lrEnd: config.lrEnd },
    { lrStart: config.lrStart * 0.4, lrEnd: config.lrEnd * 0.4 },
    { lrStart: config.lrStart * 0.1, lrEnd: config.lrEnd * 0.1 },
  ];

  for (const lr of lrTries) {
    result = runFn({ ...config, ...lr });
    if (!result.diverged) break;
    console.log(`  lr=${lr.lrStart.toFixed(2)}->${lr.lrEnd.toFixed(2)} diverged, retrying...`);
  }

  if (result.diverged) {
    console.log(`  All learning rates diverged. Skipping.`);
    allResults.push({ label, nParams: result.nParams, bestNLL: NaN, results: [], time: 0 });
    return;
  }

  console.log(`  Parameters: ${result.nParams}`);
  for (const r of result.results) {
    console.log(`  Step ${String(r.step).padStart(4)}: NLL = ${r.nll.toFixed(4)}`);
  }
  console.log(`  Time: ${(result.time / 1000).toFixed(1)}s`);

  const bestNLL = Math.min(...result.results.map(r => r.nll));
  allResults.push({ label, nParams: result.nParams, bestNLL, results: result.results, time: result.time });
}


// -----------------------------------------------------------------
// EXPERIMENT 1a: T2(R^16) baseline (large)
// -----------------------------------------------------------------
// Expected ~2.21 NLL. This is our reference point.
// Params: 27*16 + 27*(1+16+256) + 27*16 + 27 = 432 + 7371 + 432 + 27 = 8262

runAndReport(runT2Baseline, {
  label: 'T2(R^16) baseline [raw embeddings]',
  D: 16,
  steps: 200,
  batch: 300,
  evalNames: 8000,
  lrStart: 5.0,
  lrEnd: 0.5,
  reportEvery: 50,
});

// -----------------------------------------------------------------
// EXPERIMENT 1b: T2(R^8) baseline (param-matched to linear attention)
// -----------------------------------------------------------------
// Params: 27*8 + 27*(1+8+64) + 27*8 + 27 = 216 + 1971 + 216 + 27 = 2430
// This is close to LinAttn's ~2550 params, so it's a fair comparison.

runAndReport(runT2Baseline, {
  label: 'T2(R^8) baseline [param-matched]',
  D: 8,
  steps: 200,
  batch: 300,
  evalNames: 8000,
  lrStart: 5.0,
  lrEnd: 0.5,
  reportEvery: 50,
});

// -----------------------------------------------------------------
// EXPERIMENT 2: Linear attention (small) — fewer params than T2
// -----------------------------------------------------------------
// De=16, Dk=8, Dv=8:
//   E: 432, Wk: 128, Wv: 128, Wq: 128
//   Wo: 27*(1+8+8)=459, Wp: 432, b: 27
//   Total: 432+128+128+128+459+432+27 = 1734

runAndReport(runLinearAttention, {
  label: 'LinAttn De=16 Dk=8 Dv=8 [medium]',
  De: 16, Dk: 8, Dv: 8,
  steps: 200,
  batch: 300,
  evalNames: 8000,
  lrStart: 5.0,
  lrEnd: 0.5,
  reportEvery: 50,
});

// -----------------------------------------------------------------
// EXPERIMENT 3: Linear attention (matched dim to T2)
// -----------------------------------------------------------------
// De=16, Dk=16, Dv=16:
//   E: 432, Wk: 256, Wv: 256, Wq: 256
//   Wo: 27*(1+16+16)=891, Wp: 432, b: 27
//   Total: 432+256+256+256+891+432+27 = 2550

runAndReport(runLinearAttention, {
  label: 'LinAttn De=16 Dk=16 Dv=16 [matched dim]',
  De: 16, Dk: 16, Dv: 16,
  steps: 200,
  batch: 300,
  evalNames: 8000,
  lrStart: 5.0,
  lrEnd: 0.5,
  reportEvery: 50,
});


// -----------------------------------------------------------------
// EXPERIMENT 4: Projected T2 (De=16) — query readout, all grads flow
// -----------------------------------------------------------------
// De=16, Wq is 16x16:
//   E: 432, Wq: 256, Wo: 27*(1+32)=891, Wp: 432, b: 27
//   Total: 432+256+891+432+27 = 2038

runAndReport(runProjectedT2, {
  label: 'ProjT2 De=16 [query readout]',
  De: 16,
  steps: 200,
  batch: 300,
  evalNames: 8000,
  lrStart: 5.0,
  lrEnd: 0.5,
  reportEvery: 50,
});

// -----------------------------------------------------------------
// EXPERIMENT 5: Projected T2 (De=8) — param-matched comparison
// -----------------------------------------------------------------
// De=8, Wq is 8x8:
//   E: 216, Wq: 64, Wo: 27*(1+16)=459, Wp: 216, b: 27
//   Total: 216+64+459+216+27 = 982

runAndReport(runProjectedT2, {
  label: 'ProjT2 De=8 [param-matched]',
  De: 8,
  steps: 200,
  batch: 300,
  evalNames: 8000,
  lrStart: 5.0,
  lrEnd: 0.5,
  reportEvery: 50,
});


// -----------------------------------------------------------------
// EXPERIMENT 6: Multi-head ProjT2 sweep: H=1 (same as ProjT2), H=2, H=4, H=8
// -----------------------------------------------------------------
// All at D=16, same training config as ProjT2.
// H=1: E=432, Wq=256,  Wo=27*33=891,   Wp=432, b=27 -> 2,038 (matches ProjT2)
// H=2: E=432, Wq=512,  Wo=27*49=1323,  Wp=432, b=27 -> 2,726
// H=4: E=432, Wq=1024, Wo=27*81=2187,  Wp=432, b=27 -> 4,102
// H=8: E=432, Wq=2048, Wo=27*145=3915, Wp=432, b=27 -> 6,854

runAndReport(runMultiHeadProjT2, {
  label: 'MultiHead ProjT2 D=16 H=1',
  De: 16, H: 1,
  steps: 200,
  batch: 300,
  evalNames: 8000,
  lrStart: 5.0,
  lrEnd: 0.5,
  reportEvery: 50,
});

runAndReport(runMultiHeadProjT2, {
  label: 'MultiHead ProjT2 D=16 H=2',
  De: 16, H: 2,
  steps: 200,
  batch: 300,
  evalNames: 8000,
  lrStart: 5.0,
  lrEnd: 0.5,
  reportEvery: 50,
});

runAndReport(runMultiHeadProjT2, {
  label: 'MultiHead ProjT2 D=16 H=4',
  De: 16, H: 4,
  steps: 200,
  batch: 300,
  evalNames: 8000,
  lrStart: 5.0,
  lrEnd: 0.5,
  reportEvery: 50,
});

runAndReport(runMultiHeadProjT2, {
  label: 'MultiHead ProjT2 D=16 H=8',
  De: 16, H: 8,
  steps: 200,
  batch: 300,
  evalNames: 8000,
  lrStart: 5.0,
  lrEnd: 0.5,
  reportEvery: 50,
});

// -----------------------------------------------------------------
// EXPERIMENT 7: Multi-head ProjT2 H=4 with positive feature map (ELU+1)
// -----------------------------------------------------------------
// Same as H=4 but applies ELU(x)+1 to query, bridging toward softmax attention.
// ELU+1 is a positive feature map: all query components are positive, so
// the retrieval becomes a positive-weighted sum (like softmax attention).

runAndReport(runMultiHeadProjT2, {
  label: 'MultiHead ProjT2 D=16 H=4 [ELU+1]',
  De: 16, H: 4,
  positiveFeatureMap: true,
  steps: 200,
  batch: 300,
  evalNames: 8000,
  lrStart: 5.0,
  lrEnd: 0.5,
  reportEvery: 50,
});


// =====================================================================
// GRAND SUMMARY
// =====================================================================

console.log('\n\n================================================================');
console.log('GRAND SUMMARY');
console.log('================================================================');
console.log(`Count-based bigram:    NLL = ${countNLL.toFixed(4)}`);
console.log(`Karpathy MLP (~2.3),  Karpathy RNN (~2.0)\n`);

console.log('Experiment                                     | Params |  Best NLL |  Time');
console.log('-----------------------------------------------|--------|-----------|------');

for (const r of allResults) {
  const bestStr = isNaN(r.bestNLL) ? '     N/A' : `   ${r.bestNLL.toFixed(4)}`;
  const parStr = String(r.nParams).padStart(6);
  console.log(
    `${r.label.padEnd(46)} | ${parStr} | ${bestStr} | ${(r.time / 1000).toFixed(1).padStart(5)}s`
  );
}

// Detailed progression
console.log('\n\nDETAILED PROGRESSION:');
for (const r of allResults) {
  if (r.results.length === 0) continue;
  const vals = r.results.map(x => `${x.step}:${x.nll.toFixed(3)}`).join('  ');
  console.log(`  ${r.label}:`);
  console.log(`    ${vals}`);
}

// Analysis
console.log('\n\nANALYSIS:');

const t2_16 = allResults.find(r => r.label.includes('T2(R^16)'));
const t2_8 = allResults.find(r => r.label.includes('T2(R^8)'));
const linResults = allResults.filter(r => r.label.includes('LinAttn') && !isNaN(r.bestNLL));
const projResults = allResults.filter(r => r.label.includes('ProjT2') && !r.label.includes('MultiHead') && !isNaN(r.bestNLL));
const mhResults = allResults.filter(r => r.label.includes('MultiHead') && !isNaN(r.bestNLL));

if (t2_16 && !isNaN(t2_16.bestNLL)) {
  console.log(`\nT2(R^16) baseline: NLL = ${t2_16.bestNLL.toFixed(4)} with ${t2_16.nParams} params`);
}
if (t2_8 && !isNaN(t2_8.bestNLL)) {
  console.log(`T2(R^8) baseline:  NLL = ${t2_8.bestNLL.toFixed(4)} with ${t2_8.nParams} params`);
}

if (linResults.length > 0) {
  const bestLin = linResults.reduce((a, b) => b.bestNLL < a.bestNLL ? b : a);
  console.log(`Best linear attention: NLL = ${bestLin.bestNLL.toFixed(4)} with ${bestLin.nParams} params`);
  console.log(`  Config: ${bestLin.label}`);
}

if (projResults.length > 0) {
  const bestProj = projResults.reduce((a, b) => b.bestNLL < a.bestNLL ? b : a);
  console.log(`Best projected T2:    NLL = ${bestProj.bestNLL.toFixed(4)} with ${bestProj.nParams} params`);
  console.log(`  Config: ${bestProj.label}`);
}

if (mhResults.length > 0) {
  const bestMH = mhResults.reduce((a, b) => b.bestNLL < a.bestNLL ? b : a);
  console.log(`Best multi-head ProjT2: NLL = ${bestMH.bestNLL.toFixed(4)} with ${bestMH.nParams} params`);
  console.log(`  Config: ${bestMH.label}`);
}

// Compare all architectures
const allWithNLL = allResults.filter(r => !isNaN(r.bestNLL));
if (allWithNLL.length > 1) {
  const best = allWithNLL.reduce((a, b) => b.bestNLL < a.bestNLL ? b : a);
  console.log(`\nOverall best: ${best.label} with NLL = ${best.bestNLL.toFixed(4)} (${best.nParams} params)`);

  // Parameter efficiency comparison
  console.log('\nParameter efficiency (params per NLL point below count bigram):');
  for (const r of allWithNLL) {
    const improvement = countNLL - r.bestNLL;
    if (improvement > 0) {
      const efficiency = r.nParams / improvement;
      console.log(`  ${r.label}: ${efficiency.toFixed(0)} params per 0.001 NLL improvement`);
    }
  }
}

// Multi-head rank decomposition curve
if (mhResults.length > 0) {
  console.log('\n\n--- MULTI-HEAD RANK DECOMPOSITION CURVE (D=16) ---');
  console.log('Heads (H) | Params | Best NLL | Notes');
  console.log('----------|--------|----------|------');
  // Sort by number of heads
  const mhSorted = [...mhResults].sort((a, b) => a.nParams - b.nParams);
  for (const r of mhSorted) {
    const hMatch = r.label.match(/H=(\d+)/);
    const hVal = hMatch ? hMatch[1] : '?';
    const pfm = r.label.includes('ELU+1') ? ' (ELU+1 feature map)' : '';
    console.log(`    ${hVal.padStart(5)} | ${String(r.nParams).padStart(6)} | ${r.bestNLL.toFixed(4)}   |${pfm}`);
  }
  console.log('');
  console.log('Interpretation: each head adds a rank-1 readout of the D x D outer product');
  console.log('matrix. H=1 is ProjT2 (single query). As H increases toward D, we approach');
  console.log('the full-rank readout of T2 (which reads all D^2 entries via Ws).');
  console.log('The curve shows the marginal value of each additional attention head.');

  // Check for diminishing returns
  const nonPFM = mhSorted.filter(r => !r.label.includes('ELU+1'));
  if (nonPFM.length >= 2) {
    const first = nonPFM[0];
    const last = nonPFM[nonPFM.length - 1];
    const nllDrop = first.bestNLL - last.bestNLL;
    const paramIncrease = last.nParams - first.nParams;
    if (nllDrop > 0) {
      console.log(`\nGoing from ${first.nParams} to ${last.nParams} params (+${paramIncrease})`);
      console.log(`  NLL improves by ${nllDrop.toFixed(4)} (from ${first.bestNLL.toFixed(4)} to ${last.bestNLL.toFixed(4)})`);
      console.log(`  Efficiency: ${(paramIncrease / nllDrop).toFixed(0)} params per NLL point`);
    }
  }

  // Compare ELU+1 vs standard
  const pfmResult = mhResults.find(r => r.label.includes('ELU+1'));
  const stdH4 = mhResults.find(r => r.label.includes('H=4') && !r.label.includes('ELU+1'));
  if (pfmResult && stdH4) {
    const delta = pfmResult.bestNLL - stdH4.bestNLL;
    console.log('\n--- ELU+1 Positive Feature Map (H=4) ---');
    if (Math.abs(delta) < 0.01) {
      console.log(`  ELU+1 and standard are comparable (delta = ${Math.abs(delta).toFixed(4)} NLL)`);
    } else if (delta < 0) {
      console.log(`  ** ELU+1 BEATS standard by ${(-delta).toFixed(4)} NLL **`);
      console.log(`  Positive feature maps help! This bridges toward softmax attention.`);
    } else {
      console.log(`  Standard BEATS ELU+1 by ${delta.toFixed(4)} NLL`);
      console.log(`  The positivity constraint hurts — signed queries are more expressive.`);
    }
  }
}

// T2 vs ProjT2 comparison
if (t2_16 && !isNaN(t2_16.bestNLL) && projResults.length > 0) {
  const bestProj = projResults.reduce((a, b) => b.bestNLL < a.bestNLL ? b : a);
  const delta = bestProj.bestNLL - t2_16.bestNLL;
  console.log('\n--- T2 vs Projected T2 ---');
  if (delta < -0.01) {
    console.log(`  ** ProjT2 BEATS T2(R^16) by ${(-delta).toFixed(4)} NLL **`);
    console.log(`  ProjT2 uses ${bestProj.nParams} params vs T2's ${t2_16.nParams} (${((bestProj.nParams/t2_16.nParams)*100).toFixed(0)}%)`);
    console.log(`  The query-readout approach is more parameter-efficient: it compresses`);
    console.log(`  the D^2 outer product matrix into a D-dim retrieval using learned queries,`);
    console.log(`  needing far fewer output weights while maintaining expressiveness.`);
  } else if (delta < 0.01) {
    console.log(`  ProjT2 is COMPARABLE to T2(R^16) (delta = ${Math.abs(delta).toFixed(4)} NLL)`);
    console.log(`  ProjT2 uses ${bestProj.nParams} params vs T2's ${t2_16.nParams} (${((bestProj.nParams/t2_16.nParams)*100).toFixed(0)}%)`);
    console.log(`  Similar performance with far fewer parameters!`);
  } else {
    console.log(`  T2(R^16) WINS by ${delta.toFixed(4)} NLL`);
    console.log(`  The full D^2 state readout captures more than query-based retrieval.`);
    console.log(`  But ProjT2 uses only ${bestProj.nParams} vs ${t2_16.nParams} params.`);
  }
}

// T2 vs best Multi-head comparison
if (t2_16 && !isNaN(t2_16.bestNLL) && mhResults.length > 0) {
  const bestMH = mhResults.reduce((a, b) => b.bestNLL < a.bestNLL ? b : a);
  const delta = bestMH.bestNLL - t2_16.bestNLL;
  console.log('\n--- T2(R^16) vs Best Multi-Head ProjT2 ---');
  if (delta < -0.01) {
    console.log(`  ** Multi-head ProjT2 BEATS T2(R^16) by ${(-delta).toFixed(4)} NLL **`);
    console.log(`  Multi-head uses ${bestMH.nParams} params vs T2's ${t2_16.nParams} (${((bestMH.nParams/t2_16.nParams)*100).toFixed(0)}%)`);
  } else if (delta < 0.01) {
    console.log(`  Multi-head ProjT2 is COMPARABLE to T2(R^16) (delta = ${Math.abs(delta).toFixed(4)} NLL)`);
    console.log(`  Multi-head uses ${bestMH.nParams} params vs T2's ${t2_16.nParams} (${((bestMH.nParams/t2_16.nParams)*100).toFixed(0)}%)`);
  } else {
    console.log(`  T2(R^16) WINS by ${delta.toFixed(4)} NLL (full rank readout advantage)`);
    console.log(`  Multi-head uses ${bestMH.nParams} params vs T2's ${t2_16.nParams}.`);
  }
}

// ProjT2 vs LinAttn comparison
if (projResults.length > 0 && linResults.length > 0) {
  const bestProj = projResults.reduce((a, b) => b.bestNLL < a.bestNLL ? b : a);
  const bestLin = linResults.reduce((a, b) => b.bestNLL < a.bestNLL ? b : a);
  const delta = bestLin.bestNLL - bestProj.bestNLL;
  console.log('\n--- Projected T2 vs Linear Attention ---');
  if (delta > 0.01) {
    console.log(`  ** ProjT2 BEATS LinAttn by ${delta.toFixed(4)} NLL **`);
    console.log(`  This confirms the gradient hypothesis: projecting at readout (where`);
    console.log(`  gradient flows) beats projecting at accumulation (where Wk/Wv are dead).`);
    console.log(`  ProjT2 uses ${bestProj.nParams} params vs LinAttn's ${bestLin.nParams}.`);
  } else if (delta > -0.01) {
    console.log(`  ProjT2 and LinAttn are COMPARABLE (delta = ${Math.abs(delta).toFixed(4)} NLL)`);
  } else {
    console.log(`  LinAttn BEATS ProjT2 by ${(-delta).toFixed(4)} NLL`);
    console.log(`  Surprising: even without Wk/Wv gradients, LinAttn wins.`);
  }
}

// Gradient issue
console.log('\n\nGRADIENT ANALYSIS:');
console.log('Four approaches to combining projections with monoid state:');
console.log('');
console.log('  1. T2: No projections. E gets gradient from Wp*E[prev] bigram path.');
console.log('     State uses same E, so it improves for free. Simple, works well,');
console.log('     but output layer has D^2 weights (scales badly).');
console.log('');
console.log('  2. LinAttn: Project BEFORE accumulation (Wk, Wv on input side).');
console.log('     Wk and Wv get ZERO gradient without BPTT (they only affect state,');
console.log('     and we detach state from the computation graph). Dead parameters.');
console.log('');
console.log('  3. ProjT2 (H=1): Project AFTER accumulation (Wq at readout time).');
console.log('     Wq gets full gradient: loss -> Wo -> retrieved -> outer_mat^T -> dQ -> dWq.');
console.log('     This is the key insight: put learnable projections where gradient flows!');
console.log('     Same state as T2 (monoid structure preserved), but output uses query');
console.log('     retrieval (1+2D features) instead of full flattening (1+D+D^2 features).');
console.log('');
console.log('  4. Multi-Head ProjT2 (H>1): H independent query projections at readout.');
console.log('     Each head reads a rank-1 slice of the outer product matrix.');
console.log('     Output has 1+(1+H)*D features. This IS multi-head linear attention,');
console.log('     derived as the rank decomposition of the T2 readout.');
console.log('     H=1 is ProjT2, H=D is equivalent to full T2 readout (full rank).');

// Key theoretical insight
console.log('\n\nTHEORETICAL NOTE:');
console.log('All architectures implement monoid homomorphisms from');
console.log('(List Char, ++) to (State, +). The differences are in readout:');
console.log('');
console.log('  T2:       state = (1, e, e(x)e)     readout = W_s * flatten(state/count)');
console.log('  LinAttn:  state = (1, v, k(x)v)     readout = q^T * KV_mat / count');
console.log('  ProjT2:   state = (1, e, e(x)e)     readout = (Wq*e_prev)^T * outer/count');
console.log('  MH-ProjT2: state = (1, e, e(x)e)    readout = [(Wq_h*e_prev)^T * outer/count]_{h=1..H}');
console.log('');
console.log('Multi-head ProjT2 reveals that multi-head attention is the RANK DECOMPOSITION');
console.log('of the T2 output layer. Each head provides one rank-1 readout of the D x D');
console.log('outer product matrix. This connects the algebraic (T2) and neural (attention)');
console.log('perspectives: they differ only in the rank of the readout projection.');
console.log('');
console.log('The positive feature map variant (ELU+1) bridges further toward softmax');
console.log('attention: with positive queries and keys, the retrieval becomes a');
console.log('positive-weighted combination, mimicking the attention probability structure.');

console.log(`\nTotal time: ${((Date.now() - totalStart) / 1000).toFixed(1)}s`);
