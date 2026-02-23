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

const t2Result = allResults.find(r => r.label.includes('T2'));
const linResults = allResults.filter(r => r.label.includes('LinAttn') && !isNaN(r.bestNLL));

if (t2Result && !isNaN(t2Result.bestNLL)) {
  console.log(`\nT2(R^16) baseline: NLL = ${t2Result.bestNLL.toFixed(4)} with ${t2Result.nParams} params`);

  if (linResults.length > 0) {
    const bestLin = linResults.reduce((a, b) => b.bestNLL < a.bestNLL ? b : a);
    console.log(`Best linear attention: NLL = ${bestLin.bestNLL.toFixed(4)} with ${bestLin.nParams} params`);
    console.log(`  Config: ${bestLin.label}`);

    const delta = bestLin.bestNLL - t2Result.bestNLL;
    if (delta < 0) {
      console.log(`\n  ** Linear attention BEATS T2 by ${(-delta).toFixed(4)} NLL **`);
      console.log(`  The learned key/value/query projections improve over raw embeddings.`);
      console.log(`  This suggests that learning WHAT to store and WHAT to retrieve,`);
      console.log(`  within the monoid framework, adds genuine expressiveness.`);
    } else if (delta < 0.05) {
      console.log(`\n  Linear attention is COMPARABLE to T2 (delta = ${delta.toFixed(4)} NLL).`);
      console.log(`  The learned projections don't help much -- raw embeddings already`);
      console.log(`  capture the relevant structure at this scale.`);
    } else {
      console.log(`\n  T2 baseline WINS by ${delta.toFixed(4)} NLL.`);
      console.log(`  Possible explanations:`);
      console.log(`  - T2 feeds D^2 features directly to output (richer state readout)`);
      console.log(`  - Linear attention's compression through query bottleneck loses info`);
      console.log(`  - More parameters in projections = harder optimization`);
      console.log(`  - At this scale, the simpler model trains better`);
    }

    // Parameter efficiency comparison
    console.log('\nParameter efficiency (params per NLL point below count bigram):');
    for (const r of [t2Result, ...linResults]) {
      if (isNaN(r.bestNLL)) continue;
      const improvement = countNLL - r.bestNLL;
      if (improvement > 0) {
        const efficiency = r.nParams / improvement;
        console.log(`  ${r.label}: ${efficiency.toFixed(0)} params per 0.001 NLL improvement`);
      }
    }
  }
}

// Gradient issue
console.log('\n\nIMPORTANT — GRADIENT ISSUE:');
console.log('The Wk (key) and Wv (value) projection matrices receive ZERO gradient!');
console.log('This is because we do not backprop through the accumulated state (same as T2).');
console.log('In T2 this is fine: E gets gradient from the Wp*E[prev] direct bigram path,');
console.log('and the state uses the SAME E, so it improves for free.');
console.log('In linear attention, Wk and Wv only affect the state — they have no');
console.log('gradient-receiving path. They remain at random initialization.');
console.log('');
console.log('This means linear attention is handicapped: its projections are untrained.');
console.log('A fair comparison would require BPTT (backprop through time) to train Wk/Wv.');
console.log('The T2(R^8) param-matched baseline shows what T2 achieves at the same scale.');

// Key theoretical insight
console.log('\n\nTHEORETICAL NOTE:');
console.log('Both T2 and linear attention implement monoid homomorphisms from');
console.log('(List Char, ++) to (State, +). The key difference:');
console.log('');
console.log('  T2:             state += e[prev] (x) e[cur]     (raw embeddings)');
console.log('  Lin. attention: state += k[prev] (x) v[cur]     (learned projections)');
console.log('');
console.log('T2 stores e(x)e which is a SYMMETRIC function of the embedding.');
console.log('Linear attention stores k(x)v which is ASYMMETRIC: the key/value split');
console.log('lets the model learn different representations for "what happened"');
console.log('(keys) vs "what matters" (values). The query further selects what to');
console.log('retrieve based on the current context.');
console.log('');
console.log('This is the connection to attention: in standard transformers, linear');
console.log('attention computes sum_i phi(k_i) (x) v_i as state, and retrieves via');
console.log('phi(q)^T * state. Our version is identical, with phi = identity (linear');
console.log('feature map), applied to the tensor algebra framework.');

console.log(`\nTotal time: ${((Date.now() - totalStart) / 1000).toFixed(1)}s`);
