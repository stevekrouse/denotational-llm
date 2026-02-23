// tensor-bigram.js — T2(R^d) tensor algebra bigram + MLP baseline
// Pure Node.js, Float64Array, explicit backprop
//
// ═══════════════════════════════════════════════════════════════════
// THE T2(R^d) ARCHITECTURE
// ═══════════════════════════════════════════════════════════════════
//
// Represents history as an element of the truncated tensor algebra T2(V),
// where V = R^d is the embedding space. The state has three components:
//
//   state = (scalar, vector, matrix) in R x R^d x R^{d x d}
//
// Concretely (d=2): state in R^7 = (count, v0, v1, m00, m01, m10, m11)
//
// State update after observing character c with previous character p:
//   count += 1
//   vec   += E[c]                    (accumulate unigram embeddings)
//   mat   += E[p] (x) E[c]          (accumulate bigram outer products)
//
// After normalization (divide by count):
//   vec/count  = average embedding of chars seen (unigram stats in R^d)
//   mat/count  = average transition matrix (bigram stats in R^{dxd})
//
// Prediction: logits[k] = W_s[k] . norm(state) + W_p[k] . E[prev] + b[k]
//
// The W_p . E[prev] term is the standard bigram (what char did I just see?).
// The W_s . state term adds longer-range context: what chars tend to appear
// in this name (unigram) and what transitions have happened (bigram history).
//
// This is genuinely more expressive than a standard bigram because the state
// accumulates over the ENTIRE name, not just the last character.
// ═══════════════════════════════════════════════════════════════════

const fs = require('fs');

const names = fs.readFileSync('/Users/stevekrouse/Desktop/denotational-llm/names.txt', 'utf-8')
  .trim().split('\n').map(s => s.trim().toLowerCase());

console.log(`Loaded ${names.length} names`);

const VOCAB = 27;
function charToIdx(c) { return c === '.' ? 0 : c.charCodeAt(0) - 96; }
function idxToChar(i) { return i === 0 ? '.' : String.fromCharCode(96 + i); }

// Precompute training pairs
let totalExamples = 0;
for (let i = 0; i < names.length; i++) totalExamples += names[i].length + 1;
console.log(`Training examples: ${totalExamples}`);

const prevArr = new Uint8Array(totalExamples);
const targetArr = new Uint8Array(totalExamples);
const prevPrevArr = new Uint8Array(totalExamples);

let eIdx = 0;
for (let ni = 0; ni < names.length; ni++) {
  const name = names[ni];
  let pp = 0, p = 0;
  for (let ci = 0; ci <= name.length; ci++) {
    const t = ci < name.length ? charToIdx(name[ci]) : 0;
    prevPrevArr[eIdx] = pp; prevArr[eIdx] = p; targetArr[eIdx] = t;
    eIdx++; pp = p; p = t;
  }
}

const nameOffsets = new Int32Array(names.length + 1);
let noff = 0;
for (let ni = 0; ni < names.length; ni++) { nameOffsets[ni] = noff; noff += names[ni].length + 1; }
nameOffsets[names.length] = noff;

// ═══════════════════════════════════════════════════════════════════
// Count-based bigram
// ═══════════════════════════════════════════════════════════════════
function runCountBigram() {
  console.log('\n==========================================');
  console.log('Count-Based Bigram (Karpathy baseline)');
  console.log('==========================================');
  const counts = new Float64Array(VOCAB * VOCAB);
  for (let i = 0; i < totalExamples; i++) counts[prevArr[i]*VOCAB+targetArr[i]] += 1;
  const rowProbs = new Float64Array(VOCAB * VOCAB);
  for (let r = 0; r < VOCAB; r++) {
    let s = 0;
    for (let c = 0; c < VOCAB; c++) s += counts[r*VOCAB+c]+1;
    for (let c = 0; c < VOCAB; c++) rowProbs[r*VOCAB+c] = (counts[r*VOCAB+c]+1)/s;
  }
  let ll = 0;
  for (let i = 0; i < totalExamples; i++) ll += Math.log(rowProbs[prevArr[i]*VOCAB+targetArr[i]]);
  const nll = -ll / totalExamples;
  console.log(`  NLL = ${nll.toFixed(4)} (Karpathy target: ~2.454)`);
  return nll;
}

// ═══════════════════════════════════════════════════════════════════
// T2(R^d) Tensor Algebra Bigram
// ═══════════════════════════════════════════════════════════════════
function runTensorBigram() {
  console.log('\n==========================================');
  console.log('T2(R^d) Tensor Algebra Bigram (d=2)');
  console.log('==========================================');

  const D = 2, SD = 7;

  const E  = new Float64Array(VOCAB * D);
  const Ws = new Float64Array(VOCAB * SD);
  const Wp = new Float64Array(VOCAB * D);
  const b  = new Float64Array(VOCAB);
  const nParams = E.length + Ws.length + Wp.length + b.length;
  console.log(`  Parameters: ${nParams} (E:${E.length} Ws:${Ws.length} Wp:${Wp.length} b:${b.length})`);
  console.log(`  State dim: ${SD} = 1 (count) + ${D} (vec) + ${D*D} (mat)`);

  for (let i = 0; i < E.length; i++)  E[i]  = (Math.random()-0.5);
  for (let i = 0; i < Ws.length; i++) Ws[i] = (Math.random()-0.5)*0.2;
  for (let i = 0; i < Wp.length; i++) Wp[i] = (Math.random()-0.5);

  const gE = new Float64Array(VOCAB*D), gWs = new Float64Array(VOCAB*SD);
  const gWp = new Float64Array(VOCAB*D), gB = new Float64Array(VOCAB);
  const logits = new Float64Array(VOCAB), probs = new Float64Array(VOCAB);

  const STEPS = 200;
  const reportSteps = new Set([1, 10, 25, 50, 100, 200]);
  const nllHistory = [];

  // Use mini-batch of 4000 names (~28k examples) per step
  // Eval on mini-batch too (fast) but eval on full set at report steps
  const BATCH = 4000;
  const nameIdx = new Int32Array(names.length);
  for (let i = 0; i < names.length; i++) nameIdx[i] = i;

  const t0 = Date.now();

  for (let step = 1; step <= STEPS; step++) {
    // Shuffle
    for (let i = names.length-1; i > 0; i--) {
      const j = (Math.random()*(i+1))|0;
      const t = nameIdx[i]; nameIdx[i] = nameIdx[j]; nameIdx[j] = t;
    }

    gE.fill(0); gWs.fill(0); gWp.fill(0); gB.fill(0);
    let batchLL = 0, batchN = 0;

    const bn = Math.min(BATCH, names.length);
    for (let bi = 0; bi < bn; bi++) {
      const ni = nameIdx[bi];
      const start = nameOffsets[ni], end = nameOffsets[ni+1];
      batchN += end - start;

      let s0=1,s1=E[0],s2=E[1],s3=0,s4=0,s5=0,s6=0;
      let pe0=E[0],pe1=E[1],pI=0;

      for (let j = start; j < end; j++) {
        const tgt = targetArr[j];
        const inv = 1.0/s0;
        const n1=s1*inv,n2=s2*inv,n3=s3*inv,n4=s4*inv,n5=s5*inv,n6=s6*inv;

        let maxL=-1e9;
        for (let k=0;k<VOCAB;k++) {
          const ws=k*SD,wp=k*D;
          const v = b[k]+Ws[ws]+Ws[ws+1]*n1+Ws[ws+2]*n2+Ws[ws+3]*n3+Ws[ws+4]*n4+Ws[ws+5]*n5+Ws[ws+6]*n6+Wp[wp]*pe0+Wp[wp+1]*pe1;
          logits[k]=v; if(v>maxL) maxL=v;
        }
        let se=0;
        for(let k=0;k<VOCAB;k++){probs[k]=Math.exp(logits[k]-maxL);se+=probs[k];}
        const invS=1.0/se;
        for(let k=0;k<VOCAB;k++) probs[k]*=invS;

        batchLL += Math.log(probs[tgt]+1e-30);

        for (let k=0;k<VOCAB;k++) {
          const dl=(k===tgt?1:0)-probs[k];
          const ws=k*SD,wp=k*D;
          gWs[ws]+=dl;gWs[ws+1]+=dl*n1;gWs[ws+2]+=dl*n2;
          gWs[ws+3]+=dl*n3;gWs[ws+4]+=dl*n4;gWs[ws+5]+=dl*n5;gWs[ws+6]+=dl*n6;
          gWp[wp]+=dl*pe0;gWp[wp+1]+=dl*pe1;
          gB[k]+=dl;
          gE[pI*D]+=dl*Wp[wp];gE[pI*D+1]+=dl*Wp[wp+1];
        }

        const te0=E[tgt*D],te1=E[tgt*D+1];
        s0+=1;s1+=te0;s2+=te1;
        s3+=pe0*te0;s4+=pe0*te1;s5+=pe1*te0;s6+=pe1*te1;
        pI=tgt;pe0=te0;pe1=te1;
      }
    }

    // Update params
    const lr = 2.0;
    const sc = lr / batchN;
    for (let i=0;i<E.length;i++) E[i]+=sc*gE[i];
    for (let i=0;i<Ws.length;i++) Ws[i]+=sc*gWs[i];
    for (let i=0;i<Wp.length;i++) Wp[i]+=sc*gWp[i];
    for (let i=0;i<b.length;i++) b[i]+=sc*gB[i];

    if (reportSteps.has(step)) {
      // Evaluate on FULL dataset
      let fullLL = 0;
      for (let ni = 0; ni < names.length; ni++) {
        const start=nameOffsets[ni],end=nameOffsets[ni+1];
        let s0=1,s1=E[0],s2=E[1],s3=0,s4=0,s5=0,s6=0;
        let pe0=E[0],pe1=E[1];
        for (let j=start;j<end;j++) {
          const tgt=targetArr[j]; const inv=1.0/s0;
          let maxL=-1e9;
          for(let k=0;k<VOCAB;k++){
            const ws=k*SD,wp=k*D;
            logits[k]=b[k]+Ws[ws]+Ws[ws+1]*s1*inv+Ws[ws+2]*s2*inv+Ws[ws+3]*s3*inv+Ws[ws+4]*s4*inv+Ws[ws+5]*s5*inv+Ws[ws+6]*s6*inv+Wp[wp]*pe0+Wp[wp+1]*pe1;
            if(logits[k]>maxL)maxL=logits[k];
          }
          let se=0;for(let k=0;k<VOCAB;k++){probs[k]=Math.exp(logits[k]-maxL);se+=probs[k];}
          for(let k=0;k<VOCAB;k++)probs[k]/=se;
          fullLL+=Math.log(probs[tgt]+1e-30);
          const te0=E[tgt*D],te1=E[tgt*D+1];
          s0+=1;s1+=te0;s2+=te1;
          s3+=pe0*te0;s4+=pe0*te1;s5+=pe1*te0;s6+=pe1*te1;
          pe0=te0;pe1=te1;
        }
      }
      const nll = -fullLL/totalExamples;
      nllHistory.push({ step, nll });
      console.log(`  Step ${String(step).padStart(3)}: NLL = ${nll.toFixed(4)}  (batch NLL = ${(-batchLL/batchN).toFixed(4)})`);
    }
  }

  console.log(`  Training time: ${Date.now()-t0}ms`);

  // Generate (sampled)
  console.log('\n  Generated names (sampled):');
  for (let g = 0; g < 10; g++) {
    let name = '';
    let s0=1,s1=E[0],s2=E[1],s3=0,s4=0,s5=0,s6=0;
    let pe0=E[0],pe1=E[1];
    for (let step=0;step<30;step++) {
      const inv=1.0/s0;
      for(let k=0;k<VOCAB;k++){const ws=k*SD,wp=k*D;logits[k]=b[k]+Ws[ws]+Ws[ws+1]*s1*inv+Ws[ws+2]*s2*inv+Ws[ws+3]*s3*inv+Ws[ws+4]*s4*inv+Ws[ws+5]*s5*inv+Ws[ws+6]*s6*inv+Wp[wp]*pe0+Wp[wp+1]*pe1;}
      let mx=-1e9;for(let k=0;k<VOCAB;k++)if(logits[k]>mx)mx=logits[k];
      let se=0;for(let k=0;k<VOCAB;k++){probs[k]=Math.exp(logits[k]-mx);se+=probs[k];}
      for(let k=0;k<VOCAB;k++)probs[k]/=se;
      let r=Math.random(),chosen=VOCAB-1;
      for(let k=0;k<VOCAB;k++){r-=probs[k];if(r<=0){chosen=k;break;}}
      if(chosen===0)break;
      name+=idxToChar(chosen);
      const te0=E[chosen*D],te1=E[chosen*D+1];
      s0+=1;s1+=te0;s2+=te1;s3+=pe0*te0;s4+=pe0*te1;s5+=pe1*te0;s6+=pe1*te1;
      pe0=te0;pe1=te1;
    }
    console.log(`    ${g+1}. ${name}`);
  }

  return nllHistory;
}

// ═══════════════════════════════════════════════════════════════════
// MLP Baseline (context=2, embed=2, hidden=4, tanh)
// ═══════════════════════════════════════════════════════════════════
function runMLP() {
  console.log('\n==========================================');
  console.log('MLP Baseline (ctx=2, d=2, hidden=4, tanh)');
  console.log('==========================================');

  const D=2,H=4,INP=4;
  const E=new Float64Array(VOCAB*D);
  const W1=new Float64Array(H*INP), b1=new Float64Array(H);
  const W2=new Float64Array(VOCAB*H), b2=new Float64Array(VOCAB);
  const nParams = E.length+W1.length+b1.length+W2.length+b2.length;
  console.log(`  Parameters: ${nParams}`);

  for(let i=0;i<E.length;i++) E[i]=(Math.random()-0.5);
  for(let i=0;i<W1.length;i++) W1[i]=(Math.random()-0.5)*0.5;
  for(let i=0;i<W2.length;i++) W2[i]=(Math.random()-0.5)*0.3;

  // Context batching: only ~602 unique contexts
  const ctxCounts = new Float64Array(VOCAB*VOCAB*VOCAB);
  const ctxTotal = new Float64Array(VOCAB*VOCAB);
  for(let i=0;i<totalExamples;i++){
    const key=prevPrevArr[i]*VOCAB+prevArr[i];
    ctxCounts[key*VOCAB+targetArr[i]]+=1; ctxTotal[key]+=1;
  }
  let numCtx=0;
  const ctxList=new Int32Array(VOCAB*VOCAB);
  for(let i=0;i<VOCAB*VOCAB;i++) if(ctxTotal[i]>0) ctxList[numCtx++]=i;
  console.log(`  Active contexts: ${numCtx}`);

  const gE=new Float64Array(VOCAB*D),gW1=new Float64Array(H*INP),gb1=new Float64Array(H);
  const gW2=new Float64Array(VOCAB*H),gb2=new Float64Array(VOCAB);
  const logits=new Float64Array(VOCAB),probs=new Float64Array(VOCAB);

  const LR=3.0;
  const reportSteps=new Set([1,10,25,50,100,200]);
  const nllHistory=[];
  const t0=Date.now();

  for(let step=1;step<=200;step++){
    gE.fill(0);gW1.fill(0);gb1.fill(0);gW2.fill(0);gb2.fill(0);
    let totalLL=0;

    for(let ci=0;ci<numCtx;ci++){
      const key=ctxList[ci];
      const c0=(key/VOCAB)|0,c1=key%VOCAB;
      const cnt=ctxTotal[key],cOff=key*VOCAB;

      const i0=E[c0*D],i1=E[c0*D+1],i2=E[c1*D],i3=E[c1*D+1];
      const h0r=b1[0]+W1[0]*i0+W1[1]*i1+W1[2]*i2+W1[3]*i3;
      const h1r=b1[1]+W1[4]*i0+W1[5]*i1+W1[6]*i2+W1[7]*i3;
      const h2r=b1[2]+W1[8]*i0+W1[9]*i1+W1[10]*i2+W1[11]*i3;
      const h3r=b1[3]+W1[12]*i0+W1[13]*i1+W1[14]*i2+W1[15]*i3;
      const t0h=Math.tanh(h0r),t1h=Math.tanh(h1r),t2h=Math.tanh(h2r),t3h=Math.tanh(h3r);

      let maxL=-1e9;
      for(let k=0;k<VOCAB;k++){const o=k*H;const v=b2[k]+W2[o]*t0h+W2[o+1]*t1h+W2[o+2]*t2h+W2[o+3]*t3h;logits[k]=v;if(v>maxL)maxL=v;}
      let se=0;for(let k=0;k<VOCAB;k++){probs[k]=Math.exp(logits[k]-maxL);se+=probs[k];}
      for(let k=0;k<VOCAB;k++)probs[k]/=se;

      for(let k=0;k<VOCAB;k++){const c=ctxCounts[cOff+k];if(c>0)totalLL+=c*Math.log(probs[k]+1e-30);}

      let dt0=0,dt1=0,dt2=0,dt3=0;
      for(let k=0;k<VOCAB;k++){
        const dl=ctxCounts[cOff+k]-cnt*probs[k];
        gb2[k]+=dl;const o=k*H;
        gW2[o]+=dl*t0h;gW2[o+1]+=dl*t1h;gW2[o+2]+=dl*t2h;gW2[o+3]+=dl*t3h;
        dt0+=dl*W2[o];dt1+=dl*W2[o+1];dt2+=dl*W2[o+2];dt3+=dl*W2[o+3];
      }
      const dh0=dt0*(1-t0h*t0h),dh1=dt1*(1-t1h*t1h),dh2=dt2*(1-t2h*t2h),dh3=dt3*(1-t3h*t3h);
      gb1[0]+=dh0;gb1[1]+=dh1;gb1[2]+=dh2;gb1[3]+=dh3;
      gW1[0]+=dh0*i0;gW1[1]+=dh0*i1;gW1[2]+=dh0*i2;gW1[3]+=dh0*i3;
      gW1[4]+=dh1*i0;gW1[5]+=dh1*i1;gW1[6]+=dh1*i2;gW1[7]+=dh1*i3;
      gW1[8]+=dh2*i0;gW1[9]+=dh2*i1;gW1[10]+=dh2*i2;gW1[11]+=dh2*i3;
      gW1[12]+=dh3*i0;gW1[13]+=dh3*i1;gW1[14]+=dh3*i2;gW1[15]+=dh3*i3;
      const di0=dh0*W1[0]+dh1*W1[4]+dh2*W1[8]+dh3*W1[12];
      const di1=dh0*W1[1]+dh1*W1[5]+dh2*W1[9]+dh3*W1[13];
      const di2=dh0*W1[2]+dh1*W1[6]+dh2*W1[10]+dh3*W1[14];
      const di3=dh0*W1[3]+dh1*W1[7]+dh2*W1[11]+dh3*W1[15];
      gE[c0*D]+=di0;gE[c0*D+1]+=di1;gE[c1*D]+=di2;gE[c1*D+1]+=di3;
    }

    const nll=-totalLL/totalExamples;
    if(reportSteps.has(step)){nllHistory.push({step,nll});console.log(`  Step ${String(step).padStart(3)}: NLL = ${nll.toFixed(4)}`);}

    const sc=LR/totalExamples;
    for(let i=0;i<E.length;i++) E[i]+=sc*gE[i];
    for(let i=0;i<W1.length;i++) W1[i]+=sc*gW1[i];
    for(let i=0;i<b1.length;i++) b1[i]+=sc*gb1[i];
    for(let i=0;i<W2.length;i++) W2[i]+=sc*gW2[i];
    for(let i=0;i<b2.length;i++) b2[i]+=sc*gb2[i];
  }

  console.log(`  Training time: ${Date.now()-t0}ms`);

  console.log('\n  Generated names (sampled):');
  for(let g=0;g<10;g++){
    let name='',c0=0,c1=0;
    for(let step=0;step<30;step++){
      const i0=E[c0*D],i1=E[c0*D+1],i2=E[c1*D],i3=E[c1*D+1];
      const th0=Math.tanh(b1[0]+W1[0]*i0+W1[1]*i1+W1[2]*i2+W1[3]*i3);
      const th1=Math.tanh(b1[1]+W1[4]*i0+W1[5]*i1+W1[6]*i2+W1[7]*i3);
      const th2=Math.tanh(b1[2]+W1[8]*i0+W1[9]*i1+W1[10]*i2+W1[11]*i3);
      const th3=Math.tanh(b1[3]+W1[12]*i0+W1[13]*i1+W1[14]*i2+W1[15]*i3);
      let mx=-1e9;for(let k=0;k<VOCAB;k++){const o=k*H;logits[k]=b2[k]+W2[o]*th0+W2[o+1]*th1+W2[o+2]*th2+W2[o+3]*th3;if(logits[k]>mx)mx=logits[k];}
      let se=0;for(let k=0;k<VOCAB;k++){probs[k]=Math.exp(logits[k]-mx);se+=probs[k];}
      for(let k=0;k<VOCAB;k++)probs[k]/=se;
      let r=Math.random(),chosen=VOCAB-1;
      for(let k=0;k<VOCAB;k++){r-=probs[k];if(r<=0){chosen=k;break;}}
      if(chosen===0)break;name+=idxToChar(chosen);c0=c1;c1=chosen;
    }
    console.log(`    ${g+1}. ${name}`);
  }

  return nllHistory;
}

// ═══════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════
const totalStart = Date.now();
const countNLL = runCountBigram();
const tensorResults = runTensorBigram();
const mlpResults = runMLP();

console.log('\n==========================================');
console.log('SUMMARY');
console.log('==========================================');
console.log(`  Count-based bigram NLL:     ${countNLL.toFixed(4)}  (optimal MLE)`);
console.log('');
console.log('  Step  | T2(R^2) Tensor | MLP (ctx=2,h=4)');
console.log('  ------|----------------|----------------');
for(let i=0;i<tensorResults.length;i++){
  const ts=tensorResults[i],ms=mlpResults[i];
  console.log(`  ${String(ts.step).padStart(5)} | ${ts.nll.toFixed(4).padStart(14)} | ${ms.nll.toFixed(4).padStart(14)}`);
}
console.log('');
console.log(`  Tensor T2 best:  ${Math.min(...tensorResults.map(x=>x.nll)).toFixed(4)}`);
console.log(`  MLP best:        ${Math.min(...mlpResults.map(x=>x.nll)).toFixed(4)}`);
console.log(`  Count baseline:  ${countNLL.toFixed(4)}`);
console.log(`\n  Total time: ${Date.now()-totalStart}ms`);
