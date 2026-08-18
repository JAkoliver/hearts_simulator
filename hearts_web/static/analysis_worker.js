// Deep-analysis worker: drives the WASM search engine's pump with
// onnxruntime-web. All compute happens on the visitor's device.
//
// Protocol (main -> worker):
//   {type:'init'}                      -> {type:'ready', ep, kFast}
//   {type:'load', seed, dealActions}   -> {type:'loaded', deals}
//   {type:'queue', jobs:[{key, deal, actionIdx, K, playedId}]}  (appends)
//   {type:'clear'}                     drop pending jobs, reset counters;
//                                      the in-flight job still posts its
//                                      result (worth caching) but NOT a
//                                      progress line - a post-clear
//                                      progress would overwrite the main
//                                      thread's "stopped" status with a
//                                      stale frozen counter
// worker -> main:
//   {type:'result', key, deal, actionIdx, K, actions, mean, se, desync}
//   {type:'progress', done, total}
//   {type:'fatal', message}
//
// Efficiency (Phase-1 calibration): forwards chunked to <=416 rows (the
// measured WebGPU cliff is at 832); fp16 model on WebGPU (argmax-flip-free
// vs fp32), fp32 on the CPU fallback; rollout rounds fetch only the
// in-graph argmax output.

// VER busts HTTP caches for the engine glue AND its .wasm (locateFile) -
// without it, browsers happily run a stale engine forever.
const VER = 17;   // 17: an_choose_pass (live pass search)
// Fixed batch buckets: WebGPU compiles a pipeline PER TENSOR SHAPE, and
// rollout rounds shrink row counts continuously - hundreds of one-off
// shapes means hundreds of shader compiles (stalls, device pressure).
// Padding every forward to one of a few shapes compiles each kernel once.
// Mobile GPUs are compute-bound where desktop lanes are nearly free, so
// padding waste is real cost there - init{mobile} adds two intermediate
// shapes (two extra one-time compiles) to halve typical waste.
let BUCKETS = [16, 64, 208, 416];
const bucketOf = n => BUCKETS.find(b => b >= n) || 416;
importScripts('/static/ort/ort.all.min.js');
importScripts('/static/analysis_engine.js?v=' + VER);

const CHUNK = 416;
// Stall guard, per job kind. A single decision needs ~45 rounds and a
// pass ~56, but a cards-only replica plays a WHOLE MATCH sequentially
// (measured 415 rounds for 8 deals) - one guard for all of them false-
// alarmed at K=64. Bound = generous multiple of the measured need.
const MAX_ROUNDS = {cards: 4000, pass: 400, livepass: 400, trace: 400,
                    playout: 400, play: 400};
// Review's pass analysis and live pass search differ ONLY in the entry
// point; the root feed and the result shape are the same.
const isPass = k => k === 'pass' || k === 'livepass';
let M = null, policy = null, equity = null, ep = null;
let jobs = [], running = false, jobsDone = 0, jobsTotal = 0;
let clearEpoch = 0;
// Fast path fetches the in-graph argmax ('act'); if the backend ever
// produces an illegal action (engine-validated), fall back permanently to
// fetching logits and doing masked argmax in JS.
let useAct = true;
let skipFp16 = false;   // main thread remembers per-browser fp16 failures
let forceWasm = false;  // ?an_ep=wasm debug override: skip webgpu tiers
let isMobile = false;   // thread cap: phones throttle thermally

function anError() {
  // read NUL-terminated error string from the wasm heap
  let s = '', p = M._an_error_msg();
  const h = M.HEAPU8;
  for (let i = p; h[i] && i < p + 256; i++) s += String.fromCharCode(h[i]);
  return s;
}

async function init() {
  // Explicit VERSIONED urls instead of a path prefix: ORT spawns its
  // threaded sub-workers from the .mjs, whose response must carry COEP
  // - and edge caches (Cloudflare caches by extension) can pin a stale
  // pre-COEP copy under the unversioned URL indefinitely. ?v= makes
  // every header deploy a fresh edge key.
  ort.env.wasm.wasmPaths = {
    mjs: '/static/ort/ort-wasm-simd-threaded.jsep.mjs?v=' + VER,
    wasm: '/static/ort/ort-wasm-simd-threaded.jsep.wasm?v=' + VER };
  // Threaded CPU fallback (2026-08-09): multi-threaded wasm requires
  // crossOriginIsolated, which /review now serves COOP/COEP for. When
  // isolation is absent this resolves to 1 thread - byte-identical to
  // the old single-thread behavior. Cap below hardwareConcurrency so
  // the page/UI thread keeps a core; phones cap lower (thermals).
  const hw = (self.navigator && navigator.hardwareConcurrency) || 2;
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.max(1, Math.min(isMobile ? 4 : 8, hw - 1)) : 1;
  M = await AnalysisEngine({
    locateFile: f => '/static/' + f + '?v=' + VER });
  M._an_init(1);
  // (env.webgpu.device injection was tried for the Concat binding-limit
  // problem and proved read-only in this ORT build; the fix lives in the
  // MODEL now - the exported graph never needs >5 bindings per op.)
  try {
    if (forceWasm) throw new Error('wasm forced (?an_ep=wasm)');
    if (skipFp16) throw new Error('fp16 previously failed on this browser');
    if (!self.navigator || !navigator.gpu) throw new Error('no webgpu');
    // fp16 first: the net is MEASURED fp16-safe (peak activation 31 vs
    // the 65504 limit; zero NaN rows under true CUDA-half on real obs),
    // and the graph is binding-safe (<=5 buffers/op after the concat-tree
    // export). If a browser's fp16 path misbehaves anyway, the pump
    // errors tier down to the fp32 model automatically.
    policy = await ort.InferenceSession.create(
      '/static/models/perilune_policy_fp16.onnx?v=' + VER,
      { executionProviders: ['webgpu'] });
    ep = 'webgpu-fp16';
  } catch (e) {
    try {
      if (forceWasm) throw new Error('wasm forced');
      policy = await ort.InferenceSession.create(
        '/static/models/perilune_policy.onnx?v=' + VER,
        { executionProviders: ['webgpu'] });
      ep = 'webgpu';
    } catch (e2) {
      policy = await ort.InferenceSession.create(
        '/static/models/perilune_policy.onnx?v=' + VER,
        { executionProviders: ['wasm'] });
      // Honest label (UI shows it): the budget-bearer must be visible.
      ep = ort.env.wasm.numThreads > 1
        ? `wasm-threaded(${ort.env.wasm.numThreads})` : 'wasm';
    }
  }
  equity = await ort.InferenceSession.create(
    '/static/models/perilune_equity.onnx?v=' + VER,
    { executionProviders: ['wasm'] });
  postMessage({ type: 'ready', ep, ver: VER });
}

function loadMatch(seed, dealActions, startHands) {
  // startHands: per deal, 4 seats x 13 card ids - explicit hands, because
  // seed-based dealing is std::shuffle-implementation-bound and the WASM
  // libc++ build cannot reproduce MSVC-recorded deals from the seed.
  const flat = [], offsets = [0], hflat = [];
  for (const da of dealActions) { flat.push(...da); offsets.push(flat.length); }
  for (const dh of startHands) for (const seat of dh) hflat.push(...seat);
  const op = M._malloc(offsets.length * 4), ap = M._malloc(flat.length * 4),
        hp = M._malloc(hflat.length * 4);
  new Int32Array(M.HEAP32.buffer, op, offsets.length).set(offsets);
  new Int32Array(M.HEAP32.buffer, ap, flat.length).set(flat);
  new Int32Array(M.HEAP32.buffer, hp, hflat.length).set(hflat);
  const n = M._an_load_match(seed >>> 0, op, dealActions.length, ap, hp);
  M._free(op); M._free(ap); M._free(hp);
  postMessage({ type: 'loaded', deals: n });
}

// Watchdog: a wedged backend (GPU device loss, jsep hang) must surface as
// an error, not an eternally-pending await.
function withTimeout(promise, ms, what) {
  return Promise.race([promise, new Promise((_, rej) =>
    setTimeout(() => rej(new Error(`${what} timed out after ${ms}ms (ep=${ep})`)), ms))]);
}

// Cards-only replicas need SAMPLED actions (argmax would make every
// replica identical); everything else uses argmax.
async function runPolicySampled(rows) {
  const obs = new Float32Array(M.HEAPF32.buffer, M._an_obs(), rows * 556);
  const mask = new Uint8Array(M.HEAPU8.buffer, M._an_mask(), rows * 52);
  const acts = new Int32Array(rows);
  for (let off = 0; off < rows; off += CHUNK) {
    const n = Math.min(CHUNK, rows - off);
    const B = bucketOf(n);
    const obsB = new Float32Array(B * 556);
    obsB.set(obs.subarray(off * 556, (off + n) * 556));
    const maskB = new Uint8Array(B * 52);
    maskB.set(mask.subarray(off * 52, (off + n) * 52));
    for (let r = n; r < B; r++) maskB.fill(1, r * 52, (r + 1) * 52);
    const out = await withTimeout(policy.run(
      { obs: new ort.Tensor('float32', obsB, [B, 556]),
        mask: new ort.Tensor('bool', maskB, [B, 52]) }, ['logits']),
      30000, 'policy forward');
    const lg = out.logits.data;
    for (let i = 0; i < n; i++) {
      let mx = -Infinity;
      for (let c = 0; c < 52; c++) {
        if (maskB[i * 52 + c] && lg[i * 52 + c] > mx) mx = lg[i * 52 + c];
      }
      let tot = 0;
      const w = new Float64Array(52);
      for (let c = 0; c < 52; c++) {
        if (maskB[i * 52 + c]) { w[c] = Math.exp(lg[i * 52 + c] - mx); tot += w[c]; }
      }
      let r = Math.random() * tot, pick = -1;
      for (let c = 0; c < 52; c++) {
        if (!maskB[i * 52 + c]) continue;
        pick = c;
        if (r < w[c]) break;
        r -= w[c];
      }
      acts[off + i] = pick;
    }
  }
  return acts;
}

async function runPolicy(rows, wantBelief) {
  // Chunked to <=CHUNK rows; returns {belief?, acts?} with concatenated data.
  const obs = new Float32Array(M.HEAPF32.buffer, M._an_obs(), rows * 556);
  const mask = new Uint8Array(M.HEAPU8.buffer, M._an_mask(), rows * 52);
  const acts = wantBelief ? null : new Int32Array(rows);
  let belief = null;
  for (let off = 0; off < rows; off += CHUNK) {
    const n = Math.min(CHUNK, rows - off);
    const B = bucketOf(n);
    // Pad to the bucket: dummy rows are zero obs with an all-ones mask
    // (valid inputs, outputs ignored - real rows come first).
    const obsB = new Float32Array(B * 556);
    obsB.set(obs.subarray(off * 556, (off + n) * 556));
    const maskB = new Uint8Array(B * 52);
    maskB.set(mask.subarray(off * 52, (off + n) * 52));
    for (let r = n; r < B; r++) maskB.fill(1, r * 52, (r + 1) * 52);
    const maskChunk = maskB.subarray(0, n * 52);
    const out = await withTimeout(policy.run(
      { obs: new ort.Tensor('float32', obsB, [B, 556]),
        mask: new ort.Tensor('bool', maskB, [B, 52]) },
      wantBelief ? ['belief'] : (useAct ? ['act'] : ['logits'])),
      30000, 'policy forward');
    if (out.act) {
      const a = out.act.data;
      for (let i = 0; i < n; i++) acts[off + i] = Number(a[i]);
    } else if (out.logits) {
      // JS-side masked argmax (fallback path)
      const lg = out.logits.data;
      for (let i = 0; i < n; i++) {
        let best = -1, bv = -Infinity;
        for (let c = 0; c < 52; c++) {
          if (maskChunk[i * 52 + c] && lg[i * 52 + c] > bv) {
            bv = lg[i * 52 + c];
            best = c;
          }
        }
        if (best === -1 && !runPolicy.dumped) {
          // First bad row: dump everything needed to reproduce it.
          runPolicy.dumped = true;
          let nan = 0;
          for (let c = 0; c < 52; c++) if (Number.isNaN(lg[i * 52 + c])) nan++;
          console.error(`[deep-analysis] bad logits row (ep=${ep}): ` +
            `${nan}/52 NaN. obs row + logits follow for repro:`);
          console.error('OBS', JSON.stringify(
            Array.from(obsB.subarray(i * 556, (i + 1) * 556))));
          console.error('LOGITS', JSON.stringify(
            Array.from(lg.subarray(i * 52, (i + 1) * 52))));
          postMessage({ type: 'note',
                        message: `bad row: ${nan}/52 logits NaN (ep=${ep}) - dumped to console` });
        }
        acts[off + i] = best;
      }
    }
    // Root is 1 real row PADDED to a bucket: keep only the real rows'
    // belief - copying the padded output into the 156-float feed buffer
    // corrupted the wasm heap (the 'memory access out of bounds' crash).
    if (out.belief) belief = out.belief.data.subarray(0, n * 156);
  }
  return { acts, belief };
}

// Yield one MACROtask so queued control messages ('clear') can run.
// On the wasm tiers session.run resolves synchronously, which makes the
// whole pump one unbroken microtask chain - onmessage NEVER fires and a
// Stop click starves until the full sweep finishes (live bug
// 2026-08-10: the progress counter kept climbing after Stop).
// MessageChannel, not setTimeout: no nested-timer clamp.
const _yieldCh = new MessageChannel();
let _yieldRes = null;
_yieldCh.port1.onmessage = () => { const r = _yieldRes; _yieldRes = null; if (r) r(); };
const yieldMacro = () => new Promise(res => {
  _yieldRes = res;
  _yieldCh.port2.postMessage(0);
});

async function analyzeOne(job) {
  const epoch0 = clearEpoch;
  let kind = job.kind === 'trace' ? M._an_deal_trace(job.deal)
    : job.kind === 'playout' ? M._an_playout(job.deal)
    : job.kind === 'pass' ? M._an_analyze_pass(job.deal, job.seat, job.K, 10)
    : job.kind === 'livepass'
      ? M._an_choose_pass(job.deal, job.seat, job.K, job.nCand || 10)
    : job.kind === 'cards' ? M._an_cards_match(job.K)
    : M._an_analyze(job.deal, job.actionIdx, job.K);
  if (kind === -3) throw new Error('engine: ' + anError());
  if (kind < 0) return { ...job, actions: [], mean: [], se: [], desync: true };
  let rounds = 0;
  while (kind !== 0) {
    await yieldMacro();
    if (clearEpoch !== epoch0) return null;   // stopped mid-job: abandon
    if (kind === -3) throw new Error('engine: ' + anError());
    const cap = MAX_ROUNDS[job.kind] || MAX_ROUNDS.play;
    if (++rounds > cap)
      throw new Error(`stalled after ${cap} rounds in ${job.kind || 'play'} (ep=${ep})`);
    const rows = M._an_rows();
    if (kind === 1) {
      if (M._an_is_root()) {
        if (isPass(job.kind)) {
          // Pass root needs logits (candidate proposals) AND belief.
          const obs = new Float32Array(M.HEAPF32.buffer, M._an_obs(), 556);
          const mask = new Uint8Array(M.HEAPU8.buffer, M._an_mask(), 52);
          const B = bucketOf(1);
          const obsB = new Float32Array(B * 556);
          obsB.set(obs);
          const maskB = new Uint8Array(B * 52);
          maskB.set(mask);
          for (let r = 1; r < B; r++) maskB.fill(1, r * 52, (r + 1) * 52);
          const out = await withTimeout(policy.run(
            { obs: new ort.Tensor('float32', obsB, [B, 556]),
              mask: new ort.Tensor('bool', maskB, [B, 52]) },
            ['logits', 'belief']), 30000, 'pass root forward');
          const lp = M._malloc(52 * 4), bp = M._malloc(156 * 4);
          M.HEAPF32.set(out.logits.data.subarray(0, 52), lp >> 2);
          M.HEAPF32.set(out.belief.data.subarray(0, 156), bp >> 2);
          kind = M._an_feed_root_pass(lp, bp);
          M._free(lp); M._free(bp);
        } else {
          const { belief } = await runPolicy(1, true);
          const p = M._malloc(156 * 4);
          M.HEAPF32.set(belief, p >> 2);
          kind = M._an_feed_root(p);
          M._free(p);
        }
      } else if (job.kind === 'cards') {
        const acts = await runPolicySampled(rows);
        new Int32Array(M.HEAP32.buffer, M._an_act_in(), rows).set(acts);
        kind = M._an_feed_acts();
      } else {
        const { acts } = await runPolicy(rows, false);
        new Int32Array(M.HEAP32.buffer, M._an_act_in(), rows).set(acts);
        kind = M._an_feed_acts();
        if (kind === -2) {
          // Backend produced an illegal action: drop the in-graph argmax
          // fast path for this session and redo the decision via logits.
          if (!useAct) throw new Error('illegal action even via logits');
          useAct = false;
          return analyzeOne(job);
        }
      }
    } else {
      const x = new Float32Array(M.HEAPF32.buffer, M._an_eq_in(), rows * 10);
      const out = await withTimeout(equity.run(
        { x: new ort.Tensor('float32', x.slice(), [rows, 10]) }, ['logits']),
        30000, 'equity forward');
      const lg = out.logits.data;
      const dst = new Float32Array(M.HEAPF32.buffer, M._an_f_in(), rows);
      for (let i = 0; i < rows; i++) {
        let mx = -1e30, s = 0;
        for (let j = 0; j < 4; j++) mx = Math.max(mx, lg[i * 4 + j]);
        for (let j = 0; j < 4; j++) s += Math.exp(lg[i * 4 + j] - mx);
        dst[i] = Math.exp(lg[i * 4] - mx) / s;
      }
      kind = M._an_feed_equity();
    }
  }
  if (job.kind === 'cards') {
    const n = M._an_cards_n();
    const cards = [...new Int32Array(M.HEAP32.buffer,
                                     M._an_result_cards(), n * 6)];
    return { ...job, cards, desync: false };
  }
  if (isPass(job.kind)) {
    const n = M._an_result_n();
    const combos = [...new Int32Array(M.HEAP32.buffer,
                                      M._an_result_combo(), n * 3)];
    const mean = [...new Float32Array(M.HEAPF32.buffer, M._an_result_mean(), n)];
    const se = [...new Float32Array(M.HEAPF32.buffer, M._an_result_se(), n)];
    const pts = [...new Float32Array(M.HEAPF32.buffer, M._an_result_pts(), n)];
    return { ...job, combos, mean, se, pts, desync: false };
  }
  if (job.kind === 'trace') {
    const n = M._an_trace_n();
    const trace = [...new Int32Array(M.HEAP32.buffer,
                                     M._an_result_trace(), n * 4)];
    return { ...job, trace, desync: false };
  }
  if (job.kind === 'playout') {
    const playout = [...new Int32Array(M.HEAP32.buffer,
                                       M._an_result_playout(), 4)];
    return { ...job, playout, desync: false };
  }
  const n = M._an_result_n();
  const actions = [...new Int32Array(M.HEAP32.buffer, M._an_result_actions(), n)];
  const mean = [...new Float32Array(M.HEAPF32.buffer, M._an_result_mean(), n)];
  const se = [...new Float32Array(M.HEAPF32.buffer, M._an_result_se(), n)];
  const pts = [...new Float32Array(M.HEAPF32.buffer, M._an_result_pts(), n)];
  // Desync guard: the card actually played must be legal here.
  const desync = job.playedId != null && !actions.includes(job.playedId);
  return { ...job, actions, mean, se, pts, desync };
}

async function pumpQueue() {
  if (running) return;
  running = true;
  while (jobs.length) {
    const job = jobs.shift();
    const epoch = clearEpoch;
    try {
      const r = await analyzeOne(job);
      if (r === null || epoch !== clearEpoch) continue;  // cleared: drop
      jobsDone++;
      postMessage({ type: 'result', ...r });
      postMessage({ type: 'progress', done: jobsDone, total: jobsTotal });
      if (r.desync) { jobs = []; break; }   // stop on replay divergence
    } catch (e) {
      // NEVER retreat silently. Tiered: fp16-webgpu errors downgrade to
      // the fp32 model on webgpu (reported); fp32-webgpu gets ONE
      // session recreate; then fail with the ORIGINAL error - the
      // single-threaded CPU fallback would burn hours pretending to work.
      console.warn('deep-analysis job failed:', e);
      if (epoch !== clearEpoch) continue;   // stopped: don't retry/refail
      if (ep === 'webgpu-fp16') {
        postMessage({ type: 'note',
                      message: `fp16 failed (${String(e).slice(0, 100)}) - switching to fp32` });
        try {
          policy = await ort.InferenceSession.create(
            '/static/models/perilune_policy.onnx',
            { executionProviders: ['webgpu'] });
          ep = 'webgpu';
          useAct = true;
          jobs.unshift(job);
          postMessage({ type: 'ready', ep });
          continue;
        } catch (e2) { /* fall through */ }
      } else if (ep === 'webgpu' && !pumpQueue.retried) {
        pumpQueue.retried = true;
        postMessage({ type: 'note',
                      message: `webgpu error (${String(e).slice(0, 120)}) - retrying once` });
        try {
          policy = await ort.InferenceSession.create(
            '/static/models/perilune_policy.onnx',
            { executionProviders: ['webgpu'] });
          useAct = true;
          jobs.unshift(job);
          continue;
        } catch (e2) { /* fall through to fatal with the original error */ }
      }
      postMessage({ type: 'fatal', message: String(e).slice(0, 300) });
      jobs = [];
      break;
    }
  }
  running = false;
}

onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'init') {
    skipFp16 = !!m.skipFp16;
    forceWasm = !!m.forceWasm;
    isMobile = !!m.mobile;
    if (m.mobile) BUCKETS = [16, 32, 64, 128, 208, 416];
    init().catch(
      e => postMessage({ type: 'fatal', message: String(e).slice(0, 300) }));
  }
  else if (m.type === 'load') loadMatch(m.seed, m.dealActions, m.startHands);
  else if (m.type === 'queue') {
    if (m.front) jobs.unshift(...m.jobs);
    else jobs.push(...m.jobs);
    jobsTotal += m.jobs.length;
    pumpQueue();
  } else if (m.type === 'clear') {
    // Fresh counters: a later queue (e.g. a single-position deepen)
    // must count from 0/n, not continue a dead sweep's totals.
    jobs = []; jobsDone = 0; jobsTotal = 0; clearEpoch++;
  }
};
