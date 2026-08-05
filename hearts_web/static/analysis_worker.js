// Deep-analysis worker: drives the WASM search engine's pump with
// onnxruntime-web. All compute happens on the visitor's device.
//
// Protocol (main -> worker):
//   {type:'init'}                      -> {type:'ready', ep, kFast}
//   {type:'load', seed, dealActions}   -> {type:'loaded', deals}
//   {type:'queue', jobs:[{key, deal, actionIdx, K, playedId}]}  (appends)
//   {type:'clear'}                     drop pending jobs
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
const VER = 6;
// Fixed batch buckets: WebGPU compiles a pipeline PER TENSOR SHAPE, and
// rollout rounds shrink row counts continuously - hundreds of one-off
// shapes means hundreds of shader compiles (stalls, device pressure).
// Padding every forward to one of four shapes compiles each kernel once.
const BUCKETS = [16, 64, 208, 416];
const bucketOf = n => BUCKETS.find(b => b >= n) || 416;
importScripts('/static/ort/ort.all.min.js');
importScripts('/static/analysis_engine.js?v=' + VER);

const CHUNK = 416;
const MAX_ROUNDS = 400;   // stall guard: no decision needs this many forwards
let M = null, policy = null, equity = null, ep = null;
let jobs = [], running = false, jobsDone = 0, jobsTotal = 0;
// Fast path fetches the in-graph argmax ('act'); if the backend ever
// produces an illegal action (engine-validated), fall back permanently to
// fetching logits and doing masked argmax in JS.
let useAct = true;

function anError() {
  // read NUL-terminated error string from the wasm heap
  let s = '', p = M._an_error_msg();
  const h = M.HEAPU8;
  for (let i = p; h[i] && i < p + 256; i++) s += String.fromCharCode(h[i]);
  return s;
}

async function init() {
  ort.env.wasm.wasmPaths = '/static/ort/';
  M = await AnalysisEngine({
    locateFile: f => '/static/' + f + '?v=' + VER });
  M._an_init(1);
  try {
    if (!self.navigator || !navigator.gpu) throw new Error('no webgpu');
    // fp32 on WebGPU. The fp16 variant OVERFLOWS on real observations
    // (NaN logits -> illegal actions, 2026-08-05); the CPU parity check
    // was blind to it because ORT's CPU EP upcasts fp16 to fp32. Parked
    // until range-calibrated. The searchlab numbers were fp32 anyway.
    policy = await ort.InferenceSession.create(
      '/static/models/perilune_policy.onnx',
      { executionProviders: ['webgpu'] });
    ep = 'webgpu';
  } catch (e) {
    policy = await ort.InferenceSession.create(
      '/static/models/perilune_policy.onnx',
      { executionProviders: ['wasm'] });
    ep = 'wasm';
  }
  equity = await ort.InferenceSession.create(
    '/static/models/perilune_equity.onnx', { executionProviders: ['wasm'] });
  postMessage({ type: 'ready', ep });
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
        acts[off + i] = best;
      }
    }
    if (out.belief) belief = out.belief.data;   // root is always 1 row
  }
  return { acts, belief };
}

async function analyzeOne(job) {
  let kind = M._an_analyze(job.deal, job.actionIdx, job.K);
  if (kind === -3) throw new Error('engine: ' + anError());
  if (kind < 0) return { ...job, actions: [], mean: [], se: [], desync: true };
  let rounds = 0;
  while (kind !== 0) {
    if (kind === -3) throw new Error('engine: ' + anError());
    if (++rounds > MAX_ROUNDS)
      throw new Error(`stalled after ${MAX_ROUNDS} rounds (ep=${ep})`);
    const rows = M._an_rows();
    if (kind === 1) {
      if (M._an_is_root()) {
        const { belief } = await runPolicy(1, true);
        const p = M._malloc(156 * 4);
        M.HEAPF32.set(belief, p >> 2);
        kind = M._an_feed_root(p);
        M._free(p);
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
    try {
      const r = await analyzeOne(job);
      jobsDone++;
      postMessage({ type: 'result', ...r });
      postMessage({ type: 'progress', done: jobsDone, total: jobsTotal });
      if (r.desync) { jobs = []; break; }   // stop on replay divergence
    } catch (e) {
      // NEVER retreat silently: report the cause, recreate the WebGPU
      // session once (a lost device can recover), and only then give up
      // with the ORIGINAL error - a single-threaded CPU fallback would
      // burn hours pretending to work.
      console.warn('deep-analysis job failed:', e);
      if (ep === 'webgpu' && !pumpQueue.retried) {
        pumpQueue.retried = true;
        postMessage({ type: 'note',
                      message: `webgpu error (${String(e).slice(0, 120)}) - retrying once` });
        try {
          policy = await ort.InferenceSession.create(
            '/static/models/perilune_policy.onnx',
            { executionProviders: ['webgpu'] });
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
  if (m.type === 'init') init().catch(
    e => postMessage({ type: 'fatal', message: String(e).slice(0, 300) }));
  else if (m.type === 'load') loadMatch(m.seed, m.dealActions, m.startHands);
  else if (m.type === 'queue') {
    if (m.front) jobs.unshift(...m.jobs);
    else jobs.push(...m.jobs);
    jobsTotal += m.jobs.length;
    pumpQueue();
  } else if (m.type === 'clear') { jobs = []; }
};
