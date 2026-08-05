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

importScripts('/static/ort/ort.all.min.js');
importScripts('/static/analysis_engine.js');

const CHUNK = 416;
let M = null, policy = null, equity = null, ep = null;
let jobs = [], running = false, jobsDone = 0, jobsTotal = 0;

async function init() {
  ort.env.wasm.wasmPaths = '/static/ort/';
  M = await AnalysisEngine();
  M._an_init(1);
  try {
    if (!self.navigator || !navigator.gpu) throw new Error('no webgpu');
    policy = await ort.InferenceSession.create(
      '/static/models/perilune_policy_fp16.onnx',
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

async function runPolicy(rows, fetches) {
  // Chunked to <=CHUNK rows; returns {belief?, act?} with concatenated data.
  const obs = new Float32Array(M.HEAPF32.buffer, M._an_obs(), rows * 556);
  const mask = new Uint8Array(M.HEAPU8.buffer, M._an_mask(), rows * 52);
  const acts = fetches.includes('act') ? new Int32Array(rows) : null;
  let belief = null;
  for (let off = 0; off < rows; off += CHUNK) {
    const n = Math.min(CHUNK, rows - off);
    const out = await policy.run(
      { obs: new ort.Tensor('float32',
               obs.slice(off * 556, (off + n) * 556), [n, 556]),
        mask: new ort.Tensor('bool',
               mask.slice(off * 52, (off + n) * 52), [n, 52]) },
      fetches);
    if (out.act) {
      const a = out.act.data;
      for (let i = 0; i < n; i++) acts[off + i] = Number(a[i]);
    }
    if (out.belief) belief = out.belief.data;   // root is always 1 row
  }
  return { acts, belief };
}

async function analyzeOne(job) {
  let kind = M._an_analyze(job.deal, job.actionIdx, job.K);
  if (kind < 0) return { ...job, actions: [], mean: [], se: [], desync: true };
  while (kind !== 0) {
    const rows = M._an_rows();
    if (kind === 1) {
      if (M._an_is_root()) {
        const { belief } = await runPolicy(1, ['belief']);
        const p = M._malloc(156 * 4);
        M.HEAPF32.set(belief, p >> 2);
        kind = M._an_feed_root(p);
        M._free(p);
      } else {
        const { acts } = await runPolicy(rows, ['act']);
        new Int32Array(M.HEAP32.buffer, M._an_act_in(), rows).set(acts);
        kind = M._an_feed_acts();
      }
    } else {
      const x = new Float32Array(M.HEAPF32.buffer, M._an_eq_in(), rows * 10);
      const out = await equity.run(
        { x: new ort.Tensor('float32', x.slice(), [rows, 10]) }, ['logits']);
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
