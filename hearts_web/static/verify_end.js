// End-of-match search verification: re-judge the top raw-net
// disagreements with the REAL deep search (K=64) on the player's own
// hardware, upgrading the lesson card from opinion to verdict - and
// pre-seeding the review's IndexedDB cache so the cyan search layer is
// already there when the review opens (the new-user demonstration).
//
// Consent-through-clarity contract (design discussion 2026-08-09):
//  - NEVER silent: a labeled progress line with a spinner and a skip
//    link runs the whole time ("runs on your device, nothing uploaded").
//  - Auto-start ONLY on the WebGPU tier on non-mobile; the threaded-CPU
//    tier and phones get a one-tap button instead.
//  - Bounded: top 3 disagreements, K=64, 25s hard cap.
//  - Two manual skips persist auto-verify off (hearts_autoverify='0');
//    the button path remains available after that.
// Cache-compat contract with review.html (keep BOTH sides in sync):
//  db 'perilune-review' store 'evals', key `${SCHEMA}|ident|md5|K|d|i`,
//  SCHEMA v6, ident = T<code>#<match_no> or <sid>, md5 = manifest
//  policy_onnx_md5[:8]. Uses index.html globals: el().
(function () {
  const SCHEMA = 'v6';           // review.html DEEP_SCHEMA - keep in sync
  const K = 64, MAX_POS = 3, CAP_MS = 25000;
  const AUTO_KEY = 'hearts_autoverify', SKIP_KEY = 'hearts_verify_skips';
  let worker = null, capTimer = null;
  // Run-generation token: stop() bumps it, and run() re-checks after
  // every await - otherwise a skip during the loading phase (review
  // payload fetch) let the in-flight async run resume, create the
  // worker, and 'pop the search back up' (user-reported 2026-08-10).
  let runToken = 0;

  function rankVal(n) {
    const r = n.slice(0, -1);
    return {J: 9, Q: 10, K: 11, A: 12}[r] ?? (parseInt(r, 10) - 2);
  }
  const cardId = n => 'CDSH'.indexOf(n.slice(-1)) * 13 + rankVal(n);

  const area = () => el('verify-area');
  let ctx = null;
  const abort = html => {
    stop(html);
    if (ctx && ctx.onAbort) ctx.onAbort();
  };
  const isMobileUA = () =>
    /Android|iPhone|iPad|Mobile/i.test(navigator.userAgent);
  const autoAllowed = () =>
    localStorage.getItem(AUTO_KEY) !== '0'
    && !!navigator.gpu && !isMobileUA();

  function stop(finalHtml) {
    runToken++;
    if (worker) { worker.terminate(); worker = null; }
    if (capTimer) { clearTimeout(capTimer); capTimer = null; }
    if (finalHtml !== undefined && area()) area().innerHTML = finalHtml;
  }

  function skip() {
    const n = 1 + (+localStorage.getItem(SKIP_KEY) || 0);
    localStorage.setItem(SKIP_KEY, String(n));
    if (n >= 2) localStorage.setItem(AUTO_KEY, '0');
    abort('');
  }

  // Two deliberate lines, never an accidental wrap: the action line
  // (spinner + message + skip), then a quieter provenance line.
  function status(msg) {
    if (!area()) return;
    area().innerHTML = `<div class="vline"><span class="vring"></span>
        <span>${msg}</span> &middot;
        <a href="#" id="verify-skip" class="qlink">skip</a></div>
      <div class="vsub">runs on your device &middot; nothing is uploaded</div>`;
    el('verify-skip').onclick = e => { e.preventDefault(); skip(); };
  }

  function verdict(gap, se, dpts) {
    if (gap < 0.01)
      return {icon: '✓', cls: '#7BC77E', txt: 'your play was fine'};
    if (gap > 2 * se)
      return {icon: '▾', cls: '#FF7A7A',
              txt: `costs ${(100 * gap).toFixed(1)}% match win`
                   + (dpts != null && dpts > 0.05
                      ? ` (~${dpts.toFixed(1)} pts)` : '')};
    return {icon: '·', cls: '#A6A6AF', txt: 'inconclusive at K=64'};
  }

  async function idbPut(ident, md5, v) {
    try {
      const db = await new Promise((res, rej) => {
        const r = indexedDB.open('perilune-review', 1);
        r.onupgradeneeded = () => r.result.createObjectStore('evals');
        r.onsuccess = () => res(r.result);
        r.onerror = () => rej(r.error);
      });
      db.transaction('evals', 'readwrite').objectStore('evals')
        .put(v, `${SCHEMA}|${ident}|${md5}|${v.K}|${v.d}|${v.i}`);
    } catch (e) { /* cache is best-effort */ }
  }

  async function run(c) {
    ctx = c;
    const tok = ++runToken;
    status('loading deep search engine…');
    let R, md5 = 'nomd5';
    try {
      const params = new URLSearchParams({pid: c.pid || ''});
      if (c.mode === 'table') {
        params.set('code', c.tcode);
        params.set('match_no', c.matchNo);
      } else params.set('sid', c.sid);
      const r = await fetch(`/api/review?${params}`);
      if (!r.ok) throw new Error('review payload unavailable');
      R = await r.json();
      if (!R.replay) throw new Error('no replay data');
      try {
        const man = await (await fetch('/static/models/manifest.json')).json();
        md5 = (man.policy_onnx_md5 || '').slice(0, 8);
      } catch (e) {}
    } catch (e) { if (tok === runToken) abort(''); return; }
    if (tok !== runToken) return;   // skipped/cancelled while loading

    const ident = c.mode === 'table'
      ? `T${c.tcode}#${c.matchNo}` : (c.sid || '');
    const passOffset = deal => deal.pass_direction === 'hold' ? 0 : 12;
    const jobs = [];
    for (const d of c.list.slice(0, MAX_POS)) {
      if (d.idx == null || !R.deals[d.deal - 1]) continue;
      jobs.push({prio: 0, key: `${d.deal - 1}|${d.idx}`, d: d.deal - 1,
                 i: d.idx, deal: d.deal - 1,
                 actionIdx: passOffset(R.deals[d.deal - 1]) + d.idx, K,
                 playedId: cardId(d.you), dis: d});
    }
    if (!jobs.length) { abort(''); return; }

    let fp16Bad = false;
    try { fp16Bad = localStorage.getItem('perilune-fp16-bad') === 'v16'; }
    catch (e) {}
    const results = [];
    const finish = () => {
      // rendering belongs to the caller (the unified lesson panel);
      // this line vanishes and onDone draws everything
      stop('');
      if (c.onDone) c.onDone(results);
    };

    if (tok !== runToken) return;   // skipped/cancelled while preparing
    worker = new Worker('/static/analysis_worker.js?v=19');
    capTimer = setTimeout(() => abort(
      `<span style="opacity:.7">deep verification stopped - too slow on
       this device</span>`), CAP_MS);
    worker.onerror = () => abort('');
    worker.onmessage = ev => {
      const m = ev.data;
      if (m.type === 'ready') {
        worker.postMessage({type: 'load', seed: R.replay.seed,
          dealActions: R.replay.deal_actions,
          startHands: R.deals.map(dl =>
            dl.start_hands.map(h => h.map(cardId)))});
      } else if (m.type === 'loaded') {
        status(`deep-verifying your top ${jobs.length}
                moment${jobs.length > 1 ? 's' : ''}…`);
        worker.postMessage({type: 'queue',
          jobs: jobs.map(({dis, ...j}) => j)});
      } else if (m.type === 'result') {
        const job = jobs.find(j => j.d === m.d && j.i === m.i);
        if (!job) return;
        if (m.desync || !m.actions) {
          results.push({dis: job.dis, gap: null, icon: '·', cls: '#A6A6AF',
                        txt: 'could not verify'});
          if (results.length >= jobs.length) finish();
          return;
        }
        idbPut(ident, md5, {d: m.d, i: m.i, K: m.K, actions: m.actions,
                            mean: m.mean, se: m.se, pts: m.pts});
        const ip = m.actions.indexOf(job.playedId);
        let ib = 0;
        m.actions.forEach((a, k) => { if (m.mean[k] > m.mean[ib]) ib = k; });
        if (ip >= 0) {
          const gap = m.mean[ib] - m.mean[ip];
          const se = Math.sqrt(m.se[ib] ** 2 + m.se[ip] ** 2);
          const dp = m.pts ? m.pts[ip] - m.pts[ib] : null;
          results.push({dis: job.dis, gap, ...verdict(gap, se, dp)});
        } else {
          results.push({dis: job.dis, gap: null, icon: '·', cls: '#A6A6AF',
                        txt: 'could not verify'});
        }
        if (results.length >= jobs.length) finish();
        else status(`deep-verifying… ${results.length}/${jobs.length} done`);
      } else if (m.type === 'fatal') abort('');
    };
    worker.postMessage({type: 'init', skipFp16: fp16Bad,
                        mobile: isMobileUA()});
  }

  // Public entry, called from showMatchEnd's insight callback.
  // willAutoRun tells the caller whether to HOLD the lesson card (auto
  // path: card populates from the WORST verified play via onDone) or
  // render it immediately (button tier / opted out). onAbort fires on
  // every path that ends without results (skip, timeout, error) so the
  // caller can fall back to the raw-net card; cancel() (leaving the
  // screen) deliberately does not - nothing is left to populate.
  window.verifyEnd = {
    willAutoRun() { return autoAllowed(); },
    begin(c) {
      if (!area() || !c.list || !c.list.length) return;
      if (autoAllowed()) { run(c); return; }
      area().innerHTML = `<a href="#" id="verify-go" class="qlink">verify
        with deep search (~10s on your device)</a>`;
      el('verify-go').onclick = e => { e.preventDefault(); run(c); };
    },
    cancel() { stop(''); },
  };
})();
