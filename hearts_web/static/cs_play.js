// Client-search play: the AI's PLAYS are searched on a player's own
// hardware at the server-owned K, then posted back for the server to apply.
// Works in solo and at a table; at a table the HOST's device does the work.
//
// Same shape as verify_end.js - a standalone driver owning an analysis
// worker - so the normal game loop is untouched when search is off.
// index.html keeps ownership of `state` and the animation; this file talks
// to it through window.csHost, because `state` is a lexical binding there
// and no other script can assign it.
//
// WHAT PLAYERS ARE TOLD, and why (consent-through-clarity, the contract
// verify_end.js established 2026-08-09):
//   - never silent: a labelled line runs whenever a device is searching,
//     naming K, with a way out that stays live
//   - solo: these matches stay off the leaderboard, and the page says WHY
//     (choosing the AI's move needs the AI's cards, so the device has them)
//   - TABLE: every seat is warned, not just the host. The host's browser
//     holds the AI seats' cards, which is information the other humans do
//     not have, in a live game against them. That is not the host's
//     business to disclose or not - the server puts client_search in every
//     seat's view and this file renders it for all of them
//   - passes ARE searched now, at the teacher's K=24, via the engine's
//     an_choose_pass (VER 17). Review's an_analyze_pass could not do it:
//     it wants a finished deal and anchors on the pass actually made
//
// ENGINE NOTE: engine VER 17 accepts a PENDING decision directly - SeekTo
// takes action_idx == acts.size(), so the placeholder action this file used
// to append is gone. The whole match so far is sent, not just the live deal:
// pass direction is deal_count % 4 and the engine counts deals by
// replaying them, so a single-deal load passes the wrong way from deal 2
// (live bug 2026-08-17: 'illegal card 4' on deal 2).
(function () {
  // Bump BOTH this and the ?v= in index.html's <script src> together. This
  // file was pinned at v=1 through a dozen rewrites and browsers kept
  // serving the first one (2026-08-17); a test now fails if they disagree.
  const CS_VER = 5;   // keep in sync with index.html cs_play.js?v=
  let worker = null, ready = null, ep = null, engineVer = null;
  let running = false, generation = 0;

  const el = id => document.getElementById(id);
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const host = () => window.csHost;
  const isTable = () => host() && host().mode() === 'table';
  const iSearch = () => !isTable() || host().isHost();

  // ---- disclosure surface --------------------------------------------------
  // Two fixed lines, always present while search is on: an ACTIVITY line
  // that swaps between the spinner and an idle message, and the standing
  // note below it. Both have reserved height in CSS, so nothing here can
  // nudge the player's hand up or down as the text changes.
  function show(on) {
    const box = el('cs-status');
    if (box) box.style.display = on ? '' : 'none';
  }
  function activity(html) {
    const n = el('cs-activity');
    if (n) n.innerHTML = html;
  }
  function note(html) {
    const n = el('cs-note');
    if (n) n.innerHTML = html;
  }

  function standingNote() {
    const H = host();
    if (!isTable()) {
      return '<b>Search on.</b> Your device is choosing the AI’s moves — '
           + 'its passes and every card it plays — so it holds the AI’s '
           + 'cards. These matches stay off the leaderboard.';
    }
    if (H.isHost()) {
      return '<b>Search on — you are the host.</b> Your device is choosing '
           + 'the AI’s moves, so your browser holds the AI seats’ cards. '
           + 'Everyone at this table is shown that.';
    }
    return '<b>Search on at this table.</b> The host’s device is choosing '
         + 'the AI’s moves, so their browser holds the AI seats’ cards — '
         + 'information you do not have. Friendly games only.';
  }

  function idle() {
    const H = host();
    const st = H && H.state();
    if (!st || !st.client_search || st.finished) { show(false); return; }
    show(true);
    if (st.cs_fallback) {
      activity('The server is playing the AI');
      note('<b>Search off for this match.</b> It was handed back to the '
         + 'server, so the AI is playing on instinct from here.');
      return;
    }
    // Idle text rather than an empty line: a blank row where a spinner
    // sometimes appears reads as something having gone wrong.
    activity(st.your_turn
      ? '<span class="cs-dim">Your move — the AI is not searching</span>'
      : '<span class="cs-dim">Waiting for the AI’s turn</span>');
    note(standingNote());
  }

  function searching(seat, k) {
    show(true);
    note(standingNote());
    if (!iSearch()) {
      activity('<span class="vring"></span> the host’s device is searching '
             + '· K=' + k);
      return;
    }
    activity(`<span class="vring"></span> searching on your device · K=${k}`
       + ` <span class="cs-dim">(${seatName(seat)})</span>`
       + ` <a href="#" id="cs-stop">stop</a>`);
    const s = el('cs-stop');
    if (s) s.onclick = e => { e.preventDefault(); handOff('user_abort'); };
  }

  function offerHandOff(why) {
    show(true);
    activity(`Search stopped: ${esc(why)}.`
       + ` <a href="#" id="cs-hand">let the server play the AI</a>`);
    note(standingNote());
    const h = el('cs-hand');
    if (h) h.onclick = e => { e.preventDefault(); handOff('engine_error'); };
  }

  // ---- worker --------------------------------------------------------------
  function boot() {
    if (ready) return ready;
    ready = new Promise((resolve, reject) => {
      worker = new Worker('/static/analysis_worker.js?v=21');
      worker.onerror = e => reject(new Error(e.message || 'worker failed'));
      worker.onmessage = ev => {
        const m = ev.data;
        if (m.type === 'ready') { ep = m.ep; engineVer = m.ver; resolve(m); }
        else if (m.type === 'fatal') reject(new Error(m.message));
      };
      worker.postMessage({ type: 'init' });
    });
    return ready;
  }

  // One decision -> {cards:[...]}. A play is one card; a pass is a whole
  // 3-card combo, because the engine scores combos, not cards. Rejects
  // rather than guessing.
  function decide(pos) {
    const isPass = pos.phase === 'pass';
    return new Promise((resolve, reject) => {
      const onMsg = ev => {
        const m = ev.data;
        if (m.type !== 'result' && m.type !== 'fatal') return;
        worker.removeEventListener('message', onMsg);
        if (m.type === 'fatal') { reject(new Error(m.message)); return; }
        const out = isPass ? m.combos : m.actions;
        if (m.desync || !out || !out.length) {
          reject(new Error('search desync'));   // never play a stale answer
          return;
        }
        let best = 0;
        for (let i = 1; i < m.mean.length; i++)
          if (m.mean[i] > m.mean[best]) best = i;
        cacheResult(pos, m);      // free seed for the review; never awaited
        // combos arrive flat: 3 ids per candidate
        resolve(isPass ? out.slice(best * 3, best * 3 + 3)
                       : [out[best]]);
      };
      worker.addEventListener('message', onMsg);
      worker.postMessage({ type: 'load', seed: pos.seed,
                           dealActions: pos.deals,
                           startHands: pos.start_hands });
      worker.postMessage({ type: 'queue', jobs: isPass
        ? [{ key: `cs:${pos.deal_idx}:p${pos.seat}`, kind: 'livepass',
             deal: pos.deal_idx, seat: pos.seat, K: pos.k, nCand: 10 }]
        : [{ key: `cs:${pos.deal_idx}:${pos.action_idx}`, kind: 'play',
             deal: pos.deal_idx, actionIdx: pos.action_idx,
             K: pos.k, playedId: null }] });
    });
  }

  // ---- seeding the review's cache ------------------------------------------
  // Every decision here is a real K=64 search at a real position, and the
  // review runs the same engine over the same coordinates. Writing results
  // into the store review.html reads means opening the review afterwards
  // shows the search layer for the AI's moves with nothing recomputed.
  // Same db/store/key contract verify_end.js already uses - keep in sync.
  //
  // What this does NOT cover: the player's OWN moves. We only ever search
  // AI decision points, and "was my play right" is what most people open
  // the review for. This seeds the AI half, free.
  const CACHE_SCHEMA = 'v6';        // review.html DEEP_SCHEMA
  let _md5 = null;
  async function deepMd5() {
    if (_md5 !== null) return _md5;
    try {
      const man = await (await fetch('/static/models/manifest.json')).json();
      _md5 = (man.policy_onnx_md5 || '').slice(0, 8);
    } catch (e) { _md5 = 'nomd5'; }
    return _md5;
  }
  const cacheIdent = () => isTable()
    ? `T${host().code()}#${(host().state() || {}).match_no || ''}`
    : (host().sid() || '');

  async function cacheResult(pos, m) {
    try {
      // Two shapes, both review.html's own (see its result handler):
      // a pass is filed under 'ps<seat>', a play under its play index -
      // and the review indexes plays only, while the engine indexes the
      // full action list, hence the pass offset.
      const pass = pos.phase === 'pass';
      const i = pass ? 'ps' + pos.seat
                     : pos.action_idx - (pos.pass_offset || 0);
      if (!pass && i < 0) return;
      const md5 = await deepMd5();
      const db = await new Promise((res, rej) => {
        const r = indexedDB.open('perilune-review', 1);
        r.onupgradeneeded = () => r.result.createObjectStore('evals');
        r.onsuccess = () => res(r.result);
        r.onerror = () => rej(r.error);
      });
      // review.html's janitor keeps the 50 most-recently-OPENED matches
      // and never prunes an ident it has no open for. Without a stamp
      // these would accumulate forever, so register the match now and let
      // it age out on the same terms as everything else.
      try {
        const opens = JSON.parse(localStorage.getItem('hearts_rvopen') || '{}');
        if (!opens[cacheIdent()]) {
          opens[cacheIdent()] = Date.now();
          localStorage.setItem('hearts_rvopen', JSON.stringify(opens));
        }
      } catch (e) {}
      const v = pass
        ? {d: pos.deal_idx, i, K: pos.k, combos: m.combos, mean: m.mean,
           se: m.se, pts: m.pts, seat: pos.seat}
        : {d: pos.deal_idx, i, K: pos.k, actions: m.actions,
           mean: m.mean, se: m.se, pts: m.pts};
      db.transaction('evals', 'readwrite').objectStore('evals')
        .put(v, `${CACHE_SCHEMA}|${cacheIdent()}|${md5}|${v.K}|${v.d}|${v.i}`);
    } catch (e) { /* cache is best-effort; never break the game for it */ }
  }

  const routes = () => isTable()
    ? { pos: `/api/cs/table/position/${host().code()}`,
        play: `/api/cs/table/play/${host().code()}`,
        fall: `/api/cs/table/fallback/${host().code()}` }
    : { pos: `/api/cs/position/${host().sid()}`,
        play: `/api/cs/play/${host().sid()}`,
        fall: `/api/cs/fallback/${host().sid()}` };

  // ---- the loop ------------------------------------------------------------
  async function pump() {
    const H = host();
    if (running || !H) return;
    let st = H.state();
    if (!st || !st.client_search) { show(false); return; }
    if (!iSearch()) {           // a guest at a searching table: show, wait
      if (st.awaiting) searching(st.awaiting.seat, st.awaiting.k); else idle();
      return;
    }
    running = true;
    const gen = ++generation;
    try {
      while (st && st.client_search && st.awaiting && gen === generation) {
        const aw = st.awaiting;
        searching(aw.seat, aw.k);
        await boot();
        if (gen !== generation) return;
        const r = routes();
        const pos = await H.get(r.pos);
        if (gen !== generation) return;
        const t0 = Date.now();
        const picked = await decide(pos);
        if (gen !== generation) return;
        // No dwell here. animate() already sleeps pace().ai after the
        // card lands, which is the SAME single beat a non-search game
        // gets - adding another made searched moves a double beat. A
        // slow search simply pushes the card out; a fast one lands on
        // the normal rhythm, and instant lands at once.
        st = await H.post(r.play, Object.assign(
          { seat: aw.seat, k: pos.k, ms: Date.now() - t0,
            engine: { ver: engineVer, ep } },
          aw.phase === 'pass' ? { cards: picked } : { card: picked[0] }));
        // Clear the spinner BEFORE applying. apply() animates, and a deal
        // boundary parks inside showDealEnd() waiting on "Next deal" - so
        // a spinner left up here sat on the deal-end screen claiming a
        // search was running (reported 2026-08-17).
        activity('<span class="cs-dim">Playing the AI’s move</span>');
        await H.apply(st);
        st = H.state();
      }
      idle();
    } catch (e) {
      offerHandOff(e && e.message ? e.message : 'search failed');
    } finally {
      running = false;
    }
  }

  async function handOff(reason) {
    generation++;              // abandon any in-flight decision
    running = false;
    const H = host();
    if (!H) return;
    try {
      const st = await H.post(routes().fall, { reason });
      await H.apply(st);
      idle();
    } catch (e) {
      activity('Could not hand off: ' + esc(e.message || 'error'));
    }
  }

  window.csPlay = {
    sync() { return pump(); },
    stop() { generation++; running = false; show(false); },
    idle
  };
})();
