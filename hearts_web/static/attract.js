// Landing-page attract mode: two auto-driven demo panels flanking the
// menu - a mini felt replaying baked AI-vs-AI matches ("Play Hearts")
// and an auto-scrubbing top-3 policy preview ("Review Matches").
// HONESTY LAW (menu-honesty commits): these must never read as a
// resumable game - both carry explicit demo captions and live in side
// panels, never behind the menu. Data = static JSON baked by
// hearts_web/gen_demo_matches.py from the SERVED weights (all-AI
// seats, no player data). Default ON; localStorage hearts_attract='0'
// turns it off (the "hide demos" link); prefers-reduced-motion
// defaults it off. Uses index.html globals: el(), cardHTML().
// State machines: render(state) is PURE (never mutates); tick() owns
// all advancement - the only way to keep two async-paced panels from
// double-stepping.
(function () {
  const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const KEY = 'hearts_attract';
  const N_DEMOS = 3;
  const demos = [];

  function enabled() {
    const v = localStorage.getItem(KEY);
    if (v === '0') return false;
    if (v === '1') return true;
    return !REDUCED;               // default: on, unless reduced motion
  }

  // Panel cursors: i = -1 pass interstitial, 0..plays-1 a play,
  // plays.length the deal-score interstitial.
  const play = {m: 0, d: 0, i: -1, nextAt: 0};
  const rev = {m: 1 % N_DEMOS, d: 0, i: 0, nextAt: 0};

  const SEATN = ['P1', 'P2', 'P3', 'P4'];

  function dealOf(st) {
    const m = demos[st.m];
    return m ? m.deals[st.d] : null;
  }

  function advance(st) {
    st.i++;
    const deal = dealOf(st);
    if (deal && st.i > deal.plays.length) {
      st.i = -1;
      st.d++;
      if (st.d >= demos[st.m].deals.length) {
        st.d = 0;
        st.m = (st.m + 1) % N_DEMOS;
      }
    }
  }

  function advanceReviewToPlay() {
    let guard = 0;
    let deal = dealOf(rev);
    while (deal && (rev.i < 0 || rev.i >= deal.plays.length)
           && guard++ < 300) {
      advance(rev);
      deal = dealOf(rev);
    }
  }

  // Felt skeleton: on-table seat tags (name + running total, active
  // player gold) around a centered trick area. Rebuilt only when the
  // (match, deal) key changes so card entry/sweep animations survive
  // between ticks - a full innerHTML rebuild would replay every drop.
  function feltSkeleton(box, key) {
    box.innerHTML = SEATN.map((n, s) =>
      `<div class="af-tag af-t${s}" id="af-tag${s}"></div>`).join('')
      + '<div id="attract-trick"></div>';
    box.dataset.key = key;
  }

  // Running deal points per seat up to (and including) play index i:
  // completed tricks only, like the game (round scores land at trick
  // end). Hearts 1 each, QS 13.
  function dealPts(deal, i) {
    const pts = [0, 0, 0, 0];
    const done = Math.floor((i + 1) / 4);
    for (let t = 0; t < done; t++) {
      let p = 0;
      for (const pl of deal.plays.slice(4 * t, 4 * t + 4)) {
        if (pl.c === 'QS') p += 13;
        else if (pl.c.slice(-1) === 'H') p += 1;
      }
      pts[deal.winners[t]] += p;
    }
    return pts;
  }

  // The game's tag format: Name - total (+deal points)
  function updateTags(totals, pts, active) {
    for (let s = 0; s < 4; s++) {
      const t = el('af-tag' + s);
      if (!t) continue;
      t.style.display = '';
      t.innerHTML = `${SEATN[s]} &middot; ${totals[s]}
        <small>(+${pts[s]})</small>`;
      t.classList.toggle('on', s === active);
    }
  }

  function hideTags() {
    for (let s = 0; s < 4; s++) {
      const t = el('af-tag' + s);
      if (t) t.style.display = 'none';
    }
  }

  function renderPlay() {
    const match = demos[play.m];
    const deal = dealOf(play);
    if (!deal) return 1000;
    const box = el('attract-felt');
    const cap = el('attract-play-cap');
    const key = `${play.m}:${play.d}`;
    if (box.dataset.key !== key || !el('attract-trick')) {
      feltSkeleton(box, key);
    }
    const before = play.d > 0 ? match.deals[play.d - 1].totals : [0, 0, 0, 0];
    const trickBox = el('attract-trick');
    if (play.i === -1) {
      trickBox.innerHTML = '';
      updateTags(before, [0, 0, 0, 0], -1);
      cap.textContent = deal.pass_dir === 'hold'
        ? `deal ${play.d + 1} - no pass`
        : `deal ${play.d + 1} - passing ${deal.pass_dir}`;
      return 1400;
    }
    if (play.i >= deal.plays.length) {
      // deal-end scoreboard overlay: deal points -> new totals; the
      // tags leave the table while it shows (they'd double the info)
      trickBox.innerHTML = `<div class="af-scores">${deal.scores.map(
        (sc, s) => `<div><span>${SEATN[s]}</span><b>+${sc}</b>
                    <i>${deal.totals[s]}</i></div>`).join('')}</div>`;
      hideTags();
      cap.textContent = `deal ${play.d + 1} scores - lowest wins`;
      return 2600;
    }
    const p = deal.plays[play.i];
    if (play.i % 4 === 0) trickBox.innerHTML = '';   // new trick
    const div = document.createElement('div');
    div.className = `atc pos${p.s}`;
    div.innerHTML = cardHTML(p.c, 'mini');
    trickBox.appendChild(div);
    updateTags(before, dealPts(deal, play.i), p.s);
    if (play.i % 4 === 3) {
      const w = deal.winners[Math.floor(play.i / 4)];
      cap.textContent = `${SEATN[w]} takes the trick`;
      // the game's cadence: a beat to read the trick, then the sweep
      // glides all four cards toward the winner and fades
      setTimeout(() => {
        for (const c of trickBox.children) c.classList.add('sweep' + w);
      }, 500);
      return 1450;
    }
    cap.textContent = `${SEATN[p.s]} plays`;
    return 650;
  }

  function renderReview() {
    const deal = dealOf(rev);
    if (!deal || rev.i < 0 || rev.i >= deal.plays.length) return 1000;
    const p = deal.plays[rev.i];
    const maxP = Math.max(...p.t3.map(t => t[1]), 0.001);
    el('attract-rev-head').textContent =
      `deal ${rev.d + 1} - trick ${Math.floor(rev.i / 4) + 1} - `
      + `${SEATN[p.s]} to play`;
    el('attract-rev-body').innerHTML = p.t3.map((t, j) =>
      `<div class="ar-row${j === 0 ? ' pick' : ''}">
         ${cardHTML(t[0], 'mini')}
         <div class="ar-bar"><i style="width:${Math.round(100 * t[1] / maxP)}%"></i></div>
         <span>${Math.round(t[1] * 100)}%</span>
       </div>`).join('');
    return 1500;
  }

  // A LIVE game behind the menu (Return-to-current-game showing) means
  // no demos AND no re-enable link - the menu is a pause screen there,
  // not a landing page.
  function liveGame() {
    const r = el('home-resume');
    return !!r && r.style.display !== 'none';
  }

  // Visibility is re-derived every tick (the menu opens/closes through
  // several paths; polling beats hooking them all).
  function syncVisibility() {
    const live = liveGame();
    const on = enabled() && !live;
    el('attract-play').style.display = on ? '' : 'none';
    el('attract-review').style.display = on ? '' : 'none';
    el('attract-show').style.display = (!enabled() && !live) ? '' : 'none';
  }

  function tick() {
    const home = el('home');
    if (!home || home.style.display !== 'flex') return;
    syncVisibility();
    if (!enabled() || liveGame()) return;
    const panel = el('attract-play');
    if (!panel || panel.offsetParent === null) return;   // hidden (mobile)
    const now = Date.now();
    if (now >= play.nextAt) {
      play.nextAt = now + renderPlay();
      advance(play);
    }
    if (now >= rev.nextAt) {
      advanceReviewToPlay();
      rev.nextAt = now + renderReview();
      advance(rev);
    }
  }

  window.attractToggle = function (on) {
    localStorage.setItem(KEY, on ? '1' : '0');
    syncVisibility();
  };

  async function boot() {
    if (!el('attract-play')) return;
    syncVisibility();
    for (let i = 0; i < N_DEMOS; i++) {
      try {
        demos[i] = await (await fetch(`/static/demo/match_${i}.json?v=1`)).json();
      } catch (e) { /* missing demo file: panel stays quiet */ }
    }
    setInterval(tick, 200);
  }
  boot();
})();
