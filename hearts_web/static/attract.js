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
  const SLOT = ['af-s0', 'af-s1', 'af-s2', 'af-s3'];  // bottom/left/top/right

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

  function renderPlay() {
    const deal = dealOf(play);
    if (!deal) return 1000;
    const box = el('attract-felt');
    const cap = el('attract-play-cap');
    if (play.i === -1) {
      box.innerHTML = SLOT.map(c => `<div class="${c}"></div>`).join('');
      cap.textContent = deal.pass_dir === 'hold'
        ? `deal ${play.d + 1} - no pass`
        : `deal ${play.d + 1} - passing ${deal.pass_dir}`;
      return 1400;
    }
    if (play.i >= deal.plays.length) {
      box.innerHTML = `<div class="af-scores">${deal.totals.map((t, s) =>
        `<div><span>${SEATN[s]}</span><b>${t}</b></div>`).join('')}</div>`;
      cap.textContent = `deal ${play.d + 1} scores - lowest wins`;
      return 2400;
    }
    const tstart = Math.floor(play.i / 4) * 4;
    const trick = deal.plays.slice(tstart, play.i + 1);
    box.innerHTML = SLOT.map(c => `<div class="${c}"></div>`).join('');
    for (const p of trick) {
      box.getElementsByClassName(SLOT[p.s])[0].innerHTML =
        cardHTML(p.c, 'mini');
    }
    if (trick.length === 4) {
      cap.textContent =
        `${SEATN[deal.winners[Math.floor(play.i / 4)]]} takes the trick`;
      return 1300;
    }
    cap.textContent = `${SEATN[deal.plays[play.i].s]} plays`;
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

  function tick() {
    if (!enabled()) return;
    const home = el('home');
    const panel = el('attract-play');
    if (!home || home.style.display !== 'flex') return;
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

  function setVisible(on) {
    el('attract-play').style.display = on ? '' : 'none';
    el('attract-review').style.display = on ? '' : 'none';
    el('attract-show').style.display = on ? 'none' : '';
  }

  window.attractToggle = function (on) {
    localStorage.setItem(KEY, on ? '1' : '0');
    setVisible(on);
  };

  async function boot() {
    if (!el('attract-play')) return;
    setVisible(enabled());
    for (let i = 0; i < N_DEMOS; i++) {
      try {
        demos[i] = await (await fetch(`/static/demo/match_${i}.json?v=1`)).json();
      } catch (e) { /* missing demo file: panel stays quiet */ }
    }
    setInterval(tick, 200);
  }
  boot();
})();
