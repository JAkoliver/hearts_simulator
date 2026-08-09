// Shared site navigation: one "Menu ▾" control in the SAME spot on every
// page (top-right, matching the game page's Menu button position and
// styling exactly). The game page does NOT include this file - its Menu
// button opens the richer in-game home overlay, same spot, same look.
// One source of truth so five hand-copied headers can't drift.
(() => {
  const css = `
  #sitenav { position:fixed; top:8px; right:10px; z-index:60;
             font-family:system-ui, sans-serif; }
  #sitenav-btn { font-size:.75rem; padding:4px 12px; background:#26262A;
                 color:#ECECEC; border:0; border-radius:7px; cursor:pointer; }
  #sitenav-btn:hover { background:#33333A; }
  #sitenav-dd { display:none; position:absolute; right:0; top:calc(100% + 6px);
                background:#1E1E21; border:1px solid #2E2E33; border-radius:9px;
                min-width:150px; padding:5px; box-shadow:0 6px 18px rgba(0,0,0,.5); }
  #sitenav-dd.open { display:block; }
  #sitenav-dd a { display:block; padding:7px 12px; border-radius:6px;
                  color:#A6A6AF; text-decoration:none; font-size:.82rem; }
  #sitenav-dd a:hover { background:#26262A; color:#E8B923; }
  @media (max-width:700px) { body { padding-top:44px !important; } }
  `;
  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);
  const wrap = document.createElement('div');
  wrap.id = 'sitenav';
  wrap.innerHTML = `<button id="sitenav-btn">Menu ▾</button>
    <div id="sitenav-dd">
      <a href="/">Home</a>
      <a href="/leaderboard">Leaderboard</a>
      <a href="/progress">Your progress</a>
      <a href="/account">Account</a>
      <a href="/about">About</a>
    </div>`;
  document.body.appendChild(wrap);
  const dd = wrap.querySelector('#sitenav-dd');
  wrap.querySelector('#sitenav-btn').addEventListener('click', e => {
    e.stopPropagation();
    dd.classList.toggle('open');
  });
  document.addEventListener('click', () => dd.classList.remove('open'));
})();
