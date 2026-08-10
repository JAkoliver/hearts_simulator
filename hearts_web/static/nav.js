// Shared site chrome: the lighter header band (logo + Perilune centered,
// matching the game page's header) with ONE "Menu" control at the very
// right - same spot, same styling, every page. The game page does NOT
// include this file: its native header already is this band, and its
// Menu button opens the richer in-game overlay PLUS a copy of this
// dropdown (index.html #menu-dd - keep the link list and styling in
// sync with #sitenav-dd below; the game page omits 'Play' because its
// centered menu IS play).
(() => {
  const css = `
  #siteband { position:fixed; top:0; left:0; right:0; z-index:55;
              background:#18181B; padding:8px; display:flex;
              align-items:center; justify-content:center; gap:10px;
              font-family:system-ui, sans-serif; }
  #siteband .t { font-family:'Space Grotesk', system-ui, sans-serif;
                 font-size:1.25rem; font-weight:700; color:#ECECEC;
                 letter-spacing:.01em; text-decoration:none; }
  #siteband .t:hover { color:#E8B923; }
  #siteband svg { display:inline-block !important; width:20px !important;
                  height:20px !important; }
  #sitenav { position:absolute; top:50%; transform:translateY(-50%);
             right:10px; }
  #sitenav-btn { font-size:.75rem; padding:4px 12px; background:#26262A;
                 color:#ECECEC; border:0; border-radius:7px; cursor:pointer; }
  #sitenav-btn:hover { background:#33333A; }
  #sitenav-dd { display:none; position:absolute; right:0;
                top:calc(100% + 6px); background:#1E1E21;
                border:1px solid #2E2E33; border-radius:9px; min-width:150px;
                padding:5px; box-shadow:0 6px 18px rgba(0,0,0,.5); }
  #sitenav-dd.open { display:block; }
  #sitenav-dd a { display:block; padding:7px 12px; border-radius:6px;
                  color:#A6A6AF; text-decoration:none; font-size:.82rem; }
  #sitenav-dd a:hover { background:#26262A; color:#E8B923; }
  body { padding-top: 62px !important; }
  `;
  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);
  const band = document.createElement('div');
  band.id = 'siteband';
  band.innerHTML = `<a class="t" href="/?menu=1"><svg width="20" height="20"
      viewBox="0 0 32 32" style="vertical-align:-4px; margin-right:6px"><rect
      width="32" height="32" rx="7" fill="#00000055"/><path
      d="M11 27 V7 H17.5 a6.5 6.5 0 0 1 0 13 H11" fill="none"
      stroke="#ECECEC" stroke-width="3.4" stroke-linecap="round"
      stroke-linejoin="round"/><circle cx="25.2" cy="7.6" r="3.1"
      fill="#E5484D"/></svg>Perilune</a>
    <div id="sitenav"><button id="sitenav-btn">Menu</button>
      <div id="sitenav-dd">
        <a href="/?menu=1">Play</a>
        <a href="/leaderboard">Leaderboard</a>
        <a href="/progress">Your progress</a>
        <a href="/account">Account</a>
        <a href="/about">About</a>
      </div></div>`;
  document.body.appendChild(band);
  const dd = band.querySelector('#sitenav-dd');
  band.querySelector('#sitenav-btn').addEventListener('click', e => {
    e.stopPropagation();
    dd.classList.toggle('open');
  });
  document.addEventListener('click', () => dd.classList.remove('open'));
})();
