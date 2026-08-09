// Perilune service worker: repeat-visit/offline cache for IMMUTABLE
// DATA assets only - .wasm, .onnx models, fonts. Scripts (.js/.mjs)
// are deliberately NEVER cached: worker top-level scripts carry
// load-bearing response headers (COEP - a COEP page may only spawn
// workers whose script responses carry it too), and replaying a
// response cached before a header deploy reproduces the failure
// forever (bit us live 2026-08-09: the v1 cache held a pre-COEP
// analysis_worker.js and deep analysis kept failing after the server
// fix). Data fetches are header-insensitive same-origin, so caching
// them is safe; scripts are small, so the byte savings live in the
// data anyway (~30MB of models + wasm vs ~2MB of js).
//
// The server's cache regime stays authoritative: HTML is no-cache
// (never intercepted), /api/ never touched, mutable assets ride ?v=
// versions so cache-first by full URL is correct (a bump = a new key).
const CACHE = 'perilune-static-v2';   // v2: drops v1's poisoned entries

const CACHEABLE =
  /^\/static\/.+\.(wasm|onnx|woff2?|ttf|otf)(\?|$)/;

self.addEventListener('install', () => self.skipWaiting());

self.addEventListener('activate', e => e.waitUntil(
  caches.keys()
    .then(ks => Promise.all(ks.filter(k => k !== CACHE)
                              .map(k => caches.delete(k))))
    .then(() => self.clients.claim())));

self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== location.origin) return;          // same-origin only
  if (url.pathname.startsWith('/api/')) return;        // never intercept
  if (!CACHEABLE.test(url.pathname + url.search)) return;  // html/js/etc
  e.respondWith(caches.open(CACHE).then(async c => {
    const hit = await c.match(req);
    if (hit) return hit;
    const resp = await fetch(req);
    if (resp.ok && resp.type === 'basic') c.put(req, resp.clone());
    return resp;
  }));
});
