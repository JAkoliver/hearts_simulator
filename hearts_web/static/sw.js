// Perilune service worker: repeat-visit/offline cache for the IMMUTABLE
// static layer ONLY. The server's cache regime stays authoritative:
// HTML is no-cache (never intercepted here), /api/ is never touched,
// and mutable assets ride ?v= version strings - so cache-first by full
// URL (path + query) is always correct: a version bump is a different
// key. Big wins: the ~MB-scale ORT wasm binaries, ONNX models, the
// analysis engine wasm, and fonts - deep analysis loads instantly on
// repeat visits and cached matches review offline.
//
// COEP note: /review is served crossOriginIsolated (threaded-wasm
// fallback); responses replayed from this cache keep their original
// headers because real Response objects are stored, never synthesized.
const CACHE = 'perilune-static-v1';

// what may be cached: versioned assets, and the heavyweight immutable
// directories (ort runtime, models, fonts) plus any wasm binary.
const CACHEABLE =
  /^\/static\/(ort|models|fonts)\/|^\/static\/[^?]+\.wasm(\?|$)|\?v=/;

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
  if (!CACHEABLE.test(url.pathname + url.search)) return;  // HTML etc.
  e.respondWith(caches.open(CACHE).then(async c => {
    const hit = await c.match(req);
    if (hit) return hit;
    const resp = await fetch(req);
    if (resp.ok && resp.type === 'basic') c.put(req, resp.clone());
    return resp;
  }));
});
