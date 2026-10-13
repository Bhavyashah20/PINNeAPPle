// Offline support. build.py fills VERSION and PRECACHE (the app shell, models, gallery index, thumbnails and
// three.js); the light glTF files are cached the first time they are viewed.
const VERSION = "__VERSION__";
const PRECACHE = __PRECACHE__;
const CACHE = `playground-${VERSION}`;

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(keys.filter((k) => k.startsWith("playground-") && k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});

self.addEventListener("fetch", (e) => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin) return;
  // cache first (files are versioned by the cache name), then network; store what the network returns
  e.respondWith(caches.open(CACHE).then(async (c) => {
    const hit = await c.match(req, { ignoreSearch: true });
    if (hit) return hit;
    const res = await fetch(req);
    if (res.ok && res.type === "basic") c.put(req, res.clone());
    return res;
  }));
});
