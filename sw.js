self.addEventListener('install',e=>e.waitUntil(caches.open('market-ai-v1').then(c=>c.addAll(['/','/static/manifest.json']))));
self.addEventListener('fetch',e=>e.respondWith(caches.match(e.request).then(x=>x||fetch(e.request))));
