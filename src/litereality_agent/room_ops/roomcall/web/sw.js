/* The room call's service worker: it shows a scheduled meeting's reminder (a Web Push from the
   reminders cron, room_ops/roomcall/meetings.py `reminder`), even with the page closed, and opens
   the meeting's room when the reminder is clicked. Registered only when someone presses
   "Remind me" (meetings.js). No fetch handler: the page is always fetched from the network. */

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

/** A URL on this app only (the payload's is relative to the worker's scope), else the app itself. */
function appUrl(url) {
  try {
    const u = new URL(typeof url === 'string' ? url : './', self.registration.scope);
    if (u.origin === self.location.origin) return u.href;
  } catch { /* not a URL */ }
  return self.registration.scope;
}

/** The start in this device's own time zone ("15:00"), or '' when it cannot be read. */
function localTime(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  try { return new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(d); }
  catch { return ''; }
}

self.addEventListener('push', (event) => {
  let data = {};
  try { data = event.data ? event.data.json() : {}; } catch { data = {}; }
  const title = String(data.title || 'Room call').slice(0, 120);
  const at = localTime(data.start);
  // Every push must show a notification (userVisibleOnly), so a payload that cannot be read
  // still shows one.
  const body = at ? `Starts at ${at} · tap to join` : String(data.body || 'Tap to join').slice(0, 200);
  const tag = typeof data.tag === 'string' ? data.tag.slice(0, 64) : 'roomcall';
  event.waitUntil(self.registration.showNotification(title, {
    body, tag, renotify: true, icon: 'icon-192.png', badge: 'icon-192.png',
    data: { url: appUrl(data.url) },
  }));
});

/** The call room a page of this app is in (its ?room=), or null for another site or none. */
function roomOf(url) {
  try {
    const u = new URL(url);
    return u.origin === self.location.origin ? u.searchParams.get('room') : null;
  } catch { return null; }
}

self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  const url = appUrl(event.notification.data?.url);
  const room = roomOf(url);
  event.waitUntil((async () => {
    // A tab already in that call room (whatever else its link says) is brought forward rather
    // than a second one opened, which would join the room twice.
    const open = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    for (const client of open) {
      if ('focus' in client && (client.url === url || (room && roomOf(client.url) === room))) return client.focus();
    }
    return self.clients.openWindow ? self.clients.openWindow(url) : undefined;
  })());
});
