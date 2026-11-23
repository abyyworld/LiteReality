/* Scheduled meetings on the room call (server: room_ops/roomcall/meetings.py).

   Anyone in a call, or on the join card, can schedule one (Rooms › Schedule a meeting…): a title,
   a start in their own local time (sent as UTC), a length, and the room — this call room as it
   is, or a ready room from the library that the first person in switches it to. It gets a link,
   /m/<id>, which opens this page in its call room with the meeting's card on the join screen.

   Reminders degrade, never break:
   - Web Push (a browser with the Push API and a server with reminders set up): "Remind me" asks
     for notification permission on that click, subscribes this browser (sw.js shows the push,
     even with the page closed) and hands the subscription to the server; the reminders cron
     sends it 15 minutes before the start.
   - iOS Safari delivers web push only to a Home Screen web app (iOS 16.4+): there the card says
     so instead of showing a button that cannot work.
   - Anywhere, "Add to calendar": an .ics with its own 15-minute alarm.

   Everything here that does not touch the page is exported for tests (tests/test_meetings.py
   runs them under Node); `createMeetings` is the UI, wired by app.js. CSP: no inline handlers or
   styles, as in app.js. */

export const MEETING_RE = /^[A-Za-z0-9_-]{16}$/;
const ROOM_RE = /^[A-Za-z0-9_-]{1,64}$/;
const PICK_RE = /^[A-Za-z0-9_-]{1,48}$/;
export const MAX_TITLE = 80;
export const REMIND_MINUTES = 15;
const STORE_KEY = 'roomcall.reminders';
const OWNED_KEY = 'roomcall.owned';
const TOKEN_RE = /^[A-Za-z0-9_-]{16,64}$/;

/** How this browser can be reminded: 'push' (the Push API is here), 'ios-home' (iOS Safari in a
 *  tab: only a Home Screen web app gets web push), or 'none' (the calendar file still works). */
export function reminderSupport(win = globalThis) {
  const nav = win.navigator || {};
  if (win.isSecureContext !== false && 'serviceWorker' in nav && 'PushManager' in win && 'Notification' in win) return 'push';
  const ua = String(nav.userAgent || '');
  const ios = /iPad|iPhone|iPod/.test(ua) || (nav.platform === 'MacIntel' && nav.maxTouchPoints > 1);
  let standalone = nav.standalone === true;
  try { standalone ||= !!win.matchMedia?.('(display-mode: standalone)')?.matches; } catch { /* old browser */ }
  return ios && !standalone ? 'ios-home' : 'none';
}

const pad = (n) => String(n).padStart(2, '0');

/** A Date as a datetime-local input's value: the viewer's own local time, to the minute. */
export function toLocalInput(date) {
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** A datetime-local value (the viewer's local time) as the UTC ISO string the server stores, or
 *  null. Read field by field: `new Date('YYYY-MM-DDTHH:MM')` is local in some engines, UTC in old. */
export function localInputToISO(value) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?$/.exec(String(value ?? ''));
  if (!m) return null;
  const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5]);
  if (Number.isNaN(d.getTime()) || d.getMonth() !== +m[2] - 1 || d.getDate() !== +m[3]) return null;
  return d.toISOString();
}

/** A sensible first guess for a new meeting: the next half hour at least 20 minutes away. */
export function defaultStart(now = new Date()) {
  const d = new Date(now.getTime() + 20 * 60000);
  d.setSeconds(0, 0);
  const add = (30 - (d.getMinutes() % 30)) % 30;
  d.setMinutes(d.getMinutes() + add);
  return d;
}

/** "Mon 24 Nov, 15:00 GMT+1": the start in the viewer's own time zone, the zone said. */
export function formatWhen(iso, { locale, timeZone } = {}) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  try {
    return new Intl.DateTimeFormat(locale, { weekday: 'short', day: 'numeric', month: 'short', hour: 'numeric',
      minute: '2-digit', timeZoneName: 'short', ...(timeZone ? { timeZone } : {}) }).format(d);
  } catch { return d.toUTCString(); }
}

/** "in 2 h 5 min", "in 3 days", "happening now", "ended": how far the meeting is from `now`. */
export function relativeWhen(startIso, endIso, now = new Date()) {
  const start = new Date(startIso).getTime(), end = new Date(endIso).getTime(), t = now.getTime();
  if (!Number.isFinite(start)) return '';
  if (Number.isFinite(end) && t >= end) return 'ended';
  if (t >= start) return 'happening now';
  const min = Math.round((start - t) / 60000);
  if (min < 1) return 'starting now';
  if (min < 60) return `in ${min} min`;
  const h = Math.floor(min / 60), m = min % 60;
  if (h < 24) return m ? `in ${h} h ${m} min` : `in ${h} h`;
  const days = Math.round(h / 24);
  return days === 1 ? 'in 1 day' : `in ${days} days`;
}

/** base64url (padding optional) -> bytes: the VAPID key as `applicationServerKey` wants it. */
export function b64urlToBytes(s) {
  const text = String(s || '').replace(/=+$/, '');
  if (!/^[A-Za-z0-9_-]*$/.test(text)) throw new Error('not base64url');
  const b64 = text.replace(/-/g, '+').replace(/_/g, '/') + '='.repeat((4 - (text.length % 4)) % 4);
  const bin = atob(b64);
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
}

/** Whether a subscription's key (an ArrayBuffer, or none) is `bytes`. */
export function sameKey(buffer, bytes) {
  if (!buffer) return false;
  const a = new Uint8Array(buffer);
  return a.length === bytes.length && a.every((v, i) => v === bytes[i]);
}

/** The server's meeting, kept to what the card shows (or null for anything else). */
export function parseMeeting(j) {
  if (!j || typeof j !== 'object' || !MEETING_RE.test(String(j.id || ''))) return null;
  const start = new Date(j.start), end = new Date(j.end);
  if (Number.isNaN(start.getTime()) || !ROOM_RE.test(String(j.room || ''))) return null;
  const minutes = Math.max(1, Math.floor(Number(j.minutes) || 60));
  const pick = typeof j.pick === 'string' && PICK_RE.test(j.pick) ? j.pick : '';
  return {
    id: j.id, title: String(j.title || 'Meeting').slice(0, MAX_TITLE), start: start.toISOString(),
    end: (Number.isNaN(end.getTime()) ? new Date(start.getTime() + minutes * 60000) : end).toISOString(),
    minutes, room: j.room, pick, pickLabel: pick ? String(j.pick_label || pick).slice(0, 80) : '',
  };
}

/** The meeting's share link and its calendar file, on this page's origin. */
export const meetingLink = (origin, id) => `${origin}/m/${encodeURIComponent(id)}`;
export const calendarHref = (id) => `m/${encodeURIComponent(id)}.ics`;

/** The meetings this browser asked to be reminded of: {id: start}, with ones long past dropped. */
export function remindedList(storage, now = Date.now()) {
  let all = {};
  try { all = JSON.parse(storage?.getItem(STORE_KEY) || '{}') || {}; } catch { all = {}; }
  const out = {};
  for (const [id, start] of Object.entries(all)) {
    if (MEETING_RE.test(id) && Date.parse(start) > now - 86400000) out[id] = start;
  }
  return out;
}
function saveReminded(storage, map) {
  try { storage?.setItem(STORE_KEY, JSON.stringify(map)); } catch { /* private mode */ }
}

/** The meetings this browser scheduled, with their cancel tokens: {id: {token, end}}, with ones
 *  over dropped (only whoever made a meeting has its token; the server keeps only a hash). */
export function ownedList(storage, now = Date.now()) {
  let all = {};
  try { all = JSON.parse(storage?.getItem(OWNED_KEY) || '{}') || {}; } catch { all = {}; }
  const out = {};
  for (const [id, v] of Object.entries(all)) {
    if (MEETING_RE.test(id) && TOKEN_RE.test(String(v?.token || '')) && Date.parse(v?.end) > now) {
      out[id] = { token: v.token, end: v.end };
    }
  }
  return out;
}
function saveOwned(storage, map) {
  try { storage?.setItem(OWNED_KEY, JSON.stringify(map)); } catch { /* private mode */ }
}

async function errorText(res) {
  try { const j = await res.json(); if (typeof j?.detail === 'string') return j.detail.slice(0, 200); } catch { /* not JSON */ }
  return `the server said ${res.status}`;
}

function timeout(p, ms) {
  return Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error('timed out')), ms))]);
}

/* ------------------------------------------------------------------ the UI */

/** ctx: { $, toast, togglePanel, roomId: () => string, onEnabled: () => void,
 *         onPick: (libraryRoomName) => void }. */
export function createMeetings(ctx) {
  const { $, toast } = ctx;
  const storage = (() => { try { return window.localStorage; } catch { return null; } })();
  const state = { enabled: false, key: null, support: reminderSupport(window), card: null, made: null, busy: false, armed: null, armTimer: 0 };

  const reminded = (id) => id in remindedList(storage);

  /** One meeting's reminder controls: a button, an iOS hint, or nothing (calendar only). */
  function renderRemind(m, btn, hint) {
    btn.hidden = true; hint.hidden = true;
    // No key: this server sends no reminders (yet), so no hint either (calendar only).
    if (!m || Date.parse(m.start) <= Date.now() || !state.key) return;
    if (state.support === 'ios-home') {
      hint.textContent = 'Add to Home Screen to get the reminder: Share, then Add to Home Screen, and open this link from there.';
      hint.hidden = false;
      return;
    }
    if (state.support !== 'push') return;
    if (window.Notification?.permission === 'denied' && !reminded(m.id)) {
      hint.textContent = 'Notifications are blocked for this site, so no reminder here — Add to calendar works.';
      hint.hidden = false;
      return;
    }
    btn.hidden = false;
    btn.disabled = state.busy;
    const on = reminded(m.id);
    btn.textContent = on ? 'Reminder on · turn off' : 'Remind me';
    btn.setAttribute('aria-pressed', String(on));
  }

  /** "Cancel meeting": only in the browser that scheduled it, until it is over. */
  function renderCancel(m, btn) {
    btn.hidden = !(m && m.id in ownedList(storage));
    btn.disabled = state.busy;
  }

  function renderAll() {
    if (state.card) { renderRemind(state.card, $('meetRemind'), $('meetHint')); renderCancel(state.card, $('meetCardCancel')); }
    if (state.made) { renderRemind(state.made, $('meetMadeRemind'), $('meetMadeHint')); renderCancel(state.made, $('meetMadeCancel')); }
  }

  /** Two presses rather than confirm(), as Reset in app.js: embedded browsers (Zoom) may drop
   *  dialogs. The first arms the button for a few seconds. */
  async function cancelMeeting(m, btn) {
    if (!m || state.busy) return;
    const owned = ownedList(storage);
    if (!(m.id in owned)) return;
    if (state.armed !== btn) {
      clearTimeout(state.armTimer);
      for (const b of [$('meetCardCancel'), $('meetMadeCancel')]) b.textContent = 'Cancel meeting';
      state.armed = btn;
      btn.textContent = 'Cancel for everyone?';
      state.armTimer = setTimeout(() => { state.armed = null; btn.textContent = 'Cancel meeting'; }, 4000);
      toast(`Press again to cancel “${m.title}” — its link and reminders stop working`);
      return;
    }
    clearTimeout(state.armTimer); state.armed = null; btn.textContent = 'Cancel meeting';
    state.busy = true; renderAll();
    try {
      const res = await fetch(`meetings/${encodeURIComponent(m.id)}/cancel`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ token: owned[m.id].token }) });
      if (!res.ok && res.status !== 404) throw new Error(await errorText(res));
      const left = ownedList(storage); delete left[m.id]; saveOwned(storage, left);
      const map = remindedList(storage); delete map[m.id]; saveReminded(storage, map);
      if (state.card?.id === m.id) { state.card = null; $('meetCard').hidden = true; }
      if (state.made?.id === m.id) { state.made = null; ctx.togglePanel('meetPanel', false); }
      toast(`Cancelled “${m.title}”`);
    } catch (err) {
      toast(`Couldn't cancel the meeting: ${String(err?.message || err).slice(0, 160)}`, 'error');
    } finally {
      state.busy = false; renderAll();
    }
  }

  async function subscription() {
    const key = b64urlToBytes(state.key);
    await navigator.serviceWorker.register('sw.js');
    const reg = await timeout(navigator.serviceWorker.ready, 15000);
    let sub = await reg.pushManager.getSubscription();
    if (sub && !sameKey(sub.options?.applicationServerKey, key)) {   // the server's keys changed
      await sub.unsubscribe().catch(() => {});
      sub = null;
    }
    return sub || reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: key });
  }

  /** "Remind me" (or "turn off"): the permission prompt comes first, inside the click itself —
   *  Safari only shows it during the gesture, and the key was fetched before. */
  async function toggleReminder(m) {
    if (!m || state.busy) return;
    if (reminded(m.id)) { await stopReminder(m); return; }
    let permission = 'default';
    try { permission = await Notification.requestPermission(); } catch { /* very old browser */ }
    if (permission !== 'granted') {
      toast(permission === 'denied' ? 'Notifications are blocked for this site — Add to calendar works instead'
        : 'No reminder without notifications — Add to calendar works instead', 'error');
      renderAll();
      return;
    }
    state.busy = true; renderAll();
    try {
      const sub = await subscription();
      const res = await fetch(`meetings/${encodeURIComponent(m.id)}/subscribe`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(sub.toJSON()) });
      if (!res.ok) throw new Error(await errorText(res));
      const map = remindedList(storage);
      map[m.id] = m.start;
      saveReminded(storage, map);
      toast(`You'll get a reminder ${REMIND_MINUTES} minutes before “${m.title}”`);
    } catch (err) {
      toast(`Couldn't set up the reminder: ${String(err?.message || err).slice(0, 160)}`, 'error');
    } finally {
      state.busy = false; renderAll();
    }
  }

  async function stopReminder(m) {
    state.busy = true; renderAll();
    try {
      const reg = await navigator.serviceWorker.getRegistration();
      const sub = await reg?.pushManager.getSubscription();
      if (sub) {
        const res = await fetch(`meetings/${encodeURIComponent(m.id)}/unsubscribe`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ endpoint: sub.endpoint }) });
        if (!res.ok && res.status !== 404) throw new Error(await errorText(res));
      }
      const map = remindedList(storage);
      delete map[m.id];
      saveReminded(storage, map);
      toast('Reminder off');
    } catch (err) {
      toast(`Couldn't turn the reminder off: ${String(err?.message || err).slice(0, 160)}`, 'error');
    } finally {
      state.busy = false; renderAll();
    }
  }

  async function copy(text, input) {
    try { await navigator.clipboard.writeText(text); toast('Meeting link copied'); }
    catch { input.hidden = false; input.focus(); input.select(); }   // no clipboard (Zoom): copy by hand
  }

  /* -- the card for a meeting link (/m/<id> lands on ?room=…&meeting=<id>) */
  function renderCard(m) {
    $('meetTitle').textContent = m.title;
    $('meetWhen').textContent = formatWhen(m.start);
    $('meetRel').textContent = relativeWhen(m.start, m.end);
    $('meetRoom').textContent = m.pickLabel ? `in ${m.pickLabel} · ${m.minutes} min` : `${m.minutes} min`;
    $('meetIcs').href = calendarHref(m.id);
    $('meetCopyInput').value = meetingLink(location.origin, m.id);
    $('meetCopyInput').hidden = true;
    renderRemind(m, $('meetRemind'), $('meetHint'));
    renderCancel(m, $('meetCardCancel'));
    $('meetCard').hidden = false;
  }

  async function loadCard(id) {
    try {
      const res = await timeout(fetch(`meetings/${encodeURIComponent(id)}.json`, { cache: 'no-store' }), 8000);
      if (res.status === 404) { toast('That meeting is not on this server any more', 'error'); return; }
      if (!res.ok) return;
      const m = parseMeeting(await res.json());
      if (!m) return;
      state.card = m;
      renderCard(m);
      if (m.pick && Date.parse(m.end) > Date.now()) ctx.onPick?.(m.pick);
    } catch { /* offline: the page works without the card */ }
  }

  /* -- the scheduling panel */
  async function fillRooms() {
    const box = $('meetPickRow'), select = $('meetPick');
    box.hidden = true;
    try {
      const res = await timeout(fetch('rooms.json?room=' + encodeURIComponent(ctx.roomId()), { cache: 'no-store' }), 6000);
      const j = res.ok ? await res.json() : null;
      const list = (Array.isArray(j?.rooms) ? j.rooms : []).filter((r) => PICK_RE.test(String(r?.name || '')) && !r.default);
      const keep = select.value;
      const here = document.createElement('option');
      here.value = ''; here.textContent = 'This call room, as it is';
      select.replaceChildren(here, ...list.map((r) => {
        const o = document.createElement('option');
        o.value = r.name; o.textContent = String(r.label || r.name).slice(0, 60);
        return o;
      }));
      select.value = list.some((r) => r.name === keep) ? keep : '';
      box.hidden = list.length === 0;
    } catch { /* no library: this room only */ }
  }

  function openPanel() {
    if (!state.enabled) return;
    $('meetPanelTitle').textContent = 'Schedule a meeting';
    $('meetForm').hidden = false;
    $('meetMade').hidden = true;
    $('meetFormError').hidden = true;
    $('meetStart').value = toLocalInput(defaultStart());
    $('meetStart').min = toLocalInput(new Date());
    $('meetRoomName').textContent = ctx.roomId();
    $('meetZone').textContent = (() => { try { return Intl.DateTimeFormat().resolvedOptions().timeZone || ''; } catch { return ''; } })();
    ctx.togglePanel('meetPanel', true);
    $('meetTitleInput').focus();
    fillRooms();
  }

  function formError(text) {
    $('meetFormError').textContent = text;
    $('meetFormError').hidden = false;
  }

  async function submit(e) {
    e.preventDefault();
    if (state.busy) return;
    const title = $('meetTitleInput').value.trim().replace(/\s+/g, ' ');
    const start = localInputToISO($('meetStart').value);
    if (!title) { formError('Give the meeting a title.'); return; }
    if (title.length > MAX_TITLE) { formError(`At most ${MAX_TITLE} characters for the title.`); return; }
    if (!start) { formError('Pick a date and time.'); return; }
    if (Date.parse(start) < Date.now() - 60000) { formError('That time has passed — pick a later one.'); return; }
    const body = { title, start, minutes: Number($('meetMinutes').value) || 60, room: ctx.roomId() };
    if ($('meetPick').value) body.pick = $('meetPick').value;
    state.busy = true;
    $('meetCreate').disabled = true;
    try {
      const res = await fetch('meetings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      if (!res.ok) { formError(await errorText(res)); return; }
      const j = await res.json();
      const m = parseMeeting(j);
      if (!m) { formError('The server sent something odd back; try again.'); return; }
      if (TOKEN_RE.test(String(j?.token || ''))) {
        const owned = ownedList(storage);
        owned[m.id] = { token: j.token, end: m.end };
        saveOwned(storage, owned);
      }
      state.made = m;
      showMade(m);
      toast(`Scheduled “${m.title}”`);
    } catch {
      formError("Couldn't reach the server; try again.");
    } finally {
      state.busy = false;
      $('meetCreate').disabled = false;
      renderAll();
    }
  }

  function showMade(m) {
    $('meetPanelTitle').textContent = 'Meeting scheduled';
    $('meetForm').hidden = true;
    $('meetMade').hidden = false;
    $('meetMadeTitle').textContent = m.title;
    $('meetMadeWhen').textContent = `${formatWhen(m.start)} · ${relativeWhen(m.start, m.end)}`;
    $('meetLinkInput').value = meetingLink(location.origin, m.id);
    $('meetMadeIcs').href = calendarHref(m.id);
    renderRemind(m, $('meetMadeRemind'), $('meetMadeHint'));
    renderCancel(m, $('meetMadeCancel'));
  }

  /* -- wiring */
  $('meetOpenJoin').addEventListener('click', openPanel);
  $('roomsMeetBtn').addEventListener('click', openPanel);
  $('meetForm').addEventListener('submit', submit);
  $('meetCancel').addEventListener('click', () => ctx.togglePanel('meetPanel', false));
  $('meetDone').addEventListener('click', () => ctx.togglePanel('meetPanel', false));
  $('meetRemind').addEventListener('click', () => toggleReminder(state.card));
  $('meetMadeRemind').addEventListener('click', () => toggleReminder(state.made));
  $('meetCardCancel').addEventListener('click', () => cancelMeeting(state.card, $('meetCardCancel')));
  $('meetMadeCancel').addEventListener('click', () => cancelMeeting(state.made, $('meetMadeCancel')));
  $('meetCopy').addEventListener('click', () => state.card && copy(meetingLink(location.origin, state.card.id), $('meetCopyInput')));
  $('meetLinkCopy').addEventListener('click', () => state.made && copy($('meetLinkInput').value, $('meetLinkInput')));
  // Typing in the panel never walks the avatar (app.js listens for keys on the window; its own
  // check only exempts <input>, and the length is a <select>). Escape still closes it.
  $('meetPanel').addEventListener('keydown', (e) => { if (e.key !== 'Escape') e.stopPropagation(); });

  async function boot(params) {
    try {
      const res = await timeout(fetch('meetings.json', { cache: 'no-store' }), 6000);
      if (!res.ok) return;                                     // this server does not schedule
      const j = await res.json();
      state.enabled = j?.enabled === true;
      try { state.key = j?.push && b64urlToBytes(j.push).length === 65 ? j.push : null; } catch { state.key = null; }
    } catch { return; }
    if (!state.enabled) return;
    $('meetOpenJoin').hidden = false;
    $('roomsMeetBtn').hidden = false;
    ctx.onEnabled?.();
    const id = params.get('meeting');
    if (id && MEETING_RE.test(id)) await loadCard(id);
  }

  return {
    boot,
    get enabled() { return state.enabled; },
    /** The library room the meeting's link asks for (the join card preselects it). */
    pick() { return state.card && Date.parse(state.card.end) > Date.now() ? state.card.pick : ''; },
  };
}
