"""Scheduled meetings on the room call and their reminders (room_ops/roomcall/meetings.py, the
server's /meetings, /m/<id> and /push routes, web/meetings.js and web/sw.js, the reminders cron).

The cron's choices are checked as pure functions with a fake sender (no push service, no Modal);
the server through its real app with a tiny scene; the page's helpers under Node."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mujoco
import pytest
from fastapi.testclient import TestClient

from litereality_agent.room_ops.roomcall import hosted, server
from litereality_agent.room_ops.roomcall import meetings as mt
from litereality_agent.room_ops.roomcall.library import LibraryRoom, RoomLibrary

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 23, 12, 0, tzinfo=timezone.utc)
SCENE = """
<mujoco model="meetings-test">
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1"/>
    <body name="box" pos="0 0 0.1"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/></body>
  </worldbody>
</mujoco>
"""
IMPORT_MAP = '{ "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.module.js" } }'
INDEX = (f'<!doctype html><html><head><script type="importmap">{IMPORT_MAP}</script>'
         '<script type="module" src="app.js"></script></head><body></body></html>')


def _b64(data: bytes) -> str:
    return mt.b64url_encode(data)


PUBLIC_KEY = _b64(b"\x04" + bytes(range(64)))
AUTH = _b64(bytes(range(16)))


def sub(endpoint: str = "https://fcm.googleapis.com/fcm/send/abc123", *, p256dh: str = PUBLIC_KEY,
        auth: str = AUTH) -> dict:
    return {"endpoint": endpoint, "expirationTime": None, "keys": {"p256dh": p256dh, "auth": auth}}


def meeting(start: datetime, *, minutes: int = 60, title: str = "Design review", room: str = "call",
            mid: str | None = None, pick: str | None = None) -> mt.Meeting:
    return mt.Meeting(id=mid or mt.secrets.token_urlsafe(12), title=title, start=start,
                      minutes=minutes, room=room, pick=pick, pick_label="Office" if pick else None,
                      created=NOW)


# -- validation ------------------------------------------------------------------------------------

def test_a_new_meeting_is_checked_and_stored_in_utc():
    m = mt.new_meeting({"title": "  Design\n\treview‮ ", "start": "2026-11-24T15:00:00.000+01:00",
                        "room": "call"}, now=NOW)
    assert m.title == "Design review" and m.start == datetime(2026, 11, 24, 14, tzinfo=timezone.utc)
    assert m.minutes == mt.DEFAULT_MINUTES and m.pick is None and mt.MEETING_ID.match(m.id)
    assert mt.iso(m.start) == "2026-11-24T14:00:00Z"
    z = mt.new_meeting({"title": "x", "start": "2026-11-24T14:00:00Z", "room": "r", "minutes": 30},
                       now=NOW)
    assert z.start == m.start and z.end == m.start + timedelta(minutes=30)
    assert mt.new_meeting({"title": "x", "start": "2026-11-24T14:00Z", "room": "r", "pick": "office"},
                          now=NOW, picks={"office": "Office room"}).pick_label == "Office room"


@pytest.mark.parametrize("body, why", [
    ({"title": "", "start": "2026-11-24T14:00:00Z", "room": "r"}, "title"),
    ({"title": "x" * 81, "start": "2026-11-24T14:00:00Z", "room": "r"}, "80"),
    ({"title": "x", "start": "2026-11-24T14:00:00", "room": "r"}, "time zone"),
    ({"title": "x", "start": "tomorrow", "room": "r"}, "start"),
    ({"title": "x", "start": "2026-11-23T11:50:00Z", "room": "r"}, "passed"),
    ({"title": "x", "start": "2027-03-01T00:00:00Z", "room": "r"}, "90 days"),
    ({"title": "x", "start": "2026-11-24T14:00:00Z", "room": "r", "minutes": 2}, "minutes"),
    ({"title": "x", "start": "2026-11-24T14:00:00Z", "room": "r", "minutes": True}, "minutes"),
    ({"title": "x", "start": "2026-11-24T14:00:00Z", "room": "../x"}, "room"),
    ({"title": "x", "start": "2026-11-24T14:00:00Z", "room": "r", "pick": "nowhere"}, "library"),
    (["not", "an", "object"], "JSON object"),
])
def test_a_bad_meeting_is_refused(body, why):
    with pytest.raises(mt.MeetingError) as err:
        mt.new_meeting(body, now=NOW, picks={"office": "Office"})
    assert err.value.status == 400 and why in err.value.message


@pytest.mark.parametrize("endpoint", [
    "https://fcm.googleapis.com/fcm/send/abc",
    "https://updates.push.services.mozilla.com/wpush/v2/gAAA",
    "https://web.push.apple.com/QGuQyavXut",
    "https://wns2-par02p.notify.windows.com/w/?token=BQYAAA",
    "https://db5p.notify.windows.com:443/w/?token=x",
])
def test_subscriptions_from_the_known_push_services_are_taken(endpoint):
    checked = mt.check_subscription(sub(endpoint))
    assert checked == {"endpoint": endpoint, "keys": {"p256dh": PUBLIC_KEY, "auth": AUTH}}


@pytest.mark.parametrize("data", [
    sub("http://fcm.googleapis.com/fcm/send/abc"),
    sub("https://evil.example/fcm.googleapis.com/x"),
    sub("https://fcm.googleapis.com.evil.example/x"),
    sub("https://fcm.googleapis.com:8443/x"),
    sub("https://user:pw@fcm.googleapis.com/x"),
    sub("https://push.apple.com/x"),                       # a suffix needs a host before it
    sub("https://notify.windows.com/x"),
    sub("https://127.0.0.1/x"),
    sub("https://fcm.googleapis.com/" + "a" * 1100),
    sub("https://fcm.googleapis.com/a b"),
    sub(p256dh=_b64(b"\x04" + bytes(63))),                 # 64 bytes
    sub(p256dh=_b64(b"\x02" + bytes(64))),                 # not an uncompressed point
    sub(auth=_b64(bytes(15))),
    sub(auth="not base64!"),
    {"endpoint": "https://fcm.googleapis.com/x"},
    "a string",
])
def test_anything_else_is_not_a_subscription(data):
    with pytest.raises(mt.MeetingError) as err:
        mt.check_subscription(data)
    assert err.value.status == 400


def test_the_calendar_file_is_one_utc_event_with_a_15_minute_alarm():
    m = meeting(datetime(2026, 11, 24, 14, tzinfo=timezone.utc), minutes=45,
                title="Plan; review, and \\ more " + "long " * 20, mid="AbCdEfGhIjKlMnOp")
    text = mt.ics(m, "https://room.example/m/AbCdEfGhIjKlMnOp")
    assert text.endswith("\r\n") and "\n" not in text.replace("\r\n", "")
    assert all(len(line.encode()) <= 75 for line in text.split("\r\n"))
    unfolded = text.replace("\r\n ", "")
    lines = unfolded.split("\r\n")
    for needed in ("BEGIN:VCALENDAR", "VERSION:2.0", "BEGIN:VEVENT", "UID:AbCdEfGhIjKlMnOp@roomcall",
                   "DTSTART:20261124T140000Z", "DTEND:20261124T144500Z",
                   "URL:https://room.example/m/AbCdEfGhIjKlMnOp", "BEGIN:VALARM", "ACTION:DISPLAY",
                   "TRIGGER:-PT15M", "END:VALARM", "END:VEVENT", "END:VCALENDAR"):
        assert needed in lines, needed
    summary = next(line for line in lines if line.startswith("SUMMARY:"))
    assert summary.startswith("SUMMARY:Plan\\; review\\, and \\\\ more long")
    assert any(line.startswith("DESCRIPTION:") and "https://room.example/m/" in line for line in lines)


# -- the reminders cron ----------------------------------------------------------------------------

class FakeSender:
    def __init__(self, results: dict[str, str] | None = None) -> None:
        self.results = results or {}
        self.calls: list[tuple[str, dict, int]] = []

    def __call__(self, subscription: dict, payload: dict, ttl: int) -> str:
        self.calls.append((subscription["endpoint"], payload, ttl))
        result = self.results.get(subscription["endpoint"], mt.SENT)
        if result == "raise":
            raise RuntimeError("network down")
        return result


@pytest.fixture
def store(tmp_path) -> mt.MeetingStore:
    return mt.MeetingStore(tmp_path / mt.MEETINGS_DIR)


def _add(store: mt.MeetingStore, m: mt.Meeting, *endpoints: str) -> mt.Meeting:
    store.add(m, now=NOW)
    for endpoint in endpoints:
        store.subscribe(m.id, mt.check_subscription(sub(endpoint)))
    return m


def test_a_meeting_is_due_in_the_15_minutes_before_it_starts():
    start = NOW + timedelta(minutes=30)
    m = meeting(start)
    assert not mt.due(m, start - timedelta(minutes=16))
    assert not mt.due(m, start - timedelta(minutes=15, seconds=1))
    assert mt.due(m, start - timedelta(minutes=15))
    assert mt.due(m, start - timedelta(minutes=14))
    assert mt.due(m, start - timedelta(seconds=1))      # a missed round is made up, late
    assert not mt.due(m, start)                         # never once it has started
    assert not mt.expired(m, m.end + timedelta(hours=23))
    assert mt.expired(m, m.end + timedelta(days=1))


def test_plan_picks_due_unsent_subscriptions_and_meetings_long_over():
    soon = meeting(NOW + timedelta(minutes=15), mid="soon000000000000")
    later = meeting(NOW + timedelta(minutes=20), mid="later00000000000")
    started = meeting(NOW - timedelta(minutes=1), mid="started000000000")
    old = meeting(NOW - timedelta(days=2), mid="old0000000000000")
    states = [mt.MeetingState(soon, {"a" * 32: {}, "b" * 32: {}}, {"a" * 32}),
              mt.MeetingState(later, {"c" * 32: {}}),
              mt.MeetingState(started, {"d" * 32: {}}),
              mt.MeetingState(old, {"e" * 32: {}})]
    send, prune = mt.plan(states, NOW)
    assert send == [(soon, ["b" * 32])] and prune == [old.id]


def test_a_round_reminds_once_drops_dead_subscriptions_and_retries_failures(store):
    m = _add(store, meeting(NOW + timedelta(minutes=15), title="Stand-up", room="team"),
             "https://fcm.googleapis.com/fcm/send/ok",
             "https://updates.push.services.mozilla.com/wpush/v2/gone",
             "https://web.push.apple.com/flaky",
             "https://wns2-par02p.notify.windows.com/w/?token=boom")
    later = _add(store, meeting(NOW + timedelta(minutes=40)), "https://fcm.googleapis.com/fcm/send/l")
    old = _add(store, meeting(NOW + timedelta(minutes=10)), "https://fcm.googleapis.com/fcm/send/o")
    sender = FakeSender({"https://updates.push.services.mozilla.com/wpush/v2/gone": mt.GONE,
                         "https://web.push.apple.com/flaky": mt.RETRY,
                         "https://wns2-par02p.notify.windows.com/w/?token=boom": "raise"})
    report = mt.remind_round(store, sender, now=NOW + timedelta(days=3))   # everything long over
    assert set(report.pruned) == {m.id, later.id, old.id} and not store.ids()

    m = _add(store, meeting(NOW + timedelta(minutes=15), title="Stand-up", room="team"),
             "https://fcm.googleapis.com/fcm/send/ok",
             "https://updates.push.services.mozilla.com/wpush/v2/gone",
             "https://web.push.apple.com/flaky",
             "https://wns2-par02p.notify.windows.com/w/?token=boom")
    later = _add(store, meeting(NOW + timedelta(minutes=40)), "https://fcm.googleapis.com/fcm/send/l")
    report = mt.remind_round(store, sender, now=NOW)
    assert (report.sent, report.gone, report.failed) == (1, 1, 2)
    assert {call[0] for call in sender.calls} == {
        "https://fcm.googleapis.com/fcm/send/ok", "https://updates.push.services.mozilla.com/wpush/v2/gone",
        "https://web.push.apple.com/flaky", "https://wns2-par02p.notify.windows.com/w/?token=boom"}
    endpoint, payload, ttl = sender.calls[0]
    assert payload["title"] == "Starting in 15 min: Stand-up" and ttl == 15 * 60
    assert payload["url"] == f"?room=team&meeting={m.id}" and payload["tag"] == f"meeting-{m.id}"
    assert payload["start"] == mt.iso(m.start) and "UTC" in payload["body"]
    assert len(store.subs(m.id)) == 3                         # the gone one is dropped
    assert store.sent(m.id) == {mt.sub_id("https://fcm.googleapis.com/fcm/send/ok")}
    assert not store.sent(later.id)

    # The next minute: only the failures again; a browser subscribing meanwhile is reminded too.
    store.subscribe(m.id, mt.check_subscription(sub("https://fcm.googleapis.com/fcm/send/new")))
    sender.calls.clear()
    sender.results = {}
    report = mt.remind_round(store, sender, now=NOW + timedelta(minutes=1))
    assert report.sent == 3 and {c[0] for c in sender.calls} == {
        "https://web.push.apple.com/flaky", "https://wns2-par02p.notify.windows.com/w/?token=boom",
        "https://fcm.googleapis.com/fcm/send/new"}
    assert sender.calls[0][1]["title"].startswith("Starting in 14 min")
    sender.calls.clear()
    assert mt.remind_round(store, sender, now=NOW + timedelta(minutes=2)).sent == 0
    assert not sender.calls
    # And `later` in its own window, once.
    assert mt.remind_round(store, sender, now=NOW + timedelta(minutes=25)).sent == 1
    assert mt.remind_round(store, sender, now=NOW + timedelta(minutes=26)).sent == 0


def test_the_reminder_goes_out_on_the_real_cron_tick_15_minutes_before(store):
    # The cron fires on the minute plus a few seconds of start-up; start times are on the minute.
    start = NOW + timedelta(minutes=30)
    m = _add(store, meeting(start, title="Stand-up"), "https://fcm.googleapis.com/fcm/send/x")
    sender = FakeSender()
    for late in (3, 59):
        store.mark_sent(m.id, [], NOW, {})
        sender.calls.clear()
        assert mt.remind_round(store, sender, now=start - timedelta(minutes=16) + timedelta(seconds=late)).sent == 0
        assert mt.remind_round(store, sender, now=start - timedelta(minutes=15) + timedelta(seconds=late)).sent == 1
        assert [c[1]["title"] for c in sender.calls] == ["Starting in 15 min: Stand-up"]
        assert sender.calls[0][1]["minutes"] == 15


def test_a_round_sends_at_most_its_budget(store):
    m = _add(store, meeting(NOW + timedelta(minutes=15)),
             *(f"https://fcm.googleapis.com/fcm/send/{i}" for i in range(5)))
    sender = FakeSender()
    first = mt.remind_round(store, sender, now=NOW, max_sends=3)
    assert first.sent == 3 and first.deferred == 2
    assert mt.remind_round(store, sender, now=NOW, max_sends=3).sent == 2
    assert len(store.sent(m.id)) == 5 and len(sender.calls) == 5


def test_a_round_shares_its_pushes_fairly_between_meetings(store):
    a = meeting(NOW + timedelta(minutes=10), mid="aaaaaaaaaaaaaaaa")
    b = meeting(NOW + timedelta(minutes=5), mid="bbbbbbbbbbbbbbbb")
    order = mt.interleave([(a, ["1", "2", "3"]), (b, ["4"])])
    assert [(m.id[0], sid) for m, sid in order] == [("b", "4"), ("a", "1"), ("a", "2"), ("a", "3")]
    # A meeting with many subscriptions sorting first by id never starves a later one.
    big = _add(store, meeting(NOW + timedelta(minutes=14), mid="0000000000000000"),
               *(f"https://fcm.googleapis.com/fcm/send/big{i}" for i in range(10)))
    small = _add(store, meeting(NOW + timedelta(minutes=14), mid="zzzzzzzzzzzzzzzz"),
                 "https://fcm.googleapis.com/fcm/send/s1", "https://fcm.googleapis.com/fcm/send/s2")
    assert mt.remind_round(store, FakeSender(), now=NOW, max_sends=4).sent == 4
    assert len(store.sent(small.id)) == 2 and len(store.sent(big.id)) == 2


class Ticks:
    """A monotonic clock that moves `step` seconds each time it is read."""

    def __init__(self, step: float) -> None:
        self.t, self.step = 0.0, step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def test_a_round_stops_at_its_deadline_and_keeps_what_it_sent(store):
    m = _add(store, meeting(NOW + timedelta(minutes=15)),
             *(f"https://fcm.googleapis.com/fcm/send/{i}" for i in range(6)))
    sender, commits = FakeSender(), []
    # The clock moves 10 s a read: the third batch would start 30 s in, past the 25 s deadline.
    report = mt.remind_round(store, sender, now=NOW, workers=2, deadline=25, clock=Ticks(10),
                             commit=lambda: commits.append(len(store.sent(m.id))))
    assert report.sent == 4 and report.deferred == 2
    assert commits == [2, 4]                      # marked and committed after every batch
    assert len(store.sent(m.id)) == 4
    sender.calls.clear()
    assert mt.remind_round(store, sender, now=NOW + timedelta(minutes=1)).sent == 2
    assert len(sender.calls) == 2                 # the rest, and nobody twice


class Killed(BaseException):
    """Modal stopping the function at its timeout, mid-round."""


def test_a_round_cut_short_never_sends_a_reminder_twice(store):
    m = _add(store, meeting(NOW + timedelta(minutes=15)),
             *(f"https://fcm.googleapis.com/fcm/send/{i}" for i in range(5)))
    calls = []

    def dies_on_the_third(subscription, payload, ttl):
        calls.append(subscription["endpoint"])
        if len(calls) == 3:
            raise Killed
        return mt.SENT

    commits = []
    with pytest.raises(Killed):
        mt.remind_round(store, dies_on_the_third, now=NOW, workers=1,
                        commit=lambda: commits.append(1))
    assert len(store.sent(m.id)) == 2 and len(commits) == 2    # the two delivered are on record
    sender = FakeSender()
    assert mt.remind_round(store, sender, now=NOW + timedelta(minutes=1)).sent == 3
    assert not {c[0] for c in sender.calls} & set(calls[:2])


def test_subscriptions_that_keep_failing_are_dropped(store):
    bad = "https://fcm.googleapis.com/fcm/send/garbage"
    m = _add(store, meeting(NOW + timedelta(minutes=15)), bad)
    sender = FakeSender({bad: mt.RETRY})
    # Rounds where no push got through at all are not held against the subscription.
    for i in range(mt.MAX_STRIKES + 1):
        assert mt.remind_round(store, sender, now=NOW + timedelta(seconds=i)).struck == 0
    assert mt.sub_id(bad) in store.subs(m.id) and not store.strikes(m.id)
    # With other pushes going through, MAX_STRIKES failed rounds drop it.
    other = _add(store, meeting(NOW + timedelta(minutes=14)))
    for i in range(mt.MAX_STRIKES):
        store.subscribe(other.id, mt.check_subscription(sub(f"https://fcm.googleapis.com/fcm/send/ok{i}")))
        report = mt.remind_round(store, sender, now=NOW + timedelta(seconds=10 + i))
        assert report.sent == 1
        assert report.struck == (1 if i == mt.MAX_STRIKES - 1 else 0)
        if i < mt.MAX_STRIKES - 1:
            assert store.strikes(m.id) == {mt.sub_id(bad): i + 1}
    assert not store.subs(m.id) and not store.strikes(m.id)
    # A success clears the count.
    flaky = "https://fcm.googleapis.com/fcm/send/flaky"
    store.subscribe(m.id, mt.check_subscription(sub(flaky)))
    store.subscribe(other.id, mt.check_subscription(sub("https://fcm.googleapis.com/fcm/send/okx")))
    mt.remind_round(store, FakeSender({flaky: mt.RETRY}), now=NOW + timedelta(minutes=1))
    assert store.strikes(m.id) == {mt.sub_id(flaky): 1}
    mt.remind_round(store, FakeSender(), now=NOW + timedelta(minutes=2))
    assert not store.strikes(m.id) and mt.sub_id(flaky) in store.sent(m.id)


def test_unreadable_meetings_are_skipped_and_removed_once_old(store, tmp_path):
    good = _add(store, meeting(NOW + timedelta(minutes=15)), "https://fcm.googleapis.com/fcm/send/x")
    broken = store.root / "brokenbrokenbrok"
    broken.mkdir()
    (broken / mt.MEETING_FILE).write_text("{not json")
    (store.root / ".tmpfile").write_text("ignored")
    assert [m.id for m in store.all()] == [good.id]
    report = mt.remind_round(store, FakeSender(), now=NOW)
    assert report.sent == 1 and broken.exists()             # might be mid-write: kept for now
    os.utime(broken, (0, 0))
    assert "brokenbrokenbrok" in store.prune(NOW)


def test_the_cron_round_publishes_the_key_and_sends_with_the_secret(tmp_path, monkeypatch):
    root = tmp_path / "volume"
    mstore = mt.MeetingStore(root / mt.MEETINGS_DIR)
    _add(mstore, meeting(NOW + timedelta(minutes=15)), "https://fcm.googleapis.com/fcm/send/x")
    with pytest.raises(RuntimeError, match="VAPID_PRIVATE_KEY"):
        mt.cron_round(root, env={"VAPID_PUBLIC_KEY": PUBLIC_KEY, "VAPID_SUBJECT": "mailto:a@b.c"})
    seen = {}

    def fake_sender(private_key, subject):
        seen.update(private=private_key, subject=subject)
        return FakeSender()

    monkeypatch.setattr(mt, "webpush_sender", fake_sender)
    env = {"VAPID_PRIVATE_KEY": "priv", "VAPID_PUBLIC_KEY": PUBLIC_KEY, "VAPID_SUBJECT": "mailto:a@b.c"}
    report = mt.cron_round(root, env=env, now=NOW)
    assert report.sent == 1 and report.key_published and seen == {"private": "priv", "subject": "mailto:a@b.c"}
    assert mstore.push_key() == PUBLIC_KEY
    assert not mt.cron_round(root, env=env, now=NOW).key_published   # unchanged: not rewritten
    with pytest.raises(ValueError):
        mt.cron_round(root, env={**env, "VAPID_PUBLIC_KEY": "short"}, now=NOW)


def test_the_webpush_sender_maps_the_push_services_answers(monkeypatch):
    calls = []

    class WebPushException(Exception):
        def __init__(self, message, response=None):
            super().__init__(message)
            self.response = response

    def webpush(subscription, **kwargs):
        calls.append((subscription, kwargs))
        status = {"gone": 410, "missing": 404, "busy": 503}.get(subscription["endpoint"])
        if subscription["endpoint"] == "down":
            raise ConnectionError("no route")
        if status:
            raise WebPushException("refused", types.SimpleNamespace(status_code=status))

    monkeypatch.setitem(sys.modules, "pywebpush",
                        types.SimpleNamespace(webpush=webpush, WebPushException=WebPushException))
    send = mt.webpush_sender("priv", "mailto:a@b.c", timeout=3)
    results = [send({"endpoint": e}, {"title": "t"}, 900) for e in ("ok", "gone", "missing", "busy", "down")]
    assert results == [mt.SENT, mt.GONE, mt.GONE, mt.RETRY, mt.RETRY]
    _, kwargs = calls[0]
    assert kwargs["vapid_private_key"] == "priv" and kwargs["vapid_claims"] == {"sub": "mailto:a@b.c"}
    assert kwargs["ttl"] == 900 and kwargs["timeout"] == 3 and kwargs["headers"]["Urgency"] == "high"
    assert json.loads(kwargs["data"]) == {"title": "t"}


# -- the server ------------------------------------------------------------------------------------

class Clock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.now = NOW

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def web_dir(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text(INDEX)
    (web / "app.js").write_text("// app\n")
    for name in (server.MEETINGS_JS, server.SERVICE_WORKER, server.MANIFEST, *server.ICONS):
        shutil.copyfile(server.WEB_DIR / name, web / name)
    return web


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def mstore(tmp_path):
    return mt.MeetingStore(tmp_path / "volume" / mt.MEETINGS_DIR)


@pytest.fixture
def client(web_dir, clock, mstore):
    library = RoomLibrary(lambda: [LibraryRoom(name="office", label="Office room",
                                               version="office-0123456789ab", scene=Path("x"))])
    service = mt.MeetingService(mstore, clock=clock, now=lambda: clock.now)
    app = server.create_app(model=mujoco.MjModel.from_xml_string(SCENE), web_dir=web_dir,
                            settle_seconds=0.05, library=library, meetings=service)
    with TestClient(app) as c:
        yield c


def _headers_ok(response) -> None:
    headers = response.headers
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["strict-transport-security"].startswith("max-age=")
    csp = dict(part.strip().split(" ", 1) for part in headers["content-security-policy"].split(";"))
    assert "'self'" in csp["worker-src"] and csp["manifest-src"] == "'self'"
    assert "'unsafe-inline'" not in csp["script-src"]


def _create(client, **body):
    body = {"title": "Design review", "start": "2026-11-24T14:00:00Z", "room": "call", **body}
    return client.post("/meetings", json=body)


def test_scheduling_a_meeting_and_its_link(client):
    made = _create(client, minutes=45, pick="office")
    assert made.status_code == 201, made.text
    _headers_ok(made)
    m = made.json()
    assert mt.MEETING_ID.match(m["id"]) and m["start"] == "2026-11-24T14:00:00Z"
    assert m["end"] == "2026-11-24T14:45:00Z" and m["pick_label"] == "Office room"
    assert m["link"] == f"m/{m['id']}" and m["ics"] == f"m/{m['id']}.ics"
    assert m["join"] == f"?room=call&meeting={m['id']}"
    token = m.pop("token")
    assert re.fullmatch(r"[A-Za-z0-9_-]{32}", token)
    shown = client.get(f"/meetings/{m['id']}.json").json()
    assert shown == m and not {"token", "by", "owner"} & set(shown)   # only its maker has the token

    page = client.get(f"/m/{m['id']}", follow_redirects=False)
    assert page.status_code == 302 and page.headers["location"] == f"../?room=call&meeting={m['id']}"
    _headers_ok(page)
    landed = client.get(f"/m/{m['id']}")                  # where the redirect lands: the page
    assert landed.status_code == 200 and landed.text == INDEX

    cal = client.get(f"/m/{m['id']}.ics", headers={"host": "x.modal.run"})
    assert cal.status_code == 200 and cal.headers["content-type"].startswith("text/calendar")
    assert cal.headers["content-disposition"].startswith("attachment")
    _headers_ok(cal)
    assert f"URL:https://x.modal.run/m/{m['id']}" in cal.text and "TRIGGER:-PT15M" in cal.text
    local = client.get(f"/m/{m['id']}.ics", headers={"host": "localhost:8790"})
    assert f"URL:http://localhost:8790/m/{m['id']}" in local.text


@pytest.mark.parametrize("path", ["/m/unknownunknownu", "/m/unknownunknownu.ics", "/m/short",
                                  "/m/..%2f..%2fsecret", "/meetings/unknownunknownu.json",
                                  "/meetings/unknownunknownu", "/m/", "/push", "/sw.js.map",
                                  "/meetings.js.map", "/icon-1024.png"])
def test_unknown_meeting_paths_are_404_with_headers(client, path):
    response = client.get(path)
    assert response.status_code == 404
    _headers_ok(response)


def test_bad_requests_are_refused(client):
    assert _create(client, title="").status_code == 400
    assert _create(client, start="2026-11-24T14:00:00").status_code == 400
    assert _create(client, pick="nowhere").status_code == 400
    assert client.post("/meetings", content=b"{nope", headers={"content-type": "application/json"}
                       ).status_code == 400
    big = client.post("/meetings", content=b"{" + b" " * 5000 + b"}")
    assert big.status_code == 413
    _headers_ok(big)


def test_scheduling_is_rate_limited_per_address(client, clock, monkeypatch):
    monkeypatch.setattr(mt, "MAX_PER_CREATOR", 100)
    for i in range(mt.CREATE_PER_10_MIN):
        assert _create(client, room=f"r{i}").status_code == 201
    refused = _create(client, room="another")
    assert refused.status_code == 429 and int(refused.headers["retry-after"]) > 0
    clock.t += 601
    assert _create(client, room="another").status_code == 201


def test_meetings_are_capped_per_room_and_in_all(client, clock, monkeypatch):
    monkeypatch.setattr(mt, "MAX_PER_ROOM", 2)
    monkeypatch.setattr(mt, "MAX_FUTURE", 3)
    monkeypatch.setattr(mt, "CREATE_PER_10_MIN", 100)
    monkeypatch.setattr(mt, "MAX_PER_CREATOR", 100)
    assert _create(client, room="a").status_code == 201
    assert _create(client, room="a").status_code == 201
    third = _create(client, room="a")
    assert third.status_code == 429 and "2 meetings" in third.json()["detail"]
    assert _create(client, room="b").status_code == 201
    assert _create(client, room="c").status_code == 429
    # Meetings over no longer count; a day after their end they are gone from the Volume.
    clock.now = NOW + timedelta(days=2, hours=3)
    assert _create(client, room="a", start="2026-11-26T12:00:00Z").status_code == 201


def test_one_network_cannot_fill_a_room_or_the_server(client, clock, mstore, monkeypatch):
    monkeypatch.setattr(mt, "CREATE_PER_10_MIN", 100)
    made = [_create(client, room="victim", start=f"2027-02-{d:02d}T10:00:00Z") for d in range(1, 6)]
    assert all(r.status_code == 201 for r in made)
    sixth = _create(client, room="victim", start="2027-02-10T10:00:00Z")
    assert sixth.status_code == 429 and "your network" in sixth.json()["detail"]
    # The stored meeting says which network made it only as a hash.
    stored = json.loads((mstore.root / made[0].json()["id"] / mt.MEETING_FILE).read_text())
    assert stored["by"] == mt.client_hash(mt.client_key("testclient")) and "testclient" not in json.dumps(stored)
    # Another network still schedules there.
    other = mt.new_meeting({"title": "x", "start": "2027-02-11T10:00:00Z", "room": "victim"}, now=NOW,
                           by=mt.client_hash("203.0.113.9"))
    assert mstore.add(other, now=NOW) == other
    # Cancelling one makes room again.
    first = made[0].json()
    assert client.post(f"/meetings/{first['id']}/cancel", json={"token": first["token"]}).status_code == 204
    assert _create(client, room="victim", start="2027-02-10T10:00:00Z").status_code == 201


def test_only_whoever_scheduled_a_meeting_can_cancel_it(client, mstore):
    m = _create(client).json()
    good = sub("https://web.push.apple.com/QGuQyavXut")
    assert client.post(f"/meetings/{m['id']}/subscribe", json=good).status_code == 204
    near = m["token"][:-1] + ("B" if m["token"].endswith("A") else "A")
    for wrong in ({"token": "x" * 32}, {"token": 5}, {}, {"token": near}):
        refused = client.post(f"/meetings/{m['id']}/cancel", json=wrong)
        assert refused.status_code == 403, wrong
        _headers_ok(refused)
    assert client.get(f"/meetings/{m['id']}.json").status_code == 200
    done = client.post(f"/meetings/{m['id']}/cancel", json={"token": m["token"]})
    assert done.status_code == 204
    _headers_ok(done)
    assert not (mstore.root / m["id"]).exists()                    # reminders gone with it
    assert client.get(f"/meetings/{m['id']}.json").status_code == 404
    assert client.get(f"/m/{m['id']}", follow_redirects=False).status_code == 404
    assert client.post(f"/meetings/{m['id']}/cancel", json={"token": m["token"]}).status_code == 404
    assert client.post("/meetings/x/cancel", json={"token": m["token"]}).status_code == 404


def test_the_operator_can_purge_a_rooms_meetings(mstore):
    a = mstore.add(meeting(NOW + timedelta(days=3), room="flooded"), now=NOW)
    b = mstore.add(meeting(NOW + timedelta(days=4), room="flooded"), now=NOW)
    keep = mstore.add(meeting(NOW + timedelta(days=4), room="fine"), now=NOW)
    assert sorted(mstore.purge_room("flooded")) == sorted([a.id, b.id])
    assert mstore.ids() == [keep.id]


def test_subscribing_a_browser_to_a_reminder(client, mstore, clock):
    mid = _create(client).json()["id"]
    good = sub("https://web.push.apple.com/QGuQyavXut")
    assert client.post(f"/meetings/{mid}/subscribe", json=sub("https://evil.example/x")).status_code == 400
    assert client.post(f"/meetings/{mid}/subscribe", json={"keys": {}}).status_code == 400
    assert client.post("/meetings/unknownunknownu/subscribe", json=good).status_code == 404
    assert client.post("/meetings/x/subscribe", json=good).status_code == 404
    ok = client.post(f"/meetings/{mid}/subscribe", json=good)
    assert ok.status_code == 204
    _headers_ok(ok)
    assert client.post(f"/meetings/{mid}/subscribe", json=good).status_code == 204   # once only
    assert list(mstore.subs(mid).values()) == [mt.check_subscription(good)]
    off = client.post(f"/meetings/{mid}/unsubscribe", json={"endpoint": good["endpoint"]})
    assert off.status_code == 204 and not mstore.subs(mid)
    clock.now = NOW + timedelta(days=1, hours=3)               # the meeting has started
    late = client.post(f"/meetings/{mid}/subscribe", json=good)
    assert late.status_code == 409


def test_subscriptions_are_rate_limited_and_capped(client, monkeypatch):
    mid = _create(client).json()["id"]
    monkeypatch.setattr(mt, "MAX_SUBS", 2)
    for i in range(2):
        assert client.post(f"/meetings/{mid}/subscribe",
                           json=sub(f"https://fcm.googleapis.com/fcm/send/{i}")).status_code == 204
    assert client.post(f"/meetings/{mid}/subscribe",
                       json=sub("https://fcm.googleapis.com/fcm/send/9")).status_code == 429
    monkeypatch.setattr(mt, "SUBSCRIBE_PER_HOUR", 4)
    assert client.post(f"/meetings/{mid}/unsubscribe", json={"endpoint": "x"}).status_code == 204
    limited = client.post(f"/meetings/{mid}/unsubscribe", json={"endpoint": "x"})
    assert limited.status_code == 429 and limited.headers["retry-after"]


def test_one_network_cannot_take_every_reminder_slot(client, mstore, monkeypatch):
    mid = _create(client).json()["id"]
    monkeypatch.setattr(mt, "MAX_SUBS_PER_CLIENT", 2)
    for i in range(2):
        assert client.post(f"/meetings/{mid}/subscribe",
                           json=sub(f"https://fcm.googleapis.com/fcm/send/{i}")).status_code == 204
    again = client.post(f"/meetings/{mid}/subscribe", json=sub("https://fcm.googleapis.com/fcm/send/1"))
    assert again.status_code == 204                      # a browser already in keeps its slot
    third = client.post(f"/meetings/{mid}/subscribe", json=sub("https://fcm.googleapis.com/fcm/send/2"))
    assert third.status_code == 429 and "your network" in third.json()["detail"]
    # Others still get in (here: straight to the store, as another network).
    other = mt.client_hash("198.51.100.7")
    mstore.subscribe(mid, mt.check_subscription(sub("https://fcm.googleapis.com/fcm/send/9")), by=other)
    assert len(mstore.subs(mid)) == 3
    stored = json.loads(next((mstore.root / mid / mt.SUBS_DIR).glob("*.json")).read_text())
    assert set(stored) == {"endpoint", "keys", "by"}


def test_the_push_key_appears_once_the_cron_has_published_it(client, mstore, monkeypatch):
    monkeypatch.delenv("VAPID_PUBLIC_KEY", raising=False)
    missing = client.get("/push/key")
    assert missing.status_code == 404
    _headers_ok(missing)
    assert client.get("/meetings.json").json() == {"enabled": True, "push": None}
    mstore.publish_push_key(PUBLIC_KEY)
    assert client.get("/push/key").json() == {"key": PUBLIC_KEY}
    assert client.get("/meetings.json").json() == {"enabled": True, "push": PUBLIC_KEY}


def test_the_service_worker_manifest_and_icons_are_served(client):
    sw = client.get("/sw.js")
    assert sw.status_code == 200 and sw.headers["content-type"].startswith("text/javascript")
    assert sw.headers["service-worker-allowed"] == "/" and sw.headers["cache-control"] == "no-cache"
    _headers_ok(sw)
    js = client.get("/meetings.js")
    assert js.status_code == 200 and "createMeetings" in js.text
    manifest = client.get("/manifest.webmanifest")
    assert manifest.headers["content-type"].startswith("application/manifest+json")
    data = manifest.json()
    assert data["display"] == "standalone" and data["name"] == "Room call"
    for icon in data["icons"]:
        got = client.get("/" + icon["src"])
        assert got.status_code == 200 and got.headers["content-type"].startswith(icon["type"])
    touch = client.get("/apple-touch-icon.png")
    assert touch.status_code == 200 and touch.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_without_meetings_every_meeting_route_is_404(web_dir):
    app = server.create_app(model=mujoco.MjModel.from_xml_string(SCENE), web_dir=web_dir,
                            settle_seconds=0.05)
    with TestClient(app) as c:
        for response in (c.get("/meetings.json"), c.get("/push/key"), _create(c),
                         c.get("/m/unknownunknownu"), c.get("/meetings/unknownunknownu.json"),
                         c.post("/meetings/unknownunknownu/subscribe", json=sub())):
            assert response.status_code == 404
            _headers_ok(response)
        assert c.get("/sw.js").status_code == 200           # static files are there either way


# -- the real client, and the hosted wiring -----------------------------------------------------------

def test_the_real_page_has_what_meetings_js_uses():
    page = (server.WEB_DIR / "index.html").read_text("utf-8")
    script = (server.WEB_DIR / "meetings.js").read_text("utf-8")
    used = set(re.findall(r"\$\('([A-Za-z]+)'\)", script))
    assert used and not {i for i in used if f'id="{i}"' not in page}
    assert '<link rel="manifest" href="manifest.webmanifest">' in page
    assert 'href="apple-touch-icon.png"' in page
    # Still exactly one inline script (the import map): the CSP admits it by its hash.
    assert len(server.inline_script_hashes(page)) == 1
    app_js = (server.WEB_DIR / "app.js").read_text("utf-8")
    assert "import { createMeetings } from './meetings.js';" in app_js and "'meetPanel'" in app_js
    manifest = json.loads((server.WEB_DIR / server.MANIFEST).read_text("utf-8"))
    assert "start_url" not in manifest   # the page it was added from: a meeting link stays one
    assert {icon["src"] for icon in manifest["icons"]} <= set(server.ICONS)


def test_the_hosted_app_keeps_meetings_on_the_volume(tmp_path):
    from litereality_agent.room_ops.roomcall.avatars import MemoryBudget

    root = tmp_path / "volume"
    root.mkdir()
    commits = []
    app = hosted.build_app(root, scratch=tmp_path / "demo", scan_scratch=tmp_path / "scans",
                           scan_budget=MemoryBudget(1), commit=lambda: commits.append(1))
    service = app.state.meetings
    assert service.store.root == root / mt.MEETINGS_DIR
    with TestClient(app) as c:
        start = mt.iso(datetime.now(timezone.utc) + timedelta(days=1))
        made = c.post("/meetings", json={"title": "x", "start": start, "room": "call"})
        assert made.status_code == 201
    assert (root / mt.MEETINGS_DIR / made.json()["id"] / mt.MEETING_FILE).is_file() and commits
    # Never a version folder (a publish never prunes it), never what CURRENT can name.
    assert not hosted._VERSION.match(mt.MEETINGS_DIR) and not hosted._NAME.match(mt.MEETINGS_DIR)
    assert hosted.build_app(root, scratch=tmp_path / "d2", uploads="off",
                            meetings=False).state.meetings is None


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_reminders_cron_is_its_own_app_and_the_room_call_keeps_one_function():
    pytest.importorskip("modal")
    reminders = _load(REPO / mt.REMINDERS_DEPLOY_FILE, "reminders_deploy")
    assert reminders.app.name == mt.REMINDERS_APP != hosted.APP_NAME
    assert set(reminders.app.registered_functions) == {"remind"}
    roomcall = _load(REPO / hosted.DEPLOY_FILE, "roomcall_deploy_one")
    assert set(roomcall.app.registered_functions) == {hosted.FUNCTION_NAME}


def test_the_vapid_script_makes_a_matching_pair_and_never_prints_the_private_key(monkeypatch, capsys):
    script = _load(REPO / "scripts" / "roomcall" / "vapid_secret.py", "vapid_secret")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    private, public = script.make_keys()
    raw = mt.b64url_decode(private)
    assert len(raw) == 32 and mt.check_public_key(public) == public
    derived = ec.derive_private_key(int.from_bytes(raw, "big"), ec.SECP256R1()).public_key()
    assert mt.b64url_encode(derived.public_bytes(serialization.Encoding.X962,
                                                 serialization.PublicFormat.UncompressedPoint)) == public
    seen = {}

    def run(command, check=False):
        path = Path(command[command.index("--from-json") + 1])
        seen.update(command=command, mode=path.stat().st_mode & 0o777,
                    values=json.loads(path.read_text()))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(script.subprocess, "run", run)
    assert script.main(["--env", "main"]) == 0
    assert seen["command"][2:6] == ["modal", "secret", "create", mt.VAPID_SECRET]
    assert "--force" not in seen["command"] and seen["mode"] == 0o600
    values = seen["values"]
    assert set(values) == set(mt.VAPID_KEYS) and values["VAPID_SUBJECT"].startswith("mailto:")
    out = capsys.readouterr()
    assert values["VAPID_PRIVATE_KEY"] not in out.out + out.err
    assert values["VAPID_PUBLIC_KEY"] in out.out
    assert not Path(seen["command"][seen["command"].index("--from-json") + 1]).exists()
    assert script.main(["--rotate", "--subject", "https://example.org"]) == 0
    assert "--force" in seen["command"]


# -- the page's helpers, under Node ----------------------------------------------------------------

NODE = shutil.which("node")

JS_CHECKS = r"""
const m = await import(process.argv[2]);
const out = {};
const win = (o) => ({ isSecureContext: true, navigator: {}, ...o });
out.support = {
  chrome: m.reminderSupport(win({ navigator: { serviceWorker: {}, userAgent: 'Chrome' }, PushManager: 1, Notification: 1 })),
  insecure: m.reminderSupport(win({ isSecureContext: false, navigator: { serviceWorker: {} }, PushManager: 1, Notification: 1 })),
  iosTab: m.reminderSupport(win({ navigator: { userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X)' } })),
  ipadTab: m.reminderSupport(win({ navigator: { userAgent: 'Macintosh', platform: 'MacIntel', maxTouchPoints: 5 } })),
  iosOldHome: m.reminderSupport(win({ navigator: { userAgent: 'iPhone', standalone: true } })),
  iosHome: m.reminderSupport(win({ navigator: { userAgent: 'iPhone', standalone: true, serviceWorker: {} }, PushManager: 1, Notification: 1 })),
  desktopOld: m.reminderSupport(win({ navigator: { userAgent: 'Firefox' } })),
};
out.iso = m.localInputToISO('2026-11-24T15:00');
out.isoSummer = m.localInputToISO('2026-07-01T09:30');
out.bad = [m.localInputToISO('2026-02-30T10:00'), m.localInputToISO(''), m.localInputToISO('soon')];
out.local = m.toLocalInput(new Date('2026-11-24T20:00:00Z'));
out.start = m.toLocalInput(m.defaultStart(new Date('2026-11-24T20:05:00Z')));
out.start2 = m.toLocalInput(m.defaultStart(new Date('2026-11-24T20:15:00Z')));
out.when = m.formatWhen('2026-11-24T14:00:00Z', { locale: 'en-GB', timeZone: 'Europe/Berlin' });
out.rel = ['2026-11-24T14:00:00Z', '2026-11-24T16:05:00Z', '2026-11-26T14:00:00Z', '2026-11-24T13:00:00Z', '2026-11-24T12:00:00Z']
  .map((s) => m.relativeWhen(s, new Date(Date.parse(s) + 3600000).toISOString(), new Date('2026-11-24T13:50:00Z')));
const key = 'BAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4fICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8';
const bytes = m.b64urlToBytes(key);
out.key = [bytes.length, bytes[0], bytes[64], m.sameKey(bytes.buffer, bytes), m.sameKey(new Uint8Array(65).buffer, bytes), m.sameKey(null, bytes)];
out.parsed = m.parseMeeting({ id: 'AbCdEfGhIjKlMnOp', title: 'x'.repeat(100), start: '2026-11-24T14:00:00Z', end: '2026-11-24T15:00:00Z', minutes: 60, room: 'call', pick: 'office', pick_label: 'Office' });
out.rejected = [m.parseMeeting({ id: 'short', start: '2026-11-24T14:00:00Z', room: 'r' }),
  m.parseMeeting({ id: 'AbCdEfGhIjKlMnOp', start: 'soon', room: 'r' }),
  m.parseMeeting({ id: 'AbCdEfGhIjKlMnOp', start: '2026-11-24T14:00:00Z', room: '../x' })];
out.links = [m.meetingLink('https://a.example', 'AbCdEfGhIjKlMnOp'), m.calendarHref('AbCdEfGhIjKlMnOp')];
const mem = { v: JSON.stringify({ AbCdEfGhIjKlMnOp: '2026-11-24T14:00:00Z', Old0000000000000: '2026-11-01T00:00:00Z', bad: 'x' }),
              getItem() { return this.v; } };
out.reminded = m.remindedList(mem, Date.parse('2026-11-24T13:00:00Z'));
const owned = { v: JSON.stringify({ AbCdEfGhIjKlMnOp: { token: 'T'.repeat(32), end: '2026-11-24T15:00:00Z' },
  Old0000000000000: { token: 'T'.repeat(32), end: '2026-11-24T12:00:00Z' },
  NoToken000000000: { token: 'bad token', end: '2026-11-24T15:00:00Z' } }), getItem() { return this.v; } };
out.owned = m.ownedList(owned, Date.parse('2026-11-24T13:00:00Z'));
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(NODE is None, reason="needs Node to run the page's functions")
def test_the_pages_meeting_helpers(tmp_path):
    module = tmp_path / "meetings.mjs"
    shutil.copyfile(server.WEB_DIR / "meetings.js", module)
    (tmp_path / "check.mjs").write_text(JS_CHECKS)
    done = subprocess.run([NODE, str(tmp_path / "check.mjs"), module.as_uri()], capture_output=True,
                          text=True, timeout=60, env={**os.environ, "TZ": "America/New_York"})
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert out["support"] == {"chrome": "push", "insecure": "none", "iosTab": "ios-home",
                              "ipadTab": "ios-home", "iosOldHome": "none", "iosHome": "push",
                              "desktopOld": "none"}
    # The viewer's local time (New York here) goes to the server as UTC, DST included.
    assert out["iso"] == "2026-11-24T20:00:00.000Z" and out["isoSummer"] == "2026-07-01T13:30:00.000Z"
    assert out["bad"] == [None, None, None] and out["local"] == "2026-11-24T15:00"
    assert out["start"] == "2026-11-24T15:30" and out["start2"] == "2026-11-24T16:00"
    assert "15:00" in out["when"] and "Tue" in out["when"]
    assert out["rel"] == ["in 10 min", "in 2 h 15 min", "in 2 days", "happening now", "ended"]
    assert out["key"] == [65, 4, 63, True, False, False]
    assert out["parsed"] == {"id": "AbCdEfGhIjKlMnOp", "title": "x" * 80, "start": "2026-11-24T14:00:00.000Z",
                             "end": "2026-11-24T15:00:00.000Z", "minutes": 60, "room": "call",
                             "pick": "office", "pickLabel": "Office"}
    assert out["rejected"] == [None, None, None]
    assert out["links"] == ["https://a.example/m/AbCdEfGhIjKlMnOp", "m/AbCdEfGhIjKlMnOp.ics"]
    assert out["reminded"] == {"AbCdEfGhIjKlMnOp": "2026-11-24T14:00:00Z"}
    assert out["owned"] == {"AbCdEfGhIjKlMnOp": {"token": "T" * 32, "end": "2026-11-24T15:00:00Z"}}


CARD_CHECKS = r"""
// The meeting card's reminder controls, through createMeetings with a stand-in page.
const m = await import(process.argv[2]);
const [ua, push] = [process.argv[3], process.argv[4] === 'key' ? 'BAABAgMEBQYHCAkKCwwNDg8QERITFBUWFxgZGhscHR4fICEiIyQlJicoKSorLC0uLzAxMjM0NTY3ODk6Ozw9Pj8' : null];
const els = {};
const $ = (id) => (els[id] ||= { id, hidden: true, disabled: false, textContent: '', value: '', href: '',
  addEventListener() {}, setAttribute() {}, focus() {}, select() {}, replaceChildren() {} });
const start = new Date(Date.now() + 3 * 3600000).toISOString();
const store = { v: JSON.stringify({ AbCdEfGhIjKlMnOp: { token: 'T'.repeat(32), end: '2099-01-01T00:00:00Z' } }),
                getItem() { return this.v; }, setItem(k, v) { this.v = v; } };
globalThis.window = { navigator: { userAgent: ua }, isSecureContext: true, localStorage: store };
globalThis.location = { origin: 'https://room.example' };
globalThis.fetch = async (url) => {
  const body = String(url).startsWith('meetings.json') ? { enabled: true, push }
    : { id: 'AbCdEfGhIjKlMnOp', title: 'Stand-up', start, end: start, minutes: 30, room: 'team' };
  return { ok: true, status: 200, json: async () => body };
};
const api = m.createMeetings({ $, toast() {}, togglePanel() {}, roomId: () => 'team' });
await api.boot(new URLSearchParams('room=team&meeting=AbCdEfGhIjKlMnOp'));
console.log(JSON.stringify({ card: !els.meetCard.hidden, hint: els.meetHint.hidden ? null : els.meetHint.textContent,
  remind: !els.meetRemind.hidden, cancel: !els.meetCardCancel.hidden }));
"""


@pytest.mark.skipif(NODE is None, reason="needs Node to run the page's functions")
@pytest.mark.parametrize("ua, key, hint", [
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X)", "key", "Add to Home Screen"),
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X)", "none", None),   # no reminders here
    ("Mozilla/5.0 (X11; Linux x86_64) Firefox/50", "key", None),               # calendar only
])
def test_the_ios_hint_shows_only_where_reminders_are_set_up(tmp_path, ua, key, hint):
    module = tmp_path / "meetings.mjs"
    shutil.copyfile(server.WEB_DIR / "meetings.js", module)
    (tmp_path / "card.mjs").write_text(CARD_CHECKS)
    done = subprocess.run([NODE, str(tmp_path / "card.mjs"), module.as_uri(), ua, key],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert out["card"] and not out["remind"] and out["cancel"]   # this browser made it: it may cancel
    assert out["hint"] is None if hint is None else out["hint"].startswith(hint)


SW_CHECKS = r"""
const handlers = {};
const shown = [], opened = [], focused = [];
const self = globalThis.self = {
  location: { origin: 'https://room.example' },
  registration: { scope: 'https://room.example/', showNotification: (t, o) => { shown.push([t, o]); return Promise.resolve(); } },
  clients: { claim: () => Promise.resolve(), openWindow: (u) => { opened.push(u); return Promise.resolve(); },
             matchAll: async () => [
               { url: 'https://other.example/?room=team', focus() { focused.push(this.url); } },
               { url: 'https://room.example/?room=open&meeting=AbCdEfGhIjKlMnOp', focus() { focused.push(this.url); } },
               { url: 'https://room.example/?room=team&name=Ann', focus() { focused.push(this.url); } }] },
  skipWaiting() {},
  addEventListener: (t, f) => { handlers[t] = f; },
};
await import(process.argv[2]);
const waits = [];
const ev = (data) => ({ data: data === null ? null : { json: () => (typeof data === 'string' ? JSON.parse(data) : data) },
                        waitUntil: (p) => waits.push(p) });
handlers.push(ev({ title: 'Starting in 15 min: Stand-up', start: '2026-11-24T14:00:00Z', url: '?room=team&meeting=AbCdEfGhIjKlMnOp', tag: 'meeting-AbCdEfGhIjKlMnOp' }));
handlers.push(ev('{broken'));
handlers.push(ev({ title: 'x', url: 'https://evil.example/' }));
await Promise.all(waits);
const click = (url) => ({ notification: { data: { url }, close() {} }, waitUntil: (p) => waits.push(p) });
handlers.notificationclick(click(shown[0][1].data.url));      // a tab is in room team already
handlers.notificationclick(click('https://room.example/?room=open&meeting=AbCdEfGhIjKlMnOp'));
handlers.notificationclick(click('?room=empty&meeting=AbCdEfGhIjKlMnOp'));   // nobody there yet
await Promise.all(waits);
console.log(JSON.stringify({ shown, opened, focused }));
"""


@pytest.mark.skipif(NODE is None, reason="needs Node to run the service worker")
def test_the_service_worker_shows_the_reminder_and_opens_the_room(tmp_path):
    module = tmp_path / "sw.mjs"
    shutil.copyfile(server.WEB_DIR / "sw.js", module)
    (tmp_path / "check.mjs").write_text(SW_CHECKS)
    done = subprocess.run([NODE, str(tmp_path / "check.mjs"), module.as_uri()], capture_output=True,
                          text=True, timeout=60, env={**os.environ, "TZ": "Europe/Berlin", "LANG": "en_GB.UTF-8"})
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    (title, first), (title2, broken), (_, evil) = out["shown"]
    assert title == "Starting in 15 min: Stand-up" and first["tag"] == "meeting-AbCdEfGhIjKlMnOp"
    assert first["body"].startswith("Starts at 15:00")         # the device's own time zone
    assert first["data"]["url"] == "https://room.example/?room=team&meeting=AbCdEfGhIjKlMnOp"
    assert title2 == "Room call" and broken["body"] == "Tap to join"   # every push shows something
    assert evil["data"]["url"] == "https://room.example/"     # never another site
    # A tab already in the meeting's call room is brought forward (never one on another site);
    # a new one opens only where none is.
    assert out["focused"] == ["https://room.example/?room=team&name=Ann",
                              "https://room.example/?room=open&meeting=AbCdEfGhIjKlMnOp"]
    assert out["opened"] == ["https://room.example/?room=empty&meeting=AbCdEfGhIjKlMnOp"]
