"""Scheduled meetings on the room call, with a reminder 15 minutes before, even when the page is
closed.

Anyone in a call (or on the join card) can schedule one: a title, a start time, the call room it
is in (and, optionally, a ready room from the library that the first person in switches it to).
It gets an unguessable link, `/m/<id>`, that opens the room call with a card saying when it is,
with **Remind me** (Web Push) and **Add to calendar** (an .ics with its own 15-minute alarm).

Where things live (hosted: the scenes Volume; the folder is dot-named, so it is never mistaken for
a version folder and never pruned by a publish):

    .meetings/<id>/meeting.json        what was scheduled            (written once, by the web app)
    .meetings/<id>/subs/<sha>.json     one browser's push subscription (web app; cron removes dead)
    .meetings/<id>/sent.json           subscriptions already reminded  (the reminders cron only)
    .meetings/.vapid-public-key        the VAPID public key            (the reminders cron)

One writer per file: the web app and the reminders cron are separate containers on one Volume,
whose commits are last-writer-wins per file, so neither ever rewrites a file the other writes.

The reminders cron (`deploy/modal/reminders/app.py`) runs every minute: `remind_round` reminds
every subscription of a meeting starting within `REMIND_EARLY` (15 minutes) that has not been
reminded, fairly across meetings and within a time limit, drops subscriptions the push service
says are gone (404/410) or that keep failing, and removes meetings a day after they ended. Only it holds the VAPID private key (a Modal secret); it publishes the public key on the
Volume, where the web app reads it for browsers subscribing (`MeetingStore.push_key`).

Standard library only at module level: the reminders image installs pywebpush and nothing of the
room call's own stack.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import shutil
import time
import unicodedata
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

log = logging.getLogger(__name__)

MEETINGS_DIR = ".meetings"
MEETING_FILE = "meeting.json"
SUBS_DIR = "subs"
SENT_FILE = "sent.json"
PUSH_KEY_FILE = ".vapid-public-key"

# The reminders: their own small Modal app (see deploy/modal/reminders/app.py for why not a second
# function of the room call's), its secret, and what its image installs (pywebpush 2.5.0 and the
# versions it resolved to for Python 3.11; checked by sending to a local endpoint and decrypting).
REMINDERS_APP = "litereality-roomcall-reminders"
REMINDERS_DEPLOY_FILE = "deploy/modal/reminders/app.py"
VAPID_SECRET = "roomcall-vapid"
VAPID_KEYS = ("VAPID_PRIVATE_KEY", "VAPID_PUBLIC_KEY", "VAPID_SUBJECT")
REMINDER_PACKAGES = (
    "pywebpush==2.5.0",
    "py-vapid==1.9.6",
    "http-ece==1.2.1",
    "cryptography==50.0.2",
    "requests==2.34.2",
    "aiohttp==3.14.5",
)

MEETING_ID = re.compile(r"^[A-Za-z0-9_-]{16}$")          # secrets.token_urlsafe(12)
ROOM_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")            # as the server's
PICK_NAME = re.compile(r"^[A-Za-z0-9_-]{1,48}$")          # a library room's name
SUB_ID = re.compile(r"^[0-9a-f]{32}$")
_HASH = re.compile(r"^[0-9a-f]{16,64}$")                  # who made it / the cancel token's hash
MAX_TITLE = 80
DEFAULT_MINUTES = 60
MIN_MINUTES = 5
MAX_MINUTES = 8 * 60
# How far ahead a meeting may be, and how late its start may already be when it is made (a form
# filled in slowly, a clock a little off).
MAX_AHEAD = timedelta(days=90)
PAST_SLACK = timedelta(minutes=2)
# Caps: meetings not yet over, on the whole server, per call room and per network that made them
# (so filling a room's or the server's cap takes many networks, not one), and browsers per
# meeting in all and from one network (so one network cannot take every reminder slot; an office
# behind one address still fits).
MAX_FUTURE = 200
MAX_PER_ROOM = 20
MAX_PER_CREATOR = 5
MAX_SUBS = 200
MAX_SUBS_PER_CLIENT = 20
# Per client address: meetings made per 10 minutes and per day, and (un)subscribes per hour.
CREATE_PER_10_MIN = 6
CREATE_PER_DAY = 30
SUBSCRIBE_PER_HOUR = 60
MAX_BODY = 4096
# The reminder: 15 minutes before. The cron runs every minute (a few seconds past it) and sends
# to every subscription not yet reminded once the start is `REMIND_EARLY` away or less, so the
# first run that may send is the one a few seconds after start - 15 min; a run that is late or
# missed is made up by the next one; none is sent once the meeting has started.
REMIND_BEFORE = timedelta(minutes=15)
REMIND_EARLY = REMIND_BEFORE
# A meeting is removed (with its subscriptions) this long after it ended.
PRUNE_AFTER = timedelta(days=1)
# One round: at most this many pushes, `SEND_WORKERS` at a time, each with `PUSH_TIMEOUT`, and
# no new batch once `ROUND_SECONDS` have passed (the function's timeout is 120 s; the next
# minute's round carries on). A subscription whose push fails `MAX_STRIKES` times (in rounds
# where other pushes went through, so the trouble is the subscription's, not ours) is dropped.
MAX_SENDS_PER_ROUND = 1000
SEND_WORKERS = 16
PUSH_TIMEOUT = 5.0
ROUND_SECONDS = 45.0
MAX_STRIKES = 3

# The push services browsers hand out subscriptions for: Chrome/Edge-on-Android/Opera/Samsung
# (FCM), Firefox (Mozilla autopush), Safari on macOS and iOS (Apple), Edge on Windows (WNS). Only
# these are accepted, so a subscription can never make the cron POST to an address of someone's
# choosing.
PUSH_HOSTS = frozenset({"fcm.googleapis.com", "updates.push.services.mozilla.com"})
PUSH_HOST_SUFFIXES = (".push.apple.com", ".notify.windows.com")
MAX_ENDPOINT = 1024
_B64URL = re.compile(r"^[A-Za-z0-9_-]+={0,2}$")
_HOST = re.compile(r"^[a-z0-9.-]{1,253}$")


class MeetingError(Exception):
    """A refused request: an HTTP status, a message for people, and maybe a Retry-After."""

    def __init__(self, status: int, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(moment: datetime) -> str:
    """UTC, to the second, with a Z: what is stored and sent."""
    return moment.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_time(text: Any) -> datetime:
    """An ISO 8601 time with its offset (the page sends the viewer's local time as UTC) -> UTC.
    A time without an offset is refused: it would mean the server's zone, nobody's."""
    if not isinstance(text, str) or not 10 <= len(text) <= 40:
        raise ValueError("a start time is needed")
    value = text.strip()
    if value.endswith(("Z", "z")):
        value = value[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("the start time is not a date and time") from None
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("the start time needs its time zone")
    return moment.astimezone(timezone.utc).replace(microsecond=0)


def clean_title(text: Any) -> str:
    """One line of at most `MAX_TITLE` characters: whitespace runs become one space, control and
    format characters (bidi overrides included) go."""
    if not isinstance(text, str):
        raise ValueError("a title is needed")
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("C") or ch.isspace())
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("a title is needed")
    if len(text) > MAX_TITLE:
        raise ValueError(f"the title is longer than {MAX_TITLE} characters")
    return text


@dataclass(frozen=True)
class Meeting:
    id: str
    title: str
    start: datetime
    minutes: int
    room: str
    pick: str | None = None            # a library room the call room switches to at the start
    pick_label: str | None = None
    created: datetime | None = None
    by: str | None = None              # a hash of the network that made it (`client_hash`)
    owner: str | None = None           # sha256 of the cancel token handed to whoever made it

    @property
    def end(self) -> datetime:
        return self.start + timedelta(minutes=self.minutes)

    def to_json(self) -> dict:
        return {"id": self.id, "title": self.title, "start": iso(self.start),
                "minutes": self.minutes, "room": self.room, "pick": self.pick,
                "pick_label": self.pick_label,
                "created": iso(self.created) if self.created else None,
                "by": self.by, "owner": self.owner}

    @classmethod
    def from_json(cls, data: Any) -> Meeting:
        """A stored meeting, checked as strictly as a new one (the Volume is not trusted more
        than a request: a file there may be from another version of this code)."""
        if not isinstance(data, dict) or not MEETING_ID.match(str(data.get("id", ""))):
            raise ValueError("not a meeting")
        minutes = data.get("minutes")
        if isinstance(minutes, bool) or not isinstance(minutes, int):
            raise ValueError("bad minutes")
        room = data.get("room")
        if not isinstance(room, str) or not ROOM_ID.match(room):
            raise ValueError("bad room")
        pick = data.get("pick")
        if pick is not None and (not isinstance(pick, str) or not PICK_NAME.match(pick)):
            raise ValueError("bad pick")
        label = data.get("pick_label")
        created = data.get("created")
        by, owner = data.get("by"), data.get("owner")
        return cls(id=data["id"], title=clean_title(data.get("title")),
                   start=parse_time(data.get("start")),
                   minutes=max(MIN_MINUTES, min(MAX_MINUTES, minutes)), room=room, pick=pick,
                   pick_label=label[:80] if isinstance(label, str) and pick else None,
                   created=parse_time(created) if isinstance(created, str) else None,
                   by=by if isinstance(by, str) and _HASH.match(by) else None,
                   owner=owner if isinstance(owner, str) and _HASH.match(owner) else None)

    def join_query(self) -> str:
        """The page's query for this meeting: its call room and the meeting card."""
        return "?" + urlencode({"room": self.room, "meeting": self.id})

    def public(self) -> dict:
        """What a browser is told (GET /meetings/<id>.json and the create answer): never who
        made it or the cancel token's hash."""
        data = self.to_json()
        del data["by"], data["owner"]
        return {**data, "end": iso(self.end), "link": f"m/{self.id}",
                "ics": f"m/{self.id}.ics", "join": self.join_query()}

    def owned_by(self, token: Any) -> bool:
        """Whether `token` is this meeting's cancel token."""
        if not self.owner or not isinstance(token, str) or not 16 <= len(token) <= 64:
            return False
        return hmac.compare_digest(token_hash(token), self.owner)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def client_hash(key: str) -> str:
    """Who made a meeting or a subscription, as stored on the Volume: a hash of the network's
    `client_key`, never the address itself."""
    return hashlib.sha256(f"roomcall-meetings:{key}".encode()).hexdigest()[:16]


def new_meeting(body: Any, *, now: datetime | None = None, picks: dict[str, str] | None = None,
                by: str | None = None, owner: str | None = None) -> Meeting:
    """A meeting from a request's JSON: {title, start, room, minutes?, pick?}. `picks` maps the
    library's room names to their labels (a pick must be one of them). Raises MeetingError(400)."""
    now = now or _utcnow()
    if not isinstance(body, dict):
        raise MeetingError(400, "send the meeting as a JSON object")
    try:
        title = clean_title(body.get("title"))
        start = parse_time(body.get("start"))
    except ValueError as exc:
        raise MeetingError(400, str(exc)) from None
    if start < now - PAST_SLACK:
        raise MeetingError(400, "that time has passed")
    if start > now + MAX_AHEAD:
        raise MeetingError(400, f"a meeting can be at most {MAX_AHEAD.days} days ahead")
    minutes = body.get("minutes", DEFAULT_MINUTES)
    if minutes is None:
        minutes = DEFAULT_MINUTES
    if isinstance(minutes, bool) or not isinstance(minutes, int) \
            or not MIN_MINUTES <= minutes <= MAX_MINUTES:
        raise MeetingError(400, f"the length is {MIN_MINUTES} to {MAX_MINUTES} minutes")
    room = body.get("room")
    if not isinstance(room, str) or not ROOM_ID.match(room):
        raise MeetingError(400, "bad room id")
    pick = body.get("pick") or None
    label = None
    if pick is not None:
        if not isinstance(pick, str) or not PICK_NAME.match(pick) or pick not in (picks or {}):
            raise MeetingError(400, "that room is not in the library")
        label = str((picks or {})[pick])[:80]
    return Meeting(id=secrets.token_urlsafe(12), title=title, start=start, minutes=minutes,
                   room=room, pick=pick, pick_label=label, created=now.replace(microsecond=0),
                   by=by, owner=owner)


# ---------------------------------------------------------------------------------------------
# Push subscriptions
# ---------------------------------------------------------------------------------------------

def b64url_decode(text: Any) -> bytes:
    if not isinstance(text, str) or not _B64URL.match(text) or len(text) > 200:
        raise ValueError("not base64url")
    text = text.rstrip("=")
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def push_host_ok(host: str | None) -> bool:
    host = (host or "").lower()
    if not _HOST.match(host) or ".." in host:
        return False
    return host in PUSH_HOSTS or any(host.endswith(s) and len(host) > len(s)
                                     for s in PUSH_HOST_SUFFIXES)


def check_subscription(data: Any) -> dict:
    """A browser's `PushSubscription.toJSON()`, kept to what sending needs, or MeetingError(400):
    an https endpoint on a known push service, a P-256 public key (65 bytes, uncompressed) and a
    16-byte auth secret."""
    if not isinstance(data, dict):
        raise MeetingError(400, "send the subscription as a JSON object")
    endpoint = data.get("endpoint")
    if not isinstance(endpoint, str) or not 0 < len(endpoint) <= MAX_ENDPOINT \
            or any(ch.isspace() or ord(ch) < 32 for ch in endpoint):
        raise MeetingError(400, "bad push endpoint")
    try:
        parts = urlsplit(endpoint)
        port = parts.port
    except ValueError:
        raise MeetingError(400, "bad push endpoint") from None
    if parts.scheme != "https" or parts.username or parts.password or port not in (None, 443) \
            or not push_host_ok(parts.hostname):
        raise MeetingError(400, "that push service is not one this server sends to")
    keys = data.get("keys")
    if not isinstance(keys, dict):
        raise MeetingError(400, "the subscription has no keys")
    try:
        p256dh, auth = b64url_decode(keys.get("p256dh")), b64url_decode(keys.get("auth"))
    except ValueError:
        raise MeetingError(400, "the subscription's keys are not base64url") from None
    if len(p256dh) != 65 or p256dh[0] != 4 or len(auth) != 16:
        raise MeetingError(400, "the subscription's keys are the wrong size")
    return {"endpoint": endpoint, "keys": {"p256dh": b64url_encode(p256dh),
                                           "auth": b64url_encode(auth)}}


def sub_id(endpoint: str) -> str:
    return hashlib.sha256(endpoint.encode("utf-8")).hexdigest()[:32]


def check_public_key(text: Any) -> str | None:
    """A VAPID public key as browsers take it (`applicationServerKey`): base64url of the 65-byte
    uncompressed P-256 point. None for anything else."""
    try:
        raw = b64url_decode(text.strip() if isinstance(text, str) else text)
    except ValueError:
        return None
    return b64url_encode(raw) if len(raw) == 65 and raw[0] == 4 else None


# ---------------------------------------------------------------------------------------------
# The store (a folder: hosted, on the scenes Volume)
# ---------------------------------------------------------------------------------------------

@dataclass
class MeetingState:
    """A meeting as the cron sees it: its subscriptions and which were reminded."""

    meeting: Meeting
    subs: dict[str, dict] = field(default_factory=dict)
    sent: set[str] = field(default_factory=set)


class MeetingStore:
    """Meetings in a folder (`MEETINGS_DIR` on the Volume). Every write goes to a temporary file
    renamed into place, under `lock` (the Volume's: a reload refuses to run with files open),
    followed by `commit` (the Volume's, when the web app writes; the cron commits once a round).
    Blocking: the server calls it in a thread."""

    def __init__(self, root: Path, *, commit: Callable[[], Any] | None = None,
                 lock: Any = None) -> None:
        self.root = Path(root)
        self.commit = commit
        self.lock = lock if lock is not None else contextlib.nullcontext()

    # -- paths
    def _folder(self, mid: str) -> Path:
        if not MEETING_ID.match(mid or ""):
            raise ValueError("bad meeting id")
        return self.root / mid

    def _write(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}")
        tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, path)

    def _commit(self) -> None:
        if self.commit is not None:
            try:
                self.commit()
            except Exception as exc:  # noqa: BLE001 — written locally; the next commit carries it
                log.warning("roomcall: meetings: commit failed: %s", exc)

    @staticmethod
    def _read(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # -- meetings
    def get(self, mid: str) -> Meeting | None:
        try:
            folder = self._folder(mid)
        except ValueError:
            return None
        data = self._read(folder / MEETING_FILE)
        try:
            meeting = Meeting.from_json(data)
        except (ValueError, TypeError):
            return None
        return meeting if meeting.id == mid else None

    def ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if MEETING_ID.match(p.name) and p.is_dir())

    def all(self) -> list[Meeting]:
        return [m for m in (self.get(mid) for mid in self.ids()) if m is not None]

    def add(self, meeting: Meeting, *, now: datetime | None = None, max_future: int | None = None,
            max_per_room: int | None = None, max_per_creator: int | None = None) -> Meeting:
        """Store a new meeting if the caps let it in (meetings not over yet: `max_future` in all,
        `max_per_room` in its call room, `max_per_creator` made from its network);
        MeetingError(429) otherwise. Meetings long over are removed on the way (the cron does it
        too; this keeps a server without it bounded)."""
        now = now or _utcnow()
        max_future = MAX_FUTURE if max_future is None else max_future
        max_per_room = MAX_PER_ROOM if max_per_room is None else max_per_room
        max_per_creator = MAX_PER_CREATOR if max_per_creator is None else max_per_creator
        with self.lock:
            self.prune(now, commit=False)
            live = [m for m in self.all() if m.end > now]
            if meeting.by and sum(m.by == meeting.by for m in live) >= max_per_creator:
                raise MeetingError(429, f"your network already has {max_per_creator} meetings "
                                        "coming up; cancel one or wait until one has happened")
            if len(live) >= max_future:
                raise MeetingError(429, "this server has all the meetings it can hold; try again "
                                        "after some have happened")
            if sum(m.room == meeting.room for m in live) >= max_per_room:
                raise MeetingError(429, f"this room already has {max_per_room} meetings coming up")
            self._write(self._folder(meeting.id) / MEETING_FILE, meeting.to_json())
            self._commit()
        return meeting

    def remove(self, mid: str) -> None:
        shutil.rmtree(self._folder(mid), ignore_errors=True)

    def cancel(self, mid: str, token: Any) -> bool:
        """Remove a meeting (and its reminders) for whoever holds its cancel token. False: no
        such meeting; MeetingError(403): the wrong token."""
        with self.lock:
            meeting = self.get(mid)
            if meeting is None:
                return False
            if not meeting.owned_by(token):
                raise MeetingError(403, "only whoever scheduled this meeting can cancel it")
            self.remove(mid)
            self._commit()
        return True

    def purge_room(self, room: str) -> list[str]:
        """Every meeting in one call room removed (for the operator, against a flood)."""
        with self.lock:
            gone = [m.id for m in self.all() if m.room == room]
            for mid in gone:
                self.remove(mid)
            if gone:
                self._commit()
        return gone

    def prune(self, now: datetime, *, commit: bool = True) -> list[str]:
        """Remove meetings that ended more than `PRUNE_AFTER` ago, and unreadable ones."""
        removed = []
        for mid in self.ids():
            meeting = self.get(mid)
            # An unreadable folder may be one being written right now: only an old one goes.
            gone = expired(meeting, now) if meeting is not None else self._old(self.root / mid, now)
            if gone:
                self.remove(mid)
                removed.append(mid)
        if removed and commit:
            self._commit()
        return removed

    @staticmethod
    def _old(folder: Path, now: datetime) -> bool:
        try:
            return now.timestamp() - folder.stat().st_mtime > PRUNE_AFTER.total_seconds()
        except OSError:
            return True

    # -- subscriptions
    def subs(self, mid: str) -> dict[str, dict]:
        folder = self._folder(mid) / SUBS_DIR
        found: dict[str, dict] = {}
        if not folder.is_dir():
            return found
        for path in sorted(folder.glob("*.json")):
            if not SUB_ID.match(path.stem):
                continue
            data = self._read(path)
            try:
                sub = check_subscription(data)
            except MeetingError:
                continue
            if sub_id(sub["endpoint"]) == path.stem:
                found[path.stem] = sub
        return found

    def subscribe(self, mid: str, subscription: dict, *, by: str | None = None,
                  max_subs: int | None = None, max_per_client: int | None = None) -> str:
        """Store a browser's subscription; `by` (`client_hash`) is kept beside it so one network
        holds at most `max_per_client` of the meeting's `max_subs` slots. A browser already
        subscribed is always taken again (its endpoint is its slot)."""
        max_subs = MAX_SUBS if max_subs is None else max_subs
        max_per_client = MAX_SUBS_PER_CLIENT if max_per_client is None else max_per_client
        sid = sub_id(subscription["endpoint"])
        folder = self._folder(mid) / SUBS_DIR
        with self.lock:
            if not (folder / f"{sid}.json").exists():
                files = sorted(folder.glob("*.json")) if folder.is_dir() else []
                if len(files) >= max_subs:
                    raise MeetingError(429, "this meeting has all the reminders it can send")
                if by is not None:
                    mine = 0
                    for path in files:
                        data = self._read(path)
                        mine += isinstance(data, dict) and data.get("by") == by
                    if mine >= max_per_client:
                        raise MeetingError(429, f"your network has set {max_per_client} reminders "
                                                "for this meeting already")
            self._write(folder / f"{sid}.json", {**subscription, **({"by": by} if by else {})})
            self._commit()
        return sid

    def unsubscribe(self, mid: str, endpoint: str, *, commit: bool = True) -> bool:
        path = self._folder(mid) / SUBS_DIR / f"{sub_id(endpoint)}.json"
        with self.lock:
            try:
                path.unlink()
            except FileNotFoundError:
                return False
            if commit:
                self._commit()
        return True

    def drop_sub(self, mid: str, sid: str) -> None:
        if SUB_ID.match(sid):
            with contextlib.suppress(FileNotFoundError):
                (self._folder(mid) / SUBS_DIR / f"{sid}.json").unlink()

    def sent(self, mid: str) -> set[str]:
        data = self._read(self._folder(mid) / SENT_FILE)
        subs = data.get("subs") if isinstance(data, dict) else None
        return {s for s in subs if isinstance(s, str) and SUB_ID.match(s)} if isinstance(subs, list) else set()

    def strikes(self, mid: str) -> dict[str, int]:
        """How many rounds each subscription's push has failed in (the cron's, in sent.json)."""
        data = self._read(self._folder(mid) / SENT_FILE)
        fails = data.get("fails") if isinstance(data, dict) else None
        if not isinstance(fails, dict):
            return {}
        return {k: v for k, v in fails.items() if isinstance(k, str) and SUB_ID.match(k)
                and isinstance(v, int) and not isinstance(v, bool) and v > 0}

    def mark_sent(self, mid: str, sids: Iterable[str], now: datetime,
                  fails: dict[str, int] | None = None) -> None:
        """The cron's record for a meeting: who was reminded, and failures so far (kept as they
        are when `fails` is None)."""
        fails = self.strikes(mid) if fails is None else fails
        self._write(self._folder(mid) / SENT_FILE, {"subs": sorted(set(sids)), "at": iso(now),
                                                    "fails": dict(sorted(fails.items()))})

    def states(self) -> list[MeetingState]:
        return [MeetingState(m, self.subs(m.id), self.sent(m.id)) for m in self.all()]

    # -- the VAPID public key
    def push_key(self) -> str | None:
        """The key browsers subscribe with: VAPID_PUBLIC_KEY in the environment if set, else
        what the reminders cron published on the Volume. None: reminders are not set up."""
        env = check_public_key(os.environ.get("VAPID_PUBLIC_KEY"))
        if env:
            return env
        try:
            return check_public_key((self.root / PUSH_KEY_FILE).read_text(encoding="utf-8"))
        except OSError:
            return None

    def publish_push_key(self, key: str) -> bool:
        """Write the public key for the web app (the cron, every round); True when it changed."""
        key = check_public_key(key) or ""
        if not key:
            raise ValueError("VAPID_PUBLIC_KEY is not a P-256 public key in base64url")
        path = self.root / PUSH_KEY_FILE
        try:
            if path.read_text(encoding="utf-8").strip() == key:
                return False
        except OSError:
            pass
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}")
        tmp.write_text(key, encoding="utf-8")
        os.replace(tmp, path)
        return True


# ---------------------------------------------------------------------------------------------
# Rate limits (per client address, as the avatar and scan uploads do)
# ---------------------------------------------------------------------------------------------

def client_key(host: str | None) -> str:
    """Who is asking, for the per-address limits: the IP address, an IPv6 one by its /64 (as
    `avatars.client_key`; one home or phone network holds a whole /64)."""
    host = (host or "").strip()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host[:64]
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return str(ipaddress.IPv6Network((address, 64), strict=False))
    return str(address)


class _Window:
    """Times of events per key within the last `seconds` (a sliding window)."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.times: dict[str, deque[float]] = {}

    def count(self, key: str, t: float) -> int:
        recent = self.times.get(key)
        while recent and t - recent[0] >= self.seconds:
            recent.popleft()
        return len(recent) if recent else 0

    def wait(self, key: str, t: float) -> int:
        recent = self.times.get(key)
        return max(1, math.ceil(self.seconds - (t - recent[0]))) if recent else 1

    def add(self, key: str, t: float) -> None:
        self.times.setdefault(key, deque()).append(t)

    def sweep(self, t: float) -> None:
        for key, recent in list(self.times.items()):
            if not recent or t - recent[-1] >= self.seconds:
                del self.times[key]


class MeetingService:
    """What the server's routes use: the store and the per-address limits. `clock` is monotonic
    seconds for the windows and `now` the wall clock for meeting times (tests replace both)."""

    def __init__(self, store: MeetingStore, *, clock: Callable[[], float] = time.monotonic,
                 now: Callable[[], datetime] = _utcnow) -> None:
        self.store = store
        self.clock = clock
        self.now = now
        self._create_short = _Window(600)
        self._create_day = _Window(86400)
        self._subscribe = _Window(3600)

    def _limit(self, window: _Window, who: str, cap: int, message: str) -> None:
        t = self.clock()
        window.sweep(t)
        if window.count(who, t) >= cap:
            raise MeetingError(429, message, window.wait(who, t))

    def create(self, body: Any, *, client: str | None,
               picks: dict[str, str]) -> tuple[Meeting, str]:
        """The meeting and its cancel token (only its hash is stored; whoever made the meeting
        keeps the token). Blocking (the store writes): call it in a thread."""
        who = client_key(client)
        self._limit(self._create_short, who, CREATE_PER_10_MIN,
                    "you have scheduled a lot of meetings just now; wait a few minutes")
        self._limit(self._create_day, who, CREATE_PER_DAY,
                    "your network has scheduled all the meetings it can today")
        now = self.now()
        token = secrets.token_urlsafe(24)
        meeting = new_meeting(body, now=now, picks=picks, by=client_hash(who),
                              owner=token_hash(token))
        t = self.clock()
        self._create_short.add(who, t)
        self._create_day.add(who, t)
        return self.store.add(meeting, now=now), token

    def cancel(self, mid: str, body: Any, *, client: str | None) -> bool:
        """{"token"}: the meeting removed, with its reminders."""
        who = client_key(client)
        self._limit(self._subscribe, who, SUBSCRIBE_PER_HOUR,
                    "too many changes from your network; wait a while")
        self._subscribe.add(who, self.clock())
        token = body.get("token") if isinstance(body, dict) else None
        if not self.store.cancel(mid, token):
            raise MeetingError(404, "no such meeting")
        return True

    def get(self, mid: str) -> Meeting | None:
        return self.store.get(mid)

    def subscribe(self, mid: str, body: Any, *, client: str | None) -> str:
        who = client_key(client)
        self._limit(self._subscribe, who, SUBSCRIBE_PER_HOUR,
                    "too many reminder changes from your network; wait a while")
        meeting = self.store.get(mid)
        if meeting is None:
            raise MeetingError(404, "no such meeting")
        if meeting.start <= self.now():
            raise MeetingError(409, "this meeting has started: no reminder to send")
        subscription = check_subscription(body)
        self._subscribe.add(who, self.clock())
        return self.store.subscribe(mid, subscription, by=client_hash(who))

    def unsubscribe(self, mid: str, body: Any, *, client: str | None) -> bool:
        who = client_key(client)
        self._limit(self._subscribe, who, SUBSCRIBE_PER_HOUR,
                    "too many reminder changes from your network; wait a while")
        if self.store.get(mid) is None:
            raise MeetingError(404, "no such meeting")
        endpoint = body.get("endpoint") if isinstance(body, dict) else None
        if not isinstance(endpoint, str) or not 0 < len(endpoint) <= MAX_ENDPOINT:
            raise MeetingError(400, "bad push endpoint")
        self._subscribe.add(who, self.clock())
        return self.store.unsubscribe(mid, endpoint)

    def push_key(self) -> str | None:
        return self.store.push_key()


# ---------------------------------------------------------------------------------------------
# The calendar file
# ---------------------------------------------------------------------------------------------

def _ics_text(text: str) -> str:
    return (text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,")
            .replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n"))


def _fold(line: str) -> str:
    """RFC 5545 line folding: at most 75 octets a line, continuations start with a space."""
    out, current = [], b""
    for ch in line:
        encoded = ch.encode("utf-8")
        if len(current) + len(encoded) > (75 if not out else 74):
            out.append(current.decode("utf-8"))
            current = b""
        current += encoded
    out.append(current.decode("utf-8"))
    return "\r\n ".join(out)


def _ics_time(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ics(meeting: Meeting, link: str, *, now: datetime | None = None) -> str:
    """The meeting as an iCalendar file: one VEVENT in UTC, the link to join in URL, LOCATION and
    DESCRIPTION, and a display alarm 15 minutes before (the reminder for a calendar)."""
    stamp = meeting.created or now or _utcnow()
    where = f" ({meeting.pick_label})" if meeting.pick_label else ""
    lines = [
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//LiteReality//Room call//EN",
        "CALSCALE:GREGORIAN", "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{meeting.id}@roomcall",
        f"DTSTAMP:{_ics_time(stamp)}",
        f"DTSTART:{_ics_time(meeting.start)}",
        f"DTEND:{_ics_time(meeting.end)}",
        f"SUMMARY:{_ics_text(meeting.title)}",
        f"URL:{link}",
        f"LOCATION:{_ics_text(link)}",
        f"DESCRIPTION:{_ics_text(f'Room call{where}. Join: {link}')}",
        "BEGIN:VALARM", "ACTION:DISPLAY", f"DESCRIPTION:{_ics_text(meeting.title)}",
        f"TRIGGER:-PT{int(REMIND_BEFORE.total_seconds() // 60)}M", "END:VALARM",
        "END:VEVENT", "END:VCALENDAR",
    ]
    return "\r\n".join(_fold(line) for line in lines) + "\r\n"


# ---------------------------------------------------------------------------------------------
# The reminders cron
# ---------------------------------------------------------------------------------------------

def due(meeting: Meeting, now: datetime) -> bool:
    """Time to remind: the start is `REMIND_EARLY` away or less, and not yet here."""
    return meeting.start - REMIND_EARLY <= now < meeting.start


def expired(meeting: Meeting, now: datetime) -> bool:
    return now >= meeting.end + PRUNE_AFTER


def plan(states: Iterable[MeetingState], now: datetime) -> tuple[list[tuple[Meeting, list[str]]],
                                                                  list[str]]:
    """Pure: which subscriptions of which meetings to remind now, and which meetings to remove."""
    send, prune = [], []
    for state in states:
        if expired(state.meeting, now):
            prune.append(state.meeting.id)
        elif due(state.meeting, now):
            todo = [sid for sid in sorted(state.subs) if sid not in state.sent]
            if todo:
                send.append((state.meeting, todo))
    return send, prune


def reminder(meeting: Meeting, now: datetime) -> dict:
    """The push's payload (the service worker shows it, `web/sw.js`): the title says how soon
    (rounded up, so the round a few seconds after start - 15 min says 15); the body is the start
    in UTC, which the worker replaces with the device's own local time; `url` is relative to the
    worker's scope, so the cron never needs the app's address."""
    minutes = max(1, math.ceil((meeting.start - now).total_seconds() / 60))
    room = f" · {meeting.pick_label}" if meeting.pick_label else ""
    return {"title": f"Starting in {minutes} min: {meeting.title}",
            "body": f"{meeting.start:%H:%M} UTC{room} · tap to join",
            "start": iso(meeting.start), "minutes": minutes, "room": meeting.room,
            "url": meeting.join_query(), "tag": f"meeting-{meeting.id}"}


# What a sender says about one push: delivered, the subscription is gone (unsubscribed, expired:
# 404/410), or a failure worth trying again next round.
SENT, GONE, RETRY = "sent", "gone", "retry"
Sender = Callable[[dict, dict, int], str]


@dataclass
class Report:
    sent: int = 0
    gone: int = 0
    failed: int = 0
    struck: int = 0
    deferred: int = 0
    pruned: list[str] = field(default_factory=list)
    key_published: bool = False

    def __str__(self) -> str:
        return (f"reminders: {self.sent} sent, {self.gone} gone, {self.failed} to retry "
                f"({self.struck} dropped after {MAX_STRIKES} failures), {self.deferred} left for "
                f"the next round, {len(self.pruned)} meetings pruned"
                + (", public key published" if self.key_published else ""))


def interleave(to_send: list[tuple[Meeting, list[str]]]) -> list[tuple[Meeting, str]]:
    """Pure: the pushes of a round in a fair order: one per meeting in turn, the soonest meeting
    first, so a meeting with many subscriptions (or many that fail) never starves the others."""
    queues = [(m, list(sids)) for m, sids in sorted(to_send, key=lambda item: (item[0].start,
                                                                              item[0].id))]
    order: list[tuple[Meeting, str]] = []
    while queues:
        for m, sids in queues:
            order.append((m, sids.pop(0)))
        queues = [(m, sids) for m, sids in queues if sids]
    return order


def _send_one(send: Sender, sub: dict, payload: dict, ttl: int, mid: str) -> str:
    try:
        return send(sub, payload, ttl)
    except Exception as exc:  # noqa: BLE001 — one bad push never stops the round
        log.warning("reminders: sending for %s failed: %s", mid, exc)
        return RETRY


def remind_round(store: MeetingStore, send: Sender, *, now: datetime | None = None,
                 max_sends: int | None = None, deadline: float | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 commit: Callable[[], Any] | None = None, workers: int | None = None) -> Report:
    """One cron round: remind what is due (`plan`), in a fair order (`interleave`), `workers`
    pushes at a time; after each batch the meetings' reminded subscriptions are marked and
    `commit` (the Volume's) is called, so a round cut short never sends one twice. No new batch
    starts once `deadline` seconds (`ROUND_SECONDS`) have passed: the next round carries on.
    Subscriptions that are gone (404/410) are dropped; ones that failed in `MAX_STRIKES` rounds
    are dropped too, counting only rounds in which some push got an answer (a round where
    nothing did is our trouble or the network's, not the subscriptions'). Meetings long over are
    removed. `send(subscription, payload, ttl_seconds)` -> SENT, GONE or RETRY; an exception
    counts as RETRY."""
    now = now or _utcnow()
    started = clock()
    limit = ROUND_SECONDS if deadline is None else deadline
    report = Report()
    to_send, to_prune = plan(store.states(), now)
    budget = MAX_SENDS_PER_ROUND if max_sends is None else max_sends
    order = interleave(to_send)
    if len(order) > budget:
        report.deferred += len(order) - budget
        order = order[:budget]
    # Per meeting: its subscriptions, who was reminded, and failures so far (from sent.json).
    subs: dict[str, dict[str, dict]] = {}
    sent: dict[str, set[str]] = {}
    fails: dict[str, dict[str, int]] = {}
    payloads: dict[str, tuple[dict, int]] = {}
    for meeting, _ in to_send:
        subs[meeting.id] = store.subs(meeting.id)
        sent[meeting.id] = store.sent(meeting.id)
        fails[meeting.id] = store.strikes(meeting.id)
        payloads[meeting.id] = (reminder(meeting, now),
                                max(60, int((meeting.start - now).total_seconds())))
    retried: dict[str, list[str]] = {}
    size = max(1, SEND_WORKERS if workers is None else workers)
    with ThreadPoolExecutor(max_workers=size) as pool:
        for at in range(0, len(order), size):
            if clock() - started >= limit:
                report.deferred += len(order) - at
                break
            batch = [(m, sid, subs[m.id].get(sid)) for m, sid in order[at:at + size]]
            batch = [(m, sid, sub) for m, sid, sub in batch if sub is not None]
            futures = [pool.submit(_send_one, send, sub, *payloads[m.id], m.id)
                       for m, sid, sub in batch]
            touched = set()
            for (m, sid, _), future in zip(batch, futures, strict=True):
                result = future.result()
                if result == SENT:
                    sent[m.id].add(sid)
                    fails[m.id].pop(sid, None)
                    report.sent += 1
                    touched.add(m.id)
                elif result == GONE:
                    store.drop_sub(m.id, sid)
                    fails[m.id].pop(sid, None)
                    report.gone += 1
                else:
                    retried.setdefault(m.id, []).append(sid)
                    report.failed += 1
            for mid in touched:
                store.mark_sent(mid, sent[mid], now, fails[mid])
            if touched and commit is not None:
                _commit_quietly(commit)
    # Strikes: only when this round reached the push services at all.
    if retried and report.sent + report.gone > 0:
        for mid, sids in retried.items():
            for sid in sids:
                fails[mid][sid] = fails[mid].get(sid, 0) + 1
                if fails[mid][sid] >= MAX_STRIKES:
                    store.drop_sub(mid, sid)
                    del fails[mid][sid]
                    report.struck += 1
            store.mark_sent(mid, sent[mid], now, fails[mid])
    for mid in to_prune:
        store.remove(mid)
        report.pruned.append(mid)
    return report


def _commit_quietly(commit: Callable[[], Any]) -> None:
    try:
        commit()
    except Exception as exc:  # noqa: BLE001 — the round's last commit carries it
        log.warning("reminders: commit failed: %s", exc)


def webpush_sender(private_key: str, subject: str, *, timeout: float = PUSH_TIMEOUT) -> Sender:
    """A `Sender` over pywebpush with the VAPID key (base64url raw private key, as
    `scripts/roomcall/vapid_secret.py` makes it). Imported here, in the reminders image only."""
    from pywebpush import WebPushException, webpush

    def send(subscription: dict, payload: dict, ttl: int) -> str:
        try:
            webpush(subscription, data=json.dumps(payload, separators=(",", ":")),
                    vapid_private_key=private_key, vapid_claims={"sub": subject},
                    timeout=timeout, ttl=ttl, headers={"Urgency": "high"})
        except WebPushException as exc:
            status = getattr(exc.response, "status_code", None)
            if status in (404, 410):
                return GONE
            log.warning("reminders: push refused (%s): %s", status, str(exc)[:200])
            return RETRY
        except Exception as exc:  # noqa: BLE001 — network trouble: the next round tries again
            log.warning("reminders: push failed: %s", str(exc)[:200])
            return RETRY
        return SENT

    return send


def cron_round(root: Path, *, env: dict | None = None, now: datetime | None = None,
               commit: Callable[[], Any] | None = None) -> Report:
    """What the reminders function runs each minute against the mounted Volume (`root`): the
    public key published for the web app, then `remind_round` with pywebpush, committing the
    Volume (`commit`) after each batch of pushes."""
    env = os.environ if env is None else env
    missing = [k for k in VAPID_KEYS if not env.get(k)]
    if missing:
        raise RuntimeError(f"the {VAPID_SECRET} secret lacks {', '.join(missing)}")
    store = MeetingStore(Path(root) / MEETINGS_DIR)
    published = store.publish_push_key(env["VAPID_PUBLIC_KEY"])
    sender = webpush_sender(env["VAPID_PRIVATE_KEY"], env["VAPID_SUBJECT"])
    report = remind_round(store, sender, now=now, commit=commit)
    report.key_published = published
    return report
