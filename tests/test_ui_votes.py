"""Vote events: schemas per mode, the fsynced append-only log, the derived queue, resume, abandon, retract and
micro-sessions (src/lofi_rl/ui/votes.py). Votes go to temp dirs only, never data/votes."""
import errno
import json
import os
import threading
from datetime import datetime, timedelta, timezone

import pytest

import lofi_rl.ui.votes as votes_mod
from lofi_rl.ui.manifest import MODES, build_demo, load_batch
from lofi_rl.ui.votes import (Conflict, EventError, Ledger, LogBusy, active_votes, check_event, read_events,
                              read_state)

FAST = dict(bpm=240.0, sr=8000)


@pytest.fixture(scope="module")
def batches(tmp_path_factory):
    root = tmp_path_factory.mktemp("inbox") / "2026-11-03"
    return {m: load_batch(build_demo(root / f"demo-{m}", m, **FAST)) for m in MODES}


class Clock:
    def __init__(self):
        self.t = datetime(2026, 11, 10, 21, 0, 0, tzinfo=timezone(timedelta(hours=-8)))

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


def listening(r, bars="11111111"):
    return {"clips": [{"clip": c.id, "pos": c.pos, "heard_bars": bars, "heard_ms": 8000} for c in r.clips],
            "last_heard": r.clips[-1].id, "switches": 1, "play_order": list(range(1, len(r.clips) + 1)),
            "autoplay_done": True, "t_ms": {"commit": 30000}, "device": "airpods", "audio": {"sr": 8000}}


def vote(m, r, **picks):
    """A valid vote for round r of manifest m, with the picks overridable."""
    ids = r.ids
    base = {"bws4": dict(best=ids[0], worst=ids[-1]), "shootout": dict(best=ids[0], worst=ids[-1]),
            "keep3": dict(best=ids[0]), "triage": dict(tier="fine"), "ab": dict(pick=ids[0]),
            "playlist": dict(mark="heart", plays=2)}[m.mode]
    base.update(picks)
    return {"type": m.mode, "round": r.id, **listening(r), **base}


@pytest.mark.parametrize("mode", sorted(MODES))
def test_a_valid_vote_passes_for_every_mode(batches, mode):
    m, _ = batches[mode]
    out = check_event(vote(m, m.rounds[0]), m)
    assert out["type"] == mode and out["round"] == m.rounds[0].id and len(out["clips"]) == len(m.rounds[0].clips)


@pytest.mark.parametrize("mode, change, expect", [
    ("bws4", lambda v, ids: v.update(worst=ids[0]), "best and worst must be different"),
    ("bws4", lambda v, ids: v.update(worst=None), "incomplete"),
    ("bws4", lambda v, ids: v.update(worst=None, no_worst=True, tag="sour"), "goes with the worst pick"),
    ("bws4", lambda v, ids: v.update(best=None, no_best=True, tag="sax_line"), "goes with the best pick"),
    ("bws4", lambda v, ids: v.update(tag="loud"), "tag: one of"),
    ("bws4", lambda v, ids: v.update(best=ids[0], all_bad=True), "'all bad' and a best pick"),
    ("bws4", lambda v, ids: v.update(no_best=1), "no_best: must be true or false"),
    ("bws4", lambda v, ids: v.update(best="k0000000000"), "best: not a clip of this round"),
    ("bws4", lambda v, ids: v.update(score=0.9), "unknown field(s) ['score']"),
    ("bws4", lambda v, ids: v["clips"][0].update(heard_bars="1111"), "must be 8 characters of 0 and 1"),
    ("bws4", lambda v, ids: v["clips"][0].update(heard_bars="1111x111"), "must be 8 characters of 0 and 1"),
    ("bws4", lambda v, ids: v["clips"].reverse(), "clips[0].clip: expected"),
    ("bws4", lambda v, ids: v.update(switches=True), "switches: must be a whole number"),
    ("bws4", lambda v, ids: v.update(play_order=[1, 1]), "play_order"),
    ("bws4", lambda v, ids: v.update(t_ms={"commit": -5}), "t_ms.commit"),
    ("bws4", lambda v, ids: v.update(device="My AirPods!"), "device: must match"),
    ("bws4", lambda v, ids: v.update(type="triage"), "this is a bws4 batch, so it takes bws4 votes"),
    ("bws4", lambda v, ids: v.update(round="r00000000"), "round: not a round of this batch"),
    ("bws4", lambda v, ids: v.update(event_id="x1"), "event_id: must match"),
    ("shootout", lambda v, ids: v.update(tag="sour"), "unknown field(s) ['tag']"),
    ("keep3", lambda v, ids: v.update(all_bad=True), "keep one loop, or say 'none acceptable' (not both)"),
    ("keep3", lambda v, ids: v.update(best=None), "keep one loop"),
    ("triage", lambda v, ids: v.update(tier="meh"), "tier: one of"),
    ("ab", lambda v, ids: v.update(pick=None), "pick: the clip picked (forced choice)"),
    ("ab", lambda v, ids: v.update(worst=ids[0]), "unknown field(s) ['worst']"),
    ("playlist", lambda v, ids: v.update(mark="love"), "mark: one of"),
])
def test_schema_errors_are_reported(batches, mode, change, expect):
    m, _ = batches[mode]
    r = m.rounds[0]
    v = vote(m, r)
    change(v, r.ids)
    with pytest.raises(EventError) as e:
        check_event(v, m)
    assert expect in str(e.value)


def test_other_event_types_are_checked(batches):
    m, _ = batches["bws4"]
    r = m.rounds[0]
    assert check_event({"type": "session", "what": "start", "device": "laptop"}, m)["what"] == "start"
    assert check_event({"type": "abandon", "round": r.id, "why": "esc"}, m)["why"] == "esc"
    assert check_event({"type": "retract", "target": "e" + "ab" * 12, "why": "undo"}, m)["target"]
    for bad, msg in [({"type": "session", "what": "dance"}, "what: one of"),
                     ({"type": "session", "what": "round_start"}, "round_start needs the round"),
                     ({"type": "abandon", "round": r.id, "why": "bored"}, "why: one of"),
                     ({"type": "retract", "target": "nope"}, "target: the event_id"),
                     ({"type": "approve"}, "type: one of"),
                     (["not", "an", "object"], "must be a JSON object")]:
        with pytest.raises(EventError) as e:
            check_event(bad, m)
        assert msg in str(e.value)


def _ledger(batches, mode, votes_dir, clock=None):
    m, h = batches[mode]
    return Ledger(votes_dir, m, h, micro_gap_s=600, ui="testui", clock=clock or Clock())


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def test_events_are_appended_and_fsynced(batches, tmp_path, monkeypatch):
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
    led = _ledger(batches, "bws4", tmp_path)
    try:
        m, h = batches["bws4"]
        log_fd = led._log.fd
        for i, ev in enumerate([{"type": "session", "what": "start"},
                                {"type": "session", "what": "round_start", "round": m.rounds[0].id},
                                vote(m, m.rounds[0], tag="sour")]):
            before = synced.count(log_fd)
            reply = led.submit(ev)
            assert synced.count(log_fd) == before + 1                    # one fsync per event, before the reply
            lines = _lines(tmp_path / "demo-bws4.jsonl")                 # re-read from disk after each write
            assert len(lines) == i + 1 and lines[-1]["event_id"] == reply["event_id"]
        rec = lines[-1]
        hr = h.rounds[m.rounds[0].id]
        assert rec["v"] == 1 and rec["type"] == "bws4" and rec["batch"] == "demo-bws4" and rec["ui"] == "testui"
        assert rec["ts"] == "2026-11-10T21:00:00-08:00" and rec["session"] == "s1110a" and rec["micro"] == 1
        assert rec["best"] == hr.real(m.rounds[0].ids[0]) and rec["worst"] == hr.real(m.rounds[0].ids[-1])
        assert rec["tag"] == {"name": "sour", "clip": rec["worst"]}
        assert [c["oid"] for c in rec["clips"]] == list(m.rounds[0].ids)
        assert [c["clip"] for c in rec["clips"]] == [hr.real(o) for o in m.rounds[0].ids]
        assert rec["hidden"] == {"kind": "none", "of": None} and rec["index"] == 1 and rec["brief_id"]
        assert oct(os.stat(tmp_path / "demo-bws4.jsonl").st_mode & 0o777) == "0o600"
        st = read_state(tmp_path / "state.json")
        assert st["batches"]["demo-bws4"]["current"] == m.rounds[1].id and st["batches"]["demo-bws4"]["settled"] == 1
    finally:
        led.close()


def test_hidden_labels_reach_the_record(batches, tmp_path):
    m, h = batches["shootout"]
    led = _ledger(batches, "shootout", tmp_path)
    try:
        anchor_round = next(r for r in m.rounds if h.rounds[r.id].kind == "anchor")
        broken = h.rounds[anchor_round.id].anchor_clip
        other = next(c for c in anchor_round.ids if c != broken)
        led.submit(vote(m, anchor_round, best=other, worst=broken))
        rec = _lines(tmp_path / "demo-shootout.jsonl")[-1]
        assert rec["hidden"] == {"kind": "anchor", "of": None, "anchor": h.rounds[anchor_round.id].real(broken)}
        assert rec["category"] == "keys" and len(rec["options"]) == 4 and all(rec["options"])
    finally:
        led.close()


def test_state_resumes_from_the_log(batches, tmp_path):
    m, _ = batches["bws4"]
    led = _ledger(batches, "bws4", tmp_path)
    led.submit(vote(m, m.rounds[0]))
    led.submit(vote(m, m.rounds[1]))
    led.close()
    (tmp_path / "state.json").write_text("{not json")             # a damaged cursor file: the log still wins
    led2 = _ledger(batches, "bws4", tmp_path)
    try:
        st = led2.state()
        assert st["current"] == m.rounds[2].id and st["settled"] == 2 and st["index"] == 3
        assert read_state(tmp_path / "state.json")["batches"]["demo-bws4"]["current"] == m.rounds[2].id
        with pytest.raises(Conflict):
            led2.submit(vote(m, m.rounds[0]))                    # a round with a vote takes no second one
    finally:
        led2.close()


def test_a_second_server_cannot_open_the_same_log(batches, tmp_path):
    led = _ledger(batches, "bws4", tmp_path)
    try:
        with pytest.raises(LogBusy):
            _ledger(batches, "bws4", tmp_path)
    finally:
        led.close()


def test_abandoned_round_goes_to_the_end(batches, tmp_path):
    m, _ = batches["bws4"]
    ids = [r.id for r in m.rounds]
    led = _ledger(batches, "bws4", tmp_path)
    try:
        led.submit({"type": "abandon", "round": ids[0], "why": "esc"})
        assert led.state()["current"] == ids[1] and led.state()["upcoming"] == [ids[2], ids[3]]
        for rid in ids[1:]:
            led.submit(vote(m, m.round(rid)))
        assert led.state()["current"] == ids[0] and not led.state()["done"]
        led.submit(vote(m, m.rounds[0]))
        assert led.state()["done"] and led.state()["current"] is None
    finally:
        led.close()


def test_an_open_round_is_abandoned_on_the_next_start(batches, tmp_path):
    m, _ = batches["bws4"]
    led = _ledger(batches, "bws4", tmp_path)
    led.submit({"type": "session", "what": "start"})
    led.submit({"type": "session", "what": "round_start", "round": m.rounds[0].id})
    led.close()                                                    # the tab closed before any vote or abandon
    led2 = _ledger(batches, "bws4", tmp_path)
    try:
        led2.submit({"type": "session", "what": "start"})
        lines = _lines(tmp_path / "demo-bws4.jsonl")
        assert [(e["type"], e.get("why")) for e in lines[-2:]] == [("abandon", "no_commit"), ("session", None)]
        assert led2.state()["current"] == m.rounds[1].id
    finally:
        led2.close()


def test_retract_is_append_only_and_reopens_the_round(batches, tmp_path):
    m, _ = batches["bws4"]
    led = _ledger(batches, "bws4", tmp_path)
    try:
        first = led.submit(vote(m, m.rounds[0]))
        assert led.state()["last_vote"] == {"event_id": first["event_id"], "round": m.rounds[0].id}
        led.submit(vote(m, m.rounds[1]))
        back = led.submit({"type": "retract", "target": first["event_id"], "why": "undo"})
        st = back["state"]
        assert st["current"] == m.rounds[0].id and st["settled"] == 1
        lines = _lines(tmp_path / "demo-bws4.jsonl")
        assert len(lines) == 3 and lines[0]["event_id"] == first["event_id"]          # nothing rewritten
        assert lines[-1]["type"] == "retract" and lines[-1]["target"] == first["event_id"]
        assert lines[-1]["round"] == m.rounds[0].id
        with pytest.raises(Conflict):
            led.submit({"type": "retract", "target": first["event_id"]})                # already taken back
        with pytest.raises(Conflict):
            led.submit({"type": "retract", "target": "e" + "0" * 24})                    # no such vote
        led.submit(vote(m, m.rounds[0], best=m.rounds[0].ids[1]))
        act = active_votes(read_events(tmp_path / "demo-bws4.jsonl")[0])
        assert set(act) == {m.rounds[0].id, m.rounds[1].id}
        assert act[m.rounds[0].id]["best"] == batches["bws4"][1].rounds[m.rounds[0].id].real(m.rounds[0].ids[1])
    finally:
        led.close()


def test_duplicate_event_ids_are_written_once(batches, tmp_path):
    m, _ = batches["triage"]
    led = _ledger(batches, "triage", tmp_path)
    try:
        v = vote(m, m.rounds[0], event_id="e" + "1" * 24)
        assert led.submit(v)["event_id"] == "e" + "1" * 24
        again = led.submit(v)                                      # a retry after a lost reply
        assert again["duplicate"] and len(_lines(tmp_path / "demo-triage.jsonl")) == 1
        with pytest.raises(Conflict):
            led.submit({"type": "abandon", "round": m.rounds[1].id, "event_id": "e" + "1" * 24})
    finally:
        led.close()


def test_sessions_and_micro_sessions(batches, tmp_path):
    clock = Clock()
    m, _ = batches["triage"]
    led = _ledger(batches, "triage", tmp_path, clock)
    seen = []
    try:
        for step in [dict(), dict(minutes=9), dict(minutes=10), dict(minutes=3)]:
            clock.advance(**step)
            led.submit({"type": "session", "what": "pause"})
            seen.append(_lines(tmp_path / "demo-triage.jsonl")[-1])
    finally:
        led.close()
    assert [(e["session"], e["micro"]) for e in seen] == [("s1110a", 1), ("s1110a", 1), ("s1110a", 2), ("s1110a", 2)]
    clock.advance(minutes=5)                                       # restart within 10 min: same micro-session
    led = _ledger(batches, "triage", tmp_path, clock)
    led.submit({"type": "session", "what": "start"})
    led.close()
    clock.advance(minutes=45)                                      # restart after a long gap: a new session
    led = _ledger(batches, "triage", tmp_path, clock)
    led.submit({"type": "session", "what": "start"})
    led.close()
    clock.advance(days=1)
    led = _ledger(batches, "triage", tmp_path, clock)
    led.submit({"type": "session", "what": "start"})
    led.close()
    tail = [(e["session"], e["micro"]) for e in _lines(tmp_path / "demo-triage.jsonl")[-3:]]
    assert tail == [("s1110a", 2), ("s1110b", 1), ("s1111a", 1)]


def test_a_torn_last_line_is_kept_apart(batches, tmp_path):
    m, _ = batches["bws4"]
    led = _ledger(batches, "bws4", tmp_path)
    led.submit(vote(m, m.rounds[0]))
    led.close()
    with open(tmp_path / "demo-bws4.jsonl", "ab") as f:
        f.write(b'{"v":1,"type":"bws4","event_id":"e12')        # a crash mid-write
    led = _ledger(batches, "bws4", tmp_path)
    try:
        assert led.problems == ["line 2: unreadable, skipped"]
        led.submit(vote(m, m.rounds[1]))
        events, problems = read_events(tmp_path / "demo-bws4.jsonl")
        assert len(events) == 2 and problems == ["line 2: unreadable, skipped"]
        assert led.state()["settled"] == 2
    finally:
        led.close()


# ----------------------------------------------------------------------------- failed writes (integrity-1, spec-2)

class FlakyDisk:
    """os.write for one file that, once armed, fails part-way like a disk that fills up: write(2) returns a short
    count, then the next call fails with ENOSPC. "short": the next buffer is half written. "room": the disk has
    room for the first line it is given plus 10 bytes, however the writes are split."""

    def __init__(self, monkeypatch, fd):
        self.fd, self.armed, self.room, self.real = fd, None, None, os.write
        monkeypatch.setattr(os, "write", self)

    def _fail(self):
        self.armed, self.room = None, None
        raise OSError(errno.ENOSPC, "No space left on device")

    def __call__(self, fd, data):
        if fd != self.fd or not self.armed:
            return self.real(fd, data)
        data = bytes(data)
        if self.armed == "short":
            self.armed = "fail"
            return self.real(fd, data[: max(1, len(data) // 2)])
        if self.armed == "fail":
            self._fail()
        if self.room is None:
            self.room = data.index(b"\n") + 1 + 10
        if len(data) <= self.room:
            self.room -= len(data)
            return self.real(fd, data)
        if not self.room:
            self._fail()
        n, self.room = self.room, 0
        return self.real(fd, data[:n])


def _eid(ch):
    return "e" + ch * 24


def test_a_write_that_fails_part_way_is_cut_back_and_the_retry_survives_a_restart(batches, tmp_path, monkeypatch):
    """The page shows 'Not saved' and resends the same event_id. That retry is answered 200, so it must still be
    in the log after a restart (taste_pipeline §1 Resume: each event is fsynced)."""
    m, _ = batches["bws4"]
    led = _ledger(batches, "bws4", tmp_path)
    disk = FlakyDisk(monkeypatch, led._log.fd)
    led.submit(vote(m, m.rounds[0], event_id=_eid("a")))
    size = led.path.stat().st_size
    disk.armed = "short"
    with pytest.raises(OSError):
        led.submit(vote(m, m.rounds[1], event_id=_eid("b")))
    assert led.path.stat().st_size == size                          # the half line was cut off again
    assert led.state()["settled"] == 1
    again = led.submit(vote(m, m.rounds[1], event_id=_eid("b")))     # Enter: the page's retry
    assert again["ok"] and not again.get("duplicate") and again["state"]["settled"] == 2
    led.submit(vote(m, m.rounds[2], event_id=_eid("c")))
    led.close()
    events, problems = read_events(led.path)
    assert problems == [] and [e["event_id"] for e in events] == [_eid("a"), _eid("b"), _eid("c")]
    led = _ledger(batches, "bws4", tmp_path)                         # make rate again later
    try:
        assert led.problems == [] and led.state()["settled"] == 3 and led.state()["current"] == m.rounds[3].id
    finally:
        led.close()


def test_when_the_cut_back_fails_too_the_retry_starts_on_a_line_of_its_own(batches, tmp_path, monkeypatch):
    m, _ = batches["bws4"]
    led = _ledger(batches, "bws4", tmp_path)
    disk = FlakyDisk(monkeypatch, led._log.fd)

    def no_truncate(fd, n):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(os, "ftruncate", no_truncate)
    led.submit(vote(m, m.rounds[0], event_id=_eid("a")))
    disk.armed = "short"
    with pytest.raises(OSError):
        led.submit(vote(m, m.rounds[1], event_id=_eid("b")))
    assert led.submit(vote(m, m.rounds[1], event_id=_eid("b")))["state"]["settled"] == 2
    led.close()
    events, problems = read_events(led.path)
    assert [e["event_id"] for e in events] == [_eid("a"), _eid("b")] and problems == ["line 2: unreadable, skipped"]
    led = _ledger(batches, "bws4", tmp_path)
    try:
        assert led.state()["settled"] == 2 and led.state()["current"] == m.rounds[2].id
    finally:
        led.close()


def test_a_session_start_and_its_abandon_are_written_together_or_not_at_all(batches, tmp_path, monkeypatch):
    """A start while a round is open also logs abandon(no_commit). If the write fails, neither is kept: the queue
    and the session clock stay as they were, and the retry writes both."""
    m, _ = batches["bws4"]
    clock = Clock()
    led = _ledger(batches, "bws4", tmp_path, clock)
    try:
        led.submit({"type": "session", "what": "round_start", "round": m.rounds[0].id})
        before = led.state()
        disk = FlakyDisk(monkeypatch, led._log.fd)
        disk.armed = "room"                          # room for the abandon line, not for the start after it
        clock.advance(minutes=30)
        with pytest.raises(OSError):
            led.submit({"type": "session", "what": "start", "event_id": _eid("d")})
        assert len(_lines(led.path)) == 1 and led.state() == before
        assert read_state(tmp_path / "state.json")["session"]["last"] == "2026-11-10T21:00:00-08:00"
        led.submit({"type": "session", "what": "start", "event_id": _eid("d")})
        tail = _lines(led.path)[-2:]
        assert [(e["type"], e.get("why") or e.get("what")) for e in tail] == [("abandon", "no_commit"),
                                                                             ("session", "start")]
        assert led.state()["current"] == m.rounds[1].id
    finally:
        led.close()


# ----------------------------------------------------------------------------- text (integrity-4)

def test_text_that_is_not_valid_unicode_is_refused_before_anything_is_written(batches, tmp_path):
    """json.loads turns "\\ud800" into a lone surrogate, which UTF-8 cannot encode: refuse it (400), never 500."""
    m, _ = batches["bws4"]
    for ev in ({"type": "session", "what": "pause", "detail": "\ud800"},
               {"type": "retract", "target": _eid("a"), "why": "x\udfff"}):
        with pytest.raises(EventError, match="plain text"):
            check_event(ev, m)
    led = _ledger(batches, "bws4", tmp_path)
    try:
        led.submit({"type": "session", "what": "round_start", "round": m.rounds[0].id})
        before = led.state()
        with pytest.raises(EventError, match="plain text"):
            led.submit({"type": "session", "what": "start", "detail": "\ud800"})
        assert len(_lines(led.path)) == 1 and led.state() == before      # no abandon logged, the queue unchanged
    finally:
        led.close()


# ----------------------------------------------------------------------------- two servers, one votes folder (integrity-2)

def test_two_servers_on_one_votes_folder_never_reuse_a_session_id(batches, tmp_path):
    """Server A (bws4) runs from 21:00; server B (triage) starts at 21:30 after a long gap and begins a new
    session; A posts at 21:31. A server started at 22:00 must begin yet another session, never B's id again."""
    clock = Clock()
    a = _ledger(batches, "bws4", tmp_path, clock)
    b = None
    try:
        a.submit({"type": "session", "what": "start"})
        clock.advance(minutes=5)
        a.submit({"type": "session", "what": "pause"})
        clock.advance(minutes=25)
        b = _ledger(batches, "triage", tmp_path, clock)
        b.submit({"type": "session", "what": "start"})
        clock.advance(minutes=1)
        a.submit({"type": "session", "what": "resume"})
    finally:
        a.close()
        if b:
            b.close()
    clock.advance(minutes=29)
    c = _ledger(batches, "shootout", tmp_path, clock)
    try:
        c.submit({"type": "session", "what": "start"})
    finally:
        c.close()
    a_log, b_log = _lines(tmp_path / "demo-bws4.jsonl"), _lines(tmp_path / "demo-triage.jsonl")
    c_log = _lines(tmp_path / "demo-shootout.jsonl")
    assert [(e["session"], e["micro"]) for e in a_log] == [("s1110a", 1), ("s1110a", 1), ("s1110b", 1)]
    assert [(e["session"], e["micro"]) for e in b_log] == [("s1110b", 1)]      # A at 21:31 joined B's session
    assert [(e["session"], e["micro"]) for e in c_log] == [("s1110c", 1)]
    assert read_state(tmp_path / "state.json")["session"]["id"] == "s1110c"


def test_two_servers_never_undo_each_others_state(batches, tmp_path, monkeypatch):
    """Server B rewrites state.json while server A votes between B's read and B's write. A's cursor must
    survive: each server's read-merge-write is atomic with respect to the others."""
    ma, _ = batches["bws4"]
    a = _ledger(batches, "bws4", tmp_path)
    b = _ledger(batches, "triage", tmp_path)
    real_write = votes_mod.atomic_write
    other = {}

    def write_while_a_votes(path, data):
        if not other and threading.current_thread() is threading.main_thread():
            other["t"] = threading.Thread(target=a.submit, args=(vote(ma, ma.rounds[0]),))
            other["t"].start()
            other["t"].join(0.3)                          # A votes now, unless it must wait for B to finish
        real_write(path, data)

    monkeypatch.setattr(votes_mod, "atomic_write", write_while_a_votes)
    try:
        b.submit({"type": "session", "what": "pause"})
        other["t"].join(5)
        assert not other["t"].is_alive()
        st = read_state(tmp_path / "state.json")
        assert st["batches"]["demo-bws4"]["settled"] == 1 and st["batches"]["demo-bws4"]["events"] == 1
        assert st["batches"]["demo-triage"]["events"] == 1
    finally:
        a.close()
        b.close()


# ----------------------------------------------------------------------------- timing fields (spec-5, spec-8, spec-9)

@pytest.mark.parametrize("mode, picks, t_ms, kept", [
    ("bws4", lambda ids: dict(best=None, no_best=True, worst=ids[3], tag=None),
     {"best": 1000, "no_best": 2000, "worst": 3000, "tag": 4000, "commit": 9000},
     {"no_best": 2000, "worst": 3000, "commit": 9000}),
    ("bws4", lambda ids: dict(best=None, worst=None, all_bad=True),
     {"first_play": 10, "best": 1000, "all_bad": 2000, "worst": 1500, "no_worst": 1700, "commit": 3000},
     {"first_play": 10, "all_bad": 2000, "commit": 3000}),
    ("keep3", lambda ids: dict(best=None, all_bad=True), {"best": 1000, "all_bad": 2000, "commit": 3000},
     {"all_bad": 2000, "commit": 3000}),
    ("playlist", lambda ids: dict(mark="none"), {"mark": 1000, "commit": 3000}, {"commit": 3000}),
    ("ab", lambda ids: dict(pick=ids[1]), {"pick": 1000, "best": 900, "commit": 3000}, {"pick": 1000, "commit": 3000}),
])
def test_t_ms_keeps_only_the_times_of_marks_in_the_vote(batches, mode, picks, t_ms, kept):
    """A mark that was set and later cleared (T after Q, a tag pressed twice, G after a best) leaves no time."""
    m, _ = batches[mode]
    r = m.rounds[0]
    v = vote(m, r, **picks(r.ids))
    v["t_ms"] = t_ms
    assert check_event(v, m)["t_ms"] == kept


def test_playback_heartbeats_keep_a_long_round_in_its_micro_session(batches, tmp_path):
    """He lets round 1 loop for 11 minutes while working: the page posts session `playing` every 5 minutes of
    uninterrupted playback, so no 10-minute gap opens. Without playback, 11 silent minutes are still a gap."""
    clock = Clock()
    m, _ = batches["bws4"]
    led = _ledger(batches, "bws4", tmp_path, clock)
    try:
        led.submit({"type": "session", "what": "start"})
        clock.advance(seconds=90)
        led.submit({"type": "session", "what": "round_start", "round": m.rounds[0].id})
        for _ in range(2):
            clock.advance(minutes=5)
            led.submit({"type": "session", "what": "playing", "round": m.rounds[0].id})
        clock.advance(minutes=1)
        led.submit(vote(m, m.rounds[0]))
        assert (_lines(led.path)[-1]["type"], _lines(led.path)[-1]["micro"]) == ("bws4", 1)
        clock.advance(seconds=30)
        led.submit({"type": "session", "what": "round_start", "round": m.rounds[1].id})
        clock.advance(minutes=11)
        led.submit(vote(m, m.rounds[1]))
        assert _lines(led.path)[-1]["micro"] == 2
    finally:
        led.close()


def test_the_page_learns_when_its_micro_session_began(batches, tmp_path):
    """A reload or a restart within 10 minutes keeps the micro-session, so the page's 25-minute break nudge keeps
    counting from its start (and knows how many nudges it has had)."""
    clock = Clock()
    m, _ = batches["triage"]
    led = _ledger(batches, "triage", tmp_path, clock)
    try:
        st = led.submit({"type": "session", "what": "start"})["state"]
        assert (st["micro"], st["micro_start"], st["micro_nudges"]) == (1, "2026-11-10T21:00:00-08:00", 0)
        clock.advance(minutes=8)
        led.submit({"type": "session", "what": "nudge"})
        clock.advance(minutes=1)
        st = led.submit({"type": "session", "what": "start"})["state"]          # the page reloads
        assert (st["micro"], st["micro_start"], st["micro_nudges"]) == (1, "2026-11-10T21:00:00-08:00", 1)
    finally:
        led.close()
    clock.advance(minutes=5)
    led = _ledger(batches, "triage", tmp_path, clock)                             # make rate restarts
    try:
        st = led.submit({"type": "session", "what": "start"})["state"]
        assert (st["micro"], st["micro_start"], st["micro_nudges"]) == (1, "2026-11-10T21:00:00-08:00", 1)
        clock.advance(minutes=10)
        st = led.submit({"type": "session", "what": "pause"})["state"]           # a 10-minute gap: micro 2
        assert (st["micro"], st["micro_start"], st["micro_nudges"]) == (2, "2026-11-10T21:24:00-08:00", 0)
    finally:
        led.close()
