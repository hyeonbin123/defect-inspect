import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from defect_inspect import ledger

# Child process for the multi-process test: waits for the "go" file, then appends argv[3] entries.
CHILD = """
import sys, time
from pathlib import Path
from defect_inspect import ledger
path, tag, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
ready, go = Path(sys.argv[4]), Path(sys.argv[5])
ready.write_bytes(b"")
while not go.exists():
    time.sleep(0.001)
for i in range(n):
    ledger.record_test_access(path, stage=f"{tag}-{i}", config="c", commit="x")
"""


def check_ledger(path: Path, stages: set[str]) -> None:
    """Every line is one complete JSON entry, and exactly the expected entries are there."""
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    lines = raw.decode("utf-8").splitlines()
    entries = [json.loads(line) for line in lines]  # a torn or merged line fails to parse
    assert all(list(e) == ["time", "commit", "stage", "config", "note"] for e in entries)
    assert len(entries) == len(stages)
    assert {e["stage"] for e in entries} == stages
    assert not ledger.lock_path(path).exists()


def test_record_appends_one_json_line_per_call(tmp_path):
    path = tmp_path / "reports" / "test_ledger.jsonl"
    now = datetime(2026, 9, 30, 12, 0, 5, 123456, tzinfo=UTC)
    first = ledger.record_test_access(path, stage="1", config="P0", commit="abc1234", now=now)
    second = ledger.record_test_access(
        path, stage="1", config="P0-교차", note="다시 잼", commit="abc1234+dirty", now=now
    )
    assert first == {
        "time": "2026-09-30T12:00:05+00:00",
        "commit": "abc1234",
        "stage": "1",
        "config": "P0",
        "note": "",
    }
    assert list(first) == ["time", "commit", "stage", "config", "note"]
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    lines = raw.decode("utf-8").splitlines()
    assert [json.loads(line) for line in lines] == [first, second]
    assert "다시 잼" in lines[1]  # written as UTF-8, not \u escapes


def test_time_is_converted_to_utc(tmp_path):
    path = tmp_path / "ledger.jsonl"
    kst = timezone(timedelta(hours=9))
    entry = ledger.record_test_access(
        path, stage="2", config="c", commit="x", now=datetime(2026, 10, 1, 8, 30, tzinfo=kst)
    )
    assert entry["time"] == "2026-09-30T23:30:00+00:00"
    naive = ledger.record_test_access(path, stage="2", config="c", commit="x", now=datetime(2026, 10, 1, 8))
    assert naive["time"] == "2026-10-01T08:00:00+00:00"


def test_defaults_use_current_time_and_git(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "git_commit", lambda root=None: "feed123")
    before = datetime.now(UTC) - timedelta(seconds=2)
    entry = ledger.record_test_access(tmp_path / "ledger.jsonl", stage="1", config="P0")
    stamp = datetime.fromisoformat(entry["time"])
    assert stamp.utcoffset() == timedelta(0)
    assert before <= stamp <= datetime.now(UTC) + timedelta(seconds=2)
    assert entry["commit"] == "feed123"


def test_git_commit_unknown_when_git_fails(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(ledger.subprocess, "run", missing)
    assert ledger.git_commit(tmp_path) == "unknown"

    def failing(cmd, **kwargs):
        raise subprocess.CalledProcessError(128, cmd)

    monkeypatch.setattr(ledger.subprocess, "run", failing)
    assert ledger.git_commit(tmp_path) == "unknown"


def test_git_commit_in_a_real_repository(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
        subprocess.run(["git", *identity, *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    assert ledger.git_commit(repo) == "unknown"  # no commit yet
    (repo / "a.txt").write_text("a\n", encoding="utf-8", newline="\n")
    git("add", "a.txt")
    git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "first")
    clean = ledger.git_commit(repo)
    assert clean != "unknown" and not clean.endswith("+dirty")
    assert len(clean) >= 7 and int(clean, 16) >= 0
    # A file git does not track yet (a new module) changes behaviour too: it counts as dirty.
    (repo / "new_module.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    assert ledger.git_commit(repo) == clean + "+dirty"
    (repo / "new_module.py").unlink()
    assert ledger.git_commit(repo) == clean
    (repo / "a.txt").write_text("b\n", encoding="utf-8", newline="\n")
    assert ledger.git_commit(repo) == clean + "+dirty"


def test_concurrent_appends_from_threads_are_all_kept(tmp_path):
    path = tmp_path / "ledger.jsonl"
    n_threads, n_each = 8, 50
    errors: list[BaseException] = []
    start = threading.Barrier(n_threads)

    def work(tag: int) -> None:
        try:
            start.wait()
            for i in range(n_each):
                ledger.record_test_access(path, stage=f"{tag}-{i}", config="c", note="동시 쓰기", commit="x")
        except BaseException as err:  # noqa: BLE001 - reported by the assertion below
            errors.append(err)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    check_ledger(path, {f"{t}-{i}" for t in range(n_threads) for i in range(n_each)})  # 400 lines


def test_concurrent_appends_from_processes_are_all_kept(tmp_path):
    path = tmp_path / "ledger.jsonl"
    go = tmp_path / "go"
    n_procs, n_each = 4, 60
    package_parent = str(Path(ledger.__file__).resolve().parents[1])
    inherited = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([package_parent, *inherited]))
    ready = [tmp_path / f"ready-{p}" for p in range(n_procs)]
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", CHILD, str(path), str(p), str(n_each), str(ready[p]), str(go)], env=env
        )
        for p in range(n_procs)
    ]
    try:
        deadline = time.monotonic() + 60
        while not all(r.exists() for r in ready):
            assert time.monotonic() < deadline and all(p.poll() is None for p in procs)
            time.sleep(0.01)
        go.write_bytes(b"")  # all children start appending at the same moment
        assert [p.wait(timeout=120) for p in procs] == [0] * n_procs
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    check_ledger(path, {f"{p}-{i}" for p in range(n_procs) for i in range(n_each)})


def test_append_waits_for_a_lock_that_is_released(tmp_path):
    path = tmp_path / "ledger.jsonl"
    lock = ledger.lock_path(path)
    assert lock.parent == path.parent and lock != path
    lock.write_bytes(b"")  # another writer holds the lock ...
    release = threading.Timer(0.3, lock.unlink)  # ... and lets go after 0.3 s
    release.start()
    started = time.monotonic()
    ledger.record_test_access(path, stage="1", config="c", commit="x")
    assert time.monotonic() - started >= 0.25
    release.join()
    check_ledger(path, {"1"})


def test_held_lock_times_out_and_nothing_is_written(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"kept": true}\n')
    lock = ledger.lock_path(path)
    lock.write_bytes(b"")  # fresh, so it is not treated as stale
    monkeypatch.setattr(ledger, "LOCK_TIMEOUT_S", 0.2)
    with pytest.raises(TimeoutError, match="lock"):
        ledger.record_test_access(path, stage="1", config="c", commit="x")
    assert path.read_bytes() == b'{"kept": true}\n'
    assert lock.exists()  # it belongs to the other writer


def test_stale_lock_is_broken(tmp_path):
    path = tmp_path / "ledger.jsonl"
    lock = ledger.lock_path(path)
    lock.write_bytes(b"")
    old = time.time() - 2 * ledger.LOCK_STALE_S
    os.utime(lock, (old, old))  # left behind by a writer that crashed
    assert ledger.LOCK_STALE_S == 60 and ledger.LOCK_TIMEOUT_S == 10
    ledger.record_test_access(path, stage="1", config="c", commit="x")
    check_ledger(path, {"1"})


def test_lock_is_released_when_the_append_fails(tmp_path):
    path = tmp_path / "a_directory"
    path.mkdir()
    with pytest.raises(OSError):
        ledger.record_test_access(path, stage="1", config="c", commit="x")
    assert not ledger.lock_path(path).exists()


def test_entry_that_cannot_be_serialised_writes_nothing(tmp_path):
    path = tmp_path / "ledger.jsonl"
    with pytest.raises(TypeError):
        ledger.record_test_access(path, stage="1", config="c", note=object(), commit="x")
    assert not path.exists() and not ledger.lock_path(path).exists()


def test_entry_is_not_glued_to_a_torn_last_line(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"time": "cut off')  # a previous writer died in the middle of its line
    entry = ledger.record_test_access(path, stage="1", config="c", commit="x")
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    lines = raw.decode("utf-8").splitlines()
    assert lines[0] == '{"time": "cut off' and json.loads(lines[1]) == entry and len(lines) == 2


def test_git_state_is_read_before_the_lock_is_taken(tmp_path, monkeypatch):
    # The lock file sits next to the ledger inside the repository: `git status` must not see it.
    path = tmp_path / "ledger.jsonl"
    seen = []

    def fake_commit(root=None):
        seen.append(ledger.lock_path(path).exists())
        return "abc1234"

    monkeypatch.setattr(ledger, "git_commit", fake_commit)
    assert ledger.record_test_access(path, stage="1", config="c")["commit"] == "abc1234"
    assert seen == [False]
