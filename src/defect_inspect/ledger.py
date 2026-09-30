"""Append-only record of every run that reads the sealed test set."""

import contextlib
import json
import os
import subprocess
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

LOCK_TIMEOUT_S = 10.0  # how long an append waits for another writer before it gives up
LOCK_STALE_S = 60.0  # a lock file older than this was left by a writer that died; it may be broken
_LOCK_POLL_S = 0.005


def _git(args: list[str], root: Path) -> str:
    done = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=30, check=True
    )
    return done.stdout.strip()


def git_commit(root: Path | None = None) -> str:
    """Short HEAD hash, with "+dirty" when `git status --porcelain` lists anything; "unknown" on failure.

    Untracked files count as dirty (a new module changes behaviour too). A run that writes reports
    should call this once before it writes anything and pass the value on as `commit=`.
    """
    if root is None:
        from defect_inspect import paths

        root = paths.ROOT
    try:
        head = _git(["rev-parse", "--short", "HEAD"], Path(root))
        dirty = _git(["status", "--porcelain"], Path(root))
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not head:
        return "unknown"
    return head + ("+dirty" if dirty else "")


def lock_path(ledger: Path) -> Path:
    """Lock file that serialises appends to `ledger` (it exists only while one append is running)."""
    ledger = Path(ledger)
    return ledger.with_name(ledger.name + ".lock")


def _unlink_retrying(path: Path, attempts: int = 200) -> None:
    # Windows refuses to delete a file that a scanner or indexer has open; that lasts milliseconds.
    for _ in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(_LOCK_POLL_S)
    # Give up quietly: the file goes stale after LOCK_STALE_S and the next writer breaks it.


@contextlib.contextmanager
def _locked(lock: Path) -> Iterator[None]:
    """Hold `lock` for the duration of the block.

    Appending to a file is not atomic on Windows: two writers can both seek to the same end offset
    and one line overwrites the other. Creating a file with O_CREAT | O_EXCL is atomic on every
    platform, so whoever creates the lock file writes; the others retry until it is gone.
    """
    deadline = time.monotonic() + LOCK_TIMEOUT_S
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except (FileExistsError, PermissionError) as err:
            # PermissionError: on Windows a lock file that its holder is deleting right now.
            busy = err
        try:
            if time.time() - lock.stat().st_mtime > LOCK_STALE_S:
                lock.unlink(missing_ok=True)
                continue
        except OSError:
            pass  # released (or broken by someone else) in the meantime: just try again
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"could not take the ledger lock {lock} within {LOCK_TIMEOUT_S:g} s ({busy}): another "
                "run is writing the ledger (if no run is active, delete the lock file)"
            ) from busy
        time.sleep(_LOCK_POLL_S)
    try:
        yield
    finally:
        _unlink_retrying(lock)


def _append_line(ledger: Path, line: bytes) -> None:
    """Append `line` with a single write call and read it back. Call with the lock held."""
    with open(ledger, "a+b", buffering=0) as f:
        size = f.seek(0, os.SEEK_END)
        if size:
            f.seek(size - 1)
            if f.read(1) != b"\n":
                line = b"\n" + line  # a torn last line must not swallow this entry
        written = f.write(line)  # append mode: goes to the end of the file
        end = f.seek(0, os.SEEK_END)
        f.seek(max(0, end - len(line)))
        if written != len(line) or f.read(len(line)) != line:
            raise OSError(f"the entry appended to {ledger} could not be read back")


def record_test_access(
    ledger: Path,
    *,
    stage: str,
    config: str,
    note: str = "",
    commit: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Append one JSON line describing a read of the sealed test set and return it.

    Safe to call from several threads or processes at once: appends are serialised by a lock file
    next to the ledger. Raises (so that the caller stops before it reads the test set) when the
    entry cannot be written.
    """
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=UTC)  # naive timestamps are taken as UTC
    # Everything that can fail or look at the working tree happens before the lock file exists.
    entry = {
        "time": now.astimezone(UTC).isoformat(timespec="seconds"),
        "commit": git_commit() if commit is None else commit,
        "stage": stage,
        "config": config,
        "note": note,
    }
    line = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with _locked(lock_path(ledger)):
        _append_line(ledger, line)
    return entry
