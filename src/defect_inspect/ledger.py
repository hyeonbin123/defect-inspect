"""Append-only record of every run that reads the sealed test set."""

# [파일 역할] 봉인 테스트(VisA 공식 2cls_highshot test 4,328장, M2AD 테스트 이미지)를 읽은 기록을
# reports/test_ledger.jsonl에 한 줄씩 덧붙이는 장부 모듈이다.
# [왜 필요한가] 이 프로젝트는 규칙을 먼저 커밋하고 봉인 테스트는 단계가 끝날 때 정해 둔 것만 한 번
# 잰다(docs/experiments.md 공통 규칙 '봉인 테스트'). 언제·어느 커밋에서·어느 단계·어느 구성이 테스트를 봤는지
# 남겨야 '테스트를 보고 고치지 않았다'를 나중에 확인할 수 있다.
# [설계 이유] 기록을 쓰지 못하면 예외를 던져 호출한 실행이 테스트를 읽기 전에 멈추게 한다. 여러
# 실행(스레드·프로세스)이 동시에 장부에 쓸 수 있으므로 잠금 파일로 덧붙이기를 한 번에 하나씩만 하게 한다.
# [연결] run_patchcore, run_grid, run_export_eval, run_m2ad, run_dinomaly, run_perturb, run_supervised,
# run_m2ad_enrol, dinomaly_serving, cache, m2ad, figures가 record_test_access를 부르고, git_commit은 리포트의
# commit 필드를 채우는 데 쓰인다.

import contextlib
import json
import os
import subprocess
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

# 잠금 대기 시간(초). 다른 기록자가 잠금을 쥐고 있으면 10초까지 기다리다 TimeoutError를 낸다.
# docs/experiments.md(서비스 모델 교체 절의 2026-10-04 정정)에는 CPU가 아주 바쁠 때 tests/test_ledger.py의
# 동시 추가 테스트 두 개가 이 10초 안에 잠금을 얻지 못해 실패한 기록이 있다.
# LOCK_STALE_S: 이보다 오래된 잠금 파일은 기록 도중 죽은 프로세스가 남긴 것으로 보고 지운다. _LOCK_POLL_S:
# 잠금을 다시 시도하는 간격(5ms).
LOCK_TIMEOUT_S = 10.0  # how long an append waits for another writer before it gives up
LOCK_STALE_S = 60.0  # a lock file older than this was left by a writer that died; it may be broken
_LOCK_POLL_S = 0.005


# git 명령을 root에서 실행해 표준 출력을 돌려주는 작은 도우미. timeout=30과 check=True로, git이 멈추거나
# 실패하면 예외가 나서 git_commit이 "unknown"으로 처리한다.
def _git(args: list[str], root: Path) -> str:
    done = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=30, check=True
    )
    return done.stdout.strip()


# 실행 당시 코드 버전을 짧은 커밋 해시로 돌려준다. 커밋하지 않은 변경이 있으면 "+dirty"를 붙여, 리포트의
# 수치가 커밋된 코드에서 나온 것인지 구별할 수 있게 한다(experiments.md에 'be38874+dirty' 같은 기록이 있다).
# 호출 위치: record_test_access(commit을 주지 않을 때), 그리고 export, run_grid, run_export_eval, run_m2ad,
# run_dinomaly, dinomaly_serving, runtime_compare 등이 리포트를 쓰기 전에 한 번 부른다.
# 주의: git이 없거나 저장소가 아니거나 커밋이 아직 없으면 예외 대신 "unknown"을 돌려준다.
def git_commit(root: Path | None = None) -> str:
    """Short HEAD hash, with "+dirty" when `git status --porcelain` lists anything; "unknown" on failure.

    Untracked files count as dirty (a new module changes behaviour too). A run that writes reports
    should call this once before it writes anything and pass the value on as `commit=`.
    """
    # paths를 함수 안에서 늦게 가져오는 것은 테스트가 임시 저장소(root)를 넘겨 쓸 수 있게 하고, 기본값일 때만
    # 저장소 루트를 쓰기 위해서다.
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


# 장부 파일 옆에 둘 잠금 파일 경로(예: test_ledger.jsonl.lock). 테스트(tests/test_ledger.py)가 기록 뒤 잠금
# 파일이 남지 않았는지 확인할 때도 쓴다.
def lock_path(ledger: Path) -> Path:
    """Lock file that serialises appends to `ledger` (it exists only while one append is running)."""
    ledger = Path(ledger)
    return ledger.with_name(ledger.name + ".lock")


# 잠금 파일을 지우되, Windows에서 백신·색인기가 파일을 잠깐 열고 있어 PermissionError가 나면 5ms 간격으로 최대
# 200번(약 1초) 다시 시도한다.
def _unlink_retrying(path: Path, attempts: int = 200) -> None:
    # Windows refuses to delete a file that a scanner or indexer has open; that lasts milliseconds.
    for _ in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(_LOCK_POLL_S)
    # Give up quietly: the file goes stale after LOCK_STALE_S and the next writer breaks it.


# with 블록 동안 잠금 파일을 쥐는 문맥 관리자. O_CREAT | O_EXCL로 '파일이 없을 때만 만들기'를 원자적으로 해서,
# 먼저 만든 쪽만 장부에 쓰고 나머지는 잠금 파일이 사라질 때까지 다시 시도한다(docstring: Windows에서는
# 덧붙이기 자체가 원자적이지 않아 두 기록자가 같은 끝 위치에 써서 한 줄이 덮일 수 있다).
@contextlib.contextmanager
def _locked(lock: Path) -> Iterator[None]:
    """Hold `lock` for the duration of the block.

    Appending to a file is not atomic on Windows: two writers can both seek to the same end offset
    and one line overwrites the other. Creating a file with O_CREAT | O_EXCL is atomic on every
    platform, so whoever creates the lock file writes; the others retry until it is gone.
    """
    # 마감 시각은 시스템 시계 변경에 흔들리지 않는 monotonic으로 잰다. 반면 아래의 오래된 잠금 판정은 파일
    # 수정 시각(st_mtime)과 비교해야 하므로 time.time()을 쓴다.
    deadline = time.monotonic() + LOCK_TIMEOUT_S
    while True:
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except (FileExistsError, PermissionError) as err:
            # PermissionError: on Windows a lock file that its holder is deleting right now.
            busy = err
        # 잠금 파일이 LOCK_STALE_S보다 오래됐으면 죽은 기록자가 남긴 것으로 보고 지운 뒤 곧바로 다시 잡으러
        # 간다(continue). stat 도중 파일이 사라졌으면 OSError를 무시하고 그냥 다시 시도한다.
        try:
            if time.time() - lock.stat().st_mtime > LOCK_STALE_S:
                lock.unlink(missing_ok=True)
                continue
        except OSError:
            pass  # released (or broken by someone else) in the meantime: just try again
        # 마감을 넘기면 마지막으로 받은 오류(busy)를 원인으로 붙여 TimeoutError를 낸다. 메시지에 '실행 중인
        # 작업이 없으면 잠금 파일을 지우라'는 복구 방법을 적어 둔다.
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"could not take the ledger lock {lock} within {LOCK_TIMEOUT_S:g} s ({busy}): another "
                "run is writing the ledger (if no run is active, delete the lock file)"
            ) from busy
        time.sleep(_LOCK_POLL_S)
    # 블록이 예외로 끝나더라도 finally에서 잠금 파일을 반드시 지워 다음 기록자가 기다리지 않게 한다.
    try:
        yield
    finally:
        _unlink_retrying(lock)


# 잠금을 쥔 상태에서 한 줄을 덧붙이고, 방금 쓴 바이트를 다시 읽어 같은지 확인한다. 쓰기가 잘리거나 다른 내용이
# 들어갔으면 OSError로 실행을 멈추게 해, 기록되지 않은 테스트 읽기가 생기지 않게 한다.
# buffering=0으로 버퍼 없이 한 번의 write 호출로 쓴다.
def _append_line(ledger: Path, line: bytes) -> None:
    """Append `line` with a single write call and read it back. Call with the lock held."""
    with open(ledger, "a+b", buffering=0) as f:
        # 이전 기록이 줄바꿈 없이 끊겨 있으면(쓰다 죽은 경우) 앞에 줄바꿈을 붙여, 새 기록이 끊긴 줄에 이어
        # 붙어 JSON 한 줄이 깨지지 않게 한다.
        size = f.seek(0, os.SEEK_END)
        if size:
            f.seek(size - 1)
            if f.read(1) != b"\n":
                line = b"\n" + line  # a torn last line must not swallow this entry
        written = f.write(line)  # append mode: goes to the end of the file
        # 파일 끝에서 방금 쓴 길이만큼 되돌아가 읽어 쓴 내용과 비교한다(읽기 검증).
        end = f.seek(0, os.SEEK_END)
        f.seek(max(0, end - len(line)))
        if written != len(line) or f.read(len(line)) != line:
            raise OSError(f"the entry appended to {ledger} could not be read back")


# 봉인 테스트 한 번 읽기를 장부에 남기는 공개 함수. 입력: 장부 경로, 단계 이름(stage, 예: "1", "6", "7-m2ad",
# "cache"), 구성 이름(config), 메모(note), 커밋(없으면 git_commit()), 시각(없으면 현재 UTC). 출력: 기록한
# dict.
# 왜 이 방식인가: 실행 스크립트는 테스트 이미지를 읽기 전에 이 함수를 부르므로, 기록이 실패하면 예외가 나
# 테스트를 읽지 않고 멈춘다. 시각을 인자로 받는 것은 tests/test_ledger.py가 정해진 시각을 넣어 기록 형식을
# 확인하기 위해서다.
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
    # git 호출과 JSON 직렬화를 잠금을 잡기 전에 끝내, 잠금을 쥐는 시간을 짧게 하고 잠금 중 실패할 일을 줄인다.
    # Everything that can fail or look at the working tree happens before the lock file exists.
    entry = {
        "time": now.astimezone(UTC).isoformat(timespec="seconds"),
        "commit": git_commit() if commit is None else commit,
        "stage": stage,
        "config": config,
        "note": note,
    }
    # ensure_ascii=False로 메모의 한글 등을 그대로 UTF-8로 남겨 장부를 사람이 읽기 쉽게 한다. JSONL 형식(한
    # 줄에 JSON 하나)이라 덧붙이기만으로 기록이 쌓인다.
    line = (json.dumps(entry, ensure_ascii=False) + "\n").encode("utf-8")
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with _locked(lock_path(ledger)):
        _append_line(ledger, line)
    return entry
