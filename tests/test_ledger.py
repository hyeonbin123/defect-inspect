# [파일 역할] 봉인 테스트를 읽을 때마다 reports/test_ledger.jsonl에 한 줄을 남기는 장부
# 모듈(src/defect_inspect/ledger.py)을 검증한다.
# [왜 필요한가] 이 프로젝트의 측정 방식은 '규칙을 먼저 커밋하고, 봉인 테스트는 단계가 끝날 때 정해 둔 것만 한
# 번 읽는다'이다(docs/plan.md 측정 방식). 장부는 언제·어느 커밋에서·어느 단계가 테스트를 봤는지의 유일한
# 증거이므로, 줄이 빠지거나 두 줄이 섞이거나 커밋 표시가 틀리면 그 증거가 무너진다.
# [주요 구성] (1) 기록 형식(키 순서, UTF-8, UTC 시각) (2) git_commit의 '+dirty' 판정과 실패 시 "unknown" (3)
# 스레드·프로세스 동시 기록에서 줄이 하나도 빠지거나 겹치지 않는지 (4) 잠금 파일의 대기·시간 초과·오래된 잠금
# 정리·실패 후 해제 (5) 직렬화 실패·끊긴 마지막 줄 같은 예외 상황 (6) 장부 줄의 tree(작업 트리 상태: git
# status --porcelain 목록과 git diff HEAD의 SHA-256)를 임시 git 저장소에서 실제로 기록해 보는 것. tree는 이
# 저장소에 커밋되는 장부에만 붙으므로, 저장소 밖 tmp_path에 쓰는 다른 테스트의 장부 줄은 앞의 다섯 키
# 그대로다.
# [주의] 동시 기록 테스트 두 개는 CPU가 아주 바쁠 때 10초 잠금 대기(LOCK_TIMEOUT_S)를 넘겨 실패한 적이
# 있다(docs/experiments.md 서비스 모델 교체 점검의 2026-10-04 정정). 따로 돌리면 통과했다. 이 두 테스트가 보는
# 것은 줄이 빠지거나 섞이지 않는지이고 대기 시간은 아니므로, 둘 다 대기 시간을 CONCURRENT_LOCK_TIMEOUT_S로
# 늘려 돈다. 시간 초과 동작은 test_held_lock_times_out_and_nothing_is_written이 따로 본다.

import hashlib
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

# 모듈째 가져와 LOCK_TIMEOUT_S, git_commit, ledger_tree, subprocess 같은 모듈 속성을 monkeypatch로 바꿀 수
# 있게 한다. paths는 '이 저장소'의 위치(paths.ROOT)를 임시 git 저장소로 바꾸려고 가져온다.
from defect_inspect import ledger, paths

# 여러 프로세스 동시 기록 테스트에서 각 자식 프로세스가 실행할 코드. 준비 파일(ready)을 만든 뒤 go 파일이 생길
# 때까지 기다렸다가 기록을 시작하므로, 모든 자식이 같은 순간에 장부를 두드리게 된다. 프로세스 시작 시간 차이
# 때문에 사실상 순서대로 쓰는 일이 없게 하려는 장치다.
# 자식은 go 파일이 생긴 뒤 argv[3]개의 기록을 덧붙이고, 잠금 대기 시간을 argv[6]으로 받는다(동시 기록
# 테스트의 대기 시간, 위 [주의]).
CHILD = """
import sys, time
from pathlib import Path
from defect_inspect import ledger
path, tag, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
ready, go = Path(sys.argv[4]), Path(sys.argv[5])
ledger.LOCK_TIMEOUT_S = float(sys.argv[6])
ready.write_bytes(b"")
while not go.exists():
    time.sleep(0.001)
for i in range(n):
    ledger.record_test_access(path, stage=f"{tag}-{i}", config="c", commit="x")
"""

# 동시 기록 테스트 두 개의 잠금 대기 시간(초). 이 테스트들은 줄이 빠지지 않는지를 보지, 기록자가 얼마나
# 기다리는지를 보지 않는다. 기본 10초는 CPU가 바쁠 때 모자란 적이 있어 넉넉히 늘린다.
CONCURRENT_LOCK_TIMEOUT_S = 120.0

# 이 저장소에 커밋되는 장부의 키 순서: 처음부터의 다섯 키 뒤에 2026-10-29에 더한 tree.
TREE_KEYS = ["time", "commit", "stage", "config", "note", "tree"]
# 가짜 ledger_tree가 돌려주는 트리 상태(실제 장부에서 가장 흔한 모양: 앞 실행이 남긴 장부 줄이 아직 커밋되지
# 않음). 해시 값 자체는 의미 없는 자리표시자다.
FAKE_TREE = {"porcelain": [" M reports/test_ledger.jsonl"], "diff_sha256": "0" * 64}


# 테스트용 git 명령 도우미. 사용자 git 설정에 기대지 않도록 이름·메일과 서명 끄기를 명령마다 -c로 넘긴다(전역
# 설정이 없는 CI에서도 커밋이 된다).
def git_in(repo: Path, *args: str) -> bytes:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false"]
    return subprocess.run(["git", *identity, *args], cwd=repo, check=True, capture_output=True).stdout


# 장부에 적는 diff_sha256의 정의를 테스트 쪽에서 따로 계산한다: `git diff --binary HEAD`(바이너리 변경도
# 내용까지 담는 형식)의 출력 바이트 SHA-256. 색·외부 diff 도구·textconv·하위 폴더 한정(diff.relative)을 꺼서
# 사용자 설정에 덜 흔들리는 출력으로 만든다. 나중에 장부 줄을 검증하는 사람도 이 명령으로 같은 값을 다시 낼 수
# 있다.
def diff_sha256(repo: Path) -> str:
    args = ["-c", "core.quotePath=true", "diff", "--binary", "--no-color", "--no-ext-diff", "--no-textconv"]
    return hashlib.sha256(git_in(repo, *args, "--no-relative", "HEAD")).hexdigest()


# 장부 파일 전체가 온전한지 검사하는 공용 도우미. 줄바꿈이 LF로만 끝나고, 모든 줄이 JSON으로 읽히며(끊기거나
# 두 줄이 붙은 줄은 여기서 실패한다), 키 순서가 정해진 대로이고, 기대한 stage가 빠짐없이 정확히 한 번씩
# 있으며, 기록이 끝난 뒤 잠금 파일이 남지 않았는지 본다.
def check_ledger(path: Path, stages: set[str]) -> None:
    """모든 줄이 온전한 JSON 기록 하나이고, 기대한 기록이 정확히 그만큼 있는지 확인한다."""
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    lines = raw.decode("utf-8").splitlines()
    entries = [json.loads(line) for line in lines]  # 끊기거나 두 줄이 붙은 줄은 파싱에 실패한다
    assert all(list(e) == ["time", "commit", "stage", "config", "note"] for e in entries)
    assert len(entries) == len(stages)
    assert {e["stage"] for e in entries} == stages
    assert not ledger.lock_path(path).exists()


# 호출 한 번에 JSON 한 줄이 덧붙고, 반환값과 파일 내용이 같은지 본다. 시각은 초 단위로 잘리고(마이크로초
# 버림), 키 순서는 time, commit, stage, config, note로 고정이다. 한글 메모가 \u 이스케이프가 아니라 UTF-8
# 그대로 쓰여 사람이 장부를 바로 읽을 수 있는지, Windows에서도 CRLF가 섞이지 않는지 확인한다. 부모
# 폴더(reports/)가 없어도 만들어야 한다.
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
    assert "다시 잼" in lines[1]  # \u 이스케이프가 아니라 UTF-8 그대로 쓰였다


# 시간대가 붙은 시각(KST)은 UTC로 바꿔 적고, 시간대가 없는 시각은 UTC로 간주하는지 본다. 장부 시각을 한
# 기준으로 맞춰야 커밋 시각·다른 기록과 순서를 비교할 수 있다.
def test_time_is_converted_to_utc(tmp_path):
    path = tmp_path / "ledger.jsonl"
    kst = timezone(timedelta(hours=9))
    entry = ledger.record_test_access(
        path, stage="2", config="c", commit="x", now=datetime(2026, 10, 1, 8, 30, tzinfo=kst)
    )
    assert entry["time"] == "2026-09-30T23:30:00+00:00"
    naive = ledger.record_test_access(path, stage="2", config="c", commit="x", now=datetime(2026, 10, 1, 8))
    assert naive["time"] == "2026-10-01T08:00:00+00:00"


# now와 commit을 주지 않으면 현재 UTC 시각과 git_commit() 결과를 쓰는지 본다. git_commit을 가짜로 바꿔 실제
# 저장소 상태에 따라 결과가 달라지지 않게 하고, 시각은 앞뒤 2초 범위로만 확인한다.
def test_defaults_use_current_time_and_git(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "git_commit", lambda root=None: "feed123")
    before = datetime.now(UTC) - timedelta(seconds=2)
    entry = ledger.record_test_access(tmp_path / "ledger.jsonl", stage="1", config="P0")
    stamp = datetime.fromisoformat(entry["time"])
    assert stamp.utcoffset() == timedelta(0)
    assert before <= stamp <= datetime.now(UTC) + timedelta(seconds=2)
    assert entry["commit"] == "feed123"


# git이 설치되지 않았거나(FileNotFoundError) 명령이 실패해도(CalledProcessError) 예외 대신 "unknown"을
# 돌려주는지 본다. 커밋 정보를 못 얻었다는 이유로 측정 실행 전체가 멈추면 안 되기 때문이다. subprocess.run을
# 가짜로 바꿔 두 실패를 만든다.
def test_git_commit_unknown_when_git_fails(tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(ledger.subprocess, "run", missing)
    assert ledger.git_commit(tmp_path) == "unknown"

    def failing(cmd, **kwargs):
        raise subprocess.CalledProcessError(128, cmd)

    monkeypatch.setattr(ledger.subprocess, "run", failing)
    assert ledger.git_commit(tmp_path) == "unknown"


# 임시 git 저장소에서 git_commit을 실제로 돌려 본다: 커밋이 없으면 "unknown", 커밋 직후는 짧은 해시(16진수 7자
# 이상), 추적하지 않는 새 파일이나 수정된 파일이 있으면 '+dirty'다. 새 모듈 파일도 동작을 바꾸므로 dirty로
# 세야 리포트 수치가 커밋된 코드에서 나왔는지 가릴 수 있다(git_commit docstring). git이 없으면 건너뛴다.
def test_git_commit_in_a_real_repository(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    repo.mkdir()

    # 사용자 git 설정에 기대지 않도록 이름·메일을 명령마다 -c로 넘긴다(전역 설정이 없는 CI에서도 커밋이 된다).
    def git(*args):
        identity = ["-c", "user.name=test", "-c", "user.email=test@example.invalid"]
        subprocess.run(["git", *identity, *args], cwd=repo, check=True, capture_output=True)

    git("init", "-q")
    assert ledger.git_commit(repo) == "unknown"  # 아직 커밋이 없다
    (repo / "a.txt").write_text("a\n", encoding="utf-8", newline="\n")
    git("add", "a.txt")
    git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "first")
    clean = ledger.git_commit(repo)
    assert clean != "unknown" and not clean.endswith("+dirty")
    assert len(clean) >= 7 and int(clean, 16) >= 0
    # git이 아직 추적하지 않는 파일(새 모듈)도 동작을 바꾸므로 dirty로 센다.
    (repo / "new_module.py").write_text("x = 1\n", encoding="utf-8", newline="\n")
    assert ledger.git_commit(repo) == clean + "+dirty"
    (repo / "new_module.py").unlink()
    assert ledger.git_commit(repo) == clean
    (repo / "a.txt").write_text("b\n", encoding="utf-8", newline="\n")
    assert ledger.git_commit(repo) == clean + "+dirty"


# 임시 git 저장소를 '이 저장소'(paths.ROOT)로 삼아 장부를 실제로 기록해 tree 필드를 본다. 저장소는 이
# 프로젝트와 같은 모양이다: 추적되는 장부 reports/test_ledger.jsonl, 추적 파일 a.txt, 바이너리 파일 b.bin,
# scratch/를 무시하는 .gitignore. 확인하는 것:
# (1) 커밋이 아직 없는 저장소는 git diff HEAD가 실패하므로 트리를 못 읽는다(None)
# (2) 깨끗한 트리면 porcelain은 빈 목록이고 diff_sha256은 빈 출력의 SHA-256이며, tree는 맨 뒤 키다
# (3) 트리는 덧붙이기 전에 읽는다: 둘째 기록에는 첫 줄만 커밋하지 않은 변경으로 잡히고(실제 장부의 '+dirty'는
#     대부분 이것이다) 해시도 그 순간의 diff다. 잠금 파일(이 .gitignore에 없음)도 목록에 나오지 않는다
# (4) 추적 파일 수정과 추적하지 않는 새 파일이 목록에 나오고(추적 변경이 먼저, ??가 뒤), 해시는 위 정의로 따로
#     낸 값과 같으며 추적 파일 내용이 바뀌면 달라진다
# (5) 바이너리 파일의 내용 변경도 해시를 바꾼다
# (6) 추적하지 않는 파일은 이름만 남는다: 그 내용이 바뀌어도 기록은 같다(문서에 적은 한계)
# (7) 저장소 밖의 장부와 .gitignore로 무시되는 곳의 장부(테스트의 임시 장부가 이렇다)에는 tree를 적지 않는다
# git이 없으면 건너뛴다.
def test_tree_is_recorded_for_the_ledger_of_the_repository(tmp_path, monkeypatch):
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    repo = tmp_path / "repo"
    (repo / "reports").mkdir(parents=True)
    git_in(repo, "init", "-q")
    assert ledger.git_tree(repo) is None  # (1) 아직 커밋이 없다
    (repo / ".gitignore").write_text("scratch/\n", encoding="utf-8", newline="\n")
    (repo / "a.txt").write_text("a\n", encoding="utf-8", newline="\n")
    (repo / "b.bin").write_bytes(bytes(range(256)))
    path = repo / "reports" / "test_ledger.jsonl"
    path.write_bytes(b"")
    git_in(repo, "add", ".gitignore", "a.txt", "b.bin", "reports/test_ledger.jsonl")
    git_in(repo, "commit", "-q", "-m", "first")
    monkeypatch.setattr(paths, "ROOT", repo)

    clean = ledger.record_test_access(path, stage="1", config="c", commit="x")  # (2)
    assert clean["tree"] == {"porcelain": [], "diff_sha256": hashlib.sha256(b"").hexdigest()}
    assert list(clean) == TREE_KEYS
    before_second = diff_sha256(repo)  # (3) 첫 줄만 있는 지금의 diff
    second = ledger.record_test_access(path, stage="2", config="c", commit="x")
    assert second["tree"]["porcelain"] == [" M reports/test_ledger.jsonl"]
    assert second["tree"]["diff_sha256"] == before_second != clean["tree"]["diff_sha256"]
    assert diff_sha256(repo) != before_second  # 두 번째 줄이 붙은 지금의 diff는 또 다르다
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [clean, second]

    (repo / "a.txt").write_text("b\n", encoding="utf-8", newline="\n")  # (4)
    (repo / "new.txt").write_text("new\n", encoding="utf-8", newline="\n")
    edited = ledger.git_tree(repo)
    assert edited["porcelain"] == [" M a.txt", " M reports/test_ledger.jsonl", "?? new.txt"]
    assert edited["diff_sha256"] == diff_sha256(repo) != second["tree"]["diff_sha256"]
    assert ledger.ledger_tree(path) == edited == ledger.git_tree()  # 기본 저장소(paths.ROOT)도 같은 곳이다
    (repo / "b.bin").write_bytes(bytes(range(255, -1, -1)))  # (5)
    binary = ledger.git_tree(repo)
    assert binary["porcelain"] == [" M a.txt", " M b.bin", " M reports/test_ledger.jsonl", "?? new.txt"]
    assert binary["diff_sha256"] == diff_sha256(repo) != edited["diff_sha256"]
    (repo / "new.txt").write_text("changed\n", encoding="utf-8", newline="\n")  # (6)
    assert ledger.git_tree(repo) == binary

    outside = ledger.record_test_access(tmp_path / "out" / "ledger.jsonl", stage="3", config="c", commit="x")
    ignored = ledger.record_test_access(repo / "scratch" / "ledger.jsonl", stage="3", config="c", commit="x")
    assert list(outside) == list(ignored) == TREE_KEYS[:-1]  # (7)
    assert ledger.git_tree(repo) == binary  # 무시되는 곳의 장부는 목록에도 나오지 않는다


# git이 없거나(FileNotFoundError) 명령이 실패하면(CalledProcessError) git_tree는 None이고, 저장소 안의
# 장부라도 tree 없이 장부 줄을 그대로 쓰는지 본다. 트리를 못 읽었다는 이유로 측정 실행 전체가 멈추면 안
# 된다(git_commit의 "unknown"과 같은 처리). 커밋이 아직 없는 진짜 저장소(git diff HEAD 실패)도 같다.
def test_tree_is_left_out_when_git_cannot_read_it(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    path = tmp_path / "reports" / "test_ledger.jsonl"

    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    def failing(cmd, **kwargs):
        raise subprocess.CalledProcessError(128, cmd)

    with monkeypatch.context() as patched:
        patched.setattr(ledger.subprocess, "run", missing)
        assert ledger.git_tree(tmp_path) is None and ledger.ledger_tree(path) is None
        entry = ledger.record_test_access(path, stage="1", config="c", commit="x")
        patched.setattr(ledger.subprocess, "run", failing)
        assert ledger.git_tree(tmp_path) is None and ledger.ledger_tree(path) is None
    assert list(entry) == TREE_KEYS[:-1]
    assert json.loads(path.read_text(encoding="utf-8")) == entry
    if shutil.which("git") is not None:
        git_in(tmp_path, "init", "-q")  # 무시되지 않는 장부지만 커밋이 없어 diff HEAD가 실패한다
        assert ledger.ledger_tree(path) is None
        assert "tree" not in ledger.record_test_access(path, stage="2", config="c", commit="x")


# 스레드 8개가 Barrier로 동시에 출발해 50줄씩, 모두 400줄을 쓴 뒤 한 줄도 빠지거나 겹치지 않는지 본다.
# Windows에서는 덧붙이기가 원자적이지 않아 두 기록자가 같은 끝 위치에 쓰면 한 줄이 덮일 수 있으므로, 잠금
# 파일로 한 번에 하나씩만 쓰게 한 설계(ledger._locked docstring)를 확인하는 테스트다. 스레드 안의 예외는
# errors에 모아 메인 스레드에서 단언한다.
def test_concurrent_appends_from_threads_are_all_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LOCK_TIMEOUT_S", CONCURRENT_LOCK_TIMEOUT_S)
    path = tmp_path / "ledger.jsonl"
    n_threads, n_each = 8, 50
    errors: list[BaseException] = []
    start = threading.Barrier(n_threads)

    def work(tag: int) -> None:
        try:
            start.wait()
            for i in range(n_each):
                ledger.record_test_access(path, stage=f"{tag}-{i}", config="c", note="동시 쓰기", commit="x")
        except BaseException as err:  # noqa: BLE001 - 아래 단언에서 보고한다
            errors.append(err)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    check_ledger(path, {f"{t}-{i}" for t in range(n_threads) for i in range(n_each)})  # 400줄


# 같은 검사를 별도 프로세스 4개(60줄씩)로 한다. 실제로는 서로 다른 측정 실행(프로세스)이 같은 장부에 쓸 수
# 있으므로 스레드만으로는 부족하다. 자식이 설치 여부와 관계없이 이 저장소의 defect_inspect를 import하도록
# PYTHONPATH 앞에 패키지 상위 폴더를 넣는다. 자식이 모두 준비 파일을 만들 때까지 기다린 뒤 go 파일로 동시에
# 출발시키고, 끝나면 남은 자식을 kill해 테스트가 멈춘 프로세스를 남기지 않게 한다.
def test_concurrent_appends_from_processes_are_all_kept(tmp_path):
    path = tmp_path / "ledger.jsonl"
    go = tmp_path / "go"
    n_procs, n_each = 4, 60
    package_parent = str(Path(ledger.__file__).resolve().parents[1])
    inherited = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([package_parent, *inherited]))
    ready = [tmp_path / f"ready-{p}" for p in range(n_procs)]
    wait = str(CONCURRENT_LOCK_TIMEOUT_S)
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", CHILD, str(path), str(p), str(n_each), str(ready[p]), str(go), wait],
            env=env,
        )
        for p in range(n_procs)
    ]
    try:
        deadline = time.monotonic() + 60
        while not all(r.exists() for r in ready):
            assert time.monotonic() < deadline and all(p.poll() is None for p in procs)
            time.sleep(0.01)
        go.write_bytes(b"")  # 모든 자식이 같은 순간에 기록을 시작한다
        assert [p.wait(timeout=120) for p in procs] == [0] * n_procs
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    check_ledger(path, {f"{p}-{i}" for p in range(n_procs) for i in range(n_each)})


# 다른 기록자가 잠금을 쥐고 있으면(잠금 파일을 미리 만들어 흉내 냄) 바로 실패하지 않고 기다렸다가, 0.3초 뒤
# 잠금이 풀리면 기록하는지 본다. 걸린 시간이 0.25초 이상이면 실제로 기다렸다는 뜻이다. 잠금 파일은 장부와 같은
# 폴더의 다른 이름이어야 한다.
def test_append_waits_for_a_lock_that_is_released(tmp_path):
    path = tmp_path / "ledger.jsonl"
    lock = ledger.lock_path(path)
    assert lock.parent == path.parent and lock != path
    lock.write_bytes(b"")  # 다른 기록자가 잠금을 쥐고 있다가 ...
    release = threading.Timer(0.3, lock.unlink)  # ... 0.3초 뒤에 놓는다
    release.start()
    started = time.monotonic()
    ledger.record_test_access(path, stage="1", config="c", commit="x")
    assert time.monotonic() - started >= 0.25
    release.join()
    check_ledger(path, {"1"})


# 잠금이 풀리지 않으면 대기 시간(여기서는 0.2초로 줄임) 뒤 TimeoutError를 내고 장부에는 아무것도 쓰지 않는지
# 본다. 남의 잠금 파일은 지우지 않아야 한다. 기록이 실패하면 예외로 호출한 실행이 테스트를 읽기 전에 멈추는
# 것이 장부의 설계다.
def test_held_lock_times_out_and_nothing_is_written(tmp_path, monkeypatch):
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"kept": true}\n')
    lock = ledger.lock_path(path)
    lock.write_bytes(b"")  # 방금 만든 잠금이라 오래된 잠금으로 보지 않는다
    monkeypatch.setattr(ledger, "LOCK_TIMEOUT_S", 0.2)
    with pytest.raises(TimeoutError, match="lock"):
        ledger.record_test_access(path, stage="1", config="c", commit="x")
    assert path.read_bytes() == b'{"kept": true}\n'
    assert lock.exists()  # 다른 기록자의 잠금이므로 남아 있어야 한다


# 기록 도중 죽은 프로세스가 남긴 오래된 잠금 파일(수정 시각을 LOCK_STALE_S의 두 배 전으로 돌림)은 지우고
# 기록을 이어 가는지 본다. 그러지 않으면 한 번의 비정상 종료로 장부가 영원히 잠긴다. 상수 값(오래된 잠금 60초,
# 대기 10초)도 함께 고정해 무심코 바꾸지 않게 한다.
def test_stale_lock_is_broken(tmp_path):
    path = tmp_path / "ledger.jsonl"
    lock = ledger.lock_path(path)
    lock.write_bytes(b"")
    old = time.time() - 2 * ledger.LOCK_STALE_S
    os.utime(lock, (old, old))  # 비정상 종료한 기록자가 남긴 잠금
    assert ledger.LOCK_STALE_S == 60 and ledger.LOCK_TIMEOUT_S == 10
    ledger.record_test_access(path, stage="1", config="c", commit="x")
    check_ledger(path, {"1"})


# 장부 경로가 폴더라서 쓰기가 실패해도 OSError가 그대로 올라오고 잠금 파일은 지워지는지 본다(_locked의
# finally). 잠금이 남으면 다음 실행이 10초를 기다린 뒤 실패한다.
def test_lock_is_released_when_the_append_fails(tmp_path):
    path = tmp_path / "a_directory"
    path.mkdir()
    with pytest.raises(OSError):
        ledger.record_test_access(path, stage="1", config="c", commit="x")
    assert not ledger.lock_path(path).exists()


# JSON으로 바꿀 수 없는 값(object())이 들어오면 TypeError가 나고, 장부 파일도 잠금 파일도 생기지 않는지 본다.
# 직렬화를 잠금을 잡기 전에 끝내는 구현 순서를 확인하는 것이다.
def test_entry_that_cannot_be_serialised_writes_nothing(tmp_path):
    path = tmp_path / "ledger.jsonl"
    with pytest.raises(TypeError):
        ledger.record_test_access(path, stage="1", config="c", note=object(), commit="x")
    assert not path.exists() and not ledger.lock_path(path).exists()


# 이전 기록자가 줄 중간에서 죽어 마지막 줄이 줄바꿈 없이 끊겨 있으면, 새 기록 앞에 줄바꿈을 붙여 새 줄이 끊긴
# 줄에 이어 붙지 않는지 본다. 끊긴 줄은 그대로 두고(지우지 않음) 새 기록은 독립된 한 줄로 읽혀야 한다.
def test_entry_is_not_glued_to_a_torn_last_line(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(b'{"time": "cut off')  # 이전 기록자가 줄 중간에서 죽었다
    entry = ledger.record_test_access(path, stage="1", config="c", commit="x")
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    lines = raw.decode("utf-8").splitlines()
    assert lines[0] == '{"time": "cut off' and json.loads(lines[1]) == entry and len(lines) == 2


# git_commit과 ledger_tree가 잠금 파일을 만들기 전에 불리는지 본다. 잠금 파일은 저장소 안 장부 옆에 생기므로,
# 그 뒤에 git status를 보면 추적하지 않는 파일로 잡혀 깨끗한 코드도 '+dirty'로, 트리 목록에도 잠금 파일이
# 잘못 기록된다. 가짜 함수가 불린 순간 잠금 파일이 있었는지를 seen에 남겨 확인한다.
def test_git_state_is_read_before_the_lock_is_taken(tmp_path, monkeypatch):
    # 잠금 파일은 저장소 안 장부 옆에 생기므로 `git status`가 그것을 보면 안 된다.
    path = tmp_path / "ledger.jsonl"
    seen = []

    def fake_commit(root=None):
        seen.append(("commit", ledger.lock_path(path).exists()))
        return "abc1234"

    def fake_tree(where):
        seen.append(("tree", ledger.lock_path(path).exists()))
        return FAKE_TREE

    monkeypatch.setattr(ledger, "git_commit", fake_commit)
    monkeypatch.setattr(ledger, "ledger_tree", fake_tree)
    entry = ledger.record_test_access(path, stage="1", config="c")
    assert (entry["commit"], entry["tree"]) == ("abc1234", FAKE_TREE) and list(entry) == TREE_KEYS
    assert seen == [("commit", False), ("tree", False)]
