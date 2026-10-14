"""재구성 아티팩트 하나의 CPU 런타임 대조: onnxruntime 대 OpenVINO(점수 동일성, 그다음 지연).

팔(docs/experiments.md "CPU 런타임 대조", 2026-10-04 측정 전에 등록):

- R0: onnxruntime CPUExecutionProvider, intra-op 스레드는 기본값(서빙 설정)
- R1: onnxruntime, intra-op 스레드 12개(대조)
- R2: 같은 ONNX 파일을 읽는 CPU OpenVINO: PERFORMANCE_HINT=LATENCY,
  INFERENCE_PRECISION_HINT=f32, NUM_STREAMS=1

`parity`는 dev 평가 세트(겹 0 정상 + dev 결함)를 팔마다 이미지 한 장씩 채점해 점수, 맵, AUROC, 홀드아웃
판정을 R0와 비교한다(R0는 기록된 dev 실행과 비교). `latency`는 PC가 한가해질 때까지 기다린 뒤, 순서가
돌아가는 블록으로 팔들의 시간을 잰다. 실행마다 새 프로세스(`run-arm`)를 띄운다. 두 명령 모두 봉인 테스트
세트를 읽지 않는다. OpenVINO는 OpenVINO 팔을 만들 때만 가져온다(의존성 그룹 `openvino`).
"""

# [파일 역할] 검사 서비스 모델(dms-280-car, Dinomaly ViT-S 280px + CAR의 FP32 ONNX)을 CPU 런타임 세 팔로 돌려
# 비교한다. parity는 dev 평가 세트 1,398장으로 점수·맵·AUROC·임계값 판정이 같은지, latency는 한가한 PC에서
# 지연이 얼마나 다른지 잰다. run-arm은 latency가 팔 하나를 새 프로세스에서 재려고 부르는 내부 명령이다.
# [왜 필요한가] 같은 ONNX를 OpenVINO로 돌리면 더 빠르다는 근거와 AMD CPU에서는 onnxruntime이 같거나 낫다는
# 안내가 엇갈려, 측정 PC(Ryzen 5 3600)에서 직접 잰다(docs/experiments.md "CPU 런타임 대조"). 빠르더라도
# 점수가 달라지면 임계값을 쓸 수 없으므로 점수 동일성 관문을 먼저 둔다.
# [설계 이유] 세 팔 모두 같은 ReconstructionInspector(전처리, 점수·맵 추출)를 쓰고 세션만 바꾼다. OpenVINO는
# onnxruntime 세션과 같은 run(names, feeds) 모양의 감싸개(OpenVINOSession)로 끼워 넣어 나머지 코드를 그대로
# 둔다. 지연은 팔 순서를 블록마다 돌리고 실행마다 새 프로세스를 띄워, 순서 효과와 앞 팔의 스레드 풀이 끼어드는
# 것을 막는다.
# [흐름] parity: 범주마다 팔을 적재 → 이미지마다 세 팔이 차례로 채점 → R0를 6단계 기록과, R1·R2를 R0와 비교 →
# 관문 (가)~(라) → reports/runtime/parity.json. latency: 한가해질 때까지 기다림 → 리사이즈 시간 → 블록 3개 ×
# 팔 3개를 각각 새 프로세스로 → 요약과 H22 → reports/runtime/latency.json.
# [연결] inspector.ReconstructionInspector(서비스와 같은 추론 경로), calibrate.conformal_threshold(홀드아웃
# 임계값), metrics.auroc, bench.time_resize·cpu_name(4·6단계와 같은 지연 측정 방식). 결과는 README의 CPU
# 런타임 줄과 experiments.md 결과 절에 옮겨졌다.

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# openvino와 onnxruntime은 팔을 만들 때만 함수 안에서 가져온다(openvino는 별도 의존성 그룹이라 서비스 이미지에
# 없다).
import numpy as np

# 내부 모듈: 경로, 홀드아웃 임계값 계산, ONNX 파일 sha256, 재구성 모델 검사기와 메타 검사, 커밋 해시, AUROC.
from . import paths
from .calibrate import conformal_threshold
from .download import sha256_file
from .inspector import RECONSTRUCTION_PRECISIONS, ReconstructionInspector, check_reconstruction_meta
from .ledger import git_commit
from .metrics import auroc

# 등록한 세 팔. R0 = 지금 서빙 설정(onnxruntime 스레드 기본값), R1 = onnxruntime 스레드 12(SMT 포함, 설정
# 대조), R2 = OpenVINO(지연 우선 힌트, f32 정밀도 고정, 스트림 1). R2의 INFERENCE_PRECISION_HINT=f32는 추론
# 정밀도를 f32로 고정해, 다른 두 팔과 같은 FP32 계산으로 비교하게 한다.
ARMS: dict[str, dict] = {
    "R0": {"runtime": "onnxruntime", "threads": None},
    "R1": {"runtime": "onnxruntime", "threads": 12},
    "R2": {
        "runtime": "openvino",
        "config": {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32", "NUM_STREAMS": "1"},
    },
}
# 관문 기준값(규칙 그대로): ALPHA 목표 오검출률 5%, (가) 점수 상대 차이 ≤ 1e-3, (나) 맵 절대 차이 ≤ 1e-3, (다)
# dev 평균 AUROC 차이 ≤ 0.01%p, R0와 6단계 기록의 차이 ≤ 1e-6, H22의 속도 이득 10% 이상, 한가함 기준 CPU·GPU
# 10% 미만.
# OV_PROPERTIES: 컴파일된 OpenVINO 모델에서 실제로 적용된 설정(스레드 수, 하이퍼스레딩 등)을 읽어 리포트에
# 남길 속성 이름들.
REFERENCE = "R0"
ALPHA = 0.05
SCORE_REL_TOL = 1e-3  # (가) R0 대비 이미지 점수 상대 차이의 최댓값
MAP_ABS_TOL = 1e-3  # (나) R0 대비 맵 절대 차이의 최댓값
AUROC_TOL_PP = 0.01  # (다) dev 범주 평균 이미지 AUROC 차이, %p
RECORD_REL_TOL = 1e-6  # R0 대 기록된 dev 실행(같은 파일, 같은 런타임)
MIN_SPEEDUP = 0.10  # H22
CPU_IDLE_PCT = 10.0
GPU_IDLE_PCT = 10.0
OV_PROPERTIES = (
    "PERFORMANCE_HINT",
    "INFERENCE_PRECISION_HINT",
    "NUM_STREAMS",
    "INFERENCE_NUM_THREADS",
    "ENABLE_HYPER_THREADING",
    "ENABLE_CPU_PINNING",
)


# 리포트를 UTF-8·LF로 쓴다. 상위 폴더(reports/runtime)가 없으면 만든다.
def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


# ---------------------------------------------------------------- 팔


# OpenVINO를 onnxruntime 세션처럼 보이게 하는 감싸개. ReconstructionInspector가 쓰는 메서드는 run(출력 이름
# 목록, 입력 dict) 하나뿐이라 이것만 맞춘다. ONNX 파일을 IR로 변환하지 않고 직접 읽어, 세 팔이 정확히 같은
# 파일을 쓰게 한다. 추론 요청 객체 하나를 계속 재사용한다(세션 하나와 같은 조건).
class OpenVINOSession:
    """`ReconstructionInspector`가 쓰는 onnxruntime 세션의 부분을 OpenVINO로 구현한다.

    ONNX 파일을 직접 읽는다(IR로 변환하지 않음). 추론 요청 하나를 세션 하나처럼 계속 재사용한다.
    """

    def __init__(self, model_path: Path, config: dict, device: str = "CPU"):
        import openvino as ov

        core = ov.Core()
        self.compiled = core.compile_model(core.read_model(str(model_path)), device, dict(config))
        self.request = self.compiled.create_infer_request()
        self.version = ov.__version__
        self.properties = {}
        # 플러그인이 알려 주지 않는 속성은 None으로 남기고 넘어간다(실패로 보지 않음).
        for name in OV_PROPERTIES:
            try:
                self.properties[name] = str(self.compiled.get_property(name))
            except Exception:  # 이 플러그인이 알려 주지 않는 속성
                self.properties[name] = None

    # onnxruntime의 session.run과 같은 모양: 요청한 출력 이름 순서대로 numpy 배열 목록을 돌려준다.
    def run(self, names: list[str], feeds: dict) -> list[np.ndarray]:
        result = self.request.infer(feeds)
        return [np.array(result[self.compiled.output(name)]) for name in names]


# 모델 파일 경로. 서비스 아티팩트는 ONNX 하나를 12개 범주가 같이 쓰고 범주 폴더에는 임계값만 있으므로, 범주
# 폴더에 없으면 부모 폴더의 파일을 쓴다(ReconstructionInspector.load와 같은 규칙).
def _model_path(category_dir: Path, precision: str = "fp32") -> Path:
    """범주 폴더의 모델 파일. `ReconstructionInspector.load`처럼 부모 폴더에 있을 수도 있다."""
    name = RECONSTRUCTION_PRECISIONS[precision]
    path = Path(category_dir) / name
    return path if path.exists() else Path(category_dir).parent / name


# 팔 이름과 범주 폴더로 FP32 검사기를 만든다. onnxruntime 팔은 서비스와 같은 load(스레드 수만 다름)를,
# OpenVINO 팔은 메타를 같은 함수로 검사한 뒤 OpenVINOSession을 끼운 검사기를 만든다.
def load_arm(arm: str, category_dir: Path) -> ReconstructionInspector:
    """팔 하나에서 범주 하나의 FP32 검사기."""
    spec = ARMS[arm]
    if spec["runtime"] == "onnxruntime":
        return ReconstructionInspector.load(category_dir, precision="fp32", threads=spec["threads"])
    with open(Path(category_dir) / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    check_reconstruction_meta(meta)
    inspector = ReconstructionInspector(OpenVINOSession(_model_path(category_dir), spec["config"]), meta)
    inspector.precision = "fp32"
    return inspector


# 리포트에 남길 팔 정보: 런타임 이름·버전, onnxruntime 스레드 설정 또는 OpenVINO 설정과 컴파일된 실제 값.
def arm_info(arm: str, inspector: ReconstructionInspector) -> dict:
    spec = ARMS[arm]
    info: dict = {"runtime": spec["runtime"]}
    if spec["runtime"] == "onnxruntime":
        import onnxruntime as ort

        info.update(version=ort.__version__, intra_op_threads=spec["threads"])
    else:
        info.update(version=inspector.session.version, config=spec["config"])
        info["compiled"] = inspector.session.properties
    return info


# ---------------------------------------------------------------- 점수 동일성(parity)


# 최대 상대 차이. 분모가 0에 가까울 때 터지지 않게 1e-12로 막는다.
def max_rel_diff(values: np.ndarray, ref: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values - ref) / np.maximum(np.abs(ref), 1e-12)))


# 범주 하나에서 팔의 점수를 R0 점수와 비교한다(라벨 0 = 겹 0 정상, 1 = dev 결함).
# (라) 관문은 "각 팔이 자기 점수로 다시 잡은 홀드아웃 임계값"이 R0와 같은 겹 0 정상을 표시하는지 본다. 저장된
# 임계값은 R0 점수 중 하나라서, 다른 팔 점수에 그대로 쓰면 임계값 자리의 이미지가 마지막 비트 차이만으로
# 판정이 바뀔 수 있기 때문이다(docs/experiments.md 런타임 대조 (1)절). 런타임을 바꾸면 임계값도 그 런타임
# 점수로 다시 잡으므로 이것이 실제 운영에 맞는 비교다.
# 저장된 임계값을 그대로 썼을 때 판정이 바뀌는 수와, 자기 임계값에서 결함 판정이 바뀌는 수는 판정 없이 기록만
# 한다.
def category_parity(ref: np.ndarray, arm: np.ndarray, labels: np.ndarray, stored_threshold: float) -> dict:
    """범주 하나: 같은 이미지에서 한 팔의 이미지 점수를 R0의 점수와 비교한다(라벨 0 = 겹 0 정상)."""
    ref = np.asarray(ref, dtype=np.float32)
    arm = np.asarray(arm, dtype=np.float32)
    labels = np.asarray(labels)
    normal = labels == 0
    thr_ref = conformal_threshold(ref[normal], ALPHA)
    thr_arm = conformal_threshold(arm[normal], ALPHA)
    flags_ref = ref > thr_ref.value
    flags_arm = arm > thr_arm.value
    return {
        "images": int(labels.size),
        "normal": int(normal.sum()),
        "defect": int((~normal).sum()),
        "max_rel_diff": max_rel_diff(arm, ref),
        "auroc_ref": auroc(ref[normal], ref[~normal]),
        "auroc_arm": auroc(arm[normal], arm[~normal]),
        "threshold_ref": thr_ref.value,
        "threshold_arm": thr_arm.value,
        "threshold_rank": thr_ref.rank,
        # (라): 팔이 자기 점수로 잡은 홀드아웃 임계값이 R0 자신의 것과 같은 겹 0 정상을 표시하는지.
        "fold0_flags_ref": int(flags_ref[normal].sum()),
        "fold0_flags_same": bool(np.array_equal(flags_ref[normal], flags_arm[normal])),
        # 기록만 한다: 자기 임계값에서 바뀌는 결함 판정 수, 저장된 임계값에서 바뀌는 전체 이미지 판정 수.
        "defect_verdict_flips_own": int((flags_ref[~normal] != flags_arm[~normal]).sum()),
        "verdict_flips_stored": int(((ref > stored_threshold) != (arm > stored_threshold)).sum()),
    }


# 범주별 결과를 팔 하나의 관문 (가)~(라)로 모은다: 점수 상대 차이의 최댓값, 맵 절대 차이, 12개 범주 평균 AUROC
# 차이(%p), 12개 범주 모두에서 겹 0 표시가 같은지. 넷 다 통과해야 pass다.
def parity_gate(categories: dict[str, dict], map_abs: float) -> dict:
    """팔 하나의 범주별 결과를 등록된 관문 (가)~(라)로 모은다."""
    rel = max(c["max_rel_diff"] for c in categories.values())
    auroc_ref = 100.0 * float(np.mean([c["auroc_ref"] for c in categories.values()]))
    auroc_arm = 100.0 * float(np.mean([c["auroc_arm"] for c in categories.values()]))
    checks = {
        "a_score_rel": {"value": rel, "limit": SCORE_REL_TOL, "pass": rel <= SCORE_REL_TOL},
        "b_map_abs": {"value": map_abs, "limit": MAP_ABS_TOL, "pass": map_abs <= MAP_ABS_TOL},
        "c_macro_auroc_pp": {
            "value": auroc_arm - auroc_ref,
            "arm": auroc_arm,
            "ref": auroc_ref,
            "limit": AUROC_TOL_PP,
            "pass": abs(auroc_arm - auroc_ref) <= AUROC_TOL_PP,
        },
        "d_fold0_flags": {
            "same_in": sum(c["fold0_flags_same"] for c in categories.values()),
            "categories": len(categories),
            "pass": all(c["fold0_flags_same"] for c in categories.values()),
        },
    }
    return {
        "checks": checks,
        "pass": all(v["pass"] for v in checks.values()),
        "defect_verdict_flips_own": sum(c["defect_verdict_flips_own"] for c in categories.values()),
        "verdict_flips_stored": sum(c["verdict_flips_stored"] for c in categories.values()),
    }


# parity 명령. dev 평가 세트만 쓰고 봉인 테스트는 읽지 않는다. 팔 목록은 R0로 시작해야 한다(모든 비교의 기준).
def _parity_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from .cache import ImageCache
    from .splits import read_manifest, select

    art = args.artifacts
    arms = args.arms
    if arms[0] != REFERENCE or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        parser.error(f"arms must start with {REFERENCE} and be distinct names from {sorted(ARMS)}")
    started = time.perf_counter()
    manifest = read_manifest(paths.VISA_MANIFEST)
    model_path = art / RECONSTRUCTION_PRECISIONS["fp32"]
    with open(args.recorded / "run.json", encoding="utf-8") as f:
        recorded_run = json.load(f)
    size = None
    errors: dict[str, str] = {}
    per_arm: dict[str, dict] = {a: {} for a in arms}
    map_abs = {a: 0.0 for a in arms if a != REFERENCE}
    record_rel = 0.0
    seconds = {a: 0.0 for a in arms}
    info: dict[str, dict] = {}
    # R0가 적재에 실패하면 비교 자체가 불가능하므로 예외를 그대로 올린다. 다른 팔이 이 모델을 읽거나
    # 컴파일하지 못하면 그 팔만 오류로 기록하고 관문 불통과로 적는다(규칙).
    for category in args.categories:
        inspectors = {}
        for arm in arms:
            if arm in errors:
                continue
            try:
                inspectors[arm] = load_arm(arm, art / category)
            except Exception as err:  # 이 모델을 적재하지 못하는 팔은 관문을 통과하지 못한다
                if arm == REFERENCE:
                    raise
                errors[arm] = f"{type(err).__name__}: {err}"
                print(json.dumps({"arm": arm, "error": errors[arm]}), flush=True)
        for arm, insp in inspectors.items():
            info.setdefault(arm, arm_info(arm, insp))
        size = size or inspectors[REFERENCE].size
        normal = select(manifest, protocol="dev", part="eval_normal", category=category)
        defect = select(manifest, protocol="dev", part="eval_defect", category=category)
        rows = normal + defect
        labels = np.array([0] * len(normal) + [1] * len(defect), dtype=np.int8)
        # 6단계 기록(dinomaly_serving score --protocol dev)이 같은 이미지 목록인지 확인하고 그 점수를 읽는다.
        # R0가 이 기록과 같아야 환경이 바뀌지 않았다고 본다.
        with np.load(args.recorded / f"{category}.npz") as z:
            if z["eval_images"].tolist() != [r.image for r in rows]:
                raise RuntimeError(f"{args.recorded}/{category}.npz lists other images")
            recorded = z["eval_score"].astype(np.float32)
        images = ImageCache(paths.CACHE, size).images(rows)
        scores = {arm: np.empty(len(rows), dtype=np.float32) for arm in inspectors}
        # 이미지 한 장을 세 팔이 차례로 처리하고(같은 입력), 맵은 저장하지 않고 R0와의 최대 절대 차이만 누적해
        # 메모리를 아낀다. 팔별 채점 시간도 함께 더한다(참고용).
        for i, image in enumerate(images):
            maps = {}
            for arm, insp in inspectors.items():
                t0 = time.perf_counter()
                scores[arm][i], maps[arm] = insp.run(image)
                seconds[arm] += time.perf_counter() - t0
            for arm in maps:
                if arm != REFERENCE:
                    diff = float(np.max(np.abs(maps[arm] - maps[REFERENCE])))
                    map_abs[arm] = max(map_abs[arm], diff)
        record_rel = max(record_rel, max_rel_diff(scores[REFERENCE], recorded))
        # 아티팩트에 저장된 이 범주의 임계값(R0 점수로 정한 값). 판정 없이 "그대로 썼을 때 바뀌는 수"를 세는
        # 데만 쓴다.
        stored = inspectors[REFERENCE].threshold
        for arm in inspectors:
            per_arm[arm][category] = category_parity(scores[REFERENCE], scores[arm], labels, stored)
        line = {a: round(per_arm[a][category]["max_rel_diff"], 9) for a in inspectors if a != REFERENCE}
        print(json.dumps({"category": category, "images": len(rows), "max_rel_diff": line}), flush=True)

    arms_report = {}
    for arm in arms:
        if arm == REFERENCE:
            continue
        if arm in errors:
            arms_report[arm] = {"pass": False, "error": errors[arm]}
            continue
        arms_report[arm] = {
            **info[arm],
            **parity_gate(per_arm[arm], map_abs[arm]),
            "score_seconds": round(seconds[arm], 1),
            "categories": per_arm[arm],
        }
    # 리포트: 커밋, 아티팩트 경로(저장소 기준 상대 경로), ONNX sha256, R0의 기록 재현 여부, 허용치, 팔별 관문
    # 결과와 범주별 수치.
    report = {
        "check": "CPU runtime parity on the dev evaluation set (docs/experiments.md, 2026-10-04)",
        "commit": git_commit(),
        "artifacts": _rel(art),
        "onnx_sha256": sha256_file(model_path),
        "reference": {
            **info[REFERENCE],
            "arm": REFERENCE,
            "score_seconds": round(seconds[REFERENCE], 1),
            "recorded_run": _rel(args.recorded),
            "recorded_onnx_sha256": recorded_run["config"]["onnx_sha256"],
            "max_rel_diff_vs_record": record_rel,
            "matches_record": record_rel <= RECORD_REL_TOL,
        },
        "images": sum(c["images"] for c in per_arm[REFERENCE].values()),
        "tolerances": {
            "score_rel": SCORE_REL_TOL,
            "map_abs": MAP_ABS_TOL,
            "auroc_pp": AUROC_TOL_PP,
            "record_rel": RECORD_REL_TOL,
            "alpha": ALPHA,
        },
        "arms": arms_report,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    _write_json(args.out, report)
    summary = {a: v.get("pass") for a, v in arms_report.items()}
    print(json.dumps({"matches_record": report["reference"]["matches_record"], "pass": summary}), flush=True)


# ---------------------------------------------------------------- PC 부하


# Windows FILETIME 구조체(상위·하위 32비트)를 하나의 정수(100ns 단위)로 합친다.
def _filetime(ft) -> int:
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


# Windows GetSystemTimes로 (유휴 시간, 커널 + 사용자 시간)을 읽는다. 커널 시간에 유휴 시간이 포함되므로 둘의
# 비율로 전체 CPU 사용률을 낼 수 있다. 외부 패키지(psutil) 없이 ctypes만 쓴다. Windows가 아니면 None이다.
def _system_times() -> tuple[int, int] | None:
    """Windows에서 (유휴, 커널 + 사용자) 시간을 100ns 단위로 돌려준다. 커널 시간에 유휴 시간이 들어 있다."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    if not ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
    ):
        return None
    return _filetime(idle), _filetime(kernel) + _filetime(user)


# 두 시점 사이의 CPU 사용률 = 1 − (유휴 증가분 / 전체 증가분). 값을 못 읽었거나 시간이 흐르지 않았으면 None.
def busy_percent(before: tuple[int, int] | None, after: tuple[int, int] | None) -> float | None:
    if before is None or after is None or after[1] <= before[1]:
        return None
    return 100.0 * (1.0 - (after[0] - before[0]) / (after[1] - before[1]))


def cpu_percent(seconds: float) -> float | None:
    """앞으로 `seconds`초 동안의 전체 CPU 평균 사용률(읽을 수 없으면 None)."""
    before = _system_times()
    time.sleep(seconds)
    return busy_percent(before, _system_times())


# nvidia-smi로 첫 GPU의 사용률을 읽는다. 명령이 없거나 실패하면 None(GPU 조건은 건너뜀).
def gpu_percent() -> float | None:
    """nvidia-smi로 읽은 첫 GPU의 사용률, 또는 None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


# 1분 동안의 CPU 사용률과, 5초마다 12번 읽은 GPU 사용률의 평균. 한가함 판단의 단위다(규칙: 시작 전 1분).
def load_minute(step_s: float = 5.0, steps: int = 12) -> dict:
    """1분 동안의 CPU 사용률과, `step_s`초마다 읽은 GPU 사용률의 평균."""
    before = _system_times()
    gpu = []
    for _ in range(steps):
        time.sleep(step_s)
        value = gpu_percent()
        if value is not None:
            gpu.append(value)
    return {
        "cpu_pct": busy_percent(before, _system_times()),
        "gpu_pct": float(np.mean(gpu)) if gpu else None,
        "gpu_samples": len(gpu),
    }


# CPU가 10% 미만이고 GPU도 10% 미만(GPU 값을 못 읽으면 CPU만)이면 한가하다. CPU 값을 못 읽으면 한가하지 않은
# 것으로 본다.
def is_idle(minute: dict, cpu_limit: float = CPU_IDLE_PCT, gpu_limit: float = GPU_IDLE_PCT) -> bool:
    cpu, gpu = minute.get("cpu_pct"), minute.get("gpu_pct")
    return cpu is not None and cpu < cpu_limit and (gpu is None or gpu < gpu_limit)


# 1분씩 재며 한가해질 때까지 최대 max_minutes분 기다린다. 끝내 한가하지 않으면 False와 기록을 돌려주고, 호출한
# 쪽은 그대로 재되 리포트에 조건을 남긴다(규칙). 1분을 먼저 잰 뒤 그 끝 시각을 기록하도록 2026-10-04에 순서를
# 고쳤다. minute와 clock을 인자로 받아 테스트가 가짜 측정을 넣을 수 있다.
def wait_idle(max_minutes: int, minute=load_minute, clock=time.strftime) -> tuple[bool, list[dict]]:
    """PC가 한가해지거나 `max_minutes`분이 지날 때까지 1분씩 잰다."""
    log = []
    for i in range(max_minutes):
        load = minute()  # 먼저 잰다: `end`는 그 1분이 끝난 시각이다(2026-10-04 수정)
        record = {"minute": i + 1, "end": clock("%Y-%m-%d %H:%M:%S"), **load}
        record["idle"] = is_idle(record)
        log.append(record)
        print(json.dumps(record), flush=True)
        if record["idle"]:
            return True, log
    return False, log


# ---------------------------------------------------------------- 지연(latency)


# 블록 b에서는 팔 순서를 b칸 돌린다(1: R0 R1 R2, 2: R1 R2 R0, 3: R2 R0 R1). 팔 수만큼 블록을 돌면 모든 팔이
# 모든 자리를 한 번씩 맡아 순서 효과(먼저 돈 팔이 유리·불리한 것)가 고르게 나뉜다.
def block_orders(arms: list[str], blocks: int) -> list[list[str]]:
    """블록 b는 팔 순서를 b칸 돌려 돌린다. len(arms)개 블록에서 모든 팔이 모든 자리를 한 번씩 맡는다."""
    return [arms[b % len(arms) :] + arms[: b % len(arms)] for b in range(blocks)]


# 워밍업 뒤 inspector.run(정규화 + 모델 호출) 한 번씩의 시간을 ms로 모은다. 이미지는 돌아가며 쓴다.
# bench.time_reconstruction과 같은 방식이지만 요약하지 않은 원값을 돌려준다(블록을 합친 중앙값을 내기 위해).
def time_calls(inspector, images: np.ndarray, warmup: int, repeats: int) -> list[float]:
    """측정한 `inspector.run` 호출마다의 밀리초(`bench.time_reconstruction`과 같은 방식, 요약 전 원값)."""
    if len(images) == 0 or warmup < 0 or repeats < 1:
        raise ValueError("need images, warmup >= 0 and repeats >= 1")
    out = []
    for i in range(warmup + repeats):
        image = images[i % len(images)]
        t0 = time.perf_counter()
        inspector.run(image)
        t1 = time.perf_counter()
        if i >= warmup:
            out.append((t1 - t0) * 1e3)
    return out


# 지연 측정 이미지: 범주(기본 pcb1)의 dev 정상 풀 앞 n장(4·6단계와 같은 조건).
def _latency_images(category: str, size: int, n: int) -> np.ndarray:
    from .cache import ImageCache
    from .splits import read_manifest, select

    rows = select(read_manifest(paths.VISA_MANIFEST), protocol="dev", part="pool_normal", category=category)
    return ImageCache(paths.CACHE, size).images(rows[:n])


# run-arm 명령: 이 프로세스에서 팔 하나를 적재하고 시간을 잰 뒤 결과를 JSON 한 줄로 표준 출력에 낸다.
# latency가 마지막 줄을 읽어 모은다.
def _run_arm_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    t0 = time.perf_counter()
    inspector = load_arm(args.arm, args.artifacts / args.category)
    load_s = time.perf_counter() - t0
    images = _latency_images(args.category, inspector.size, args.images)
    ms = time_calls(inspector, images, args.warmup, args.repeats)
    print(
        json.dumps({"arm": args.arm, "load_s": load_s, "ms": ms, **arm_info(args.arm, inspector)}), flush=True
    )


# 팔마다 세 블록의 모든 측정(150회)을 합친 중앙값에 리사이즈 시간을 더한 값이 팔의 지연이다. 속도 이득 = 1 −
# (팔 지연 / R0 지연). 블록별 중앙값과 p95도 남겨 블록 사이 흔들림(부하)을 볼 수 있게 한다.
def summarise_latency(runs: list[dict], resize_ms: float, arms: list[str]) -> dict:
    """팔마다: 모든 블록의 측정 호출을 합친 중앙값 + 리사이즈 시간, 그리고 R0 대비 속도 이득."""
    out: dict = {}
    for arm in arms:
        mine = [r for r in runs if r["arm"] == arm]
        pooled = [v for r in mine for v in r["ms"]]
        median = float(statistics.median(pooled))
        out[arm] = {
            "inference_median_ms": median,
            "inference_p95_ms": float(np.percentile(pooled, 95)),
            "block_medians_ms": [float(statistics.median(r["ms"])) for r in mine],
            "timed_calls": len(pooled),
            "total_with_resize_ms": resize_ms + median,
        }
    ref = out[REFERENCE]["total_with_resize_ms"]
    for arm in arms:
        out[arm]["speedup_vs_r0"] = 1.0 - out[arm]["total_with_resize_ms"] / ref
        out[arm]["inference_speedup_vs_r0"] = (
            1.0 - out[arm]["inference_median_ms"] / out[REFERENCE]["inference_median_ms"]
        )
    return out


# H22 판정: 관문을 통과하고 같은 세션에서 10% 이상 빠르면 후보. R0가 6단계 기록을 재현하지 못했다면 환경이
# 바뀐 것이므로 어느 팔의 관문도 인정하지 않는다. parity 리포트가 없으면 관문 결과는 None으로 둔다.
def h22(parity: dict | None, latency: dict) -> dict:
    """도전 팔마다 H22: 점수 동일성 관문을 통과하고, 이 세션에서 R0보다 10% 이상 빠른가.

    점수 동일성 관문은 R0 자신이 기록된 dev 점수를 재현했을 때만 인정한다.
    """
    out = {}
    for arm, lat in latency.items():
        if arm == REFERENCE:
            continue
        passed = None
        if parity is not None:
            reference_ok = bool(parity.get("reference", {}).get("matches_record", False))
            passed = reference_ok and bool(parity.get("arms", {}).get(arm, {}).get("pass", False))
        faster = lat["speedup_vs_r0"] >= MIN_SPEEDUP
        out[arm] = {
            "parity_pass": passed,
            "speedup_vs_r0": lat["speedup_vs_r0"],
            "faster_by_10pct": faster,
            "candidate": bool(passed and faster),
        }
    return out


# run-arm을 새 파이썬 프로세스로 부를 명령줄. 같은 인터프리터(sys.executable)로 같은 모듈을 실행한다.
def _run_arm_cmd(args: argparse.Namespace, arm: str) -> list[str]:
    options = {
        "--arm": arm,
        "--artifacts": str(args.artifacts),
        "--category": args.category,
        "--images": str(args.images),
        "--warmup": str(args.warmup),
        "--repeats": str(args.repeats),
    }
    cmd = [sys.executable, "-m", "defect_inspect.runtime_compare", "run-arm"]
    for name, value in options.items():
        cmd += [name, value]
    return cmd


# 리포트(커밋되는 파일)에 로컬 절대 경로가 들어가지 않도록 저장소 안 경로는 상대 경로로 바꾼다.
def _rel(path: Path) -> str:
    """저장소 안에 있는 경로면 저장소 기준 상대 경로로 바꾼다(리포트에 로컬 경로 앞부분이 남지 않게)."""
    try:
        return Path(path).resolve().relative_to(paths.ROOT).as_posix()
    except ValueError:
        return str(path)


# latency 명령. 한가함을 기다린 뒤 세션 시작 시각을 적고, 1500×1000 사진을 입력 크기로 줄이는 시간을 한 번
# 재서 모든 팔에 더한다.
def _latency_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    from .bench import cpu_name, time_resize

    arms = args.arms
    if arms[0] != REFERENCE or len(set(arms)) != len(arms) or not set(arms) <= set(ARMS):
        parser.error(f"arms must start with {REFERENCE} and be distinct names from {sorted(ARMS)}")
    with open(args.artifacts / args.category / "meta.json", encoding="utf-8") as f:
        size = int(json.load(f)["img_size"])
    idle, wait_log = wait_idle(args.wait_max_min)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    resize_ms = time_resize(size)
    runs = []
    for b, order in enumerate(block_orders(arms, args.blocks)):
        for arm in order:
            # 실행마다 직전 5초 CPU 사용률과 GPU 사용률을 남긴다. 측정 중 다른 작업이 끼어들었는지 나중에
            # 확인하기 위해서다(experiments.md 결과 절의 정정이 이 값을 부하 근거 중 하나로 썼다). 실행별
            # 시작·끝 시각은 남기지 않아, 그 정정에서는 블록 시각을 실행 시간의 합으로 추정했다.
            cpu5, gpu = cpu_percent(5.0), gpu_percent()
            # 팔 실행 하나 = 새 프로세스: 모델 적재, 워밍업 5회, 50회 측정. 앞 팔의 스레드 풀이 남아 끼어들지
            # 않게 한다(규칙).
            done = subprocess.run(_run_arm_cmd(args, arm), capture_output=True, text=True, check=False)
            if done.returncode != 0:
                raise RuntimeError(f"run-arm {arm} failed: {done.stderr[-2000:]}")
            run = json.loads(done.stdout.strip().splitlines()[-1])
            run.update(block=b + 1, cpu5_before_pct=cpu5, gpu_before_pct=gpu)
            runs.append(run)
            print(
                json.dumps(
                    {k: v for k, v in run.items() if k != "ms"} | {"median_ms": statistics.median(run["ms"])}
                ),
                flush=True,
            )
    # 세션이 끝난 뒤 1분 동안의 CPU·GPU 사용률도 남긴다(측정 직후의 부하 확인).
    after = load_minute()
    summary = summarise_latency(runs, resize_ms, arms)
    parity = None
    if args.parity is not None and args.parity.exists():
        with open(args.parity, encoding="utf-8") as f:
            parity = json.load(f)
    report = {
        "check": "CPU runtime latency (docs/experiments.md, 2026-10-04)",
        "commit": git_commit(),
        "artifacts": _rel(args.artifacts),
        "onnx_sha256": sha256_file(args.artifacts / RECONSTRUCTION_PRECISIONS["fp32"]),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "category": args.category,
        "images": args.images,
        "warmup": args.warmup,
        "repeats_per_run": args.repeats,
        "blocks": block_orders(arms, args.blocks),
        "idle": {
            "reached": idle,
            "cpu_limit_pct": CPU_IDLE_PCT,
            "gpu_limit_pct": GPU_IDLE_PCT,
            "wait": wait_log,
        },
        "started": started,
        "after_minute": after,
        "resize_ms": resize_ms,
        "arms": summary,
        "H22": h22(parity, summary),
        "parity_report": None if parity is None else _rel(args.parity),
        "runs": runs,
    }
    _write_json(args.out, report)
    print(json.dumps({"arms": summary, "H22": report["H22"]}, indent=1), flush=True)


# 명령행 진입점: parity, latency, run-arm 세 하위 명령. 기본 아티팩트는 artifacts/dms-280-car, 기록은
# outputs/dms-280-car-onnx-fp32-dev, 리포트는 reports/runtime/. 범주 목록은 visa에서 함수 안에서 가져온다.
def main(argv: list[str] | None = None) -> None:
    from .visa import CATEGORIES

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    default_art = paths.ROOT / "artifacts" / "dms-280-car"

    par = commands.add_parser("parity", help="모든 팔로 dev 평가 세트를 채점해 R0와 비교한다")
    par.add_argument("--artifacts", type=Path, default=default_art)
    par.add_argument("--recorded", type=Path, default=paths.OUTPUTS / "dms-280-car-onnx-fp32-dev")
    par.add_argument("--arms", nargs="+", default=list(ARMS))
    par.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    par.add_argument("--out", type=Path, default=paths.REPORTS / "runtime" / "parity.json")

    lat = commands.add_parser("latency", help="한가한 PC에서 순서를 돌리는 블록으로 팔들의 지연을 잰다")
    lat.add_argument("--artifacts", type=Path, default=default_art)
    lat.add_argument("--arms", nargs="+", default=list(ARMS))
    lat.add_argument("--category", default="pcb1")
    lat.add_argument("--images", type=int, default=50)
    lat.add_argument("--warmup", type=int, default=5)
    lat.add_argument("--repeats", type=int, default=50)
    lat.add_argument("--blocks", type=int, default=3)
    lat.add_argument("--wait-max-min", type=int, default=60, help="PC가 한가해지기를 기다릴 최대 분")
    lat.add_argument("--parity", type=Path, default=paths.REPORTS / "runtime" / "parity.json")
    lat.add_argument("--out", type=Path, default=paths.REPORTS / "runtime" / "latency.json")

    one = commands.add_parser("run-arm", help="(latency가 내부에서 씀) 이 프로세스에서 팔 하나의 시간을 잰다")
    one.add_argument("--arm", choices=sorted(ARMS), required=True)
    one.add_argument("--artifacts", type=Path, required=True)
    one.add_argument("--category", default="pcb1")
    one.add_argument("--images", type=int, default=50)
    one.add_argument("--warmup", type=int, default=5)
    one.add_argument("--repeats", type=int, default=50)

    args = parser.parse_args(argv)
    {"parity": _parity_command, "latency": _latency_command, "run-arm": _run_arm_command}[args.command](
        args, parser
    )


if __name__ == "__main__":
    main()
