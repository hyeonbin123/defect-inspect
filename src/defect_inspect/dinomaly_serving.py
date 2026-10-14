"""Dinomaly 모델의 CPU 파이프라인: ONNX 내보내기, onnxruntime 채점, 서비스용 임계값.

`export`는 아티팩트 세트를 쓴다: `model_fp32.onnx`(이미지를 넣으면 `score`와 `map`이 나옴,
`dinomaly_model.ServingGraph` 참고), 선택으로 `model_int8_dynamic.onnx`(MatMul 노드의 동적 INT8 양자화),
범주마다 `meta.json`(`kind: reconstruction`). 학습한 모델은 `run_dinomaly train`에서 온다. `--untrained`는
그 설정의 갓 초기화한 모델을 내보내는데, 지연은 가중치와 무관하므로 6단계 지연 관문에는 이것으로 충분하다.
갓 내보낸 세트의 임계값은 `calibrate`가 `score` 실행으로 정하기 전까지 자리 표시 값이다
(`calibration.strategy: none`).

`score`는 절차의 평가 세트(test 절차에서는 홀드아웃 임계값을 정할 겹 0 정상도)를 CPU의 onnxruntime으로
채점한다. 선택으로 3단계 합성 조건 하나를 평가 이미지에 적용할 수 있다. 2단계 공통 형식의
`<범주>.npz`(맵 없음)와 `run.json`을 쓴다. torch가 필요한 것은 `export`뿐이다.
"""

# [파일 역할] 6단계 Dinomaly ViT-S(DMS) 모델을 CPU 서빙용으로 바꾸고 재는 파이프라인이다. 명령 세 개로 나뉜다.
# - export: 학습한 모델(또는 학습 전 모델)을 ONNX 아티팩트 세트로 내보낸다. 범주마다 meta.json(kind:
#   reconstruction)을 쓰고, 선택으로 동적 INT8 모델도 만든다.
# - score: dev 또는 봉인 test 절차의 평가 세트를 onnxruntime CPU로 채점해 2단계와 같은 공통 형식(.npz)으로
#   저장한다. 6단계 H16·H17의 판정은 이 점수로 한다(docs/experiments.md 6단계 "봉인 테스트" 2.).
# - calibrate: score 실행에 저장된 겹 0 정상 점수로 범주별 홀드아웃 임계값을 계산해 아티팩트의 meta.json에
#   적는다. 검사 서비스가 쓰는 임계값이 이것이다.
# [설계 이유] 학습 전 모델로도 export할 수 있게 한 것은 지연이 가중치와 무관해서다. 학습하기 전에 CPU 200ms
# 관문(관문 1)을 먼저 열어 통과한 크기만 학습했다. torch가 필요한 것은 export뿐이고, score와 calibrate는
# onnxruntime만으로 돈다(서빙 환경과 같은 조건).
# [연결] 모델 구조는 run_dinomaly(CONFIGS, build_model, load_model)와 dinomaly_model.ServingGraph에서, CPU
# 추론은 inspector.ReconstructionInspector에서 온다. score 결과(outputs/<이름>-onnx-<정밀도>-<절차>)는
# analyze_stage6, drift, runtime_compare가 읽는다.

import argparse
import json
import time
from pathlib import Path

import numpy as np

# 프로젝트 내부 모듈: paths(기본 경로), ImageCache(리사이즈 캐시), conformal_threshold(홀드아웃 임계값),
# conditions(3단계 교란을 dev 보조 채점에 적용), sha256_file(ONNX 지문 기록), inspector(서빙과 같은
# 전처리·검사기·meta 검사), ledger(봉인 테스트 장부와 커밋 기록), splits(절차별 행 선택), CATEGORIES(VisA 12개
# 범주).
from . import paths
from .cache import ImageCache
from .calibrate import conformal_threshold
from .conditions import CLEAN, apply_condition, condition_names
from .download import sha256_file
from .inspector import (
    ARTIFACT_VERSION,
    RECONSTRUCTION,
    RECONSTRUCTION_PRECISIONS,
    ReconstructionInspector,
    check_reconstruction_meta,
    preprocess,
)
from .ledger import git_commit, record_test_access
from .splits import ManifestRow, read_manifest, select
from .visa import CATEGORIES

# MAP_SIZE: 내보낸 그래프가 내는 이상 맵의 크기(픽셀 지표·히트맵의 256 격자). PARITY_TOL: 내보낼 때 torch와
# onnxruntime의 차이 허용치(점수는 상대, 맵은 절대 1e-3, 6단계 관문 1 규칙). 실제로는 학습 전 모델 6개가
# 2.4e-7·6.0e-7 이하, dms-280-car가 8.0e-7·5.1e-7이었다.
MAP_SIZE = 256
PARITY_TOL = 1e-3


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _read_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as err:
        raise ValueError(f"{path} is missing or not readable JSON ({err})") from err


# ---------------------------------------------------------------- export (torch 필요)


# ONNX는 opset 18이고, 내보낸 뒤 onnx.checker로 형식을 확인한다. 내보내기가 조용히 다른 계산을 하는 일을 막는
# 관문이다. 반환값(최대 차이)은 artifacts.json에 남는다.
def export_onnx(model, img_size: int, path: Path, *, opset: int = 18, probes: int = 2) -> dict:
    """`model`(CPU·fp32의 DinomalyModel)을 `img_size` 이미지 한 장용 `ServingGraph` ONNX로 내보낸다.

    무작위 이미지 `probes`장(시드 0)에서 onnxruntime을 torch와 비교해, 점수가 PARITY_TOL(상대)보다,
    또는 맵이 PARITY_TOL(절대)보다 크게 다르면 RuntimeError를 낸다. 가장 큰 차이를 돌려준다.
    """
    import onnx
    import torch

    from .dinomaly_model import ServingGraph

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # CPU·fp32·eval 모드로 바꾼 뒤 [1, 3, S, S] 예시 입력으로 추적한다. dynamo=False로 TorchScript 기반
    # 내보내기를 쓴다. 입력 이름은 image, 출력은 score와 map이고 서빙 쪽 검사기가 이 이름으로 부른다.
    graph = ServingGraph(model.to("cpu").float().eval()).eval()
    example = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            graph,
            (example,),
            str(path),
            input_names=["image"],
            output_names=["score", "map"],
            opset_version=opset,
            dynamo=False,
        )
    onnx.checker.check_model(str(path))
    session = _session(path)
    rng = np.random.default_rng(0)
    score_rel = map_abs = 0.0
    # 시드 0 무작위 uint8 이미지를 서빙과 같은 preprocess로 정규화해 torch 그래프와 onnxruntime 세션에 똑같이
    # 넣는다.
    for _ in range(probes):
        image = rng.integers(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        x = preprocess(image, img_size)
        with torch.no_grad():
            ref_score, ref_map = (t.numpy() for t in graph(torch.from_numpy(x)))
        score, amap = session.run(["score", "map"], {"image": x})
        # 점수는 상대 차이(0에 가까운 점수는 1e-12로 막음), 맵은 절대 차이의 최댓값을 모은다.
        score_rel = max(
            score_rel, float(np.max(np.abs(score - ref_score) / np.maximum(np.abs(ref_score), 1e-12)))
        )
        map_abs = max(map_abs, float(np.max(np.abs(amap - ref_map))))
    parity = {"score_rel_diff": score_rel, "map_abs_diff": map_abs, "probes": probes, "tolerance": PARITY_TOL}
    if score_rel > PARITY_TOL or map_abs > PARITY_TOL:
        raise RuntimeError(f"ONNX export differs from torch: {parity}")
    return parity


# onnxruntime CPU 세션을 만든다. threads를 주지 않으면 onnxruntime 기본 스레드 수(물리 코어 수 6)를
# 쓰고, 이것이 서빙 설정이다(CPU 런타임 대조의 R0).
def _session(path: Path, threads: int | None = None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    if threads is not None:
        options.intra_op_num_threads = int(threads)
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


# 6단계에서 판정 없는 보조로 쟀다: 18% 빨랐지만(123.2ms) dev AUROC가 1.44%p 떨어져 서빙은 FP32로 둔다. 4단계
# PatchCore의 정적 양자화(QDQ, MinMax)와는 방식이 다르다.
def quantize_dynamic_matmul(fp32_path: Path, out_path: Path) -> Path:
    """MatMul 노드의 동적 INT8 양자화(가중치는 QInt8, 활성값은 호출마다 양자화)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        str(fp32_path), str(out_path), op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8
    )
    return Path(out_path)


# 아티팩트 세트: model_fp32.onnx(+ 선택으로 model_int8_dynamic.onnx), 범주마다 meta.json, 전체 정보
# artifacts.json. 12개 범주가 모델 파일 하나를 같이 쓰고 범주 폴더에는 임계값만 다르다(뱅크 없음). 처음에는
# 임계값을 0.0, calibration.strategy를 none으로 적어 두고 calibrate가 채운다. meta에 커밋과 ONNX sha256을 넣어
# 어떤 모델로 만든 임계값인지 추적한다. 쓰기 전에 check_reconstruction_meta로 서비스가 읽을 수 있는 형식인지
# 확인한다.
def build_artifacts(
    model,
    *,
    name: str,
    img_size: int,
    out_dir: Path,
    source: dict,
    categories: list[str],
    int8_dynamic: bool = False,
) -> dict:
    """`model`을 내보내고, 모든 범주의 meta를 자리 표시 임계값으로 쓴다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fp32 = out_dir / RECONSTRUCTION_PRECISIONS["fp32"]
    # 내보내기와 동일성 확인. 허용치를 넘으면 여기서 예외로 멈추므로 범주 meta가 쓰이지 않는다.
    parity = export_onnx(model, img_size, fp32)
    info: dict = {
        "export_s": round(time.perf_counter() - t0, 1),
        "parity": parity,
        "int8_dynamic": {"available": False, "reason": "not requested"},
    }
    int8_path = out_dir / RECONSTRUCTION_PRECISIONS["int8-dynamic"]
    # 예전 INT8 파일은 새 FP32 모델에서 나온 것이 아니므로, 이번에 만들지 않더라도 먼저 지워 둔다.
    int8_path.unlink(missing_ok=True)
    if int8_dynamic:
        quantize_dynamic_matmul(fp32, int8_path)
        info["int8_dynamic"] = {"available": True, "op_types": ["MatMul"], "weight_type": "QInt8"}
    commit = git_commit()
    model_sha256 = sha256_file(fp32)
    # 범주마다 meta.json을 쓴다. 모델 파일은 세트 루트에 하나뿐이고, 범주 폴더는 그 모델을 가리키는 설정과
    # 임계값만 가진다.
    for category in categories:
        meta = {
            "version": ARTIFACT_VERSION,
            "kind": RECONSTRUCTION,
            "name": name,
            "category": category,
            "img_size": int(img_size),
            "map_size": MAP_SIZE,
            "threshold": 0.0,
            "calibration": {"strategy": "none"},
            "commit": commit,
            "source": {**source, "onnx_sha256": model_sha256},
        }
        check_reconstruction_meta(meta)
        _write_json(out_dir / category / "meta.json", meta)
    _write_json(
        out_dir / "artifacts.json",
        {"name": name, "img_size": int(img_size), "commit": commit, "source": source, **info},
    )
    return info


# export 명령. --model(학습한 model.pt)과 --untrained 중 정확히 하나를 받는다.
# - --untrained: torch.manual_seed(seed) 뒤 그 설정의 새 모델을 만든다(관문 1 지연 측정용,
#   artifacts/<설정>-untrained).
# - --model: 저장된 모델의 설정 이름이 --config와 다르면 거절한다(다른 크기 모델을 잘못 내보내는 실수 방지).
#   학습 스텝, AMP 여부, 모델 sha256을 source에 남긴다.
def _export_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    import torch

    from . import run_dinomaly

    cfg = run_dinomaly.CONFIGS[args.config]
    if args.untrained == (args.model is not None):
        parser.error("pass either --model (a trained model) or --untrained")
    out_dir = args.out or paths.ROOT / "artifacts" / (cfg.name + ("-untrained" if args.untrained else ""))
    # 학습 전 모델은 전역 torch 시드를 고정한 뒤 만들어, 같은 시드면 같은 초기 가중치가 나온다(build_model이
    # 전역 생성기로 디코더를 초기화).
    if args.untrained:
        torch.manual_seed(args.seed)
        model = run_dinomaly.build_model(cfg.encoder, **cfg.build_options()).eval()
        source = {"config": cfg.name, "untrained": True, "seed": args.seed, "options": cfg.build_options()}
    else:
        if not args.model.exists():
            parser.error(f"{args.model} does not exist")
        model, saved = run_dinomaly.load_model(args.model)
        if saved.get("config", run_dinomaly.NAME) != cfg.name:
            parser.error(f"{args.model} holds a {saved.get('config', run_dinomaly.NAME)!r} model")
        source = {
            "config": cfg.name,
            "untrained": False,
            "model": str(args.model),
            "model_sha256": sha256_file(args.model),
            "train_steps": saved["steps"],
            "train_amp": saved["amp"],
            "options": saved.get("options", {}),
        }
    info = build_artifacts(
        model,
        name=cfg.name,
        img_size=cfg.img_size,
        out_dir=out_dir,
        source=source,
        categories=args.categories,
        int8_dynamic=args.int8_dynamic,
    )
    print(json.dumps({"out": str(out_dir), **info}, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- 채점 (onnxruntime만 필요)


# 평가 세트는 절차가 정한다(dev: 겹 0 정상 + dev 결함, test: 봉인 테스트). 보정 정상은 test 절차일 때만 겹 0
# 정상이다(6단계 규칙).
def _rows(
    manifest: list[ManifestRow], protocol: str, category: str, allow_test: bool
) -> tuple[list[ManifestRow], list[ManifestRow], list[ManifestRow]]:
    """범주 하나의 (평가 정상, 평가 결함, 보정 정상) 행."""
    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    # 홀드아웃 임계값은 어떤 모델도 학습에 쓰지 않은 겹 0 정상으로 정한다. dev 절차에서는 그것이 곧 평가
    # 정상이라 따로 읽지 않는다.
    cal = (
        select(manifest, protocol="dev", part="eval_normal", category=category) if protocol == "test" else []
    )
    return eval_normal, eval_defect, cal


# 보정 점수는 교란 없이, 평가 이미지만 condition(예:
# shift-1, blur-2)을 적용해 채점한다. 그래서 dev 보조 표의 교란 칸은 "교란 없는 정상으로 정한 임계값을 교란된
# 같은 정상 몇 장이 넘는가"를 보여 준다. 맵은 저장하지 않는다(AUPRO 등 픽셀 지표는 torch 실행 값을 쓴다). npz
# 키는 2단계 공통 형식(eval_images, eval_labels, eval_defect_types, eval_score, cal_score, cal_images)이라
# compare.load_common으로 그대로 읽힌다.
def score_category(
    inspector: ReconstructionInspector,
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache,
    allow_test: bool,
    out_dir: Path,
    *,
    condition: str = CLEAN,
) -> dict:
    """범주 하나를 CPU 파이프라인으로 채점해 `<범주>.npz`(공통 형식, 맵 없음)를 쓴다."""
    eval_normal, eval_defect, cal_rows = _rows(manifest, protocol, category, allow_test)
    rows = eval_normal + eval_defect
    labels = np.array([0] * len(eval_normal) + [1] * len(eval_defect), dtype=np.int8)
    t0 = time.perf_counter()
    # 보정 정상(test 절차의 겹 0)은 교란 없이 채점한다. 임계값은 늘 원래 촬영 조건의 정상으로 정한다.
    cal_score = inspector.scores(cache.images(cal_rows)) if cal_rows else np.empty(0, dtype=np.float32)
    t1 = time.perf_counter()
    # 평가 이미지에만 교란 조건을 준다(clean이면 복사본 그대로).
    images = apply_condition(cache.images(rows), condition) if rows else np.empty((0, 1, 1, 3), np.uint8)
    eval_score = inspector.scores(images)
    t2 = time.perf_counter()
    np.savez(
        out_dir / f"{category}.npz",
        eval_images=np.array([r.image for r in rows]),
        eval_labels=labels,
        eval_defect_types=np.array([r.defect_types for r in rows]),
        eval_score=eval_score,
        cal_score=cal_score,
        cal_images=np.array([r.image for r in cal_rows], dtype=np.str_),
    )
    return {
        "category": category,
        "eval_normal": len(eval_normal),
        "eval_defect": len(eval_defect),
        "cal": len(cal_rows),
        "threshold_in_meta": inspector.threshold,
        "timing": {"cal_score_s": round(t1 - t0, 2), "eval_score_s": round(t2 - t1, 2)},
    }


# score 명령. 봉인 테스트 규칙을 지키는 순서가 중요하다.
# 1) test 절차는 --allow-test와 --stage가 둘 다 있어야 한다.
# 2) 조건 이름, 아티팩트, 모델 파일, 범주를 확인하고 모든 검사기를 먼저 만든다. 깨진 아티팩트는 이미지를
#    하나도 읽기 전에 멈추므로 장부에 헛된 줄이 남지 않는다.
# 3) 같은 출력 폴더의 예전 결과를 지우고(--overwrite 없이 완성된 실행을 덮어쓰지 않음), test면 장부에 기록한
#    뒤에 채점한다.
# 4) run.json에 커밋, ONNX sha256, 정밀도, 스레드, 조건, 보정 방식을 남긴다. calibrate는 이 sha256으로 같은
#    모델의 점수인지 확인한다.
def _score_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.protocol == "test" and (not args.allow_test or not args.stage):
        parser.error("--protocol test needs --allow-test and --stage")
    if args.condition not in condition_names():
        parser.error(f"unknown condition {args.condition!r}; one of {condition_names()}")
    art = args.artifacts
    try:
        info = _read_json(art / "artifacts.json")
    except ValueError as err:
        parser.error(str(err))
    # 출력 폴더와 장부 config 이름에 조건을 붙여, 조건별 실행이 서로 덮어쓰지 않게 한다(예:
    # outputs/dms-280-car-onnx-fp32-dev-blur-2).
    suffix = "" if args.condition == CLEAN else f"-{args.condition}"
    out_dir = args.out or paths.OUTPUTS / f"{info['name']}-onnx-{args.precision}-{args.protocol}{suffix}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    model_path = art / RECONSTRUCTION_PRECISIONS[args.precision]
    if not model_path.exists():
        parser.error(f"{model_path} does not exist")
    manifest = read_manifest(paths.VISA_MANIFEST)
    known = {r.category for r in manifest}
    bad = [c for c in args.categories if c not in known or not (art / c / "meta.json").exists()]
    if bad or not args.categories or len(set(args.categories)) != len(args.categories):
        parser.error(f"categories {bad or args.categories} are unknown, repeated or not in {art}")
    # 이미지를 읽기 전에 모든 검사기를 만든다: 깨진 아티팩트는 장부에 기록되기 전에 실행을 멈춘다.
    try:
        inspectors = {
            c: ReconstructionInspector.load(art / c, precision=args.precision, threads=args.threads)
            for c in args.categories
        }
    except ValueError as err:
        parser.error(str(err))
    # 모든 범주가 같은 입력 크기여야 한 크기의 리사이즈 캐시로 채점할 수 있다.
    sizes = {i.size for i in inspectors.values()}
    if len(sizes) != 1:
        parser.error(f"the categories of {art} have different input sizes {sorted(sizes)}")
    commit = git_commit()
    cache = ImageCache(paths.CACHE, sizes.pop())
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in [out_dir / "run.json", *out_dir.glob("*.npz")]:
        path.unlink(missing_ok=True)  # 예전 실행의 파일이 이번 실행에 섞이면 안 된다
    if args.protocol == "test":
        record_test_access(
            paths.TEST_LEDGER,
            stage=args.stage,
            config=f"{info['name']}-onnx-{args.precision}" + suffix,
            note=args.note,
            commit=commit,
        )
    summaries = []
    started = time.perf_counter()
    for category in args.categories:
        summary = score_category(
            inspectors[category],
            args.protocol,
            category,
            manifest,
            cache,
            args.allow_test,
            out_dir,
            condition=args.condition,
        )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    run = {
        "method": "dinomaly-onnx",
        "protocol": args.protocol,
        "commit": commit,
        "config": {
            "name": info["name"],
            "img_size": info["img_size"],
            "artifacts": str(art),
            "precision": args.precision,
            "onnx_sha256": sha256_file(model_path),
            "source": info.get("source"),
            "threads": args.threads,
            "condition": args.condition,
            "calibration": "fold-0 normals (hold-out)" if args.protocol == "test" else "none",
        },
        "categories": summaries,
        "total_s": round(time.perf_counter() - started, 1),
    }
    _write_json(out_dir / "run.json", run)


# ---------------------------------------------------------------- 서비스용 임계값


# calibrate 명령: 서비스용 임계값을 아티팩트에 적는다. 교란 없는 test 절차의 score 실행만 받는다(그 실행에만
# 겹 0 정상 점수가 있다). 점수를 낸 모델이 지금 아티팩트의 FP32 모델과 같은지 sha256으로 확인한다. 범주마다 겹
# 0 점수(60~121장)에서 목표 α(기본 5%)의 conformal 임계값을 구해 threshold, alpha, calibration(holdout, n,
# 보장 여부)을 meta.json에 쓴다. 6단계에서는 12개 범주 모두 보장 조건(n ≥ 19)을 만족했다.
# 서비스 교체 점검은 서비스의 /categories 임계값이 이 계산과 같은지 확인했다(docs/experiments.md "서비스 모델
# 교체 점검").
def _calibrate_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    art = args.artifacts
    try:
        run = _read_json(args.scores / "run.json")
    except ValueError as err:
        parser.error(str(err))
    if run.get("protocol") != "test" or run.get("config", {}).get("condition") != CLEAN:
        parser.error("thresholds come from a clean test-protocol score run (its fold-0 normal scores)")
    fp32 = art / RECONSTRUCTION_PRECISIONS["fp32"]
    if run["config"].get("precision") != "fp32" or run["config"].get("onnx_sha256") != sha256_file(fp32):
        parser.error(f"{args.scores} was not scored with the fp32 model of {art}")
    written = {}
    for summary in run["categories"]:
        category = summary["category"]
        with np.load(args.scores / f"{category}.npz") as z:
            cal = z["cal_score"]
        # 범주의 겹 0 정상 점수로 홀드아웃 임계값을 정하고, 보장 여부와 표본 수를 meta에 함께 남긴다.
        thr = conformal_threshold(cal, args.alpha)
        meta = _read_json(art / category / "meta.json")
        meta.update(
            threshold=float(thr.value),
            alpha=float(args.alpha),
            calibration={"strategy": "holdout", "n": int(thr.n), "guaranteed": bool(thr.guaranteed)},
        )
        meta["source"] = {**meta.get("source", {}), "scores": str(args.scores)}
        check_reconstruction_meta(meta)
        _write_json(art / category / "meta.json", meta)
        written[category] = meta["threshold"]
    print(json.dumps(written, ensure_ascii=False), flush=True)


# 명령행 입구: export, score, calibrate 세 하위 명령을 만들고 이름으로 처리 함수를 고른다. 실행 순서는
# README의 6단계 재현 명령에 있다(export 뒤 score, 마지막에 calibrate).
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    exp = commands.add_parser("export", help="Dinomaly 모델을 아티팩트 세트로 내보낸다")
    exp.add_argument("--config", required=True, help="run_dinomaly 설정 이름, 예: dms-280")
    exp.add_argument("--model", type=Path, default=None, help="run_dinomaly train이 저장한 model.pt")
    exp.add_argument("--untrained", action="store_true", help="갓 초기화한 모델(지연 측정용)")
    exp.add_argument("--seed", type=int, default=0, help="--untrained의 초기 가중치 시드")
    exp.add_argument("--int8-dynamic", action="store_true", help="동적 INT8 모델도 쓴다")
    exp.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    exp.add_argument("--out", type=Path, default=None, help="기본값: artifacts/<설정>[-untrained]")

    score = commands.add_parser("score", help="절차의 평가 세트를 CPU의 onnxruntime으로 채점한다")
    score.add_argument("--artifacts", type=Path, required=True)
    score.add_argument("--protocol", choices=["dev", "test"], required=True)
    score.add_argument("--precision", choices=sorted(RECONSTRUCTION_PRECISIONS), default="fp32")
    score.add_argument("--condition", default=CLEAN, help="평가 이미지에 적용할 3단계 교란 조건")
    score.add_argument(
        "--threads", type=int, default=None, help="onnxruntime 연산 안 스레드 수(기본값: onnxruntime 자체 값)"
    )
    score.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    score.add_argument("--out", type=Path, default=None)
    score.add_argument("--allow-test", action="store_true", help="봉인 테스트 세트를 읽는다(장부에 기록)")
    score.add_argument("--stage", default="", help="테스트 장부에 적을 단계 이름(--allow-test와 함께 필수)")
    score.add_argument("--note", default="")
    score.add_argument("--overwrite", action="store_true")

    cal = commands.add_parser("calibrate", help="교란 없는 test 절차 score 실행으로 홀드아웃 임계값을 정한다")
    cal.add_argument("--artifacts", type=Path, required=True)
    cal.add_argument("--scores", type=Path, required=True)
    cal.add_argument("--alpha", type=float, default=0.05)

    args = parser.parse_args(argv)
    {"export": _export_command, "score": _score_command, "calibrate": _calibrate_command}[args.command](
        args, parser
    )


if __name__ == "__main__":
    main()
