"""Score the evaluation set and the cross-fitted pool with the exported CPU pipeline.

The torch pipeline (GPU fp16) produced the banks and the thresholds of a `run_grid --save-banks` run. This
run repeats the scoring with onnxruntime on the CPU (fp32 and INT8 models) against the same banks, so
that the drift of scores, thresholds and false alarm rates caused by the export can be measured.
"""

# [파일 역할] 4단계 CPU 재채점 스크립트. run_grid --save-banks 실행이 GPU torch 파이프라인(fp16)으로 만든
# 뱅크를 그대로 쓰되, 채점만 onnxruntime CPU 파이프라인(inspector.Inspector, FP32·INT8 모델)으로 다시 해서
# 평가 세트 점수와 교차 적합 풀 점수를 저장한다.
# [왜 필요한가] 서빙 구성을 CPU로 옮겼을 때 점수·임계값·오검출률이 얼마나 달라지는지 재야 H14(INT8 임계값을
# 그대로 쓰면 오검출률이 2%p 넘게 달라진다)와 H15(CPU 지연 + 오검출률 + 정확도)를 판정할 수
# 있다(experiments.md 4단계 'test 절차'). 같은 뱅크·같은 이미지로 채점하므로 차이는 내보내기(ONNX)와
# 정밀도에서만 나온다.
# [결과] README: FP32 CPU 점수는 GPU 점수와 중앙값 0.012% 차이로 AUROC와 오검출률이 같았고, INT8은 점수가
# 중앙값 24% 달라져 FP32 임계값으로 정상의 68%를 불량으로 판정했다.
# [흐름] 인자 점검 → check_inputs(이미지를 읽지 않고 격자 실행·아티팩트·모델·이미지 목록이 서로 맞는지 확인) →
# 장부 기록(test만) → 정밀도 × 범주마다 run_category → npz와 run.json.
# [연결] 입력은 run_grid(뱅크·torch 점수)와 export(아티팩트)의 출력, 출력은 analyze_export가 읽는다.

import argparse
import json
import time
from pathlib import Path

# Inspector: 서비스와 같은 CPU 채점 경로. check_meta·check_model_io: 아티팩트와 모델이 맞는지 이미지를 읽기
# 전에 확인한다. ratio_name·ratio_rows: run_grid와 같은 비율 폴더 이름·뱅크 행 수 규칙.
import numpy as np

from . import paths
from .cache import ImageCache
from .inspector import PRECISIONS, Inspector, check_meta, check_model_io
from .ledger import git_commit, record_test_access
from .run_grid import ratio_name, ratio_rows
from .splits import ManifestRow, pool_folds, read_manifest, select

# 아티팩트 meta와 격자 실행 config에서 반드시 같아야 하는 값(백본, 입력 크기, 재가중 k, 맵 σ). 다르면 두
# 파이프라인의 점수 차이가 내보내기 탓인지 설정 탓인지 가를 수 없다.
# Artifact meta values that must equal the grid run's: they decide the scores of both pipelines.
_CONFIG_KEYS = ("backbone", "img_size", "reweight_k", "sigma")


# JSON을 읽되 파일이 없거나 깨졌으면 경로를 담은 ValueError로 바꿔, main이 parser.error로 깔끔히 알리게 한다.
def _read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as err:
        raise ValueError(f"{path} is missing or not readable JSON ({err})") from err


# .npy 헤더만 읽어 배열 모양을 알아낸다. 큰 뱅크 파일을 실제로 읽지 않고 행 수·차원을 점검하기 위해서다(형식
# 버전 1.0과 2.0 헤더를 구분).
def _npy_shape(path: Path) -> tuple[int, ...]:
    """Shape of a .npy file from its header; the data is not read."""
    try:
        with open(path, "rb") as f:
            version = np.lib.format.read_magic(f)
            read_header = (
                np.lib.format.read_array_header_1_0
                if version == (1, 0)
                else np.lib.format.read_array_header_2_0
            )
            return tuple(read_header(f)[0])
    except (OSError, ValueError) as err:
        raise ValueError(f"{path} is missing or not a readable array ({err})") from err


# 이 절차에 있어야 하는 뱅크 이름: 전체 풀 뱅크(full)와 겹마다 그 겹을 뺀 뱅크(dev는 겹 1~4, test는 겹 0~4).
def _bank_names(protocol: str) -> list[str]:
    return ["full", *(f"minus_fold_{fold}" for fold in pool_folds(protocol))]


# 범주의 (풀 정상, 평가 정상, 평가 결함) 행을 매니페스트 순서로 고른다. run_grid와 같은 select 호출이라 같은
# 이미지가 같은 순서로 나온다. 봉인 테스트는 allow_test가 있어야 나온다.
def _rows(
    manifest: list[ManifestRow], protocol: str, category: str, allow_test: bool
) -> tuple[list[ManifestRow], list[ManifestRow], list[ManifestRow]]:
    """(pool normals, evaluation normals, evaluation defects) of a category, in manifest order."""
    pool = select(manifest, protocol=protocol, part="pool_normal", category=category)
    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    return pool, eval_normal, eval_defect


# 격자 실행이 저장한 torch 점수(평가, 교차 적합 풀)를 읽으면서, 그 점수가 지금 매니페스트가 고르는 바로 그
# 이미지들(같은 순서, 같은 개수)의 것인지 확인한다. 이미지가 어긋나면 비교 자체가 무의미하므로 멈춘다.
def _check_same_images(npz_path: Path, category: str, pool: list[ManifestRow], eval_rows: list[ManifestRow]):
    """The grid run's torch scores (evaluation, cross-fitted pool), checked to be of these images."""
    try:
        with np.load(npz_path) as z:
            eval_images, pool_images = z["eval_images"].tolist(), z["pool_images"].tolist()
            torch_eval, torch_oof = z["eval_score_full"], z["pool_score_oof"]
    except (OSError, KeyError, ValueError) as err:
        raise ValueError(f"{npz_path} is missing or not a grid run's score file ({err!r})") from err
    if eval_images != [r.image for r in eval_rows] or pool_images != [r.image for r in pool]:
        raise ValueError(f"{category}: the grid run scored other images than the manifest selects")
    if len(torch_eval) != len(eval_rows) or len(torch_oof) != len(pool):
        raise ValueError(f"{category}: {npz_path.name} has scores for another number of images")
    return torch_eval, torch_oof


# 실행에 필요한 모든 것을 이미지 없이 점검하고 범주별 아티팩트 meta를 돌려준다: 비율과 범주가 격자 실행에
# 있는지, 요청한 정밀도의 모델 파일이 있는지, 아티팩트가 같은 설정·비율·절차로 만들어졌는지, 뱅크 파일 모양이
# 비율에 충분한지, 저장된 torch 점수의 이미지 목록이 맞는지.
# test 절차는 이 점검을 장부 기록 전에 해서, 입력 문제로 실패한 실행이 봉인 테스트 기록만 남기지 않게
# 한다(docstring). 커밋 84a26c4 메시지의 'run_export_eval checks ratio, categories and files before the ledger
# line'에 해당한다.
def check_inputs(
    grid_dir: Path,
    grid: dict,
    ratio: float,
    artifacts: Path,
    precisions: list[str],
    categories: list[str],
    manifest: list[ManifestRow],
    allow_test: bool,
) -> dict[str, dict]:
    """Check everything a run needs, without reading an image; returns the artifact meta per category.

    Raises `ValueError` naming the first problem: a ratio the grid run does not have, missing models,
    banks or score files, artifacts that were built from another setting or ratio, image lists that
    differ from the grid run's. A test-protocol run calls this before it writes the ledger line.
    """
    # 격자 실행에 이 비율과 범주들이 있는지 먼저 확인한다(비율은 폴더 이름 규칙으로 비교).
    protocol = grid["protocol"]
    if ratio_name(ratio) not in {ratio_name(r) for r in grid["ratios"]}:
        raise ValueError(f"ratio {ratio:g} is not one of the grid run's ratios {grid['ratios']}")
    unknown = [c for c in categories if c not in grid["categories"]]
    if unknown:
        raise ValueError(f"the grid run has no categories {unknown}")
    # INT8 모델이 없으면 export가 artifacts.json에 남긴 이유(예: DINOv2는 정적 양자화 불가, 규칙 변경 5)를
    # 메시지에 붙이고 --precision fp32만 쓰라고 안내한다.
    for precision in precisions:
        model = artifacts / PRECISIONS[precision]
        if not model.is_file():
            hint = ""
            if precision == "int8":
                reason = None
                if (artifacts / "artifacts.json").is_file():
                    reason = _read_json(artifacts / "artifacts.json").get("int8", {}).get("reason")
                hint = f" (export: {reason})" if reason else ""
                hint += "; pass --precision fp32 to score the fp32 model only"
            raise ValueError(f"the artifact set has no {model.name}{hint}")

    # 범주마다 아티팩트 meta를 읽어 형식을 점검하고 격자 실행 값과 대조한다.
    ratio_dir = grid_dir / ratio_name(ratio)
    ratio_run = _read_json(ratio_dir / "run.json")
    config = ratio_run["config"]
    infos = {info["category"]: info for info in ratio_run["categories"]}
    metas = {}
    for category in categories:
        if category not in infos:
            raise ValueError(f"{ratio_dir / 'run.json'} has no category {category}")
        meta = _read_json(artifacts / category / "meta.json")
        try:
            check_meta(meta)
        except ValueError as err:
            raise ValueError(f"{artifacts / category / 'meta.json'}: {err}") from err
        # 아티팩트 meta의 백본·입력 크기·재가중 k·σ·특징 차원·패치 격자가 격자 실행의 값과 같은지 비교하고,
        # 다르면 다른 값만 골라 보여 준다.
        expected = {key: config[key] for key in _CONFIG_KEYS}
        expected.update(dim=infos[category]["dim"], grid=list(infos[category]["grid"]))
        found = {key: list(meta[key]) if key == "grid" else meta[key] for key in expected}
        if found != expected:
            differing = {k: (found[k], expected[k]) for k in expected if found[k] != expected[k]}
            raise ValueError(
                f"{category}: the artifact was built from another setting than the grid run at ratio "
                f"{ratio:g} (artifact, grid run): {differing}"
            )
        # meta에 출처(source)가 있으면 아티팩트를 만든 비율과 절차도 이 실행과 같아야 한다. 예를 들어 dev
        # 격자로 만든 아티팩트를 test 격자 재채점에 쓰는 실수를 막는다.
        source = meta.get("source")
        if source is not None:
            if ratio_name(source.get("ratio", ratio)) != ratio_name(ratio):
                raise ValueError(
                    f"{category}: the artifact was built at ratio {source['ratio']:g}, not {ratio:g}"
                )
            if source.get("protocol", protocol) != protocol:
                raise ValueError(
                    f"{category}: the artifact was built from a {source['protocol']} run, "
                    f"the grid run is a {protocol} run"
                )

        # 뱅크마다 원래 패치 수(n_features)로 이 비율의 행 수를 계산하고, 저장된 뱅크가 그 이상 행을 가졌는지
        # 헤더만 보고 확인한다.
        sources = _read_json(grid_dir / "banks" / f"{category}.json")["n_features"]
        for bank_name in _bank_names(protocol):
            if bank_name not in sources:
                raise ValueError(f"banks/{category}.json does not list the bank {bank_name}")
            path = grid_dir / "banks" / f"{category}_{bank_name}.npy"
            shape = _npy_shape(path)
            rows = ratio_rows(sources[bank_name], ratio)
            if len(shape) != 2 or shape[1] != meta["dim"] or shape[0] < rows:
                raise ValueError(
                    f"{path} has shape {shape}; ratio {ratio:g} needs {rows} rows of dim {meta['dim']}"
                )

        # 마지막으로 매니페스트가 고르는 이미지와 격자 실행이 저장한 torch 점수의 이미지가 같은지
        # 확인한다(이미지 자체는 읽지 않음).
        pool, eval_normal, eval_defect = _rows(manifest, protocol, category, allow_test)
        _check_same_images(ratio_dir / f"{category}.npz", category, pool, eval_normal + eval_defect)
        metas[category] = meta
    return metas


# 범주 하나를 한 세션(한 정밀도)으로 재채점한다. 평가 세트는 전체 풀 뱅크의 앞 rows행으로, 풀 이미지는 각자
# 겹을 뺀 뱅크로 채점해 교차 적합 점수를 만든다(run_grid와 같은 구조). 같은 이미지의 torch 점수도 함께 돌려줘,
# 분석에서 바로 비교하게 한다.
def run_category(
    session,
    meta: dict,
    grid_dir: Path,
    ratio: float,
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache: ImageCache,
    allow_test: bool,
) -> dict[str, np.ndarray]:
    """Evaluation scores under the full bank prefix and cross-fitted pool scores, all from `session`."""
    # run_grid와 같은 행 선택이라 평가·풀 이미지가 같은 순서로 나온다. folds는 풀 이미지의 겹 번호다.
    pool, eval_normal, eval_defect = _rows(manifest, protocol, category, allow_test)
    eval_rows = eval_normal + eval_defect
    folds = np.array([r.fold for r in pool], dtype=np.int64)
    # 이미지를 읽기 전에 비교 대상 torch 점수가 같은 이미지의 것인지 먼저 확인한다.
    # Before any image is read: the torch scores this run is compared with must be of the same images.
    torch_eval, torch_oof = _check_same_images(
        grid_dir / ratio_name(ratio) / f"{category}.npz", category, pool, eval_rows
    )
    sources = _read_json(grid_dir / "banks" / f"{category}.json")["n_features"]

    # 저장된 뱅크에서 이 비율의 앞 rows행만 잘라 Inspector를 만든다. 뱅크가 모자라면(격자의 가장 큰 비율보다
    # 큰 비율) 조용히 뱅크 전체를 쓰지 않도록 멈춘다. ONNX 세션은 범주·뱅크가 바뀌어도 하나를 같이 쓴다.
    def inspector_for(bank_name: str) -> Inspector:
        bank = np.load(grid_dir / "banks" / f"{category}_{bank_name}.npy")
        rows = ratio_rows(sources[bank_name], ratio)
        if bank.shape[0] < rows:  # a ratio above the grid's largest would silently use the whole bank
            raise ValueError(
                f"{category}_{bank_name}.npy has {bank.shape[0]} rows, ratio {ratio:g} needs {rows}"
            )
        return Inspector(session, bank[:rows], meta)

    # 전체 뱅크 검사기를 이미지를 읽기 전에 만들어 뱅크 문제가 먼저 드러나게 한다. 평가 점수를 낸 뒤에는 바로
    # 지워 메모리를 아낀다.
    full = inspector_for("full")  # built before the first image is read
    eval_scores = full.scores(cache.images(eval_rows))
    del full
    # 교차 적합: 겹마다 그 겹을 뺀 뱅크로 만든 검사기로 그 겹 이미지를 채점한다. 모든 풀 이미지가 채워졌는지
    # NaN으로 확인한다.
    oof = np.full(len(pool), np.nan, dtype=np.float32)
    pool_images = cache.images(pool)
    for fold in pool_folds(protocol):
        held = np.flatnonzero(folds == fold)
        oof[held] = inspector_for(f"minus_fold_{fold}").scores(pool_images[held])
    if np.isnan(oof).any():
        raise RuntimeError("cross-fitting did not cover every pool image")

    # CPU 점수와 같은 이미지의 torch 점수를 한 파일에 담아, analyze_export가 점수 차이·임계값 차이·오검출률
    # 변화를 바로 계산하게 한다.
    return {
        "eval_images": np.array([r.image for r in eval_rows]),
        "eval_labels": np.array([0] * len(eval_normal) + [1] * len(eval_defect), dtype=np.int8),
        "eval_score_full": eval_scores,
        "pool_images": np.array([r.image for r in pool]),
        "pool_folds": folds,
        "pool_score_oof": oof,
        "torch_eval_score": torch_eval,
        "torch_pool_score_oof": torch_oof,
    }


# 명령줄 진입점. 예(README 4단계): `python -m defect_inspect.run_export_eval --grid
# outputs/grid-wrn50-256-test --ratio 0.01 --artifacts artifacts/serving-wrn50-256-r0.01 --precision fp32 int8
# --allow-test --stage 4`. 절차는 격자 실행의 run.json에서 가져오고, test면 --allow-test와 --stage가 있어야
# 한다. --threads로 onnxruntime 스레드 수를 고정할 수 있다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", type=Path, required=True, help="a run_grid --save-banks run directory")
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--artifacts", type=Path, required=True, help="artifact set built by export.py")
    parser.add_argument("--precision", nargs="+", default=["fp32", "int8"], choices=sorted(PRECISIONS))
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--out", type=Path, default=None, help="default: outputs/export-<artifact set>-<protocol>"
    )
    parser.add_argument("--allow-test", action="store_true", help="read the sealed test set (logged)")
    parser.add_argument(
        "--stage", default="", help="stage label for the test ledger (required with --allow-test)"
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    import onnxruntime as ort

    # 격자 실행의 run.json에서 절차를 읽는다. test 절차 격자면 --allow-test·--stage가 필요하고, --save-banks
    # 없이 돈 격자는 뱅크가 없어 쓸 수 없다.
    try:
        grid = _read_json(args.grid / "run.json")
    except ValueError as err:
        parser.error(str(err))
    protocol = grid["protocol"]
    if protocol == "test" and (not args.allow_test or not args.stage):
        parser.error("the grid run is a test-protocol run: --allow-test and --stage are needed")
    if not grid.get("save_banks"):
        parser.error(f"{args.grid} was run without --save-banks")
    categories = args.categories or grid["categories"]
    # 같은 정밀도를 두 번 적어도 한 번만 돌도록 순서를 지키며 중복을 없앤다.
    precisions = list(dict.fromkeys(args.precision))
    name = args.artifacts.name
    out_dir = args.out or paths.OUTPUTS / f"export-{name}-{protocol}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")

    # 테스트 이미지를 건드리지 않고 실패할 수 있는 모든 점검(입력 파일, 모델 적재, 모델 입출력과 meta의 일치,
    # 캐시 존재)을 장부 기록 전에 끝낸다.
    # Everything that can fail without touching a test image comes before the ledger line: the ratio,
    # the files of the grid run and of the artifact set, the image lists, the models and the cache.
    manifest = read_manifest(paths.VISA_MANIFEST)
    sessions = {}
    try:
        metas = check_inputs(
            args.grid, grid, args.ratio, args.artifacts, precisions, categories, manifest, args.allow_test
        )
        # 정밀도마다 onnxruntime 세션을 하나 만들고(--threads로 스레드 고정 가능), 모든 범주의 meta와 모델
        # 입출력이 맞는지 확인한다.
        for precision in precisions:
            options = ort.SessionOptions()
            if args.threads is not None:
                options.intra_op_num_threads = args.threads
            model = args.artifacts / PRECISIONS[precision]
            try:
                sessions[precision] = ort.InferenceSession(
                    str(model), sess_options=options, providers=["CPUExecutionProvider"]
                )
            except Exception as err:  # onnxruntime has its own exception types for files it cannot load
                raise ValueError(f"{model} is not a loadable ONNX model: {err}") from err
            for category in categories:
                try:
                    check_model_io(sessions[precision], metas[category])
                except ValueError as err:
                    raise ValueError(f"{model.name} does not fit the artifact of {category}: {err}") from err
    except ValueError as err:
        parser.error(str(err))
    size = metas[categories[0]]["img_size"]
    try:
        cache = ImageCache(paths.CACHE, size)
    except FileNotFoundError:
        command = f"python -m defect_inspect.cache --size {size}"
        parser.error(f"no {size} px image cache: build it with `{command}`")
    commit = git_commit()
    if protocol == "test":
        record_test_access(
            paths.TEST_LEDGER, stage=args.stage, config=f"export-{name}", note=args.note, commit=commit
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    # 이전 실행의 run.json(완성 표시)을 먼저 지우고 맨 마지막에 다시 써서, 중간에 끊긴 실행이 완성된 것처럼
    # 읽히지 않게 한다.
    (out_dir / "run.json").unlink(missing_ok=True)

    started = time.perf_counter()
    timings = []
    # 정밀도마다 하위 폴더(fp32/, int8/)에 범주별 npz를 쓴다. 범주별 소요 시간과 이미지 수를 출력하고
    # run.json에 남긴다(지연 측정은 bench.py가 따로 한다).
    for precision in precisions:
        (out_dir / precision).mkdir(parents=True, exist_ok=True)
        for category in categories:
            t0 = time.perf_counter()
            arrays = run_category(
                sessions[precision],
                metas[category],
                args.grid,
                args.ratio,
                protocol,
                category,
                manifest,
                cache,
                args.allow_test,
            )
            np.savez(out_dir / precision / f"{category}.npz", **arrays)
            line = {
                "precision": precision,
                "category": category,
                "seconds": round(time.perf_counter() - t0, 1),
            }
            line["images"] = int(len(arrays["eval_images"]) + len(arrays["pool_images"]))
            timings.append(line)
            print(json.dumps(line), flush=True)
    run = {
        "artifacts": name,
        "grid": args.grid.name,
        "ratio": args.ratio,
        "protocol": protocol,
        "precisions": precisions,
        "categories": list(categories),
        "threads": args.threads,
        "commit": commit,
        "runs": timings,
        "total_s": round(time.perf_counter() - started, 1),
    }
    with open(out_dir / "run.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
