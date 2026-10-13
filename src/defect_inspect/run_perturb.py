"""Score the evaluation set of a finished run again under the synthetic conditions of stage 3.

Nothing is trained or calibrated here: the memory bank (PatchCore) or the model (Dinomaly) and the scores
the thresholds are fixed from all come from the source run. Per category this writes
`<out>/<category>.npz` with the image scores under every condition (clean first). Tables are made by
`defect_inspect.analyze_perturb`.
"""

# [파일 역할] 3단계 3-A 실행기(python -m defect_inspect.run_perturb). 이미 끝난 실행(원본 실행: PatchCore의
# run_patchcore 또는 Dinomaly의 run_dinomaly eval)의 평가 세트를 합성 교란 15가지(밝기·감마·흐림·이동·JPEG ×
# 강도 3)와 교란 없는 조건으로 다시 채점해 저장한다. 7단계에서는 p0-c·d-s-c(이미지별 특징 중심화)의 dev
# 점검에도 쓴다.
# [설계 이유] 질문은 "촬영 조건이 조금 바뀌면 이미 고정한 임계값의 오검출률이 어떻게 되나"다. 그래서 여기서는
# 아무것도 다시 학습·보정하지 않고, 원본 실행의 뱅크(또는 모델)와 임계값용 점수를 그대로
# 가져온다(docs/experiments.md 3-A). 교란 없는 조건을 먼저 채점해 원본 점수와 같은지(상대 차이 0.1% 미만)
# 확인한 뒤에만 나머지 조건을 채점한다. 같지 않으면 임계값이 지금 채점하는 모델의 것이 아니라는 뜻이라 멈춘다.
# [흐름] main: 인자·원본 실행 검사(이미지 없이 할 수 있는 검사를 모두 먼저) → 채점기 준비(PatchCore 뱅크 또는
# Dinomaly 모델) → (test면) 장부 기록 → 범주마다 run_category(조건별 채점) → run.json.
# [연결] conditions.apply_condition(교란), patchcore.score_images / run_dinomaly.predict(채점). 결과는
# analyze_perturb(3단계 표), analyze_m2ad_enrol visa-dev(7단계 E1 점검)가 읽는다.

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

# torch, patchcore, run_dinomaly는 필요할 때 함수 안에서 가져온다(torch 없이 도는 CI에서 import 가능하게,
# Dinomaly는 anomalib이 필요해서 dm을 쓸 때만).
import numpy as np

# 내부 모듈: 교란 이름과 적용(conditions), 구성 확인(configs.same_config), 모델 파일
# 해시(download.sha256_file), 봉인 테스트 장부, 분할 선택, 범주 목록.
from . import paths
from .cache import ImageCache
from .conditions import CLEAN, apply_condition, condition_names
from .configs import get_config, same_config
from .download import sha256_file
from .ledger import git_commit, record_test_access
from .splits import ManifestRow, read_manifest, select
from .visa import CATEGORIES

# 다시 채점할 수 있는 방법: PatchCore 구성 네 가지와 Dinomaly(dm).
# CLEAN_RTOL: 교란 없는 점수가 원본 점수와 이 상대 차이 안에 들어야 한다(규칙의 허용 0.1%, 실제 3-A 결과는 0).
# p0-c and d-s-c: the stage 7 (E1) centred variants, dev protocol only by the rules.
PATCHCORE_METHODS = ("p0", "d-s", "p0-c", "d-s-c")
DINOMALY = "dm"
METHODS = (*PATCHCORE_METHODS, DINOMALY)
# The clean scores must reproduce the source run's scores within this relative difference.
CLEAN_RTOL = 1e-3

# 채점기 타입: 이미지 배열을 받아 이미지 점수를 돌려주는 함수. PatchCore와 Dinomaly를 같은 모양으로 감싸
# run_category가 방법을 몰라도 되게 한다.
Scorer = Callable[[np.ndarray], np.ndarray]  # uint8 images [N, S, S, 3] -> image scores [N]


# 교란 없는 점수가 원본과 다를 때 내는 전용 예외. 뱅크·모델·이미지 중 하나가 원본 실행과 달라 원본의 임계값을
# 쓸 수 없다는 뜻이다.
class CleanScoreMismatch(RuntimeError):
    """The clean scores differ from the source run: its thresholds do not belong to what is scored here."""


# 원본 실행에서 범주 하나에 대해 가져오는 것: 평가 이미지 경로, 교란 없는 평가 점수(대조용), 임계값 보정 점수.
@dataclass(frozen=True)
class Source:
    """What a perturbation run takes from one category of the run whose thresholds it reuses."""

    eval_images: np.ndarray  # str [N], image paths of the evaluation set
    eval_score: np.ndarray  # float32 [N], scores of the evaluated model on the unchanged images
    cal_score: np.ndarray  # float32, scores the thresholds are fixed from (may be empty)


# 원본 .npz에서 읽을 열 이름. Dinomaly(2단계 공통 형식)는 eval_score와 cal_score(홀드아웃), PatchCore는 전체
# 뱅크의 평가 점수와 교차 적합 점수다. 3-A 결과의 "P0·D-S 교차 적합, DM 홀드아웃" 임계값이 여기서 정해진다.
def source_keys(method: str) -> tuple[str, str]:
    """npz keys of (evaluation scores, calibration scores) in the source run of `method`."""
    if method == DINOMALY:
        return "eval_score", "cal_score"  # common stage 2 format
    return "eval_score_full", "pool_score_oof"  # run_patchcore: full bank, cross-fitted pool scores


# 원본 실행의 범주 파일을 읽고 이미지 없이 할 수 있는 검사(열 존재, 모양, 유한값)를 한다. 끝나지 않은 실행이나
# 다른 방법의 실행을 원본으로 잘못 넘긴 경우를 여기서 잡는다.
def load_source(source_dir: Path, category: str, method: str) -> Source:
    """One category of the source run, checked as far as the file alone allows (no image is needed)."""
    score_key, cal_key = source_keys(method)
    path = Path(source_dir) / f"{category}.npz"
    with np.load(path) as z:
        missing = [k for k in ("eval_images", score_key, cal_key) if k not in z.files]
        if missing:
            raise KeyError(f"{path} has no {missing}: it is not a finished {method} run")
        source = Source(
            eval_images=z["eval_images"],
            eval_score=z[score_key].astype(np.float32),
            cal_score=z[cal_key].astype(np.float32),
        )
    n = len(source.eval_images)
    if source.eval_score.shape != (n,) or source.cal_score.ndim != 1:
        raise ValueError(
            f"{path}: {score_key} {source.eval_score.shape} and {cal_key} {source.cal_score.shape} are "
            f"not one score per image for {n} evaluation images and a flat list of calibration scores"
        )
    if not (np.isfinite(source.eval_score).all() and np.isfinite(source.cal_score).all()):
        raise ValueError(f"{path}: {score_key} or {cal_key} is not finite")
    return source


# 매니페스트에서 평가 세트를 고르고, 원본 실행이 채점한 이미지 목록과 순서까지 같은지 확인한다. 이미지를 읽기
# 전에 매니페스트만으로 하므로, 다른 절차·다른 매니페스트로 만든 원본이면 봉인 테스트를 읽기 전에 멈춘다.
def evaluation_rows(
    protocol: str, category: str, manifest: list[ManifestRow], source: Source, allow_test: bool
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    """The (normal, defect) evaluation rows of a category, which must be the images `source` scored.

    Reads the manifest only, so a source that does not fit is found before any image is touched.
    """
    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    rows = eval_normal + eval_defect
    if not rows:
        raise ValueError(f"{category}: the manifest has no evaluation images for the {protocol} protocol")
    if [r.image for r in rows] != [str(image) for image in source.eval_images]:
        raise ValueError(
            f"{category}: the source run did not evaluate the {protocol} evaluation set of this manifest"
        )
    return eval_normal, eval_defect


# 이미지별 상대 차이의 최댓값. 분모가 0이면 나눗셈이 터지지 않게 float64의 가장 작은 양수(tiny)로 막는다.
def max_rel_diff(scores: np.ndarray, reference: np.ndarray) -> float:
    """Largest `|score - reference| / |reference|` over the images (0.0 for an empty set)."""
    a = np.asarray(scores, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"score shapes differ: {a.shape} and {b.shape}")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), np.finfo(np.float64).tiny)))


# 저장된 전체 뱅크로 채점하는 함수를 만든다. 배치 크기·재가중 이웃 수·맵 σ는 원본과 같은 구성 값을 써서 교란
# 없는 점수가 원본과 같게 나오게 한다.
def patchcore_scorer(cfg, extractor, bank, device: str) -> Scorer:
    """Image scores against a saved memory bank, with the batch size and scoring options of `cfg`."""
    from .patchcore import score_images

    def score(images: np.ndarray) -> np.ndarray:
        return score_images(
            extractor,
            bank,
            images,
            batch_size=cfg.batch_size,
            device=device,
            reweight_k=cfg.reweight_k,
            sigma=cfg.sigma,
        ).image_scores

    return score


# run_dinomaly를 늦게 가져온다. Dinomaly 모델을 만들려면 anomalib이 필요해서, PatchCore만 돌릴 때는 그
# 의존성을 요구하지 않는다.
def _dinomaly():
    """`run_dinomaly`, imported only when the dm method is used (building its model needs anomalib)."""
    from . import run_dinomaly

    return run_dinomaly


# run_dinomaly train이 쓴 model.pt를 CPU에서 eval 모드로 읽는다. 지금의 run_dinomaly에는 load_model이 있어
# 그것을 쓰고, 없으면 2단계 초기 파일 형식(인코더 이름 + 학습한 파라미터)으로 직접 복원한다.
def load_dinomaly(path: Path):
    """The model written by `run_dinomaly train`, on the CPU in eval mode."""
    dm = _dinomaly()
    loader = getattr(dm, "load_model", None)
    if loader is not None:
        return loader(path)[0]
    # The file layout of contract-stage2: the trainable state with the encoder name and step count.
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = dm.build_model(saved["encoder"])
    # 저장된 키가 학습 가능한 파라미터 이름과 정확히 같은지 확인한다. 고정된 인코더 가중치는 파일에 없으므로
    # strict=False로 읽되, 빠지거나 남는 키가 있으면 다른 모델의 파일로 보고 멈춘다.
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    if set(saved["state"]) != expected:
        odd = sorted(set(saved["state"]) ^ expected)
        raise ValueError(f"{path} does not hold the trainable parameters of {saved['encoder']}: {odd[:3]}")
    model.load_state_dict(saved["state"], strict=False)
    return model.eval()


# Dinomaly 모델로 채점하는 함수를 만든다. 배치 크기와 amp(fp16 추론 여부)는 원본 실행과 같은 값을
# 받는다(main에서 원본 run.json으로 정함).
def dinomaly_scorer(model, device: str, batch_size: int, amp: bool) -> Scorer:
    predict = _dinomaly().predict

    def score(images: np.ndarray) -> np.ndarray:
        return predict(model, images, batch_size=batch_size, device=device, amp=amp).image_scores

    return score


# 채점할 조건 목록. 기본은 등록된 16개 전부(clean 먼저)이고, 일부만 고르면(--conditions, 7단계 E1 점검의
# clean·brightness-3·gamma-3 등) 등록 순서를 지킨다. clean은 원본 점수와의 대조에 꼭 필요하므로 빠지면
# 거절한다.
def select_conditions(requested: list[str] | None = None) -> list[str]:
    """The conditions to score, in the registered order: all of them, or `requested` (clean included)."""
    names = condition_names()
    if requested is None:
        return names
    unknown = sorted(set(requested) - set(names))
    if unknown:
        raise ValueError(f"unknown conditions {unknown}; known: {names}")
    if CLEAN not in requested:
        raise ValueError(f"the {CLEAN!r} condition is needed: it checks the source run's scores")
    return [name for name in names if name in requested]


# 범주 하나의 평가 세트를 조건마다 교란해 채점하고 <범주>.npz로 쓴다. 저장 내용: 이미지 경로·라벨, 조건 이름,
# 점수 [조건, 이미지], 원본의 보정 점수(analyze_perturb가 임계값을 다시 계산할 수 있게).
# 호출 위치: main, tests/test_run_perturb.py.
def run_category(
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache: ImageCache,
    score: Scorer,
    source: Source,
    allow_test: bool,
    out_dir: Path,
    *,
    rtol: float = CLEAN_RTOL,
    conditions: list[str] | None = None,
) -> dict:
    """Score one category's evaluation set under every condition and write `<category>.npz`.

    `conditions` defaults to all of them; a subset keeps the registered order and must hold the clean
    condition. That comes first and must reproduce `source.eval_score` (CleanScoreMismatch otherwise,
    before any other condition is scored).
    """
    eval_normal, eval_defect = evaluation_rows(protocol, category, manifest, source, allow_test)
    rows = eval_normal + eval_defect
    names = [r.image for r in rows]
    labels = np.array([0] * len(eval_normal) + [1] * len(eval_defect), dtype=np.int8)
    conditions = select_conditions(conditions)
    # clean이 맨 앞이어야 대조가 실패했을 때 다른 조건을 채점하는 시간을 쓰지 않는다.
    if conditions[0] != CLEAN:
        raise RuntimeError("the clean condition must come first")

    # 원본 이미지(모델 입력 크기로 줄인 캐시)를 한 번 읽고, 조건마다 apply_condition으로 교란한 복사본을
    # 만들어 채점한다. 교란 시간과 채점 시간을 따로 잰다.
    images = cache.images(rows)
    scores = np.empty((len(conditions), len(rows)), dtype=np.float32)
    clean_diff = float("nan")
    perturb_s = score_s = 0.0
    for k, name in enumerate(conditions):
        t0 = time.perf_counter()
        changed = apply_condition(images, name)
        t1 = time.perf_counter()
        result = np.asarray(score(changed), dtype=np.float32)
        t2 = time.perf_counter()
        perturb_s += t1 - t0
        score_s += t2 - t1
        # 점수 개수와 유한값을 확인한다. 흐림·JPEG 같은 강한 교란에서 점수가 비정상이 되면 조용히 저장하지
        # 않고 멈춘다.
        if result.shape != (len(rows),):
            raise ValueError(f"{category}/{name}: expected {len(rows)} scores, got shape {result.shape}")
        if not np.isfinite(result).all():
            raise FloatingPointError(f"{category}/{name}: non-finite image scores")
        scores[k] = result
        # 교란 없는 점수가 원본 점수와 상대 0.1% 안에서 같아야 원본의 임계값을 이 점수에 그대로 쓸 수 있다.
        # not (x < rtol)로 써서 NaN도 실패로 처리한다.
        if name == CLEAN:
            clean_diff = max_rel_diff(result, source.eval_score)
            if not clean_diff < rtol:
                raise CleanScoreMismatch(
                    f"{category}: the clean scores differ from the source run by up to {clean_diff:.3g} "
                    f"(relative; limit {rtol:g}). The bank or model, or the images, are not the ones the "
                    "source run's thresholds were fixed with."
                )

    np.savez(
        out_dir / f"{category}.npz",
        eval_images=np.array(names),
        eval_labels=labels,
        conditions=np.array(conditions),
        scores=scores,
        cal_score=np.asarray(source.cal_score, dtype=np.float32),
    )
    return {
        "category": category,
        "eval_normal": len(eval_normal),
        "eval_defect": len(eval_defect),
        "cal": int(len(source.cal_score)),
        "clean_max_rel_diff": clean_diff,
        "timing": {"perturb_s": round(perturb_s, 2), "score_s": round(score_s, 2)},
    }


# 저장된 뱅크의 행 수만 읽는다. mmap으로 헤더만 보고, 바로 del로 매핑을 풀어 Windows에서 파일이 잠긴 채 남지
# 않게 한다.
def _bank_rows(path: Path) -> int:
    """Number of rows of a saved memory bank, read from the file header only."""
    bank = np.load(path, mmap_mode="r")
    rows = int(bank.shape[0])
    del bank  # unmaps: Windows cannot replace a file that is still mapped
    return rows


# 출력 폴더가 원본 폴더와 같은지 확인한다(.., 링크, 대소문자 차이까지). 같으면 원본의 run.json과 .npz를
# 덮어쓰게 되므로 main이 거절한다.
def _same_dir(a: Path, b: Path) -> bool:
    """Whether two paths name one directory (also through `..`, links or another spelling of the case)."""
    if a.exists() and b.exists():
        return a.samefile(b)
    return a.resolve() == b.resolve()


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


# 명령행 진입점. 원본 실행에 대한 검사를 이미지 없이 모두 끝낸 뒤에야 장부를 쓰고 봉인 테스트 이미지를 읽는다.
# 검사: 절차가 같은지, 범주별 점수 파일이 있는지, 평가 이미지 목록이 매니페스트와 같은지, (PatchCore) 구성이
# 같은지와 뱅크 파일·행 수, (Dinomaly) 모델 파일의 sha256이 원본과 같은지.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--protocol", choices=["dev", "test"], required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="finished run whose bank or model scores are reused (default: outputs/<method>-<protocol>)",
    )
    parser.add_argument("--model", type=Path, default=None, help="dm only (default: outputs/dm/model.pt)")
    parser.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/perturb-<method>-<protocol>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-test", action="store_true", help="read the sealed test set (logged)")
    parser.add_argument(
        "--stage", default="", help="stage label for the test ledger (required with --allow-test)"
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--conditions", nargs="+", default=None, help="a subset in the registered order (clean included)"
    )
    args = parser.parse_args(argv)

    if args.protocol == "test" and (not args.allow_test or not args.stage):
        parser.error("--protocol test needs --allow-test and --stage")
    try:
        conditions = select_conditions(args.conditions)
    except ValueError as err:
        parser.error(str(err))
    if args.model is not None and args.method != DINOMALY:
        parser.error("--model is only used with --method dm")
    source_dir = args.source or paths.OUTPUTS / f"{args.method}-{args.protocol}"
    out_dir = args.out or paths.OUTPUTS / f"perturb-{args.method}-{args.protocol}"
    if _same_dir(out_dir, source_dir):
        # Both runs write run.json and <category>.npz: the source run would be replaced file by file.
        parser.error(f"--out is the source run {source_dir}: the perturbation run would overwrite it")
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    if not (source_dir / "run.json").exists():
        parser.error(f"{source_dir} holds no finished run (run.json is missing); see --source")
    with open(source_dir / "run.json", encoding="utf-8") as f:
        source_meta = json.load(f)
    # dev 실행의 임계값으로 test 이미지를 채점하는 식의 섞임을 막는다. 임계값은 같은 절차의 원본에서 와야
    # 한다.
    if source_meta.get("protocol") != args.protocol:
        parser.error(
            f"{source_dir} is a {source_meta.get('protocol')!r} run: thresholds of the {args.protocol} "
            "protocol are needed"
        )

    commit = git_commit()
    manifest = read_manifest(paths.VISA_MANIFEST)
    known = {r.category for r in manifest}
    unknown = [c for c in args.categories if c not in known]
    if unknown or not args.categories:
        parser.error(f"unknown or missing categories {unknown}; the manifest has {sorted(known)}")
    absent = [c for c in args.categories if not (source_dir / f"{c}.npz").exists()]
    if absent:
        parser.error(f"{source_dir} has no scores for {absent}")
    # Every category's source scores and image list, not only the first one's: none of this needs an
    # image, so a source that does not fit must stop the run before a test read is recorded or made.
    sources: dict[str, Source] = {}
    for category in args.categories:
        try:
            sources[category] = load_source(source_dir, category, args.method)
            evaluation_rows(args.protocol, category, manifest, sources[category], args.allow_test)
        except (KeyError, ValueError) as err:
            parser.error(str(err.args[0]))

    import torch

    source_config = source_meta.get("config") or {}
    # Dinomaly: 모델 파일의 sha256이 원본 run.json에 적힌 값과 다르면 거절한다(기록이 없으면 통과). 원본과
    # 같은 배치 크기·정밀도로 채점해야 교란 없는 점수를 비교할 수 있다. 모델 하나를 모든 범주가 같이 쓰므로
    # scorer_for는 늘 같은 채점기를 돌려준다.
    if args.method == DINOMALY:
        dm = _dinomaly()
        img_size = int(dm.IMG_SIZE)
        model_path = args.model or paths.OUTPUTS / DINOMALY / "model.pt"
        if not model_path.exists():
            parser.error(f"{model_path} does not exist: run `python -m defect_inspect.run_dinomaly train`")
        model_sha256 = sha256_file(model_path)
        if source_config.get("model_sha256") not in (None, model_sha256):
            parser.error(f"{model_path} is not the model that {source_dir} was scored with (sha256 differs)")
        # Same batches and precision as the source run, so that the clean scores can be compared.
        batch_size = int(source_config.get("batch_size", 32))
        amp = bool(source_config.get("amp", True))
        model = load_dinomaly(model_path).to(args.device)
        scorer = dinomaly_scorer(model, args.device, batch_size, amp)
        config = {
            "name": DINOMALY,
            "img_size": img_size,
            "model": str(model_path),
            "model_sha256": model_sha256,
            "batch_size": batch_size,
            "amp": amp,
        }

        def scorer_for(category: str) -> Scorer:
            return scorer
    else:
        # PatchCore: 원본이 같은 등록 구성으로 만들어졌는지(same_config), 뱅크 파일(--save-bank로 남긴 것)이
        # 있는지, 뱅크 행 수가 원본 run.json의 기록과 같은지 확인한다. 행 수가 다르면 다른 실행이 남긴 뱅크로
        # 보고 멈춘다.
        cfg = get_config(args.method)
        img_size = cfg.img_size
        if not same_config(source_config, cfg):
            parser.error(
                f"{source_dir} was not made with the registered config {cfg.name!r}: {source_config}"
            )
        no_bank = [c for c in args.categories if not (source_dir / f"{c}_bank.npy").exists()]
        if no_bank:
            parser.error(
                f"{source_dir} has no memory bank (<category>_bank.npy) for {no_bank}: make the source run "
                f"with `python -m defect_inspect.run_patchcore --config {cfg.name} --protocol "
                f"{args.protocol} --save-bank`"
            )
        # A bank left over from another run is caught here, before the test images are read.
        recorded = {
            info.get("category"): (info.get("bank_rows") or {}).get("full")
            for info in source_meta.get("categories", [])
        }
        stale = [
            c
            for c in args.categories
            if recorded.get(c) is not None and _bank_rows(source_dir / f"{c}_bank.npy") != recorded[c]
        ]
        if stale:
            parser.error(
                f"the memory bank of {stale} in {source_dir} does not have the number of rows that its "
                "run.json records: it was written by another run"
            )
        from .backbones import make_extractor

        extractor = make_extractor(cfg.backbone, img_size=cfg.img_size, centre=cfg.centre)
        extractor = extractor.to(args.device).eval()
        config = asdict(cfg)

        # 범주마다 그 범주의 뱅크를 읽어 채점기를 만든다(뱅크는 범주별).
        def scorer_for(category: str) -> Scorer:
            bank = torch.from_numpy(np.load(source_dir / f"{category}_bank.npy"))
            return patchcore_scorer(cfg, extractor, bank, args.device)

    cache = ImageCache(paths.CACHE, img_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 다시 돌리는 실행은 끝날 때까지 끝난 실행으로 보이지 않게 run.json부터 지운다.
    (out_dir / "run.json").unlink(missing_ok=True)  # a run that is being redone is not a finished run
    if args.protocol == "test":
        # The source run (config, protocol, every category's scores, image list and bank or the model)
        # has been checked without an image: the read starts here.
        record_test_access(
            paths.TEST_LEDGER,
            stage=args.stage,
            config=f"perturb-{args.method}",
            note=args.note,
            commit=commit,
        )

    cuda = torch.device(args.device).type == "cuda"
    summaries = []
    started = time.perf_counter()
    for category in args.categories:
        t0 = time.perf_counter()
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        info = run_category(
            args.protocol,
            category,
            manifest,
            cache,
            scorer_for(category),
            sources[category],
            args.allow_test,
            out_dir,
            conditions=conditions,
        )
        info["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else 0.0
        info["seconds"] = round(time.perf_counter() - t0, 1)
        summaries.append(info)
        print(json.dumps(info, ensure_ascii=False), flush=True)

    # run.json에 원본 실행의 경로·커밋·장치와 대조 허용치를 함께 남겨, 이 결과가 어느 임계값에 대한 것인지
    # 추적할 수 있게 한다.
    run = {
        "method": args.method,
        "protocol": args.protocol,
        "commit": commit,
        "device": args.device,
        "source": str(source_dir),
        "source_commit": source_meta.get("commit"),
        "source_device": source_meta.get("device"),
        "config": config,
        "conditions": conditions,
        "clean_rtol": CLEAN_RTOL,
        "categories": summaries,
        "total_s": round(time.perf_counter() - started, 1),
    }
    _write_json(out_dir / "run.json", run)


if __name__ == "__main__":
    main()
