"""M2AD의 PatchCore 검사기: (범주, 시점)마다 하나씩, 합성 교란과 실제 조명 변화에서 채점한다.

검사기마다 `<out>/<category>_<view>.npz`를 쓴다: 기준 조명(S), 거기에 건 합성 교란 15조건(P), 실제 다른
조명 아홉 가지(R)에서의 테스트 시편 점수, 교차 적합 임계값, 그리고 새 조명마다 그 조명의 정상 이미지로
재보정한 뒤의 점수와 임계값. 표는 `defect_inspect.analyze_m2ad`가 만든다.
규칙: docs/experiments.md, 3-B단계.
"""

# [파일 역할] 3-B 실행 스크립트. M2AD의 (범주, 시점)마다 PatchCore 검사기 하나(모두 6개)를 기준 조명 I01의
# 학습 시편으로 만들고, 테스트 시편을 세 종류 조건 (S) 기준 조명, (P) 기준 조명에 합성 교란 15가지, (R) 실제
# 다른 조명 아홉 가지에서 채점한다. 새 조명마다 그 조명의 정상 n장(8, 30)을 뱅크에 더해 다시 보정한 점수도
# 남긴다.
# [왜 필요한가] 질문 3(합성 교란이 실제 조명 변화를 대신하는가, H11)과 재보정 효과(H12)를 같은 시편·같은
# 검사기에서 비교하려면, 합성 교란과 실제 조명을 한 실행 안에서 같은 뱅크·같은 임계값으로 채점해야
# 한다(experiments.md 3-B).
# [결과] README: 조명만 바꿨는데 아홉 조건 중 일곱에서 정상이 전부 불량으로 나왔고(오검출률 96.9%), 가장 센
# 합성 교란은 +82.5%p(흐림)여서 H11은 기각됐다. 새 조명의 정상 30장으로 재보정하면 4.4%로 돌아왔다(H12 지지,
# 8장으로도 3.5%).
# [흐름] 인자 점검 → 메타데이터·캐시·백본 준비 → 장부 기록(전체 실행만) → 검사기마다 run_inspector(기준 뱅크와
# 교차 적합 임계값 → 조건별 채점 → 새 조명별 재보정) → npz와 run.json. 표는 analyze_m2ad가 만든다.
# [연결] run_m2ad_enrol(7단계)이 crossfit, inspector_rows, _same_specimens, _threshold_info, _MAP_SIZE를
# 그대로 가져다 써서 3-B와 같은 절차를 유지한다.

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

# Threshold·conformal_threshold: 교차 적합 점수로 목표 5% 임계값을 정한다. apply_condition·condition_names:
# 3단계 합성 교란 15조건(VisA 3-A와 같은 정의). get_config: 등록된 PatchCore 구성(p0, d-s). m2ad 모듈:
# 메타데이터, 시편 겹, 재보정 시편, 캐시. SealedTestError: 허락 없이 테스트 이미지를 읽으려 할 때 내는 예외.
import numpy as np

from . import paths
from .calibrate import Threshold, conformal_threshold
from .conditions import apply_condition, condition_names
from .configs import PatchCoreConfig, get_config
from .ledger import git_commit, record_test_access
from .m2ad import (
    CATEGORIES,
    ILLUMINATIONS,
    N_FOLDS,
    RECAL_SIZES,
    REFERENCE,
    VIEWS,
    M2adCache,
    M2adRow,
    read_meta,
    recal_specimens,
    specimen_folds,
)
from .splits import SealedTestError

# ALPHA: 목표 오검출률 5%(공통 규칙). METHODS: 3-B 대상 방법 P0(WideResNet-50, 256px)과 D-S(DINOv2 ViT-S,
# 448px). Dinomaly는 조건마다 다시 학습해야 해서 하지 않는다(experiments.md 3-B).
ALPHA = 0.05
METHODS = ("p0", "d-s")
# M2AD에는 공식 검증 분할이 없으므로, 등록한 코드 점검은 Motor 시점 000 검사기 하나에서 조건 S와 I02만
# 돌린다(experiments.md 3-B): 검사기 하나, 기준 조건 하나, 실제 조명 하나. 결과는 '기준 조명 오검출 0/20,
# I02 오검출 20/20'으로 기록돼 있다.
CHECK_INSPECTOR = ("Motor", "000")
CHECK_ILLUMINATIONS = ("02",)
# 3-B는 이미지 점수만 쓰고 점수 맵을 저장하지 않으므로, 맵 크기를 32로 줄여 계산을 아낀다. 이미지 점수는 맵을
# 어떤 크기로 맞추든 달라지지 않는다.
_MAP_SIZE = 32


# 기준 조명 I01을 뺀 실제 조명 아홉 가지(I02~I10).
def real_illuminations() -> list[str]:
    return [light for light in ILLUMINATIONS if light != REFERENCE]


# 조건 이름 목록: S(기준 조명), P:<교란 이름> 15개, R:<조명> 9개 순서. 점검 실행(check)은 S와 R:02만. 이
# 순서가 npz의 scores 행 순서다.
def condition_list(check: bool = False) -> list[str]:
    """`S`, 이어서 합성 교란 15조건의 `P:<name>`, 이어서 조명 02..10의 `R:<LL>`."""
    if check:
        return ["S", *(f"R:{light}" for light in CHECK_ILLUMINATIONS)]
    synthetic = condition_names(include_clean=False)
    return ["S", *(f"P:{name}" for name in synthetic), *(f"R:{light}" for light in real_illuminations())]


# 재보정 결과의 키 목록: '<조명>:<n>'(예: 02:8, 02:30). 조명이 바깥 순서, 재보정 장수가 안쪽 순서다.
def recal_key_list(recal_sizes: Sequence[int] = RECAL_SIZES) -> list[str]:
    """새 조명마다, 재보정 장수마다 `<LL>:<n>`. 조명이 바깥 순서다."""
    return [f"{light}:{n}" for light in real_illuminations() for n in recal_sizes]


# 검사기 하나(범주, 시점)가 한 분할·한 조명에서 보는 이미지들: 시편마다 한 장, 시편 이름 순. 조건마다 같은
# 시편이 같은 행에 오도록 이름으로 정렬하고, 한 시편에 이미지가 둘 이상이면 멈춘다.
def inspector_rows(
    rows: Sequence[M2adRow], category: str, view: str, split: str, illumination: str
) -> list[M2adRow]:
    """검사기 하나가 한 조명에서 보는 이미지: 시편마다 한 장, 시편 이름 순."""
    wanted = (category, view, split, illumination)
    picked = sorted(
        (r for r in rows if (r.category, r.view, r.split, r.illumination) == wanted), key=lambda r: r.specimen
    )
    names = [r.specimen for r in picked]
    if len(set(names)) != len(names):
        raise ValueError(f"{wanted}: a specimen has more than one image")
    return picked


# 다른 조명의 이미지 목록이 기준 조명과 같은 시편(같은 순서)인지 확인한다. 그래야 조건 사이 비교가 같은
# 시편끼리의 비교가 되고, 시편 단위 군집 부트스트랩이 성립한다.
def _same_specimens(picked: list[M2adRow], names: list[str], what: str) -> None:
    if [r.specimen for r in picked] != names:
        raise ValueError(f"{what}: the specimens differ from those under illumination {REFERENCE}")


# 주어진 특징으로 전체 뱅크를 만들고, 겹마다 그 겹을 뺀 뱅크로 그 겹 이미지를 채점해 교차 적합 점수를 모은 뒤,
# 그 점수로 목표 5% conformal 임계값을 정한다. 출력: (전체 뱅크, 교차 적합 점수, 임계값, 뱅크별 행 수).
# 겹은 시편 단위라 한 시편의 기준 조명 이미지와 새 조명 이미지가 같은 겹에 들어가 함께 빠진다(재보정 때 중요).
# 뱅크 만들기·채점은 함수로 받아, run_m2ad_enrol이 같은 절차를 다른 이미지 묶음에 다시 쓴다.
def crossfit(
    build: Callable, score: Callable, feats, images: np.ndarray, folds: np.ndarray
) -> tuple[object, np.ndarray, Threshold, dict[str, int]]:
    """모든 이미지로 만든 뱅크, 그 이미지들의 교차 적합 점수, 그 점수로 정한 conformal 임계값.

    `feats`는 [M, P, D](torch), `images`는 uint8 [M, S, S, 3], `folds`는 int [M]이다. 이미지마다 자기 겹을
    빼고 만든 뱅크로 채점한다. 한 시편의 이미지들은 같은 겹이라 함께 빠진다.
    """
    import torch

    dim = feats.shape[2]
    bank = build(feats.reshape(-1, dim))
    oof = np.full(len(images), np.nan, dtype=np.float32)
    bank_rows = {"full": int(bank.shape[0])}
    # 이미지가 없는 겹(시편이 적을 때)은 건너뛴다. 끝에서 NaN이 남았는지로 모든 이미지가 채점됐는지 확인한다.
    for fold in range(N_FOLDS):
        held = np.flatnonzero(folds == fold)
        if held.size == 0:
            continue
        fold_bank = build(feats[torch.from_numpy(folds != fold)].reshape(-1, dim))
        oof[held] = score(fold_bank, images[held])
        bank_rows[f"minus_fold_{fold}"] = int(fold_bank.shape[0])
    if np.isnan(oof).any():
        raise RuntimeError("cross-fitting did not cover every image")
    return bank, oof, conformal_threshold(oof, ALPHA), bank_rows


# 임계값 기록용 dict: 값, 순위 l, 보정 표본 수 n, 보장 여부(l > n이면 보장 불가).
def _threshold_info(thr: Threshold) -> dict:
    return {"value": thr.value, "rank": thr.rank, "n": thr.n, "guaranteed": thr.guaranteed}


# 검사기 하나(범주, 시점)를 만들고 보정·채점·재보정해 <범주>_<시점>.npz를 쓰고 요약 dict를 돌려준다.
# 봉인 규칙: 테스트 이미지는 allow_test=True(--allow-test)이거나, 등록된 점검(check=True이고
# CHECK_INSPECTOR)일 때만 읽는다. 그 밖에는 SealedTestError로 멈춘다.
def run_inspector(
    cfg: PatchCoreConfig,
    category: str,
    view: str,
    rows: Sequence[M2adRow],
    cache: M2adCache,
    extractor,
    device: str,
    out_dir: Path,
    *,
    allow_test: bool = False,
    check: bool = False,
    recal_sizes: Sequence[int] = RECAL_SIZES,
) -> dict:
    """(범주, 시점) 하나의 검사기를 만들고 보정·채점·재보정한 뒤 npz를 쓴다.

    테스트 이미지는 `allow_test=True`일 때, 또는 등록된 점검(CHECK_INSPECTOR에서 `check=True`: 조건 S와
    R:02만, 재보정 없음)에서만 읽는다.
    """
    import torch

    from .patchcore import build_bank, collect_features, score_images

    # 봉인 검사는 함수 첫머리에서 한다. allow_test는 진릿값이 아니라 정확히 True여야 하고, 점검 실행은 등록된
    # 검사기 하나에서만 테스트 이미지를 읽을 수 있다.
    if allow_test is not True and not (check and (category, view) == CHECK_INSPECTOR):
        raise SealedTestError(
            "M2AD test images are read only by the registered check (--check: "
            f"{CHECK_INSPECTOR[0]} {CHECK_INSPECTOR[1]}, S and R:02) or with allow_test=True (--allow-test)"
        )

    # 구성의 채점 설정으로 이미지 점수만 돌려주는 도우미(맵 크기는 _MAP_SIZE).
    def score(bank, images: np.ndarray) -> np.ndarray:
        return score_images(
            extractor,
            bank,
            images,
            batch_size=cfg.batch_size,
            device=device,
            reweight_k=cfg.reweight_k,
            sigma=cfg.sigma,
            map_size=_MAP_SIZE,
        ).image_scores

    # 구성의 코어셋 비율(10%)과 seed로 뱅크를 만드는 도우미.
    def build(features):
        return build_bank(features, cfg.coreset_ratio, seed=cfg.seed, device=device)

    # 이미지들의 패치 특징을 [이미지 수, 패치 수, D]로 돌려준다. 이미지 단위로 겹을 고르기 위한 모양이다.
    def features(images: np.ndarray):
        feats, (grid_h, grid_w) = collect_features(
            extractor, images, batch_size=cfg.batch_size, device=device
        )
        return feats.view(len(images), grid_h * grid_w, feats.shape[1]), (grid_h, grid_w)

    # 검사기마다 걸린 시간과 GPU 최대 메모리를 잰다.
    started = time.perf_counter()
    on_cuda = str(device).startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    # 1~2단계: 학습 시편 30개의 기준 조명 이미지 30장으로 뱅크를 만들고, 시편 이름 SHA-256 순 5겹 교차
    # 적합으로 임계값을 정한다(experiments.md 3-B '기준 조명', '임계값').
    train_ref = inspector_rows(rows, category, view, "train", REFERENCE)
    if not train_ref:
        raise ValueError(f"no train images for {category} {view} under illumination {REFERENCE}")
    # 학습 시편 이름 순서대로 겹 번호를 매긴다. 이 folds 배열이 기준 조명 이미지와 재보정 이미지 모두의 겹이
    # 된다.
    train_names = [r.specimen for r in train_ref]
    fold_of = specimen_folds(train_names)
    folds = np.array([fold_of[name] for name in train_names], dtype=np.int64)
    ref_images = cache.images(train_ref)
    ref_feats, grid = features(ref_images)
    bank, cal_score, thr, bank_rows = crossfit(build, score, ref_feats, ref_images, folds)

    # 3단계: 테스트 시편에서 조건별 채점. 먼저 기준 조명 이미지와 라벨을 읽는다. 테스트 이미지를 읽는 곳은 이
    # 3단계뿐이다.
    test_ref = inspector_rows(rows, category, view, "test", REFERENCE)
    if not test_ref:
        raise ValueError(f"no test images for {category} {view} under illumination {REFERENCE}")
    test_names = [r.specimen for r in test_ref]
    test_ref_images = cache.images(test_ref)
    ref_labels = np.array([r.label for r in test_ref], dtype=np.int8)

    # 결과 배열을 미리 잡는다: 조건 × 테스트 시편 점수·라벨, 재보정 키 × 테스트 시편 점수·라벨·임계값.
    conditions = condition_list(check)
    recal_keys = [] if check else recal_key_list(recal_sizes)
    n_test = len(test_ref)
    scores = np.empty((len(conditions), n_test), dtype=np.float32)
    labels = np.empty((len(conditions), n_test), dtype=np.int8)
    recal_scores = np.empty((len(recal_keys), n_test), dtype=np.float32)
    recal_labels = np.empty((len(recal_keys), n_test), dtype=np.int8)
    recal_thresholds = np.empty(len(recal_keys), dtype=np.float64)
    recal_info: dict[str, dict] = {}

    # 조건마다 채점한다. S는 기준 조명 그대로, P는 기준 조명 이미지에 합성 교란을 건 것(라벨은 같다), R은 실제
    # 다른 조명에서 찍은 같은 시편의 이미지(시편이 같은지 확인하고 그 조명의 라벨을 쓴다). 모두 기준 뱅크로
    # 채점하고 임계값은 기준 조명에서 정한 것 하나를 쓴다.
    for k, condition in enumerate(conditions):
        # 조건 이름 'S', 'P:<교란>', 'R:<조명>'을 종류와 인자로 나눈다.
        kind, _, arg = condition.partition(":")
        if kind == "S":
            images, shown = test_ref_images, ref_labels
        elif kind == "P":
            images, shown = apply_condition(test_ref_images, arg), ref_labels
        else:
            test_rows = inspector_rows(rows, category, view, "test", arg)
            _same_specimens(test_rows, test_names, f"{category} {view} test, illumination {arg}")
            images = cache.images(test_rows)
            shown = np.array([r.label for r in test_rows], dtype=np.int8)
        scores[k] = score(bank, images)
        labels[k] = shown
        # 재보정은 실제 조명 조건에서만 한다(합성 교란과 점검 실행은 건너뜀).
        if kind != "R" or check:
            continue

        # 4단계(실제 조명 R에서만, 점검 실행은 제외): 새 조명 L의 학습 시편 앞 n개(SHA-256 순, n = 8, 30)
        # 이미지를 기준 조명 30장에 더해 뱅크를 다시 만들고, 더한 이미지까지 넣은 시편 단위 5겹 교차 적합으로
        # 임계값을 다시 정한 뒤, 조명 L의 테스트 이미지를 채점한다(experiments.md 3-B '재보정'). 새 이미지의
        # 겹은 같은 시편의 기준 조명 이미지와 같다. 코드에서 새 조명 L은 `arg`다.
        new_rows = inspector_rows(rows, category, view, "train", arg)
        _same_specimens(new_rows, train_names, f"{category} {view} train, illumination {arg}")
        new_images = cache.images(new_rows)
        # 새 조명 학습 이미지의 특징은 조명마다 한 번만 뽑고, 재보정 장수(8, 30)마다 그중 앞 n개 시편만 골라
        # 쓴다.
        new_feats, _ = features(new_images)
        for n in recal_sizes:
            chosen = recal_specimens(train_names, n)
            pick = np.array([train_names.index(name) for name in chosen], dtype=np.int64)
            bank_n, oof_n, thr_n, rows_n = crossfit(
                build,
                score,
                torch.cat([ref_feats, new_feats[torch.from_numpy(pick)]]),
                np.concatenate([ref_images, new_images[pick]]),
                np.concatenate([folds, folds[pick]]),
            )
            # 재보정 뱅크로 조명 L의 테스트 이미지를 다시 채점하고, 점수·라벨·임계값을 재보정 키의 행에
            # 넣는다.
            key = f"{arg}:{n}"
            j = recal_keys.index(key)
            recal_scores[j] = score(bank_n, images)
            recal_labels[j] = shown
            recal_thresholds[j] = thr_n.value
            recal_info[key] = {"images": len(oof_n), "bank_rows": rows_n, "threshold": _threshold_info(thr_n)}
            # 재보정 뱅크는 조명·장수마다 새로 만드므로 바로 놓아 GPU 메모리를 아낀다.
            del bank_n

    # 검사기별 결과 파일: 테스트 시편 이름·불량 여부, 조건 이름, 조건별 점수·라벨, 기준 임계값과 교차 적합
    # 보정 점수, 재보정 키·점수·임계값·라벨. 시편 단위 군집 부트스트랩을 analyze_m2ad가 이 파일로 한다.
    np.savez(
        out_dir / f"{category}_{view}.npz",
        specimens=np.array(test_names),
        object_anomaly=np.array([r.object_anomaly for r in test_ref], dtype=np.int8),
        conditions=np.array(conditions),
        scores=scores,
        labels=labels,
        threshold=np.float64(thr.value),
        cal_score=cal_score,
        recal_keys=np.array(recal_keys, dtype=str),
        recal_scores=recal_scores,
        recal_thresholds=recal_thresholds,
        recal_labels=recal_labels,
    )
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if on_cuda else 0.0
    # run.json에 들어갈 요약: 시편 수, 패치 격자·차원, 시편별 겹, 뱅크 행 수, 임계값 정보, 재보정 정보, 최대
    # VRAM, 소요 시간.
    return {
        "category": category,
        "view": view,
        "train_specimens": len(train_names),
        "test_specimens": n_test,
        "test_anomalous_specimens": int(sum(r.object_anomaly for r in test_ref)),
        "grid": list(grid),
        "dim": int(ref_feats.shape[2]),
        "folds": {name: int(fold_of[name]) for name in train_names},
        "bank_rows": bank_rows,
        "threshold": _threshold_info(thr),
        "recal": recal_info,
        "peak_vram_mb": round(peak_mb, 1),
        "seconds": round(time.perf_counter() - started, 1),
    }


# 구성의 백본 추출기를 만든다. centre는 7단계 E1(이미지별 특징 중심화) 설정에서만 켜지고 p0·d-s에서는 꺼져
# 있다.
def _make_extractor(cfg: PatchCoreConfig, device: str):
    from .backbones import make_extractor

    return make_extractor(cfg.backbone, img_size=cfg.img_size, centre=cfg.centre).to(device).eval()


# 명령줄 진입점. 전체 측정: `python -m defect_inspect.run_m2ad --method p0 --allow-test --stage
# 3-m2ad`(README). --check는 등록한 코드 점검이라 <out>-check 폴더에 쓰고 장부에 남기지 않으며, 전체 측정은
# --allow-test와 --stage가 있어야 하고 장부에 한 줄 남긴다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=METHODS, required=True, help="PatchCore 구성")
    parser.add_argument(
        "--check",
        action="store_true",
        help="등록한 코드 점검: Motor 000 검사기에서 S와 R:02만 돌려 <out>-check에 쓴다",
    )
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--out", type=Path, default=None, help="기본값: outputs/m2ad-<method>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-test", action="store_true", help="M2AD 테스트 이미지를 읽는다(장부에 기록)")
    parser.add_argument(
        "--stage", default="", help="봉인 테스트 장부에 남길 단계 이름(--allow-test와 함께 필수)"
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jsons", type=Path, default=None, help="기본값: data/raw/m2ad/jsons.zip")
    parser.add_argument("--cache", type=Path, default=None, help="캐시 폴더(기본값: data/cache)")
    args = parser.parse_args(argv)

    # 인자 점검: 전체 측정은 M2AD 테스트 이미지를 읽으므로 --allow-test와 --stage가 필요하다. 점검 실행은 출력
    # 폴더 이름에 -check를 붙여 전체 측정 결과와 섞이지 않게 한다.
    cfg = get_config(args.method)
    if not args.check and (not args.allow_test or not args.stage):
        parser.error("the full run reads the M2AD test images: it needs --allow-test and --stage")
    out_dir = args.out or paths.OUTPUTS / f"m2ad-{args.method}"
    if args.check:
        out_dir = out_dir.with_name(out_dir.name + "-check")
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")

    # 메타데이터·캐시·백본 준비처럼 테스트 이미지 없이 실패할 수 있는 일을 장부 기록 전에 끝낸다.
    rows = read_meta(args.jsons or paths.RAW / "m2ad" / "jsons.zip")
    cache = M2adCache(args.cache or paths.CACHE, cfg.img_size)
    extractor = _make_extractor(cfg, args.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    # 점검 실행은 등록된 검사기 하나만, 전체 측정은 고른 범주 × 세 시점 검사기를 돌리고 장부에 남긴다.
    if args.check:
        inspectors = [CHECK_INSPECTOR]
    else:
        inspectors = [
            (category, view) for category in CATEGORIES if category in args.categories for view in VIEWS
        ]
        record_test_access(
            paths.TEST_LEDGER, stage=args.stage, config=f"m2ad-{args.method}", note=args.note, commit=commit
        )
    # --overwrite로 다시 돌릴 때, 중간에 끊긴 재실행이 그것이 대신하는 완성된 실행처럼 보이면 안 되므로 옛
    # run.json(완성 표시)을 지운다. 지우는 시점은 장부 기록 뒤다: 장부에 기록하지 못한 재실행은 아무것도 읽지
    # 않고 이전 실행 결과를 그대로 둔다.
    (out_dir / "run.json").unlink(missing_ok=True)

    summaries = []
    started = time.perf_counter()
    for category, view in inspectors:
        info = run_inspector(
            cfg,
            category,
            view,
            rows,
            cache,
            extractor,
            args.device,
            out_dir,
            # 점검 실행에서는 --allow-test를 줘도 allow_test를 끄고 check 규칙(Motor 000, S와 R:02)으로만
            # 테스트 이미지를 읽게 한다.
            allow_test=args.allow_test and not args.check,
            check=args.check,
        )
        summaries.append(info)
        print(json.dumps({k: v for k, v in info.items() if k not in ("folds", "recal")}), flush=True)

    # 실행 전체 기록: 방법·구성, 커밋, 장치(코어셋 선택이 장치에 따라 반올림 수준으로 달라지므로 남김), 점검
    # 여부, 목표 α, 기준 조명, 조건·재보정 키 목록, 검사기별 요약.
    run = {
        "method": args.method,
        "config": asdict(cfg),
        "commit": commit,
        # 코어셋 선택은 장치에 따라 부동소수 반올림 수준으로 달라진다.
        "device": args.device,
        "check": args.check,
        "alpha": ALPHA,
        "reference": REFERENCE,
        "conditions": condition_list(args.check),
        "recal_keys": [] if args.check else recal_key_list(),
        "inspectors": summaries,
        "total_s": round(time.perf_counter() - started, 1),
    }
    with open(out_dir / "run.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
