"""7단계: M2AD 검사기의 다조명 등록과 이미지별 특징 중심화.

후보(방법 P0, D-S마다): e0은 3-B와 같다(I01에서 찍은 학습 시편 30개로 뱅크와 임계값). e1은 이미지별 특징
중심화. e2는 I01과 두 조명 묶음 중 하나를 등록한다(묶음을 맞바꾼 두 팔이라 모든 조명이 한 번씩 못 본 조명이
된다). e3은 둘 다.

`val`은 학습 시편만 읽는다: 시편 단위 바깥 겹마다, 나머지 겹으로 뱅크와 중첩 교차 적합 임계값을 만들고
빼 둔 시편을 조명 열 가지 모두에서 채점한다. `test`는 봉인된 M2AD 테스트 이미지를 읽는다(장부에 기록):
e0과 `val`이 고른 후보, 그리고 --loop면 e0 위의 폐루프 재등록 팔. 표와 판정은
`defect_inspect.analyze_m2ad_enrol`, 규칙은 docs/experiments.md 7단계.
"""

# [파일 역할] 7단계 실행기(python -m defect_inspect.run_m2ad_enrol val|test). M2AD의 검사기 6개(Motor·Bird ×
# 시점 000·120·240)에서, 조명이 바뀌어도 재보정 없이 버티는 후보 네 가지(E0 그대로, E1 이미지별 특징 중심화,
# E2 여러 조명 미리 등록, E3 둘 다)를 채점하고, test에서는 경보 뒤 무라벨 이미지로 자동 재등록하는 폐루프
# 팔(--loop)도 잰다.
# [왜 필요한가] 3-B에서 기준 조명 I01로만 만든 검사기는 다른 실제 조명에서 정상의 96.9%를 불량으로 판정했다.
# 사람이 새 조명의 정상으로 재보정하면 돌아오지만 언제 해야 하는지 알 수 없다(docs/experiments.md 7단계
# 머리말). 그래서 (1) 미리 대비하는 방법과 (2) 경보 뒤 자동으로 회복하는 방법을 미리 정한 규칙으로 잰다.
# [설계 이유] 봉인 test를 한 번만 읽도록 후보 선택은 학습 시편만 쓰는 val에서 한다. val은 바깥 5겹(시편
# 단위)으로 평가 시편을 빼고, 남은 시편으로 뱅크와 "중첩" 교차 적합 임계값을 만들어 평가 시편이 뱅크에도
# 임계값에도 들어가지 않게 한다(test와 같은 구조). E2·E3는 아홉 조명을 두 묶음(A 4개, B 5개)으로 나눠 한
# 묶음을 등록하고 다른 묶음을 "못 본 조명"으로 재며, 두 팔을 바꿔 모든 조명이 한 번씩 못 본 조명이 되게 한다.
# [흐름] main → (val) validate_inspector를 후보 × 검사기마다 / (test) val.json의 선택과 E0을
# sealed_inspector로, --loop면 loop_inspector → run.json.
# [연결] 3-B의 run_m2ad에서 교차 적합(crossfit)과 행 선택 함수를 그대로 가져다 쓴다(3-B 코드는 고치지 않음).
# m2ad.py(메타데이터, 캐시, 시편 겹, 조명 묶음), backbones.CentredPatches(E1). 결과는
# analyze_m2ad_enrol(val/test 표와 H18·H19 판정)이 읽는다.

import argparse
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

# torch, patchcore, backbones는 실제로 채점할 때 함수 안에서 가져온다(torch 없이 도는 CI에서도 import
# 가능하게).
import numpy as np

# m2ad: 범주·시점·조명 목록, 기준 조명(I01), 시편 겹(specimen_folds), 앞 n개 시편(recal_specimens), 조명
# 묶음(illumination_groups), 캐시. run_m2ad: 3-B와 같은 교차 적합·행 선택·임계값 기록을 그대로 써서 E0이 3-B와
# 같은 절차가 되게 한다(test의 stage3b_match 대조). splits: 봉인 예외와 SHA-256 정렬 키.
from . import paths
from .configs import PatchCoreConfig, get_config
from .ledger import git_commit, record_test_access
from .m2ad import (
    CATEGORIES,
    ILLUMINATIONS,
    N_FOLDS,
    REFERENCE,
    VIEWS,
    M2adCache,
    M2adRow,
    illumination_groups,
    read_meta,
    recal_specimens,
    specimen_folds,
)
from .run_m2ad import _MAP_SIZE, _same_specimens, _threshold_info, inspector_rows
from .run_m2ad import crossfit as _crossfit
from .splits import SealedTestError, path_key

# 규칙의 상수.
# - METHODS: 3-B와 같은 두 PatchCore 구성(P0 = WRN-50 256px, D-S = DINOv2 ViT-S 448px)
# - CANDIDATES: 단순한 순서. val에서 오검출 수가 같으면 앞의 것(단순한 쪽)을 고른다(규칙의 선택 규칙)
# - CENTRE, MULTI: 후보마다 중심화와 다조명 등록을 켤지
# - LOOP_N, LOOP_TRIM, LOOP_ARMS: 폐루프에서 경보 뒤 무라벨 20장, 그중 점수 상위 20%(4장)를 버림, 팔은 거르기
#   + 결함 0/1/2장과 대조(거르지 않음 + 결함 2장)
# - INSPECTORS: 검사기 이름 "범주_시점" 6개
METHODS = ("p0", "d-s")
CANDIDATES = ("e0", "e1", "e2", "e3")  # 단순한 순서(`val`에서 동률이면 앞의 것)
CENTRE = {"e0": False, "e1": True, "e2": False, "e3": True}
MULTI = {"e0": False, "e1": False, "e2": True, "e3": True}
# 폐루프 팔(test에서만, e0 위): 새 조명의 무라벨 이미지 n장, 그중 k장이 결함.
LOOP_N = 20
LOOP_TRIM = 0.2  # 강건 거르기가 버리는 묶음 비율: 묶음 안 점수가 가장 높은 것부터
LOOP_ARMS = (("trim", 0), ("trim", 1), ("trim", 2), ("none", 2))
INSPECTORS = tuple(f"{category}_{view}" for category in CATEGORIES for view in VIEWS)


# 후보의 팔 목록(팔 이름, 등록 조명). 단일 조명 후보(E0, E1)는 I01만 등록한 팔 하나(ref), 다조명 후보(E2,
# E3)는 I01 + 묶음 A인 팔 A와 I01 + 묶음 B인 팔 B다. 묶음은 조명 이름의 SHA-256 순으로 고정된다(A = 03, 07,
# 08, 09 / B = 02, 04, 05, 06, 10).
def arms(candidate: str) -> list[tuple[str, tuple[str, ...]]]:
    """(팔, 등록 조명) 목록: 팔 `ref` 하나(I01만) 또는 팔 `A`와 `B`(I01 + 묶음 하나)."""
    if candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate {candidate!r}; known: {CANDIDATES}")
    if not MULTI[candidate]:
        return [("ref", (REFERENCE,))]
    a, b = illumination_groups()
    return [("A", (REFERENCE, *a)), ("B", (REFERENCE, *b))]


# 팔 × 조명 10개의 등록 여부 표. npz에 같이 저장해 분석 코드가 "못 본 조명"과 "등록한 조명"을 이 표만 보고
# 가를 수 있게 한다.
def enrolled_matrix(candidate: str) -> np.ndarray:
    """bool [팔 수, 10]: 팔마다 어느 조명을 등록하는지(열은 ILLUMINATIONS 순서)."""
    return np.array([[light in enrolled for light in ILLUMINATIONS] for _, enrolled in arms(candidate)])


# 폐루프 결과의 행 이름 "조명:거르기:k"(새 조명 9개 × 팔 4개 = 36행). 저장 배열의 행 순서를 이 목록으로
# 고정한다.
def loop_keys() -> list[str]:
    """새 조명마다, 폐루프 팔마다 `<LL>:<filter>:<k>`. 조명이 바깥 순서다."""
    return [f"{light}:{filt}:{k}" for light in ILLUMINATIONS if light != REFERENCE for filt, k in LOOP_ARMS]


# E1·E3용 특징 추출기: 다른 추출기의 패치 특징에서 이미지마다 패치 평균 벡터를 뺀다(backbones.CentredPatches).
# 학습하는 것은 없고 뱅크와 질의에 같은 변환이 걸린다.
def centred(extractor):
    from .backbones import CentredPatches

    return CentredPatches(extractor)


# 구성 하나(백본, 코어셋 비율, 시드, 재가중 k, σ, 배치)의 옵션을 묶어 뱅크 만들기·채점·특징 추출·교차 적합을
# 같은 설정으로 부르게 하는 작은 도우미. 후보마다 추출기(그대로 / 중심화)만 바꿔 새로 만든다.
class Tools:
    """구성 하나의 옵션으로 뱅크 만들기, 채점, 교차 적합, 특징 추출을 한다."""

    def __init__(self, cfg: PatchCoreConfig, extractor, device: str):
        self.cfg, self.extractor, self.device = cfg, extractor, device

    def build(self, features):
        from .patchcore import build_bank

        return build_bank(features, self.cfg.coreset_ratio, seed=self.cfg.seed, device=self.device)

    # 이미지 점수만 쓴다. 맵 크기는 run_m2ad의 _MAP_SIZE(32)로 작게 둔다. 이미지 점수는 맵을 키우는 크기와
    # 무관하다(run_m2ad의 주석).
    def score(self, bank, images: np.ndarray) -> np.ndarray:
        from .patchcore import score_images

        return score_images(
            self.extractor,
            bank,
            images,
            batch_size=self.cfg.batch_size,
            device=self.device,
            reweight_k=self.cfg.reweight_k,
            sigma=self.cfg.sigma,
            map_size=_MAP_SIZE,
        ).image_scores

    # 특징을 [이미지, 패치, 차원]으로 돌려준다. 이미지 단위로 골라 뱅크를 만들 수 있게 이 모양으로 둔다.
    def features(self, images: np.ndarray):
        from .patchcore import collect_features

        feats, (grid_h, grid_w) = collect_features(
            self.extractor, images, batch_size=self.cfg.batch_size, device=self.device
        )
        return feats.view(len(images), grid_h * grid_w, feats.shape[1])

    def crossfit(self, features, images: np.ndarray, folds: np.ndarray):
        """이 구성으로 부르는 `run_m2ad.crossfit`: (전체 뱅크, 교차 적합 점수, 임계값, 뱅크 행 수)."""
        return _crossfit(self.build, self.score, features, images, folds)


# 검사기 하나의 학습 이미지를 조명 10개 모두에 대해 고른다. 모든 조명에 같은 시편이 같은 순서로 있어야 "시편
# 단위 겹"이 조명 사이에서 맞으므로, 다르면 _same_specimens가 멈춘다. _test_rows는 같은 일을 test 시편에 한다.
def _train_rows(rows: Sequence[M2adRow], category: str, view: str) -> tuple[dict, list[str]]:
    """검사기 하나의 조명별 학습 행(모든 조명에서 같은 시편, 이름 순)."""
    train = {light: inspector_rows(rows, category, view, "train", light) for light in ILLUMINATIONS}
    names = [r.specimen for r in train[REFERENCE]]
    if not names:
        raise ValueError(f"no train images for {category} {view} under illumination {REFERENCE}")
    for light, picked in train.items():
        _same_specimens(picked, names, f"{category} {view} train, illumination {light}")
    return train, names


def _test_rows(rows: Sequence[M2adRow], category: str, view: str) -> tuple[dict, list[str]]:
    test = {light: inspector_rows(rows, category, view, "test", light) for light in ILLUMINATIONS}
    names = [r.specimen for r in test[REFERENCE]]
    if not names:
        raise ValueError(f"no test images for {category} {view} under illumination {REFERENCE}")
    for light, picked in test.items():
        _same_specimens(picked, names, f"{category} {view} test, illumination {light}")
    return test, names


# 검사기마다 VRAM 최대 사용량을 따로 재려고 카운터를 초기화하고(GPU일 때만), _peak_mb가 그 값을 MB로 돌려준다.
def _peak_reset(device: str) -> bool:
    import torch

    on_cuda = str(device).startswith("cuda")
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    return on_cuda


def _peak_mb(on_cuda: bool) -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**20, 1) if on_cuda else 0.0


# 검증(val): 학습 시편 30개만 쓴다. 바깥 겹 f마다 f 밖 시편 24개의 등록 조명 이미지로 뱅크를 만들고, 같은
# 이미지의 나머지 4겹 교차 적합 점수로 임계값을 정한다(중첩). 그다음 겹 f 시편 6개를 조명 10개 모두에서 그
# 뱅크로 채점한다. 저장: 점수 [팔, 조명, 시편], 임계값 [팔, 겹], 보정 표본 수.
# 학습 시편은 모두 정상이라 여기서는 오검출률만 볼 수 있다(검출률은 test의 가드로만 본다, 규칙).
# 호출 위치: main(val), tests/test_run_m2ad_enrol.py.
def validate_inspector(
    tools: Tools,
    candidate: str,
    category: str,
    view: str,
    rows: Sequence[M2adRow],
    cache: M2adCache,
    out_dir: Path,
) -> dict:
    """후보 하나를 검사기 하나에서 학습 시편만으로 검증하고 npz를 쓴다.

    팔마다, 바깥 겹 f마다: 뱅크 = f 밖 시편의 그 팔 등록 조명 이미지, 임계값 = 그 이미지들을 나머지
    겹으로 교차 적합(중첩)한 점수의 conformal 임계값. 그 뱅크로 f의 시편을 조명 열 가지 모두에서 채점한다.
    """
    import torch

    started = time.perf_counter()
    on_cuda = _peak_reset(tools.device)
    train, names = _train_rows(rows, category, view)
    fold_of = specimen_folds(names)
    folds = np.array([fold_of[name] for name in names], dtype=np.int64)
    arm_list = arms(candidate)
    # 특징은 등록에 필요한 조명만 뽑는다(채점은 score_images가 이미지로 직접 한다). 이미지는 평가에 열 조명이
    # 모두 필요하므로 전부 읽는다.
    needed = sorted({light for _, enrolled in arm_list for light in enrolled})
    images = {light: cache.images(train[light]) for light in ILLUMINATIONS}
    feats = {light: tools.features(images[light]) for light in needed}

    n_lights, n_spec = len(ILLUMINATIONS), len(names)
    scores = np.full((len(arm_list), n_lights, n_spec), np.nan, dtype=np.float32)
    thresholds = np.full((len(arm_list), N_FOLDS), np.nan, dtype=np.float64)
    cal_n = np.zeros((len(arm_list), N_FOLDS), dtype=np.int64)
    bank_rows: dict[str, dict] = {}
    for a, (arm, enrolled) in enumerate(arm_list):
        for f in range(N_FOLDS):
            held = np.flatnonzero(folds == f)
            if held.size == 0:
                continue
            # 겹 f를 뺀 시편들의 등록 조명 이미지를 이어 붙인다. 한 시편의 여러 조명 이미지가 같은 겹 번호를
            # 받도록 folds를 조명 수만큼 반복해, 안쪽 교차 적합에서도 시편째 함께 빠지게 한다(같은 물체의 다른
            # 조명 사진이 뱅크에 남아 점수를 낮추는 누수를 막음).
            keep = folds != f
            keep_t = torch.from_numpy(keep)
            sub_feats = torch.cat([feats[light][keep_t] for light in enrolled])
            sub_images = np.concatenate([images[light][keep] for light in enrolled])
            sub_folds = np.concatenate([folds[keep]] * len(enrolled))
            bank, _, thr, rows_f = tools.crossfit(sub_feats, sub_images, sub_folds)
            thresholds[a, f], cal_n[a, f] = thr.value, thr.n
            bank_rows[f"{arm}:{f}"] = rows_f
            # 평가 시편의 열 조명 이미지를 한 번에 채점하고 [조명, 시편] 모양으로 되돌려 넣는다.
            shown = np.concatenate([images[light][held] for light in ILLUMINATIONS])
            scores[a][:, held] = tools.score(bank, shown).reshape(n_lights, held.size)
            del bank
    # NaN으로 채워 시작했으므로, 남은 NaN이 있으면 어떤 시편·겹이 빠졌다는 뜻이라 저장하지 않고 멈춘다.
    if np.isnan(scores).any() or np.isnan(thresholds).any():
        raise RuntimeError("validation did not cover every specimen and fold")

    np.savez(
        out_dir / f"{category}_{view}_{candidate}.npz",
        specimens=np.array(names),
        folds=folds,
        lights=np.array(ILLUMINATIONS),
        arms=np.array([arm for arm, _ in arm_list]),
        enrolled=enrolled_matrix(candidate),
        scores=scores,
        thresholds=thresholds,
        cal_n=cal_n,
    )
    return {
        "category": category,
        "view": view,
        "candidate": candidate,
        "specimens": n_spec,
        "arms": {arm: list(enrolled) for arm, enrolled in arm_list},
        "bank_rows": bank_rows,
        "peak_vram_mb": _peak_mb(on_cuda),
        "seconds": round(time.perf_counter() - started, 1),
    }


# 봉인 test: 팔마다 학습 시편 30개 전부의 등록 조명 이미지로 뱅크를 만들고, 시편 단위 5겹 교차 적합 점수로
# 목표 5% 임계값을 정한 뒤 test 시편을 조명 10개에서 채점한다. 저장: 점수 [팔, 조명, 시편], 라벨 [조명,
# 시편](0 정상, 1 결함, -1 제외), 팔별 임계값과 보정 점수.
# allow_test가 True 객체가 아니면 이미지를 읽기 전에 SealedTestError로 멈춘다.
# 호출 위치: main(test), tests/test_run_m2ad_enrol.py.
def sealed_inspector(
    tools: Tools,
    candidate: str,
    category: str,
    view: str,
    rows: Sequence[M2adRow],
    cache: M2adCache,
    out_dir: Path,
    *,
    allow_test: bool = False,
) -> dict:
    """후보 하나를 검사기 하나에서 봉인 테스트 시편으로 재고 npz를 쓴다.

    팔마다: 뱅크 = 모든 학습 시편의 그 팔 등록 조명 이미지, 임계값 = 그 이미지들을 시편 겹으로 교차
    적합한 점수의 conformal 임계값. 테스트 시편을 조명마다 채점한다(3-B처럼 조명 하나의 테스트 시편 전부를
    한 번에 채점).
    """
    import torch

    if allow_test is not True:
        raise SealedTestError("the M2AD test images are read only with allow_test=True (--allow-test)")
    started = time.perf_counter()
    on_cuda = _peak_reset(tools.device)
    train, names = _train_rows(rows, category, view)
    fold_of = specimen_folds(names)
    folds = np.array([fold_of[name] for name in names], dtype=np.int64)
    arm_list = arms(candidate)
    needed = sorted({light for _, enrolled in arm_list for light in enrolled})
    train_images = {light: cache.images(train[light]) for light in needed}
    feats = {light: tools.features(train_images[light]) for light in needed}

    test, test_names = _test_rows(rows, category, view)
    # 라벨은 조명마다 다를 수 있다(같은 불량 시편이라도 조명에 따라 결함이 "보임"과 "제외"를 오간다, 3-B
    # 결과). 그래서 [조명, 시편]으로 저장한다.
    labels = np.array([[r.label for r in test[light]] for light in ILLUMINATIONS], dtype=np.int8)
    scores = np.full((len(arm_list), len(ILLUMINATIONS), len(test_names)), np.nan, dtype=np.float32)
    thresholds = np.full(len(arm_list), np.nan, dtype=np.float64)
    cal_scores, info_arms = {}, {}
    test_images = {}
    for a, (arm, enrolled) in enumerate(arm_list):
        bank, oof, thr, bank_rows = tools.crossfit(
            torch.cat([feats[light] for light in enrolled]),
            np.concatenate([train_images[light] for light in enrolled]),
            np.concatenate([folds] * len(enrolled)),
        )
        thresholds[a] = thr.value
        cal_scores[f"cal_score_{arm}"] = oof
        info_arms[arm] = {
            "enrolled": list(enrolled),
            "threshold": _threshold_info(thr),
            "bank_rows": bank_rows,
        }
        # test 이미지는 조명마다 한 번만 읽어 두고 팔 사이에 재사용한다. 3-B처럼 조명 하나의 test 시편 전부를
        # 한 번에 채점한다.
        for li, light in enumerate(ILLUMINATIONS):
            if light not in test_images:
                test_images[light] = cache.images(test[light])
            scores[a, li] = tools.score(bank, test_images[light])
        del bank
    if np.isnan(scores).any():
        raise RuntimeError("not every test image was scored")

    np.savez(
        out_dir / f"{category}_{view}_{candidate}.npz",
        specimens=np.array(test_names),
        object_anomaly=np.array([r.object_anomaly for r in test[REFERENCE]], dtype=np.int8),
        lights=np.array(ILLUMINATIONS),
        arms=np.array([arm for arm, _ in arm_list]),
        enrolled=enrolled_matrix(candidate),
        scores=scores,
        labels=labels,
        thresholds=thresholds,
        **cal_scores,
    )
    return {
        "category": category,
        "view": view,
        "candidate": candidate,
        "train_specimens": len(names),
        "test_specimens": len(test_names),
        "folds": {name: int(fold_of[name]) for name in names},
        "arms": info_arms,
        "peak_vram_mb": _peak_mb(on_cuda),
        "seconds": round(time.perf_counter() - started, 1),
    }


# 폐루프 자동 재등록(E0, test, 판정 없음). 새 조명마다 경보 뒤 무라벨 20장을 흉내 낸다: 학습 시편 SHA-256 순
# 앞 20−k개(정상)와, 그 조명에서 결함이 보이는 test 불량 시편 앞 k개. trim 팔은 묶음 안 교차 적합 점수가 높은
# 4장을 버리고, 나머지를 I01 뱅크와 임계값에 더한 뒤 그 조명의 test 이미지를 채점한다.
# 등록에 쓴 불량 시편은 검출률에서 빼려고 라벨을 -1로 바꾼다(자기 자신을 뱅크에 넣었으니 못 잡는 것이
# 당연하므로).
# 호출 위치: main(test --loop), tests/test_run_m2ad_enrol.py.
def loop_inspector(
    tools: Tools,
    category: str,
    view: str,
    rows: Sequence[M2adRow],
    cache: M2adCache,
    out_dir: Path,
    *,
    allow_test: bool = False,
) -> dict:
    """검사기 하나에서 e0 위의 폐루프 재등록을 재고 `<category>_<view>_loop.npz`를 쓴다.

    새 조명 L과 팔 (filter, k)마다: 무라벨 이미지 LOOP_N장의 묶음 = SHA-256 순 앞 LOOP_N - k개 학습
    시편(정상) + L에서 결함이 보이는 테스트 시편을 시편 이름 SHA-256 순으로 앞 k개. `trim`은 묶음의 이미지마다
    묶음의 다른 겹(묶음 안 시편 겹)으로 만든 뱅크로 채점해 가장 높은 ceil(LOOP_TRIM * LOOP_N)장을 버리고,
    `none`은 모두 남긴다. 남긴 이미지는 I01 뱅크와 교차 적합에 더해진다(학습 시편은 원래 겹을 쓰고, 결함
    시편은 자기들끼리의 SHA-256 순위로 겹을 받는다). 그다음 L의 테스트 이미지를 채점하며, 등록에 들어간 결함
    시편은 검출률에서 뺀다(라벨 -1).
    """
    import torch

    if allow_test is not True:
        raise SealedTestError("the closed-loop arm reads M2AD test images: allow_test=True (--allow-test)")
    started = time.perf_counter()
    on_cuda = _peak_reset(tools.device)
    train, names = _train_rows(rows, category, view)
    fold_of = specimen_folds(names)
    folds = np.array([fold_of[name] for name in names], dtype=np.int64)
    ref_images = cache.images(train[REFERENCE])
    ref_feats = tools.features(ref_images)
    test, test_names = _test_rows(rows, category, view)
    # 버릴 장수 = ⌈0.2 × 20⌉ = 4. 결함이 섞이지 않은 묶음(k = 0)에서도 늘 4장을 버린다.
    n_drop = math.ceil(LOOP_TRIM * LOOP_N)

    keys = loop_keys()
    scores = np.full((len(keys), len(test_names)), np.nan, dtype=np.float32)
    labels = np.zeros((len(keys), len(test_names)), dtype=np.int8)
    thresholds = np.full(len(keys), np.nan, dtype=np.float64)
    info: dict[str, dict] = {}
    for light in ILLUMINATIONS:
        if light == REFERENCE:
            continue
        test_images = cache.images(test[light])
        shown = np.array([r.label for r in test[light]], dtype=np.int8)
        defect_order = sorted((r for r in test[light] if r.label == 1), key=lambda r: path_key(r.specimen))
        for filt, k in LOOP_ARMS:
            # 정상 쪽은 3-B 재보정과 같은 SHA-256 순서의 앞 20−k개 학습 시편, 결함 쪽은 그 조명에서 라벨 1인
            # test 불량 시편을 시편 이름 SHA-256 순으로 앞 k개다. 결함이 k개보다 적으면 규칙대로 만들 수
            # 없으므로 멈춘다.
            normal_names = recal_specimens(names, LOOP_N - k)
            normal_rows = [train[light][names.index(name)] for name in normal_names]
            defect_rows = defect_order[:k]
            if len(defect_rows) < k:
                raise ValueError(f"{category} {view} I{light}: fewer than {k} visible defects to mix in")
            batch_rows = normal_rows + defect_rows
            batch_names = [r.specimen for r in batch_rows]
            batch_images = cache.images(batch_rows)
            batch_feats = tools.features(batch_images)
            is_defect = np.array([False] * len(normal_rows) + [True] * len(defect_rows))
            # 강건 거르기: 묶음 20장을 시편 SHA-256 순위로 5겹으로 나누고, 겹마다 나머지 겹 이미지로 만든
            # 뱅크로 채점해(묶음 안 교차 적합) 점수가 가장 높은 4장을 버린다. 결함은 정상보다 점수가 높을
            # 것이므로 섞인 결함을 걸러 내려는 장치다. none 팔은 아무것도 버리지 않는다(대조).
            if filt == "trim":
                within = specimen_folds(batch_names)
                within_folds = np.array([within[name] for name in batch_names], dtype=np.int64)
                _, batch_oof, _, _ = tools.crossfit(batch_feats, batch_images, within_folds)
                dropped = np.argsort(-batch_oof, kind="stable")[:n_drop]
            elif filt == "none":
                dropped = np.zeros(0, dtype=np.int64)
            else:
                raise ValueError(f"unknown filter {filt!r}")
            kept = np.setdiff1d(np.arange(len(batch_rows)), dropped)
            # 남긴 이미지를 I01 뱅크(30장)에 더하고 더한 이미지까지 넣어 교차 적합 임계값을 다시 정한다. 학습
            # 시편은 원래 겹을 그대로 쓰고, 섞인 불량 시편은 자기들끼리의 SHA-256 순위로 겹을 받는다(규칙).
            defect_fold = specimen_folds([r.specimen for r in defect_rows]) if defect_rows else {}
            batch_folds = np.array(
                [defect_fold[name] if name in defect_fold else fold_of[name] for name in batch_names],
                dtype=np.int64,
            )
            bank, oof, thr, bank_rows = tools.crossfit(
                torch.cat([ref_feats, batch_feats[torch.from_numpy(kept)]]),
                np.concatenate([ref_images, batch_images[kept]]),
                np.concatenate([folds, batch_folds[kept]]),
            )
            key = f"{light}:{filt}:{k}"
            j = keys.index(key)
            scores[j] = tools.score(bank, test_images)
            labels[j] = shown
            # 등록에 들어간 불량 시편은 이 팔의 검출률 계산에서 뺀다(-1). 거르기가 버렸어도 묶음에 들어갔던
            # 시편이라 똑같이 뺀다.
            enrolled_defects = {r.specimen for r in defect_rows}
            labels[j, [i for i, name in enumerate(test_names) if name in enrolled_defects]] = -1
            thresholds[j] = thr.value
            # 팔별 기록: 묶음 크기, 섞인 결함 수, 거르기가 버린 결함·정상 수, 등록된 장수, 임계값, 뱅크 행 수.
            # "섞인 결함 가운데 거르기가 버린 수"는 규칙의 지표다.
            info[key] = {
                "batch": len(batch_rows),
                "defects_in_batch": int(is_defect.sum()),
                "dropped_defects": int(is_defect[dropped].sum()),
                "dropped_normals": int((~is_defect[dropped]).sum()),
                "enrolled": len(oof),
                "threshold": _threshold_info(thr),
                "bank_rows": bank_rows,
            }
            del bank
    np.savez(
        out_dir / f"{category}_{view}_loop.npz",
        specimens=np.array(test_names),
        object_anomaly=np.array([r.object_anomaly for r in test[REFERENCE]], dtype=np.int8),
        loop_keys=np.array(keys),
        scores=scores,
        labels=labels,
        thresholds=thresholds,
    )
    return {
        "category": category,
        "view": view,
        "loop": info,
        "peak_vram_mb": _peak_mb(on_cuda),
        "seconds": round(time.perf_counter() - started, 1),
    }


# 구성의 백본으로 중심화하지 않은 추출기를 만든다. 중심화 추출기는 main에서 이것을 centred로 감싸 만든다(같은
# 백본 가중치를 공유).
def _make_extractor(cfg: PatchCoreConfig, device: str):
    from .backbones import make_extractor

    return make_extractor(cfg.backbone, img_size=cfg.img_size).to(device).eval()


# analyze_m2ad_enrol val이 reports/stage7/val.json에 적은 방법별 선택 후보를 읽는다. 규칙상 val 결과를 커밋한
# 뒤에만 test를 돌리므로, test가 고를 후보는 이 파일에서만 온다.
def read_pick(val_report: Path, method: str) -> str:
    """`analyze_m2ad_enrol val`이 `method`에 대해 고른 후보."""
    with open(val_report, encoding="utf-8") as f:
        report = json.load(f)
    try:
        pick = report["methods"][method]["pick"]
    except (KeyError, TypeError):
        raise ValueError(f"{val_report} has no pick for {method!r}") from None
    if pick not in CANDIDATES:
        raise ValueError(f"{val_report}: unknown pick {pick!r} for {method!r}")
    return pick


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


# 요청한 검사기 이름을 검사하고 INSPECTORS 순서의 (범주, 시점) 목록으로 돌려준다.
def _inspectors(names: Sequence[str]) -> list[tuple[str, str]]:
    unknown = [name for name in names if name not in INSPECTORS]
    if unknown:
        raise ValueError(f"unknown inspectors {unknown}; known: {list(INSPECTORS)}")
    return [tuple(name.split("_")) for name in INSPECTORS if name in names]


# 명령행 진입점. val과 test 두 하위 명령이 같은 기본 인자를 갖고, val은 후보·검사기를 고를 수 있으며(점검용),
# test는 늘 검사기 6개 전부와 "E0 + val 선택"만 잰다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("val", "test"):
        p = sub.add_parser(name)
        p.add_argument("--method", choices=METHODS, required=True)
        p.add_argument("--out", type=Path, default=None, help=f"기본값: outputs/m2ad-enrol-{name}-<method>")
        p.add_argument("--device", default="cuda")
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--jsons", type=Path, default=None, help="기본값: data/raw/m2ad/jsons.zip")
        p.add_argument("--cache", type=Path, default=None, help="캐시 폴더(기본값: data/cache)")
    val = sub.choices["val"]
    val.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    val.add_argument("--inspectors", nargs="+", default=list(INSPECTORS), help="검사기 이름. 예: Motor_000")
    test = sub.choices["test"]
    test.add_argument(
        "--pick-from", type=Path, required=True, help="val 선택이 적힌 파일(reports/stage7/val.json)"
    )
    test.add_argument("--loop", action="store_true", help="e0 위의 폐루프 팔도 함께 잰다")
    test.add_argument("--allow-test", action="store_true", help="M2AD 테스트 이미지를 읽는다(장부에 기록)")
    test.add_argument("--stage", default="", help="봉인 테스트 장부에 남길 단계 이름")
    test.add_argument("--note", default="")
    args = parser.parse_args(argv)

    cfg = get_config(args.method)
    out_dir = args.out or paths.OUTPUTS / f"m2ad-enrol-{args.command}-{args.method}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    if args.command == "test":
        if not args.allow_test or not args.stage:
            parser.error("test reads the M2AD test images: it needs --allow-test and --stage")
        try:
            pick = read_pick(args.pick_from, args.method)
        except (OSError, ValueError) as err:
            parser.error(str(err))
        # E0과 선택 후보를 같은 실행에서 잰다. 선택이 E0이면 dict.fromkeys가 중복을 없애 E0 하나만 남는다.
        candidates = list(dict.fromkeys(["e0", pick]))
        inspectors = _inspectors(INSPECTORS)
    else:
        candidates = list(args.candidates)
        try:
            inspectors = _inspectors(args.inspectors)
        except ValueError as err:
            parser.error(str(err))

    # 메타데이터·캐시·추출기 준비처럼 실패할 수 있는 일을 장부 기록 전에 끝낸다. 장부(config
    # m2ad-enrol-<방법>, stage 7-m2ad)에는 후보와 폐루프 여부를 메모로 남긴다. 테스트 이미지를 건드리지 않고
    # 실패할 수 있는 일은 모두 장부 줄보다 먼저 한다.
    rows = read_meta(args.jsons or paths.RAW / "m2ad" / "jsons.zip")
    cache = M2adCache(args.cache or paths.CACHE, cfg.img_size)
    plain = _make_extractor(cfg, args.device)
    # 추출기 두 개(그대로, 중심화)를 미리 만들어 두고 후보의 CENTRE 값으로 고른다.
    extractors = {False: plain, True: centred(plain).to(args.device).eval()}
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    if args.command == "test":
        note = args.note or f"stage 7 candidates {candidates}" + (", closed loop on e0" if args.loop else "")
        record_test_access(
            paths.TEST_LEDGER, stage=args.stage, config=f"m2ad-enrol-{args.method}", note=note, commit=commit
        )
    # 다시 돌리는 실행이 끝날 때까지 끝난 실행으로 보이지 않게 run.json을 먼저 지운다.
    (out_dir / "run.json").unlink(missing_ok=True)

    summaries = []
    started = time.perf_counter()
    for candidate in candidates:
        tools = Tools(cfg, extractors[CENTRE[candidate]], args.device)
        for category, view in inspectors:
            if args.command == "val":
                info = validate_inspector(tools, candidate, category, view, rows, cache, out_dir)
            else:
                info = sealed_inspector(
                    tools, candidate, category, view, rows, cache, out_dir, allow_test=args.allow_test
                )
            summaries.append(info)
            print(json.dumps({k: info[k] for k in ("category", "view", "candidate", "seconds")}), flush=True)
    loop = []
    if args.command == "test" and args.loop:
        tools = Tools(cfg, extractors[False], args.device)
        for category, view in inspectors:
            info = loop_inspector(tools, category, view, rows, cache, out_dir, allow_test=args.allow_test)
            loop.append(info)
            print(json.dumps({"loop": f"{category}_{view}", "seconds": info["seconds"]}), flush=True)

    # run.json에 구성·커밋·장치(코어셋 선택은 장치에 따라 반올림 수준에서 달라짐), 후보, 조명 묶음, 폐루프
    # 설정, 검사기별 요약을 남긴다. analyze_m2ad_enrol이 이 파일로 npz 목록과 폐루프 여부를 안다.
    a, b = illumination_groups()
    _write_json(
        out_dir / "run.json",
        {
            "command": args.command,
            "method": args.method,
            "config": asdict(cfg),
            "commit": commit,
            # 코어셋 선택은 장치에 따라 부동소수 반올림 수준으로 달라진다.
            "device": args.device,
            "candidates": candidates,
            "inspectors": [f"{category}_{view}" for category, view in inspectors],
            "groups": {"A": list(a), "B": list(b)},
            "loop": {"n": LOOP_N, "trim": LOOP_TRIM, "arms": [list(arm) for arm in LOOP_ARMS]}
            if loop
            else None,
            "runs": summaries,
            "loop_runs": loop,
            "total_s": round(time.perf_counter() - started, 1),
        },
    )


if __name__ == "__main__":
    main()
