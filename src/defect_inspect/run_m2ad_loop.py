"""8-B단계: 7단계 폐루프 오검출률의 원인 분해, 조건별 보정, 사람이 상위 m장만 확인하는 거르기(P0).

경보 뒤 무라벨 묶음(학습 시편 정상 n − k장 + test 불량 시편 k장)을 세 가지로 등록하고(none, trim, label) 두
가지 임계값(mixed: I01 + 더한 이미지 전체, cond: 더한 새 조명 이미지만)으로 새 조명의 test를 잰다. 봉인된 M2AD
test 이미지를 읽으므로 장부에 남긴다. 규칙: docs/experiments.md 8단계 8-B.
"""

# [파일 역할] 8-B 실행기(`python -m defect_inspect.run_m2ad_loop --allow-test --stage 8-m2ad-loop`). 7단계
# 폐루프(무라벨 20장에서 점수 상위 4장을 버리고 I01 뱅크에 더함)는 결함이 없어도 오검출률이 9.4%였다. 저장된
# 기록상 그 임계값은 I01 30장 + 새 조명 16장의 혼합 교차 적합 점수에서 두 번째로 큰 값이라, 새 조명만 놓고는
# 목표 5%가 보장되지 않는다. 이 실행은 (1) 거르지 않은 대조, (2) 새 조명 이미지만으로 정한 조건별 임계값,
# (3) 상위 m = 4장만 사람이 확인해 실제 결함만 빼는 거르기를 같은 묶음에서 잰다.
# [설계 이유] 7단계 거르기 팔(trim, k = 0)을 같은 코드 순서로 다시 돌려 저장값과 같은지 확인하고(재현 관문), n
# = 30·k = 0의 none 뱅크가 3-B 사람 재보정 n = 30과 같은 이미지·같은 겹이라는 점도 함께 확인한다. 한 묶음의
# 교차 적합 점수에서 mixed와 cond 두 임계값을 모두 내므로, 임계값 방식의 효과는 같은 뱅크·같은 점수에서
# 갈린다.
# [흐름] main → 검사기마다 loop_inspector(조명 L × 묶음 (n, k) × 등록 방식: 묶음 이미지 → (trim·label이면)
# 묶음 안 교차 적합 순위 → I01 + 남긴 이미지 교차 적합 → 두 임계값 → L test 채점) → npz와 run.json. 표와 H25
# 판정은 analyze_stage8 loop.
# [연결] run_m2ad_enrol(Tools, 행 선택, 장치별 VRAM 기록), m2ad(시편 겹, 앞 n개 시편), calibrate(conformal
# 임계값).

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import paths
from .calibrate import conformal_threshold
from .configs import get_config
from .ledger import git_commit, record_test_access
from .m2ad import ILLUMINATIONS, REFERENCE, M2adCache, read_meta, recal_specimens, specimen_folds
from .run_m2ad import ALPHA, _threshold_info
from .run_m2ad_enrol import (
    INSPECTORS,
    LOOP_TRIM,
    Tools,
    _inspectors,
    _make_extractor,
    _peak_mb,
    _peak_reset,
    _test_rows,
    _train_rows,
)
from .splits import SealedTestError, path_key

# 규칙의 상수. HUMAN_M: 사람이 확인하는 상위 장수(측정 전에 4로 고정). ARMS: (묶음 크기 n, 섞은 결함 k, 등록
# 방식). k = 0의 label은 아무것도 빼지 않아 none과 같으므로 넣지 않는다. trim은 7단계 재현(n = 20, k = 0)만.
HUMAN_M = 4
ENROLMENTS = ("none", "trim", "label")
ARMS = (
    (20, 0, "none"),
    (20, 2, "none"),
    (20, 1, "label"),
    (20, 2, "label"),
    (20, 0, "trim"),
    (30, 0, "none"),
    (30, 2, "none"),
    (30, 1, "label"),
    (30, 2, "label"),
    (30, 3, "label"),
)
NEW_LIGHTS = tuple(light for light in ILLUMINATIONS if light != REFERENCE)


# 결과 행 이름 "조명:n:k:등록" (예: 02:30:2:label). 저장 배열의 행 순서를 이 목록으로 고정한다.
def arm_keys() -> list[str]:
    return [f"{light}:{n}:{k}:{enrol}" for light in NEW_LIGHTS for n, k, enrol in ARMS]


# 묶음 안 순위로 뺄 이미지를 고른다. trim은 점수 상위 ⌈0.2n⌉장을 모두, label은 상위 HUMAN_M장 가운데 실제
# 결함만. none은 아무것도 빼지 않는다. 순위는 7단계와 같은 안정 정렬(점수 내림차순, 같으면 앞의 것).
def dropped_indices(enrol: str, within_scores: np.ndarray | None, is_defect: np.ndarray) -> np.ndarray:
    """등록 방식마다 묶음에서 뺄 이미지 번호(오름차순)."""
    if enrol == "none":
        return np.zeros(0, dtype=np.int64)
    if within_scores is None:
        raise ValueError(f"{enrol} needs the within-batch scores")
    order = np.argsort(-np.asarray(within_scores), kind="stable")
    if enrol == "trim":
        return np.sort(order[: math.ceil(LOOP_TRIM * len(is_defect))])
    if enrol == "label":
        top = order[:HUMAN_M]
        return np.sort(top[np.asarray(is_defect)[top]])
    raise ValueError(f"unknown enrolment {enrol!r}")


# 검사기 하나의 8-B 실행. 저장: 행(arm_keys)마다 L test 점수, 라벨(묶음에 들어간 불량 시편은 -1), mixed·cond
# 임계값. run.json용 기록: 묶음 크기, 섞인 결함, 뺀 결함·정상, 사람 라벨 수, 두 임계값의 순위·n·보장 여부.
def loop_inspector(
    tools: Tools, category: str, view: str, rows, cache, out_dir: Path, *, allow_test=False
) -> dict:
    """검사기 하나에서 8-B 팔들을 재고 `<category>_<view>.npz`를 쓴다."""
    import torch

    if allow_test is not True:
        raise SealedTestError("stage 8-B reads M2AD test images: allow_test=True (--allow-test)")
    started = time.perf_counter()
    on_cuda = _peak_reset(tools.device)
    train, names = _train_rows(rows, category, view)
    fold_of = specimen_folds(names)
    folds = np.array([fold_of[name] for name in names], dtype=np.int64)
    ref_images = cache.images(train[REFERENCE])
    ref_feats = tools.features(ref_images)
    test, test_names = _test_rows(rows, category, view)

    keys = arm_keys()
    scores = np.full((len(keys), len(test_names)), np.nan, dtype=np.float32)
    labels = np.zeros((len(keys), len(test_names)), dtype=np.int8)
    thr_mixed = np.full(len(keys), np.nan, dtype=np.float64)
    thr_cond = np.full(len(keys), np.nan, dtype=np.float64)
    info: dict[str, dict] = {}
    for light in NEW_LIGHTS:
        test_images = cache.images(test[light])
        shown = np.array([r.label for r in test[light]], dtype=np.int8)
        defect_order = sorted((r for r in test[light] if r.label == 1), key=lambda r: path_key(r.specimen))
        batches = sorted({(n, k) for n, k, _ in ARMS})
        for n, k in batches:
            # 7단계와 같은 묶음: 정상 = 학습 시편 SHA-256 순 앞 n − k개, 결함 = 그 조명에서 결함이 보이는 test
            # 불량 시편 SHA-256 순 앞 k개.
            normal_rows = [train[light][names.index(name)] for name in recal_specimens(names, n - k)]
            defect_rows = defect_order[:k]
            if len(defect_rows) < k:
                raise ValueError(f"{category} {view} I{light}: fewer than {k} visible defects to mix in")
            batch_rows = normal_rows + defect_rows
            batch_names = [r.specimen for r in batch_rows]
            batch_images = cache.images(batch_rows)
            batch_feats = tools.features(batch_images)
            is_defect = np.array([False] * len(normal_rows) + [True] * len(defect_rows))
            enrols = [enrol for nn, kk, enrol in ARMS if (nn, kk) == (n, k)]
            within = None
            if any(enrol != "none" for enrol in enrols):
                # 묶음 안 교차 적합(묶음 시편 SHA-256 순위로 5겹): 7단계 거르기와 같은 순위 점수.
                within_fold = specimen_folds(batch_names)
                within_folds = np.array([within_fold[name] for name in batch_names], dtype=np.int64)
                _, within, _, _ = tools.crossfit(batch_feats, batch_images, within_folds)
            defect_fold = specimen_folds([r.specimen for r in defect_rows]) if defect_rows else {}
            batch_folds = np.array(
                [defect_fold[name] if name in defect_fold else fold_of[name] for name in batch_names],
                dtype=np.int64,
            )
            enrolled_defects = {r.specimen for r in defect_rows}
            for enrol in enrols:
                dropped = dropped_indices(enrol, within, is_defect)
                kept = np.setdiff1d(np.arange(len(batch_rows)), dropped)
                bank, oof, thr, bank_rows = tools.crossfit(
                    torch.cat([ref_feats, batch_feats[torch.from_numpy(kept)]]),
                    np.concatenate([ref_images, batch_images[kept]]),
                    np.concatenate([folds, batch_folds[kept]]),
                )
                # 조건별 임계값: 같은 교차 적합 점수에서 더한 조명 L 이미지(뒤쪽 len(kept)개)의 점수만 쓴다.
                cond = conformal_threshold(oof[len(ref_images) :], ALPHA)
                key = f"{light}:{n}:{k}:{enrol}"
                j = keys.index(key)
                scores[j] = tools.score(bank, test_images)
                labels[j] = shown
                labels[j, [i for i, name in enumerate(test_names) if name in enrolled_defects]] = -1
                thr_mixed[j], thr_cond[j] = thr.value, cond.value
                info[key] = {
                    "batch": len(batch_rows),
                    "defects_in_batch": int(is_defect.sum()),
                    "dropped_defects": int(is_defect[dropped].sum()),
                    "dropped_normals": int((~is_defect[dropped]).sum()),
                    "human_labels": HUMAN_M if enrol == "label" else 0,
                    "enrolled": int(len(kept)),
                    "threshold_mixed": _threshold_info(thr),
                    "threshold_cond": _threshold_info(cond),
                    "bank_rows": bank_rows,
                }
                del bank
    if np.isnan(scores).any():
        raise RuntimeError(f"{category} {view}: not every arm was scored")
    np.savez(
        out_dir / f"{category}_{view}.npz",
        specimens=np.array(test_names),
        object_anomaly=np.array([r.object_anomaly for r in test[REFERENCE]], dtype=np.int8),
        arm_keys=np.array(keys),
        scores=scores,
        labels=labels,
        thresholds_mixed=thr_mixed,
        thresholds_cond=thr_cond,
    )
    return {
        "category": category,
        "view": view,
        "arms": info,
        "peak_vram_mb": _peak_mb(on_cuda),
        "seconds": round(time.perf_counter() - started, 1),
    }


# 명령줄 진입점. 메타데이터·캐시·추출기 준비처럼 test 이미지 없이 실패할 수 있는 일을 장부 기록 전에 끝낸다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="8-B: 폐루프 원인 분해, 조건별 보정, 사람 확인 거르기(P0).")
    parser.add_argument("--out", type=Path, default=None, help="기본값: outputs/m2ad-loop-p0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-test", action="store_true", help="M2AD 테스트 이미지를 읽는다(장부에 기록)")
    parser.add_argument("--stage", default="", help="봉인 테스트 장부에 남길 단계 이름(8-m2ad-loop)")
    parser.add_argument("--note", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jsons", type=Path, default=None, help="기본값: data/raw/m2ad/jsons.zip")
    parser.add_argument("--cache", type=Path, default=None, help="캐시 폴더(기본값: data/cache)")
    args = parser.parse_args(argv)
    if not args.allow_test or not args.stage:
        parser.error("stage 8-B reads the M2AD test images: it needs --allow-test and --stage")

    cfg = get_config("p0")
    out_dir = args.out or paths.OUTPUTS / "m2ad-loop-p0"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    rows = read_meta(args.jsons or paths.RAW / "m2ad" / "jsons.zip")
    cache = M2adCache(args.cache or paths.CACHE, cfg.img_size)
    tools = Tools(cfg, _make_extractor(cfg, args.device), args.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    record_test_access(
        paths.TEST_LEDGER,
        stage=args.stage,
        config="m2ad-loop-p0",
        note=args.note or f"stage 8-B closed-loop arms {len(ARMS)} per light, human check m={HUMAN_M}",
        commit=commit,
    )
    (out_dir / "run.json").unlink(missing_ok=True)

    summaries = []
    started = time.perf_counter()
    for category, view in _inspectors(INSPECTORS):
        info = loop_inspector(tools, category, view, rows, cache, out_dir, allow_test=args.allow_test)
        summaries.append(info)
        print(json.dumps({"inspector": f"{category}_{view}", "seconds": info["seconds"]}), flush=True)
    with open(out_dir / "run.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            {
                "method": "p0",
                "config": asdict(cfg),
                "commit": commit,
                "device": args.device,
                "human_m": HUMAN_M,
                "trim": LOOP_TRIM,
                "arms": [list(arm) for arm in ARMS],
                "inspectors": summaries,
                "total_s": round(time.perf_counter() - started, 1),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
