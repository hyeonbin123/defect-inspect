"""8-A단계: 뱅크는 그대로 두고 새 조명의 정상 n장으로 임계값만 다시 잡는 절차(서비스 `/calibrate`)를 잰다.

검사기마다 3-B의 E0 겹 뱅크(I01 학습 시편 30장에서 시편 겹 하나를 뺀 뱅크 다섯 개)를 다시 만들고, 학습 시편의
새 조명 이미지를 자기 시편을 뺀 겹 뱅크로 채점해 `<out>/<검사기>.npz`에 쓴다. test 점수는 3-B 실행에 저장된
것을 분석 단계(`analyze_stage8 calibrate`)가 그대로 쓰므로 이 실행은 M2AD test 이미지를 읽지 않는다.
규칙: docs/experiments.md 8단계 8-A.
"""

# [파일 역할] 8-A 실행기(`python -m defect_inspect.run_m2ad_threshold --method p0|d-s`). 서비스의
# `/calibrate`는 모델(뱅크)을 그대로 두고 그 조건의 정상 사진으로 임계값만 다시 잡는다. 이 절차가 실제 조명
# 변화에서 무엇을 하는지 재려면 "뱅크에 없는 물건을 새 조명에서 찍은 사진의 점수"가 필요하다. M2AD 학습 시편은
# I01 사진이 뱅크에 들어 있으므로, 그 시편을 뺀 겹 뱅크(3-B 교차 적합과 같은 뱅크)로 새 조명 사진을 채점한다.
# [동일성 관문] 저장된 3-B test 점수(전체 E0 뱅크)와 새 보정 점수(겹 뱅크)를 함께 쓰려면 다시 만든 겹 뱅크가
# 3-B와 같아야 한다. 그래서 다시 만든 겹 뱅크의 I01 교차 적합 점수를 3-B의 `cal_score`와 비교해 최대 상대
# 차이와 임계값 일치를 기록하고, 어긋나면 그 검사기 결과를 쓰지 않고 멈춘다(규칙).
# [흐름] main → 검사기마다 rebuild_inspector(I01 특징 → 겹 뱅크 5개 → I01 교차 적합 점수 대조 → 새 조명 9개의
# 학습 이미지 채점) → npz와 run.json. 판정은 analyze_stage8 calibrate.
# [연결] run_m2ad_enrol.Tools(구성 하나의 뱅크·채점·특징), _train_rows(조명마다 같은 시편 순서),
# m2ad(시편 겹, 조명 목록, 캐시). 3-B·7단계 코드는 고치지 않는다.

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import paths
from .calibrate import conformal_threshold
from .configs import get_config
from .ledger import git_commit
from .m2ad import CATEGORIES, ILLUMINATIONS, N_FOLDS, REFERENCE, VIEWS, M2adCache, read_meta, specimen_folds
from .run_m2ad import ALPHA
from .run_m2ad_enrol import INSPECTORS, Tools, _make_extractor, _peak_mb, _peak_reset, _train_rows

METHODS = ("p0", "d-s")
# 동일성 관문의 허용치: 검사기마다 I01 교차 적합 점수의 최대 상대 차이(규칙: 1e-6 이하, 7단계에서는 0이었다).
REL_TOL = 1e-6
# 새 조명 아홉 가지(I02~I10). 저장 배열의 행 순서다.
NEW_LIGHTS = tuple(light for light in ILLUMINATIONS if light != REFERENCE)


# 저장된 값과 다시 낸 값의 최대 상대 차이. 0으로 나누지 않게 분모에 아주 작은 값을 둔다.
def max_rel_diff(new: np.ndarray, old: np.ndarray) -> float:
    new, old = np.asarray(new, np.float64), np.asarray(old, np.float64)
    if new.shape != old.shape:
        raise ValueError(f"shapes differ: {new.shape} and {old.shape}")
    return float(np.max(np.abs(new - old) / np.maximum(np.abs(old), 1e-12))) if new.size else 0.0


# 검사기 하나: I01 학습 이미지로 겹 뱅크 5개를 다시 만들어 I01 교차 적합 점수를 3-B 저장값과 대조하고, 새 조명
# 아홉 가지의 학습 이미지를 자기 시편을 뺀 겹 뱅크로 채점한다. 저장: 시편 이름(이름 순), 겹, 새 조명 점수
# [9, 30], I01 교차 적합 점수와 대조 결과.
# 관문이 어긋나면 RuntimeError로 멈춘다(그 검사기의 결과 파일을 쓰기 전에). stop_on_gate=False(코드 경로
# 점검용)면 멈추지 않고 gate.ok = False로 기록한 채 파일을 쓴다. 분석은 그런 실행을 받지 않는다.
def rebuild_inspector(
    tools: Tools,
    category: str,
    view: str,
    rows,
    cache,
    out_dir: Path,
    ref_dir: Path,
    *,
    stop_on_gate: bool = True,
) -> dict:
    """3-B E0 겹 뱅크를 다시 만들어 대조하고, 새 조명 학습 이미지의 교차 적합 점수를 쓴다."""
    import torch

    started = time.perf_counter()
    on_cuda = _peak_reset(tools.device)
    train, names = _train_rows(rows, category, view)
    fold_of = specimen_folds(names)
    folds = np.array([fold_of[name] for name in names], dtype=np.int64)
    ref_images = cache.images(train[REFERENCE])
    ref_feats = tools.features(ref_images)
    new_images = {light: cache.images(train[light]) for light in NEW_LIGHTS}
    dim = ref_feats.shape[2]

    ref_oof = np.full(len(names), np.nan, dtype=np.float32)
    cal = np.full((len(NEW_LIGHTS), len(names)), np.nan, dtype=np.float32)
    bank_rows = {}
    # 3-B의 crossfit과 같은 겹 뱅크: 겹 f를 뺀 시편의 I01 특징으로 같은 구성·seed로 만든다. 같은 겹 뱅크로 그
    # 겹 시편의 I01(대조용)과 새 조명 이미지(보정 점수)를 채점한다.
    for fold in range(N_FOLDS):
        held = np.flatnonzero(folds == fold)
        if held.size == 0:
            continue
        bank = tools.build(ref_feats[torch.from_numpy(folds != fold)].reshape(-1, dim))
        bank_rows[f"minus_fold_{fold}"] = int(bank.shape[0])
        ref_oof[held] = tools.score(bank, ref_images[held])
        for i, light in enumerate(NEW_LIGHTS):
            cal[i, held] = tools.score(bank, new_images[light][held])
        del bank
    if np.isnan(ref_oof).any() or np.isnan(cal).any():
        raise RuntimeError(f"{category} {view}: not every training image was scored")

    # 동일성 관문: 3-B가 저장한 I01 교차 적합 점수와 그 임계값.
    with np.load(Path(ref_dir) / f"{category}_{view}.npz") as z:
        stored, stored_thr = z["cal_score"], float(z["threshold"])
    diff = max_rel_diff(ref_oof, stored)
    same_thr = conformal_threshold(ref_oof, ALPHA).value == stored_thr
    ok = bool(diff <= REL_TOL and same_thr)
    gate = {"max_rel_diff": diff, "tolerance": REL_TOL, "threshold_equal": bool(same_thr), "ok": ok}
    if not ok and stop_on_gate:
        raise RuntimeError(f"{category} {view}: the rebuilt fold banks differ from stage 3-B: {gate}")

    np.savez(
        out_dir / f"{category}_{view}.npz",
        specimens=np.array(names),
        folds=folds,
        lights=np.array(NEW_LIGHTS),
        cal_scores=cal,
        ref_oof=ref_oof,
    )
    return {
        "category": category,
        "view": view,
        "train_specimens": len(names),
        "gate": gate,
        "bank_rows": bank_rows,
        "peak_vram_mb": _peak_mb(on_cuda),
        "seconds": round(time.perf_counter() - started, 1),
    }


# 명령줄 진입점. M2AD test 이미지를 읽지 않으므로 장부에 남기지 않는다. --inspectors는 점검용으로 줄일 때만
# 쓴다(규칙의 실행은 검사기 6개 전부). --no-gate-stop은 다른 장치(CPU)에서 코드 경로만 점검할 때 쓰며, 관문이
# 어긋나도 멈추지 않고 기록만 한다(그 실행의 결과는 측정에 쓰지 않는다).
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="8-A: E0 겹 뱅크로 새 조명 학습 이미지를 채점(임계값만 다시 잡기)."
    )
    parser.add_argument("--method", choices=METHODS, required=True, help="PatchCore 구성(3-B와 같다)")
    parser.add_argument("--out", type=Path, default=None, help="기본값: outputs/m2ad-threshold-<method>")
    parser.add_argument("--ref", type=Path, default=None, help="3-B 실행 폴더(기본값: outputs/m2ad-<method>)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--inspectors", nargs="+", default=list(INSPECTORS), help="검사기 이름. 예: Motor_000"
    )
    parser.add_argument(
        "--no-gate-stop",
        action="store_true",
        help="동일성 관문이 어긋나도 멈추지 않는다(다른 장치에서 코드 경로만 점검할 때, 측정에는 안 씀)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jsons", type=Path, default=None, help="기본값: data/raw/m2ad/jsons.zip")
    parser.add_argument("--cache", type=Path, default=None, help="캐시 폴더(기본값: data/cache)")
    args = parser.parse_args(argv)

    cfg = get_config(args.method)
    out_dir = args.out or paths.OUTPUTS / f"m2ad-threshold-{args.method}"
    ref_dir = args.ref or paths.OUTPUTS / f"m2ad-{args.method}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    unknown = [name for name in args.inspectors if name not in INSPECTORS]
    if unknown:
        parser.error(f"unknown inspectors {unknown}; known: {list(INSPECTORS)}")
    rows = read_meta(args.jsons or paths.RAW / "m2ad" / "jsons.zip")
    cache = M2adCache(args.cache or paths.CACHE, cfg.img_size)
    tools = Tools(cfg, _make_extractor(cfg, args.device), args.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").unlink(missing_ok=True)
    commit = git_commit()

    summaries = []
    started = time.perf_counter()
    for category in CATEGORIES:
        for view in VIEWS:
            if f"{category}_{view}" not in args.inspectors:
                continue
            info = rebuild_inspector(
                tools, category, view, rows, cache, out_dir, ref_dir, stop_on_gate=not args.no_gate_stop
            )
            summaries.append(info)
            print(json.dumps({k: v for k, v in info.items() if k != "bank_rows"}), flush=True)

    with open(out_dir / "run.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(
            {
                "method": args.method,
                "config": asdict(cfg),
                "commit": commit,
                # 코어셋 선택은 장치에 따라 부동소수 반올림 수준으로 달라진다(관문이 같은 장치를 요구하는
                # 이유).
                "device": args.device,
                "reference_run": str(ref_dir),
                "no_gate_stop": bool(args.no_gate_stop),
                "lights": list(NEW_LIGHTS),
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
