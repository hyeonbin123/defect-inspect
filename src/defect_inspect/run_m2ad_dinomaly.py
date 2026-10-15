"""8-C단계: 서빙 모델 구성(Dinomaly ViT-S, dms-280-car)을 M2AD에서 학습해 조명 변화에서 잰다.

`train`은 팔(e0, e2-a, e2-b)마다 전체 모델 하나와 시편 겹 모델 다섯 개를 학습한다. `score-train`은 겹 모델로
학습 시편 열 조명 이미지의 교차 적합 점수를 낸다(봉인 없음). `test`는 세 팔의 전체 모델로 봉인된 M2AD test
이미지를 채점한다(장부에 한 줄). 표와 H24 판정은 `analyze_stage8 dms-train|dms-test`.
규칙: docs/experiments.md 8단계 8-C.
"""

# [파일 역할] 8-C 실행기. 6단계에서 CPU 서빙 모델로 뽑힌 구성(ViT-S reg4, 280px, 인코더 10블록, 고정 입력
# 크기, CAR, 패치 어텐션)을 그대로 쓰되 학습 이미지를 M2AD 학습 시편으로 바꾼다. 검사기 6개(Motor·Bird × 시점
# 000·120·240)를 모델 하나로 학습한다(VisA 12개 범주를 모델 하나로 학습한 것과 같다).
# [설계 이유] Dinomaly는 메모리 뱅크가 없어 교차 적합 점수를 얻으려면 겹마다 다시 학습해야 한다. 그래서 팔마다
# 전체 모델 + 겹 모델 5개(겹 f 시편의 모든 이미지를 뺌)를 학습하고, 임계값은 등록 조명 학습 이미지의 교차
# 적합 점수로 정한다(3-B·7단계와 같은 구조). 홀드아웃(검사기당 6장)은 목표 5%를 보장할 수 없어 쓰지 않았다.
# 스텝 수(3,000 × 배치 16)는 측정 전에 정했고 M2AD 점수를 보고 바꾸지 않는다(규칙).
# [흐름] train: 팔·모델마다 학습 이미지 모으기 → build_model(seed 0) → run_dinomaly.train(fp16) → save_model.
# score-train: 겹 모델로 그 겹 시편의 열 조명 학습 이미지 채점(fp32) → <팔>/train_scores.npz.
# test: 장부 기록 → 전체 모델로 test 시편 열 조명 채점(fp32) → test/<팔>_<검사기>.npz.
# [연결] run_dinomaly(CONFIGS, build_model, train, predict, save_model, load_model), m2ad(시편 겹, 조명 묶음,
# 280px 캐시), run_m2ad_enrol(조명마다 같은 시편 순서의 행 고르기).

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import paths
from .ledger import git_commit, record_test_access
from .m2ad import (
    CATEGORIES,
    ILLUMINATIONS,
    N_FOLDS,
    REFERENCE,
    VIEWS,
    M2adCache,
    illumination_groups,
    read_meta,
    specimen_folds,
)
from .run_m2ad_enrol import _test_rows, _train_rows

# 규칙의 상수: 구성 이름, 학습 스텝·배치·seed, 채점 배치. 팔 이름과 모델 이름(full, fold0..4).
CONFIG = "dms-280-car"
STEPS = 3000
BATCH = 16
SEED = 0
SCORE_BATCH = 32
ARMS = ("e0", "e2-a", "e2-b")
MODELS = ("full", *(f"fold{f}" for f in range(N_FOLDS)))
INSPECTORS = tuple((category, view) for category in CATEGORIES for view in VIEWS)


# 팔의 등록 조명: e0은 I01만, e2-a는 I01 + 묶음 A, e2-b는 I01 + 묶음 B(7단계와 같은 묶음).
def arm_lights(arm: str) -> tuple[str, ...]:
    a, b = illumination_groups()
    table = {"e0": (REFERENCE,), "e2-a": (REFERENCE, *a), "e2-b": (REFERENCE, *b)}
    if arm not in table:
        raise ValueError(f"unknown arm {arm!r}; known: {ARMS}")
    return table[arm]


# 검사기마다 조명별 학습 행과 시편 겹. 시편 겹은 범주마다 같다(같은 학습 시편이 세 시점에 나온다).
def train_plan(rows) -> dict[tuple[str, str], dict]:
    plan = {}
    for category, view in INSPECTORS:
        train, names = _train_rows(rows, category, view)
        fold_of = specimen_folds(names)
        plan[(category, view)] = {
            "rows": train,
            "names": names,
            "folds": np.array([fold_of[name] for name in names], dtype=np.int64),
        }
    return plan


# 모델 하나의 학습 이미지: 검사기 6개 × 등록 조명 × (겹 f 모델이면 겹 f를 뺀) 학습 시편. 검사기 순서, 조명
# 순서, 시편 이름 순으로 늘어놓는다(배치 순서는 seed로 따로 섞인다).
def training_rows(plan: dict, arm: str, model: str) -> list:
    lights = arm_lights(arm)
    held = None if model == "full" else int(model.removeprefix("fold"))
    picked = []
    for key in INSPECTORS:
        p = plan[key]
        for light in lights:
            for i, row in enumerate(p["rows"][light]):
                if held is None or p["folds"][i] != held:
                    picked.append(row)
    return picked


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


# train 명령: 팔 하나의 모델 6개를 차례로 학습한다. 이미 끝난 모델(<모델>.json이 있음)은 건너뛰어, 중간에
# 끊겨도 이어서 돌릴 수 있다. 손실이 유한하지 않으면 run_dinomaly.train이 멈춘다(관문).
def train_command(args) -> None:
    import torch

    from .run_dinomaly import CONFIGS, OPTIMIZER, SCHEDULER, amp_applied, build_model, save_model, train

    cfg = CONFIGS[CONFIG]
    options = cfg.build_options()
    rows = read_meta(args.jsons or paths.RAW / "m2ad" / "jsons.zip")
    cache = M2adCache(args.cache or paths.CACHE, cfg.img_size)
    plan = train_plan(rows)
    out_dir = args.out / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    amp = amp_applied(True, args.device)
    for model_name in args.models:
        done = out_dir / f"{model_name}.json"
        if done.exists() and not args.overwrite:
            print(f"{done} exists, skipping", flush=True)
            continue
        for suffix in (".json", "_log.jsonl", ".pt"):
            (out_dir / f"{model_name}{suffix}").unlink(missing_ok=True)
        picked = training_rows(plan, args.arm, model_name)
        images = cache.images(picked)
        started = time.perf_counter()
        torch.manual_seed(SEED)  # 병목과 디코더의 초기 가중치(6단계와 같다)
        model = build_model(cfg.encoder, **options)
        log = train(
            model,
            images,
            steps=args.steps,
            batch_size=BATCH,
            device=args.device,
            amp=amp,
            seed=SEED,
            log_every=args.log_every,
            log_path=out_dir / f"{model_name}_log.jsonl",
        )
        save_model(
            out_dir / f"{model_name}.pt",
            model,
            encoder=cfg.encoder,
            steps=args.steps,
            amp=amp,
            config=cfg.name,
            options=options,
        )
        info = {
            "arm": args.arm,
            "model": model_name,
            "lights": list(arm_lights(args.arm)),
            "commit": git_commit(),
            "device": args.device,
            "config": {
                "name": cfg.name,
                "encoder": cfg.encoder,
                "img_size": cfg.img_size,
                "steps": args.steps,
                "batch_size": BATCH,
                "amp": amp,
                "seed": SEED,
                "options": options,
                "optimizer": {"name": "StableAdamW", **OPTIMIZER},
                "scheduler": {"name": "WarmCosineScheduler", "total_iters": args.steps, **SCHEDULER},
            },
            "train_images": len(picked),
            "final_loss": log[-1]["loss"],
            "skipped_steps": log[-1]["skipped"],
            "peak_vram_mb": log[-1]["vram_mb"],
            "train_s": log[-1]["seconds"],
            "total_s": round(time.perf_counter() - started, 1),
        }
        _write_json(done, info)
        print(json.dumps(info, ensure_ascii=False), flush=True)
        del model


# 저장한 모델을 읽어 채점 장치로 올린다. 학습이 끝나지 않은 모델(<모델>.json 없음)은 쓰지 않는다.
def _load(out_dir: Path, model_name: str, device: str):
    from .run_dinomaly import load_model

    if not (out_dir / f"{model_name}.json").exists():
        raise FileNotFoundError(f"{out_dir / model_name} has not finished training")
    model, _ = load_model(out_dir / f"{model_name}.pt")
    return model.to(device).eval()


# score-train 명령: 팔 하나의 겹 모델로 그 겹 시편의 열 조명 학습 이미지를 채점한다(fp32). 저장:
# scores [검사기 6, 조명 10, 시편 30], 시편 이름 [6, 30], 겹 [6, 30]. 학습 시편만 읽는다.
def score_train_command(args) -> None:
    from .run_dinomaly import CONFIGS, predict

    cfg = CONFIGS[CONFIG]
    rows = read_meta(args.jsons or paths.RAW / "m2ad" / "jsons.zip")
    cache = M2adCache(args.cache or paths.CACHE, cfg.img_size)
    plan = train_plan(rows)
    out_dir = args.out / args.arm
    n = len(next(iter(plan.values()))["names"])
    scores = np.full((len(INSPECTORS), len(ILLUMINATIONS), n), np.nan, dtype=np.float32)
    started = time.perf_counter()
    for f in range(N_FOLDS):
        model = _load(out_dir, f"fold{f}", args.device)
        for i, key in enumerate(INSPECTORS):
            held = np.flatnonzero(plan[key]["folds"] == f)
            if held.size == 0:
                continue
            for li, light in enumerate(ILLUMINATIONS):
                images = cache.images([plan[key]["rows"][light][j] for j in held])
                scores[i, li, held] = predict(
                    model, images, batch_size=SCORE_BATCH, device=args.device, amp=False
                ).image_scores
        del model
    if np.isnan(scores).any():
        raise RuntimeError("not every training image was scored")
    np.savez(
        out_dir / "train_scores.npz",
        inspectors=np.array([f"{c}_{v}" for c, v in INSPECTORS]),
        lights=np.array(ILLUMINATIONS),
        enrolled=np.array([light in arm_lights(args.arm) for light in ILLUMINATIONS]),
        specimens=np.array([plan[key]["names"] for key in INSPECTORS]),
        folds=np.stack([plan[key]["folds"] for key in INSPECTORS]),
        scores=scores,
    )
    _write_json(
        out_dir / "train_scores.json",
        {
            "arm": args.arm,
            "commit": git_commit(),
            "device": args.device,
            "seconds": round(time.perf_counter() - started, 1),
        },
    )


# test 명령: 세 팔의 전체 모델로 test 시편을 열 조명에서 채점한다(fp32). 모든 팔의 모델이 끝났는지 먼저
# 확인하고 장부에 한 줄 남긴 뒤 test를 읽는다. 저장: test/<팔>_<검사기>.npz(scores [10, n], labels [10, n],
# 시편, 불량 여부).
def test_command(args, parser) -> None:
    from .run_dinomaly import CONFIGS, predict

    if args.allow_test is not True or not args.stage:
        parser.error("test reads the M2AD test images: it needs --allow-test and --stage")
    cfg = CONFIGS[CONFIG]
    rows = read_meta(args.jsons or paths.RAW / "m2ad" / "jsons.zip")
    cache = M2adCache(args.cache or paths.CACHE, cfg.img_size)
    for arm in ARMS:
        for model_name in MODELS:
            if not (args.out / arm / f"{model_name}.json").exists():
                parser.error(f"{args.out / arm / model_name} has not finished training")
        if not (args.out / arm / "train_scores.npz").exists():
            parser.error(f"run score-train for {arm} first (its record is committed before the test)")
    out_dir = args.out / "test"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    record_test_access(
        paths.TEST_LEDGER,
        stage=args.stage,
        config=f"m2ad-{CONFIG}",
        note=args.note or f"stage 8-C arms {list(ARMS)}, full models, fp32 scoring",
        commit=commit,
    )
    (out_dir / "run.json").unlink(missing_ok=True)
    started = time.perf_counter()
    tests = {key: _test_rows(rows, *key) for key in INSPECTORS}
    for arm in ARMS:
        model = _load(args.out / arm, "full", args.device)
        for category, view in INSPECTORS:
            test, names = tests[(category, view)]
            scores = np.stack(
                [
                    predict(
                        model,
                        cache.images(test[light]),
                        batch_size=SCORE_BATCH,
                        device=args.device,
                        amp=False,
                    ).image_scores
                    for light in ILLUMINATIONS
                ]
            )
            np.savez(
                out_dir / f"{arm}_{category}_{view}.npz",
                specimens=np.array(names),
                object_anomaly=np.array([r.object_anomaly for r in test[REFERENCE]], dtype=np.int8),
                lights=np.array(ILLUMINATIONS),
                scores=scores,
                labels=np.array([[r.label for r in test[light]] for light in ILLUMINATIONS], dtype=np.int8),
            )
        del model
    _write_json(
        out_dir / "run.json",
        {
            "config": CONFIG,
            "commit": commit,
            "device": args.device,
            "arms": list(ARMS),
            "inspectors": [f"{c}_{v}" for c, v in INSPECTORS],
            "seconds": round(time.perf_counter() - started, 1),
        },
    )


# 명령줄 진입점. 모든 하위 명령이 같은 출력 폴더(outputs/m2ad-dms)를 쓴다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "score-train", "test"):
        p = sub.add_parser(name)
        p.add_argument(
            "--out", type=Path, default=paths.OUTPUTS / "m2ad-dms", help="기본값: outputs/m2ad-dms"
        )
        p.add_argument("--device", default="cuda")
        p.add_argument("--jsons", type=Path, default=None, help="기본값: data/raw/m2ad/jsons.zip")
        p.add_argument("--cache", type=Path, default=None, help="캐시 폴더(기본값: data/cache, 280px 캐시)")
    for name in ("train", "score-train"):
        sub.choices[name].add_argument("--arm", choices=ARMS, required=True, help="학습 이미지 팔")
    train_p = sub.choices["train"]
    train_p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS), help="학습할 모델")
    train_p.add_argument("--steps", type=int, default=STEPS, help="학습 스텝(규칙: 3,000. 점검 때만 줄인다)")
    train_p.add_argument("--log-every", type=int, default=100)
    train_p.add_argument("--overwrite", action="store_true", help="끝난 모델도 다시 학습한다")
    test_p = sub.choices["test"]
    test_p.add_argument("--allow-test", action="store_true", help="M2AD 테스트 이미지를 읽는다(장부에 기록)")
    test_p.add_argument("--stage", default="", help="봉인 테스트 장부에 남길 단계 이름(8-m2ad)")
    test_p.add_argument("--note", default="")
    test_p.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "train":
        if args.steps < 1:
            parser.error("--steps must be at least 1")
        train_command(args)
    elif args.command == "score-train":
        score_train_command(args)
    else:
        test_command(args, parser)


if __name__ == "__main__":
    main()
