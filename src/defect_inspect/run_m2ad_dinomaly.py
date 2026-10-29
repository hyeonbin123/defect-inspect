"""8-C·8-D단계: 서빙 모델 구성(Dinomaly ViT-S, dms-280-car)의 M2AD 조명 변화와 한 조명 재등록.

8-C: `train`은 팔(e0, e2-a, e2-b)마다 전체 모델 하나와 시편 겹 모델 다섯 개를 학습한다. `score-train`은 겹
모델로 학습 시편 열 조명 이미지의 교차 적합 점수를 낸다(봉인 없음). `test`는 세 팔의 전체 모델로 봉인된 M2AD
test 이미지를 채점한다(장부에 한 줄). 표와 H24 판정은 `analyze_stage8 dms-train|dms-test`.
8-D: `finetune`은 새 조명 L마다 8-C E0의 모델 6개를 I01 + L 학습 이미지로 300스텝 미세조정한다.
`score-finetune`은 미세조정한 겹 모델로 I01·L 학습 시편의 교차 적합 점수를 내고(봉인 없음), `test-finetune`은
아홉 미세조정 전체 모델로 I01·L test를 채점한다(장부에 한 줄). 표와 H28 판정은
`analyze_stage8 dms-ft-train|dms-ft`.
규칙: docs/experiments.md 8단계 8-C, 8-D.
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
# [8-D 한 조명 재등록] Dinomaly에는 뱅크가 없어 3-B의 '새 조명 정상을 뱅크에 더하기'에 해당하는 절차가 곧
# 학습이다. 처음부터 다시 학습(8-C E2, 3,000스텝)하는 대신 8-C E0의 저장된 모델을 출발점으로 I01 + 새 조명 L
# 이미지로 짧게(300스텝, run_dinomaly.FINETUNE_SCHEDULE) 미세조정한다. 겹 f 모델은 E0 겹 f 모델(겹 f 시편을 본
# 적 없음)에서 출발해 겹 f 시편을 뺀 이미지로만 미세조정하므로, 그 겹 시편의 점수는 끝까지 교차 적합 점수다.
# finetune: 조명·모델마다 E0 모델 불러오기·출발 모델 확인 → train(FINETUNE_SCHEDULE, fp16) → ft/<L>/<모델>.pt.
# score-finetune: 미세조정 겹 모델로 그 겹 시편의 I01·L 학습 이미지 채점(fp32) → ft/<L>/train_scores.npz.
# test-finetune: 장부 기록 → 아홉 미세조정 전체 모델로 test 시편의 I01·L 채점(fp32) →
# ft/test/<L>_<검사기>.npz.
# [연결] run_dinomaly(CONFIGS, build_model, train, predict, save_model, load_model, FINETUNE_SCHEDULE,
# optimizer_settings), m2ad(시편 겹, 조명 묶음, 280px 캐시), run_m2ad_enrol(조명마다 같은 시편 순서의 행
# 고르기), download.sha256_file(출발 모델 파일 지문).

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import paths
from .download import sha256_file
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
# 8-D 상수(규칙 docs/experiments.md 8-D). FT_ARM: 출력 하위 폴더 이름(<out>/ft/<조명>/, 봉인 test는
# <out>/ft/test/). FT_START: 출발 모델의 팔(8-C E0, <out>/e0의 모델 6개). FT_STEPS: 미세조정 스텝 300(배치는
# BATCH 16 그대로, 8-C 3,000스텝의 1/10이라 GPU 약 30초). FT_LOG_EVERY: 짧은 학습이라 10스텝마다 기록한다.
# NEW_LIGHTS: 재등록할 새 조명(I01이 아닌 아홉 개). FT_LEDGER_CONFIG: 봉인 test 장부의 config 값(8-C 값에
# -ft).
FT_ARM = "ft"
FT_START = "e0"
FT_STEPS = 300
FT_LOG_EVERY = 10
NEW_LIGHTS = tuple(light for light in ILLUMINATIONS if light != REFERENCE)
FT_LEDGER_CONFIG = f"m2ad-{CONFIG}-ft"


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


# 모델 하나의 학습 이미지: 검사기 6개 × 주어진 조명 × (겹 f 모델이면 겹 f를 뺀) 학습 시편. 검사기 순서, 조명
# 순서, 시편 이름 순으로 늘어놓는다(배치 순서는 seed로 따로 섞인다). 8-C 팔(training_rows)과 8-D
# 미세조정(finetune_rows)이 같이 쓴다.
def _rows(plan: dict, lights: tuple[str, ...], model: str) -> list:
    held = None if model == "full" else int(model.removeprefix("fold"))
    picked = []
    for key in INSPECTORS:
        p = plan[key]
        for light in lights:
            for i, row in enumerate(p["rows"][light]):
                if held is None or p["folds"][i] != held:
                    picked.append(row)
    return picked


# 8-C 팔 모델 하나의 학습 이미지(등록 조명은 arm_lights).
def training_rows(plan: dict, arm: str, model: str) -> list:
    return _rows(plan, arm_lights(arm), model)


# 8-D의 미세조정 조명: 기준 조명 I01과 새 조명 L. I01을 함께 넣는 것은 미세조정이 기준 조명 검출을 잊지 않게
# 하려는 것이다(규칙의 망각 가드가 그 결과를 잰다). L은 I01이 아닌 아홉 조명 가운데 하나여야 한다.
def finetune_lights(light: str) -> tuple[str, str]:
    """8-D에서 조명 `light`를 다시 등록할 때 쓰는 조명: (I01, `light`)."""
    if light not in NEW_LIGHTS:
        raise ValueError(f"unknown new illumination {light!r}; known: {NEW_LIGHTS}")
    return (REFERENCE, light)


# 8-D 미세조정 모델 하나의 이미지: 검사기 6개 × (I01, L) × 학습 시편(겹 f 모델이면 겹 f 시편을 뺀 24개). 전체
# 모델은 360장, 겹 모델은 288장이다. 모델 이름이 MODELS에 없으면(예: fold9) 아무 시편도 빠지지 않는 조용한
# 실수가 되므로 거절한다.
def finetune_rows(plan: dict, light: str, model: str) -> list:
    """8-D에서 조명 `light`, 모델 `model`(full 또는 fold0~4)을 미세조정할 학습 행."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; known: {MODELS}")
    return _rows(plan, finetune_lights(light), model)


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


# train 명령: 팔 하나의 모델 6개를 차례로 학습한다. 이미 끝난 모델(<모델>.json이 있음)은 건너뛰어, 중간에
# 끊겨도 이어서 돌릴 수 있다. 손실이 유한하지 않으면 run_dinomaly.train이 멈춘다(관문).
def train_command(args) -> None:
    import torch

    from .run_dinomaly import (
        CONFIGS,
        OPTIMIZER,
        SCHEDULER,
        amp_applied,
        build_model,
        encoder_weights,
        runtime_env,
        save_model,
        train,
    )

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
        # env·encoder_weights: 같은 seed 0으로도 비트 단위 재현이 보장되지 않으므로(run_dinomaly.runtime_env
        # 주석) 학습 환경과 인코더 가중치 파일을 남긴다. 2026-10-29 전에 학습한 8-C 모델 기록에는 없다.
        info = {
            "arm": args.arm,
            "model": model_name,
            "lights": list(arm_lights(args.arm)),
            "commit": git_commit(),
            "device": args.device,
            "env": runtime_env(args.device),
            "encoder_weights": encoder_weights(cfg.encoder),
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


# 8-D 출발 모델 확인: <start>/<모델>.json이 8-C E0 팔의 같은 모델 기록이고(학습이 끝났다는 표시이기도 하다),
# 불러온 모델 파일이 8-C 규칙의 구성(dms-280-car, 같은 build 옵션)으로 STEPS(3,000)스텝 학습한 것이어야 한다.
# 점검용 짧은 학습(--steps 2 등)이나 다른 팔의 모델에서 출발하면 미세조정 결과가 규칙의 질문과 달라지므로
# 학습 전에 멈춘다. 돌려주는 값은 기록에 남길 출발 모델 정보(파일 이름, sha256, 스텝, 학습 커밋)다.
def _check_start(start_dir: Path, model_name: str, meta: dict, options: dict) -> dict:
    record_path = start_dir / f"{model_name}.json"
    with open(record_path, encoding="utf-8") as f:
        record = json.load(f)
    if record.get("arm") != FT_START or record.get("model") != model_name:
        raise ValueError(
            f"start model {record_path} is not the {FT_START} {model_name} model of stage 8-C: "
            f"arm {record.get('arm')!r}, model {record.get('model')!r}"
        )
    if meta.get("config") != CONFIG or meta.get("options") != options or meta.get("steps") != STEPS:
        raise ValueError(
            f"start model {start_dir / model_name}.pt is not a {CONFIG} model trained for {STEPS} steps: "
            f"config {meta.get('config')!r}, steps {meta.get('steps')!r}, options {meta.get('options')!r}"
        )
    return {
        "arm": FT_START,
        "model": model_name,
        "file": f"{start_dir.name}/{model_name}.pt",
        "sha256": sha256_file(start_dir / f"{model_name}.pt"),
        "steps": int(meta["steps"]),
        "commit": record.get("commit"),
    }


# finetune 명령: 새 조명마다 모델 6개(또는 --models)를 8-C E0의 같은 이름 모델에서 출발해 I01 + L 이미지로
# 미세조정한다(FT_STEPS × 배치 16, seed 0, 일정 FINETUNE_SCHEDULE, CUDA면 fp16 autocast + 패치 어텐션 — 출발
# 모델의 구성 옵션 그대로). 옵티마이저는 새 상태다(8-C 학습의 옵티마이저 상태는 저장하지 않았다). 이미 끝난
# 모델(<out>/ft/<L>/<모델>.json이 있음)은 건너뛰어, 54번 학습이 중간에 끊겨도 같은 명령으로 이어 간다. 손실이
# 유한하지 않으면 run_dinomaly.train이 멈춘다. 출발 모델 폴더는 기본 <out>/e0이고 --start로 바꿀 수 있다
# (출력만 임시 폴더로 보내는 점검 실행용). 출력: <모델>.pt, <모델>_log.jsonl, <모델>.json(완료 표시이자 기록).
def finetune_command(args) -> None:
    from .run_dinomaly import (
        CONFIGS,
        FINETUNE_SCHEDULE,
        amp_applied,
        encoder_weights,
        load_model,
        optimizer_settings,
        runtime_env,
        save_model,
        train,
    )

    cfg = CONFIGS[CONFIG]
    options = cfg.build_options()
    rows = read_meta(args.jsons or paths.RAW / "m2ad" / "jsons.zip")
    cache = M2adCache(args.cache or paths.CACHE, cfg.img_size)
    plan = train_plan(rows)
    start_dir = args.start or args.out / FT_START
    amp = amp_applied(True, args.device)
    optimizer, scheduler = optimizer_settings(FINETUNE_SCHEDULE)
    for light in args.light:
        out_dir = args.out / FT_ARM / light
        out_dir.mkdir(parents=True, exist_ok=True)
        for model_name in args.models:
            done = out_dir / f"{model_name}.json"
            if done.exists() and not args.overwrite:
                print(f"{done} exists, skipping", flush=True)
                continue
            for suffix in (".json", "_log.jsonl", ".pt"):
                (out_dir / f"{model_name}{suffix}").unlink(missing_ok=True)
            if not (start_dir / f"{model_name}.json").exists():
                raise FileNotFoundError(
                    f"{start_dir / model_name} has not finished training (stage 8-C {FT_START})"
                )
            picked = finetune_rows(plan, light, model_name)
            images = cache.images(picked)
            started = time.perf_counter()
            model, meta = load_model(start_dir / f"{model_name}.pt")
            start = _check_start(start_dir, model_name, meta, options)
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
                schedule=FINETUNE_SCHEDULE,
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
            # 기록: 조명·모델, 출발 모델(파일 지문 포함), 실행 환경, 실제로 쓴 옵티마이저·스케줄러 설정. 8-C
            # 학습 기록과 같은 키 순서에 light와 start를 더했다.
            info = {
                "arm": FT_ARM,
                "light": light,
                "model": model_name,
                "lights": list(finetune_lights(light)),
                "commit": git_commit(),
                "device": args.device,
                "env": runtime_env(args.device),
                "encoder_weights": encoder_weights(cfg.encoder),
                "start": start,
                "config": {
                    "name": cfg.name,
                    "encoder": cfg.encoder,
                    "img_size": cfg.img_size,
                    "steps": args.steps,
                    "batch_size": BATCH,
                    "amp": amp,
                    "seed": SEED,
                    "options": options,
                    "optimizer": {"name": "StableAdamW", **optimizer},
                    "scheduler": {"name": "WarmCosineScheduler", "total_iters": args.steps, **scheduler},
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


# 겹 모델 다섯 개(<out_dir>/fold0..4)로 각 겹 시편의 학습 이미지를 주어진 조명에서 채점한다(교차 적합, fp32).
# 겹 f 모델은 겹 f 시편을 학습에 쓰지 않았으므로 모든 점수가 '그 시편을 본 적 없는 모델'의 점수다. 출력:
# scores [검사기 6, 조명 수, 시편 30] float32. 빈 칸이 남으면(시편이 어느 겹에도 없음) 멈춘다. 8-C
# score-train(열 조명)과 8-D score-finetune(I01, L)이 같이 쓴다.
def _cross_fit(plan: dict, cache, out_dir: Path, lights: tuple[str, ...], device: str) -> np.ndarray:
    from .run_dinomaly import predict

    n = len(next(iter(plan.values()))["names"])
    scores = np.full((len(INSPECTORS), len(lights), n), np.nan, dtype=np.float32)
    for f in range(N_FOLDS):
        model = _load(out_dir, f"fold{f}", device)
        for i, key in enumerate(INSPECTORS):
            held = np.flatnonzero(plan[key]["folds"] == f)
            if held.size == 0:
                continue
            for li, light in enumerate(lights):
                images = cache.images([plan[key]["rows"][light][j] for j in held])
                scores[i, li, held] = predict(
                    model, images, batch_size=SCORE_BATCH, device=device, amp=False
                ).image_scores
        del model
    if np.isnan(scores).any():
        raise RuntimeError("not every training image was scored")
    return scores


# score-train 명령: 팔 하나의 겹 모델로 그 겹 시편의 열 조명 학습 이미지를 채점한다(fp32). 저장:
# scores [검사기 6, 조명 10, 시편 30], 시편 이름 [6, 30], 겹 [6, 30]. 학습 시편만 읽는다.
def score_train_command(args) -> None:
    from .run_dinomaly import CONFIGS

    cfg = CONFIGS[CONFIG]
    rows = read_meta(args.jsons or paths.RAW / "m2ad" / "jsons.zip")
    cache = M2adCache(args.cache or paths.CACHE, cfg.img_size)
    plan = train_plan(rows)
    out_dir = args.out / args.arm
    started = time.perf_counter()
    scores = _cross_fit(plan, cache, out_dir, ILLUMINATIONS, args.device)
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


# score-finetune 명령: 새 조명마다 미세조정한 겹 모델로 그 겹 시편의 I01·L 학습 이미지를 채점한다(fp32, 교차
# 적합). 저장(<out>/ft/<L>/train_scores.npz): scores [검사기 6, (I01, L), 시편 30], 시편 이름 [6, 30], 겹 [6,
# 30], 조명 이름. 학습 시편만 읽으므로 장부 줄이 없다. 이 점수로 analyze_stage8이 조건별(L, I01)·혼합 임계값을
# 정한다.
def score_finetune_command(args) -> None:
    from .run_dinomaly import CONFIGS

    cfg = CONFIGS[CONFIG]
    rows = read_meta(args.jsons or paths.RAW / "m2ad" / "jsons.zip")
    cache = M2adCache(args.cache or paths.CACHE, cfg.img_size)
    plan = train_plan(rows)
    for light in args.light:
        out_dir = args.out / FT_ARM / light
        lights = finetune_lights(light)
        started = time.perf_counter()
        scores = _cross_fit(plan, cache, out_dir, lights, args.device)
        np.savez(
            out_dir / "train_scores.npz",
            inspectors=np.array([f"{c}_{v}" for c, v in INSPECTORS]),
            lights=np.array(lights),
            specimens=np.array([plan[key]["names"] for key in INSPECTORS]),
            folds=np.stack([plan[key]["folds"] for key in INSPECTORS]),
            scores=scores,
        )
        _write_json(
            out_dir / "train_scores.json",
            {
                "arm": FT_ARM,
                "light": light,
                "commit": git_commit(),
                "device": args.device,
                "seconds": round(time.perf_counter() - started, 1),
            },
        )


# test 명령: 세 팔의 전체 모델로 test 시편을 열 조명에서 채점한다(fp32). 모든 팔의 모델이 끝났는지 먼저
# 확인하고 장부에 한 줄 남긴 뒤 test를 읽는다. 저장: test/<팔>_<검사기>.npz(scores [10, n], labels [10, n],
# 시편, 불량 여부).
def test_command(args, parser) -> None:
    from .run_dinomaly import CONFIGS, encoder_weights, predict, runtime_env

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
    # 채점 환경과 인코더 가중치 지문(고정 인코더는 _load가 사전학습 가중치에서 다시 만든다).
    _write_json(
        out_dir / "run.json",
        {
            "config": CONFIG,
            "commit": commit,
            "device": args.device,
            "env": runtime_env(args.device),
            "encoder_weights": encoder_weights(cfg.encoder),
            "arms": list(ARMS),
            "inspectors": [f"{c}_{v}" for c, v in INSPECTORS],
            "seconds": round(time.perf_counter() - started, 1),
        },
    )


# test-finetune 명령(8-D 봉인 test, 실행 한 번): 아홉 새 조명 L마다 미세조정 전체 모델로 test 시편의 I01과 L
# 이미지를 채점한다(fp32). 54개 모델이 모두 끝났는지와 아홉 조명의 학습 시편 점수가 모두 있는지 먼저 확인하고
# (규칙: 학습 시편 기록을 커밋한 뒤에 test를 읽는다), 장부에 한 줄 남긴 뒤 test를 읽는다. 저장:
# <out>/ft/test/<L>_<검사기>.npz(scores [2, n], labels [2, n], 첫 행 I01·둘째 행 L, 시편, 불량 여부)와
# run.json. 끝난 실행(run.json)이 있으면 --overwrite 없이 다시 읽지 않는다. 장부 기록이 먼저이고 이전 run.json
# 지우기가 그다음이다(기록 없이 결과만 사라지는 일이 없게, 8-C test와 같은 순서).
def test_finetune_command(args, parser) -> None:
    from .run_dinomaly import CONFIGS, encoder_weights, predict, runtime_env

    if args.allow_test is not True or not args.stage:
        parser.error("test-finetune reads the M2AD test images: it needs --allow-test and --stage")
    ft_dir = args.out / FT_ARM
    for light in NEW_LIGHTS:
        for model_name in MODELS:
            if not (ft_dir / light / f"{model_name}.json").exists():
                parser.error(f"{ft_dir / light / model_name} has not finished fine-tuning")
        if not (ft_dir / light / "train_scores.npz").exists():
            parser.error(f"run score-finetune for {light} first (its record is committed before the test)")
    out_dir = ft_dir / "test"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    cfg = CONFIGS[CONFIG]
    rows = read_meta(args.jsons or paths.RAW / "m2ad" / "jsons.zip")
    cache = M2adCache(args.cache or paths.CACHE, cfg.img_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    record_test_access(
        paths.TEST_LEDGER,
        stage=args.stage,
        config=FT_LEDGER_CONFIG,
        note=args.note
        or f"stage 8-D fine-tuned full models, lights {list(NEW_LIGHTS)}, I01 and L, fp32 scoring",
        commit=commit,
    )
    (out_dir / "run.json").unlink(missing_ok=True)
    started = time.perf_counter()
    tests = {key: _test_rows(rows, *key) for key in INSPECTORS}
    for light in NEW_LIGHTS:
        lights = finetune_lights(light)
        model = _load(ft_dir / light, "full", args.device)
        for category, view in INSPECTORS:
            test, names = tests[(category, view)]
            scores = np.stack(
                [
                    predict(
                        model,
                        cache.images(test[lt]),
                        batch_size=SCORE_BATCH,
                        device=args.device,
                        amp=False,
                    ).image_scores
                    for lt in lights
                ]
            )
            np.savez(
                out_dir / f"{light}_{category}_{view}.npz",
                specimens=np.array(names),
                object_anomaly=np.array([r.object_anomaly for r in test[REFERENCE]], dtype=np.int8),
                lights=np.array(lights),
                scores=scores,
                labels=np.array([[r.label for r in test[lt]] for lt in lights], dtype=np.int8),
            )
        del model
    _write_json(
        out_dir / "run.json",
        {
            "config": CONFIG,
            "ledger_config": FT_LEDGER_CONFIG,
            "commit": commit,
            "device": args.device,
            "env": runtime_env(args.device),
            "encoder_weights": encoder_weights(cfg.encoder),
            "model": "full",
            "lights": list(NEW_LIGHTS),
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
    # 8-D 하위 명령. 출력은 <out>/ft/ 아래이고, 출발 모델은 <out>/e0(8-C E0)이다.
    ft_help = {
        "finetune": "8-D: 새 조명마다 8-C E0 모델 6개를 I01 + 그 조명 학습 이미지로 300스텝 미세조정한다",
        "score-finetune": "8-D: 미세조정 겹 모델로 I01·새 조명 학습 시편의 교차 적합 점수를 낸다(봉인 없음)",
        "test-finetune": "8-D: 아홉 미세조정 전체 모델로 봉인 test의 I01·새 조명을 채점한다(장부에 한 줄)",
    }
    for name, text in ft_help.items():
        p = sub.add_parser(name, help=text, description=text)
        p.add_argument(
            "--out", type=Path, default=paths.OUTPUTS / "m2ad-dms", help="기본값: outputs/m2ad-dms"
        )
        p.add_argument("--device", default="cuda", help="학습·채점 장치(기본값: cuda)")
        p.add_argument("--jsons", type=Path, default=None, help="기본값: data/raw/m2ad/jsons.zip")
        p.add_argument("--cache", type=Path, default=None, help="캐시 폴더(기본값: data/cache, 280px 캐시)")
    for name in ("finetune", "score-finetune"):
        sub.choices[name].add_argument(
            "--light",
            nargs="+",
            choices=NEW_LIGHTS,
            required=True,
            help=(
                "다시 등록할 새 조명(02~10, 여럿이면 적은 순서대로 처리). "
                "조명마다 I01 + 그 조명 이미지로 미세조정·채점한다"
            ),
        )
    ft_p = sub.choices["finetune"]
    ft_p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS), help="미세조정할 모델")
    ft_p.add_argument(
        "--steps", type=int, default=FT_STEPS, help="미세조정 스텝(규칙: 300. 점검 때만 줄인다)"
    )
    ft_p.add_argument("--log-every", type=int, default=FT_LOG_EVERY, help="학습 기록 간격(스텝, 기본값: 10)")
    ft_p.add_argument(
        "--start",
        type=Path,
        default=None,
        help="출발 모델 폴더(기본값: <out>/e0, 8-C E0 팔). 점검 실행에서 출력만 다른 폴더로 보낼 때 쓴다",
    )
    ft_p.add_argument("--overwrite", action="store_true", help="끝난 모델도 다시 미세조정한다")
    tf_p = sub.choices["test-finetune"]
    tf_p.add_argument("--allow-test", action="store_true", help="M2AD 테스트 이미지를 읽는다(장부에 기록)")
    tf_p.add_argument("--stage", default="", help="봉인 테스트 장부에 남길 단계 이름(8-m2ad-ft)")
    tf_p.add_argument("--note", default="", help="장부 줄에 남길 메모(기본값: 자동 설명)")
    tf_p.add_argument("--overwrite", action="store_true", help="끝난 봉인 test 실행이 있어도 다시 채점한다")
    args = parser.parse_args(argv)
    if args.command in ("finetune", "score-finetune") and len(set(args.light)) != len(args.light):
        parser.error("--light lists an illumination more than once")
    if args.command == "train":
        if args.steps < 1:
            parser.error("--steps must be at least 1")
        train_command(args)
    elif args.command == "score-train":
        score_train_command(args)
    elif args.command == "test":
        test_command(args, parser)
    elif args.command == "finetune":
        if args.steps < 1:
            parser.error("--steps must be at least 1")
        finetune_command(args)
    elif args.command == "score-finetune":
        score_finetune_command(args)
    else:
        test_finetune_command(args, parser)


if __name__ == "__main__":
    main()
