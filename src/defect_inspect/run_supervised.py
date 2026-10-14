"""(범주, k, 시드)마다 지도 학습 패치 헤드를 학습하고 평가 세트를 채점한다.

특징은 고정한 DINOv2 ViT-S/14(448px)에서 나오며 범주마다 한 번만 뽑는다. (k, 시드)마다
`<out>/k{k}-s{seed}/<category>.npz`(2단계 공통 형식 + 학습에 쓴 결함과 고른 에폭)와 `<category>_maps.npy`를
쓴다. `<out>/run.json`은 모든 실행을 나열하고 호출 전체가 끝났음을 표시한다. 출력 폴더에는 늘 한 호출의
결과만 있다: 다시 돌리면(`--overwrite`) 이전 호출이 쓴 것을 먼저 지운다. 규칙: docs/experiments.md의
"2단계 / 지도 학습" 절.
"""

# [파일 역할] 2단계 지도 학습의 실행기(python -m defect_inspect.run_supervised). 범주마다 DINOv2
# ViT-S/14(448px) 특징을 한 번 뽑고, (k, 시드) 조합마다 supervised.train_head로 헤드를 학습해 평가 세트를
# 채점한 뒤 픽셀 지표까지 계산해 저장한다.
# [설계 이유] 특징 추출이 가장 비싸므로 범주당 한 번만 하고 144회(12 범주 × k 4 × 시드 3) 학습은 그 특징을
# 재사용한다. (k, 시드)마다 폴더를 따로 두어 각 폴더가 Dinomaly와 같은 "2단계 공통 형식" 실행 하나가 되게
# 했다. 그래서 compare.py가 같은 읽기 함수(load_common)로 지도 학습과 Dinomaly를 비교한다.
# [흐름] main: 인자 검사 → 매니페스트·캐시·백본 준비 → 이전 결과 정리 → (test면) 봉인 테스트 장부 기록 →
# 범주마다 run_category → (k, 시드)별 run.json과 전체 run.json 기록(끝났다는 표시).
# [연결] supervised.py(헤드), splits.py(행 선택·라벨 부분집합), cache.ImageCache(리사이즈 캐시), metrics(픽셀
# AUROC·AUPRO), ledger(봉인 테스트 기록). 결과는 compare.py --supervised가
# outputs/sup-<절차>/k{k}-s{seed}/에서 읽는다.

import argparse
import json
import re
import time
from dataclasses import asdict, replace
from pathlib import Path

# torch는 모듈 수준에서 가져오지 않는다(torch 없이 도는 CI에서도 이 모듈을 import할 수 있게). run_category와
# main 안에서 늦게 가져온다.
import numpy as np

# 프로젝트 내부 모듈: 경로 상수, 리사이즈 캐시 읽기, 봉인 테스트 장부와 커밋 해시, 픽셀 지표, 분할 선택, 범주
# 목록.
from . import paths
from .cache import ImageCache
from .ledger import git_commit, record_test_access
from .metrics import aupro, pixel_auroc, pro_histograms
from .splits import K_VALUES, SEEDS, ManifestRow, label_subset, read_manifest, select
from .visa import CATEGORIES

# 규칙에 고정한 값: 백본 DINOv2 ViT-S/14, 입력 448px(패치 격자 32×32). PRO_BINS는 AUPRO 부트스트랩용
# 히스토그램 칸 수, EXTRACT_BATCH는 특징 추출 배치 크기다. _RUN_DIR_NAME은 이 모듈이 만든 결과 폴더(k5-s0
# 등)만 알아보는 정규식이다.
METHOD = "supervised"
BACKBONE = "dinov2_vits14"
IMG_SIZE = 448
PRO_BINS = 2000
EXTRACT_BATCH = 32
_RUN_DIR_NAME = re.compile(r"k\d+-s\d+")


# (k, 시드) 한 실행의 출력 폴더 경로: <out>/k{k}-s{seed}.
def run_dir(out_dir: Path, k: int, seed: int) -> Path:
    return Path(out_dir) / f"k{k}-s{seed}"


# 출력 폴더에 이전 실행이 남긴 결과 파일 목록. 이 모듈이 쓰는 이름(run.json, <범주>.npz, <범주>_maps.npy)만
# 세고 다른 파일은 건드리지 않는다.
# 호출 위치: main(덮어쓰기 허락 검사), clear_previous_results.
def previous_results(out_dir: Path) -> list[Path]:
    """이전 호출이 `out_dir`의 `k{k}-s{seed}` 폴더들에 남긴 파일(정렬 순서).

    이 모듈이 쓰는 것만 센다: 그런 폴더 바로 안의 `run.json`, `<category>.npz`, `<category>_maps.npy`.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    found: list[Path] = []
    for folder in sorted(out_dir.iterdir()):
        if not folder.is_dir() or not _RUN_DIR_NAME.fullmatch(folder.name):
            continue
        for path in sorted(folder.iterdir()):
            is_result = path.name == "run.json" or path.suffix == ".npz" or path.name.endswith("_maps.npy")
            if is_result and path.is_file():
                found.append(path)
    return found


# 다시 돌리기(--overwrite) 전에 이전 결과를 지운다. compare.load_common은 run.json을 보지 않고 폴더의 *.npz를
# 전부 읽으므로, 다른 커밋·다른 k·시드·범주로 돌린 예전 파일이 남아 있으면 새 결과와 섞여 분석된다(docstring).
# 빈 폴더만 지우고 그 밖의 파일은 남긴다.
def clear_previous_results(out_dir: Path) -> int:
    """`<out_dir>/run.json`과 `previous_results`의 모든 파일을 지우고, 뒤의 것의 개수를 돌려준다.

    분석은 run.json을 보지 않고 `k{k}-s{seed}/<category>.npz`를 읽으므로, 이전 호출(다른 커밋, 다른 k·시드·
    범주)의 결과가 새 결과 옆에 남아 있으면 안 된다. 지운 뒤 빈 폴더는 지우고, 그 밖의 파일은 그대로 둔다.
    """
    out_dir = Path(out_dir)
    # 맨 먼저 지워, 정리하거나 다시 돌리는 중인 폴더가 끝난 것처럼 보이는 순간이 없게 한다.
    (out_dir / "run.json").unlink(missing_ok=True)
    found = previous_results(out_dir)
    for path in found:
        path.unlink()
    for folder in sorted({path.parent for path in found}):
        if not any(folder.iterdir()):
            folder.rmdir()
    return len(found)


# 범주의 라벨 풀(학습용 결함은 여기서만 뽑는다, 범주당 40장)을 매니페스트 순서로 돌려준다. 특징을 한 번에 뽑아
# 두고 label_subset이 고른 이미지를 인덱스로 찾아 쓰기 위해 순서를 고정한다.
def label_pool(manifest: list[ManifestRow], category: str) -> list[ManifestRow]:
    """범주의 라벨 풀(학습용 결함은 여기서만 나온다). 매니페스트 순서다."""
    return [r for r in manifest if r.category == category and r.role == "label_pool"]


# JSON을 UTF-8·LF·들여쓰기 2칸으로 쓴다(한글이 \u 이스케이프로 바뀌지 않게 ensure_ascii=False).
def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


# 범주 하나를 처리한다. 특징을 한 번 뽑고 모든 (k, 시드)에 대해 학습 → 채점 → 픽셀 지표 → 저장을 반복한다.
# 반환은 범주 정보와 실행별 정보 목록이다.
# 호출 위치: main, tests/test_run_supervised.py.
def run_category(
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache: ImageCache,
    extractor,
    device: str,
    allow_test: bool,
    out_dir: Path,
    *,
    ks: tuple[int, ...] = K_VALUES,
    seeds: tuple[int, ...] = SEEDS,
    train_cfg=None,
) -> tuple[dict, list[dict]]:
    """범주 하나의 특징을 한 번 뽑고, 모든 (k, 시드)에 대해 학습하고 평가한다.

    범주 정보와 실행마다 정보 dict 하나씩을 돌려준다. `train_cfg`(기본값 `TrainConfig()`)는 seed만 그
    실행의 시드로 바꿔 쓴다.
    """
    import torch

    from . import supervised

    base_cfg = supervised.TrainConfig() if train_cfg is None else train_cfg
    # 봉인 테스트 관문(select의 allow_test)을 가장 먼저 지나게 해서, 허락 없이 test를 부르면 특징 추출 같은
    # 비싼 작업 전에 SealedTestError로 멈춘다(평가 부분을 맨 먼저 고르는 이유).
    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    eval_rows = eval_normal + eval_defect
    eval_labels = np.array([0] * len(eval_normal) + [1] * len(eval_defect), dtype=np.int8)
    # 학습·검증 세트는 절차와 관계없이 늘 dev 쪽에서 가져온다. 학습 정상 = 겹 1~4, 검증 = 겹 0 정상 + dev 결함
    # 20장(규칙의 검증 항목). test 절차에서 바뀌는 것은 평가 세트(봉인 테스트)뿐이다. 그래서 test 절차에서도
    # 학습에 봉인 테스트가 섞일 수 없다. 두 절차 모두에서 학습 정상은 겹 1-4, 검증은 겹 0 + dev 결함이다.
    train_normal = select(manifest, protocol="dev", part="pool_normal", category=category)
    val_normal = select(manifest, protocol="dev", part="eval_normal", category=category)
    val_defect = select(manifest, protocol="dev", part="eval_defect", category=category)
    val_rows = val_normal + val_defect
    val_labels = np.array([0] * len(val_normal) + [1] * len(val_defect), dtype=np.int8)
    # 라벨 풀 전체의 특징을 한 번만 뽑아 두고, (k, 시드)별 부분집합(label_subset, 포함 관계)은 이미지 → 행
    # 번호 표(pool_index)로 골라 쓴다.
    pool_rows = label_pool(manifest, category)
    pool_index = {r.image: i for i, r in enumerate(pool_rows)}
    subsets = {(k, seed): label_subset(manifest, category, k, seed) for k in ks for seed in seeds}

    # VRAM 최대 사용량을 범주 정보에 남기려고 측정 카운터를 범주마다 초기화한다(VRAM을 넘기면 공유 메모리로
    # 넘쳐 크게 느려지므로 기록해 둔다).
    on_cuda = torch.device(device).type == "cuda"
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()

    def features(rows: list[ManifestRow]) -> torch.Tensor:
        return supervised.extract(extractor, cache.images(rows), batch_size=EXTRACT_BATCH, device=device)

    t0 = time.perf_counter()
    normal_feats = features(train_normal)
    val_feats = features(val_rows)
    pool_feats = features(pool_rows) if pool_rows else normal_feats[:0]
    # dev 절차에서는 평가 세트가 곧 검증 세트(겹 0 정상 + dev 결함)이므로 특징을 다시 뽑지 않고 그대로 쓴다.
    # 둘이 다르면 코드가 잘못된 것이라 멈춘다.
    if protocol == "dev":
        if eval_rows != val_rows:
            raise RuntimeError("the dev evaluation set must be the validation set")
        eval_feats = val_feats
    else:
        eval_feats = features(eval_rows)
    features_s = time.perf_counter() - t0
    grid_h, grid_w, dim = (int(v) for v in normal_feats.shape[1:])
    if grid_h != grid_w:
        raise ValueError(f"expected a square patch grid, got {grid_h}x{grid_w}")

    # 정상 이미지의 마스크는 전부 0이라 정답도 전부 정상 패치다. 라벨 풀 결함의 정답은 픽셀 마스크를 32×32로
    # 최대 풀링한 것이다. 평가 마스크는 픽셀 지표용으로 bool로 둔다.
    normal_targets = supervised.patch_targets(cache.masks(train_normal), grid_h)
    pool_targets = (
        supervised.patch_targets(cache.masks(pool_rows), grid_h) if pool_rows else normal_targets[:0]
    )
    eval_masks = cache.masks(eval_rows).astype(bool)

    runs: list[dict] = []
    for k in ks:
        for seed in seeds:
            started = time.perf_counter()
            # 학습 세트 = 정상 겹 1~4 전부 + 라벨 풀에서 시드 순서로 앞 k장. 시드는 결함 선택과 학습(가중치
            # 초기화·순서·뒤집기) 모두에 같은 값을 쓴다(replace로 TrainConfig의 seed만 바꿈).
            defects = subsets[(k, seed)]
            index = np.array([pool_index[r.image] for r in defects], dtype=np.int64)
            train_feats = torch.cat([normal_feats, pool_feats[torch.from_numpy(index)]])
            train_targets = np.concatenate([normal_targets, pool_targets[index]])
            cfg = replace(base_cfg, seed=seed)

            t0 = time.perf_counter()
            result = supervised.train_head(
                train_feats, train_targets, val_feats, val_labels, cfg, device=device
            )
            train_s = time.perf_counter() - t0
            # 다음 (k, 시드)로 넘어가기 전에 이어 붙인 학습 특징을 바로 풀어 메모리를 아낀다.
            del train_feats

            t0 = time.perf_counter()
            scored = supervised.predict(result.head, eval_feats, device=device)
            # 임계값 보정 점수(cal_score): test 절차에서는 헤드가 학습하지 않은 겹 0 정상의 점수로 임계값을
            # 정한다(규칙: 홀드아웃). dev 절차에서는 겹 0 정상이 평가 정상 자체이므로 보정 점수를 비워 둔다.
            if protocol == "test":
                # 임계값은 헤드가 학습하지 않은 겹 0 정상으로 정한다.
                cal_score = supervised.predict(
                    result.head, val_feats[: len(val_normal)], device=device
                ).image_scores
            else:
                # dev에서는 겹 0 정상이 평가 정상 자체다.
                cal_score = np.empty(0, dtype=np.float32)
            predict_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            # 픽셀 지표는 float32로 바꿔 계산하고 바로 버린다. pro_histograms는 compare.py가 AUPRO의
            # 부트스트랩 구간을 낼 때 쓸 히스토그램이다.
            maps = scored.maps.astype(np.float32)
            px_auroc = pixel_auroc(maps, eval_masks)
            au_pro = aupro(maps, eval_masks)
            hist = pro_histograms(maps, eval_masks, bins=PRO_BINS)
            del maps
            metrics_s = time.perf_counter() - t0

            target_dir = run_dir(out_dir, k, seed)
            target_dir.mkdir(parents=True, exist_ok=True)
            # 검증하지 않은 에폭의 val_auroc(None)는 숫자 배열에 넣을 수 있게 NaN으로 바꾼다.
            val_auroc = [np.nan if h["val_auroc"] is None else h["val_auroc"] for h in result.history]
            # 압축해서 저장한다: PRO 히스토그램이 크고 거의 비어 있다(읽을 때는 평소처럼 np.load로 읽는다).
            np.savez_compressed(
                target_dir / f"{category}.npz",
                eval_images=np.array([r.image for r in eval_rows]),
                eval_labels=eval_labels,
                eval_defect_types=np.array([r.defect_types for r in eval_rows]),
                eval_score=scored.image_scores,
                cal_score=cal_score,
                pixel_auroc=np.float64(px_auroc),
                aupro=np.float64(au_pro),
                pro_edges=hist.edges,
                pro_normal=hist.normal,
                pro_components=hist.components,
                pro_component_image=hist.component_image,
                # 학습에 쓴 결함 k장과 그 유형을 함께 저장한다. compare.unseen_masks가 이 유형과 겹치지 않는
                # 테스트 결함만 남겨 H7(학습에서 보지 못한 결함 유형)을 계산한다.
                train_defect_images=np.array([r.image for r in defects], dtype=np.str_),
                train_defect_types=np.array([r.defect_types for r in defects], dtype=np.str_),
                best_epoch=np.int64(result.best_epoch),
                best_val_auroc=np.float64(result.best_val_auroc),
                history_loss=np.array([h["loss"] for h in result.history], dtype=np.float64),
                history_val_auroc=np.array(val_auroc, dtype=np.float64),
            )
            # 점수 맵(float16 [N, 256, 256])은 .npz와 별도의 .npy 파일로 저장한다(공통 형식의
            # <범주>_maps.npy).
            np.save(target_dir / f"{category}_maps.npy", scored.maps)
            info = {
                "category": category,
                "k": int(k),
                "seed": int(seed),
                "best_epoch": int(result.best_epoch),
                "best_val_auroc": float(result.best_val_auroc),
                "n_train_normal": len(train_normal),
                "n_train_defect": len(defects),
                "pos_weight": supervised.pos_weight(train_targets, cfg.max_pos_weight),
                "final_loss": float(result.history[-1]["loss"]),
                "train_s": round(train_s, 2),
                "predict_s": round(predict_s, 2),
                "metrics_s": round(metrics_s, 2),
                "seconds": round(time.perf_counter() - started, 2),
            }
            runs.append(info)
            print(json.dumps(info, ensure_ascii=False), flush=True)

    # 범주 정보: 각 부분의 장수, 특징 격자·차원, 특징 추출 시간, VRAM 최대 사용량. run.json에 들어가 재현과
    # 예산 점검에 쓴다.
    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if on_cuda else 0.0
    category_info = {
        "category": category,
        "train_normal": len(train_normal),
        "val_normal": len(val_normal),
        "val_defect": len(val_defect),
        "label_pool": len(pool_rows),
        "eval_normal": len(eval_normal),
        "eval_defect": len(eval_defect),
        "grid": [grid_h, grid_w],
        "dim": dim,
        "features_s": round(features_s, 2),
        "peak_vram_mb": round(peak_mb, 1),
    }
    return category_info, runs


# 명령행 진입점. 실행 순서에 안전장치가 겹겹이 있다: (1) test 절차는 --allow-test와 --stage가 둘 다 있어야
# 한다 (2) 같은 값이 두 번 들어간 목록을 거절 (3) 끝난 실행이나 끝나지 않은 결과가 있으면 --overwrite 없이는
# 멈춤 (4) 범주와 k 범위 검사 (5) 장부 기록 전에 실패할 수 있는 준비를 모두 끝냄.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", choices=["dev", "test"], default="dev")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--ks", nargs="+", type=int, default=list(K_VALUES), help="범주당 학습 결함 장수 k")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--out", type=Path, default=None, help="기본값: outputs/sup-<protocol>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-test", action="store_true", help="봉인 테스트 세트를 읽는다(장부에 기록)")
    parser.add_argument(
        "--stage", default="", help="봉인 테스트 장부에 남길 단계 이름(--allow-test와 함께 필수)"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    # 봉인 테스트는 단계 이름(--stage)과 함께만 읽는다. 장부에 어느 단계에서 읽었는지 남기기 위해서다.
    if args.protocol == "test" and (not args.allow_test or not args.stage):
        parser.error("--protocol test needs --allow-test and --stage")
    for name in ("categories", "ks", "seeds"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            parser.error(f"--{name} lists a value more than once: {values}")
    out_dir = args.out or paths.OUTPUTS / f"sup-{args.protocol}"
    finished = out_dir / "run.json"
    # run.json은 실행 전체가 끝났다는 표시다. 끝난 결과를 실수로 덮어쓰거나, 중간에 멈춘 결과 위에 새 결과를
    # 섞어 쓰지 않도록 둘 다 --overwrite를 요구한다.
    if finished.exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    if previous_results(out_dir) and not args.overwrite:
        parser.error(
            f"{out_dir} holds results of an unfinished run; pass --overwrite to remove them and start again"
        )

    manifest = read_manifest(paths.VISA_MANIFEST)
    for category in args.categories:
        if not any(r.category == category for r in manifest):
            parser.error(f"category {category!r} is not in {paths.VISA_MANIFEST}")
        n_pool = len(label_pool(manifest, category))
        if min(args.ks) < 0 or max(args.ks) > n_pool:
            parser.error(f"--ks must be within 0..{n_pool} (the label pool of {category})")

    import torch

    from .backbones import make_extractor
    from .supervised import TrainConfig

    # 캐시 열기·백본 적재처럼 실패할 수 있는 일을 장부 기록보다 먼저 한다. 그래야 장부에 "봉인 테스트를
    # 읽었다"는 줄이 남았는데 실제로는 아무것도 재지 못한 경우가 줄어든다.
    cache = ImageCache(paths.CACHE, IMG_SIZE)
    extractor = make_extractor(BACKBONE, img_size=IMG_SIZE).to(args.device).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    # 중간에 끊긴 재실행이 끝난 것처럼 보이면 안 되고, 이전 호출의 파일이 이번 결과로 오인되어서도 안 된다(위
    # 검사 때문에 이 정리는 --overwrite일 때만 일어난다).
    removed = clear_previous_results(out_dir)
    if removed:
        print(json.dumps({"removed_previous_results": removed}), flush=True)
    # 결과에 남길 커밋 해시(작업 트리가 더러우면 +dirty)와 torch 스레드 수. 학습은 CPU에서 같은 스레드 수일
    # 때만 비트 단위로 재현되므로(train_head docstring) 스레드 수를 함께 기록한다.
    commit = git_commit()
    torch_threads = int(torch.get_num_threads())
    config = {
        "backbone": BACKBONE,
        "img_size": IMG_SIZE,
        "ks": list(args.ks),
        "seeds": list(args.seeds),
        "train": asdict(TrainConfig()),  # "seed"는 실행마다 그 실행의 시드로 바뀐다
    }
    # 봉인 테스트를 읽기 직전에 reports/test_ledger.jsonl에 한 줄(단계, 구성, k·시드·범주 수, 커밋)을 남긴다.
    if args.protocol == "test":
        record_test_access(
            paths.TEST_LEDGER,
            stage=args.stage,
            config=METHOD,
            note=f"ks={list(args.ks)} seeds={list(args.seeds)} categories={len(args.categories)}",
            commit=commit,
        )

    started = time.perf_counter()
    categories: list[dict] = []
    runs: list[dict] = []
    for category in args.categories:
        t0 = time.perf_counter()
        info, category_runs = run_category(
            args.protocol,
            category,
            manifest,
            cache,
            extractor,
            args.device,
            args.allow_test,
            out_dir,
            ks=tuple(args.ks),
            seeds=tuple(args.seeds),
        )
        info["total_s"] = round(time.perf_counter() - t0, 1)
        categories.append(info)
        runs.extend(category_runs)
        print(json.dumps(info, ensure_ascii=False), flush=True)

    header = {
        "method": METHOD,
        "protocol": args.protocol,
        "commit": commit,
        "device": args.device,
        # 학습은 CPU에서 스레드 수가 같을 때만 비트 단위로 재현된다(`train_head` 참고).
        "torch_threads": torch_threads,
    }
    # (k, 시드) 폴더마다 자기 실행 정보만 담은 run.json을 쓴다. 마지막으로 out_dir/run.json(전체 요약)을 써서
    # 실행 전체가 끝났음을 표시한다. 이 파일이 맨 끝에 써지므로, 중간에 멈춘 실행은 끝난 것으로 보이지 않는다.
    # k{k}-s{seed} 폴더 하나하나가 그 자체로 공통 형식의 실행 폴더가 된다.
    for k in args.ks:
        for seed in args.seeds:
            own = [r for r in runs if r["k"] == k and r["seed"] == seed]
            sub = {
                **header,
                "config": {**config, "k": int(k), "seed": int(seed)},
                "categories": own,
                "total_s": round(sum(r["seconds"] for r in own), 1),
            }
            _write_json(run_dir(out_dir, k, seed) / "run.json", sub)
    run = {
        **header,
        "config": config,
        "categories": categories,
        "runs": runs,
        "total_s": round(time.perf_counter() - started, 1),
    }
    _write_json(finished, run)


if __name__ == "__main__":
    main()
