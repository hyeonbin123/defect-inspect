"""One PatchCore setting (backbone + input size) at several coreset ratios, with cross-fitted scores.

Greedy k-center selection is nested: the bank for a smaller ratio is a prefix of the bank built for the
largest ratio. So the expensive selection runs once per bank (the full bank and one bank per fold), and
every ratio only re-scores. Each ratio gets its own folder in the `run_patchcore` layout, readable by
`compare.load_patchcore`.
"""

# [파일 역할] 4단계 격자 실행 스크립트. PatchCore 설정 하나(백본 + 입력 크기)를 코어셋 비율 여러 개(기본 10%,
# 1%, 0.1%)로 채점하고, 비율마다 교차 적합 점수와 지표를 run_patchcore와 같은 폴더 구조로 저장한다.
# [왜 필요한가] 질문 4는 '메모리 뱅크 크기·입력 해상도·백본이 정확도와 CPU 지연을 어떻게 바꾸는가'다. 다섯
# 설정(wrn50-256/384, dinov2_vits14-252/392/448) × 세 비율을 dev로 재서 CPU 200ms 안에서 dev AUROC가 가장 높은
# 서빙 구성을 고르고, 그 설정만 test로 한 번 잰다(experiments.md 4단계).
# [설계 이유] 탐욕적 k-center 선택은 앞에서부터 포함 관계라, 작은 비율의 뱅크는 가장 큰 비율 뱅크의
# 앞부분이다(docstring). 그래서 비싼 코어셋 선택은 뱅크마다(전체 풀 뱅크 1개 + 겹을 뺀 뱅크 겹 수만큼) 한 번만
# 하고, 비율마다 앞부분을 잘라 채점만 다시 한다.
# [흐름] 인자 점검 → 매니페스트·캐시 점검(테스트 이미지를 읽기 전) → 장부 기록(test만) → 범주마다
# run_category(특징 → 뱅크들 → 비율별 채점·교차 적합·지표 저장) → 비율별 run.json과 최상위 run.json.
# [연결] 출력은 compare.load_patchcore·analyze_grid가 읽고, --save-banks로 저장한 뱅크는
# export.build_artifacts와 run_export_eval이 읽는다. ratio_name·ratio_rows는 export, run_export_eval,
# analyze_grid가 가져다 쓴다.

import argparse
import json
import time
from pathlib import Path

# 프로젝트 모듈: 경로(paths), 리사이즈 캐시(ImageCache), 구성 묶음(PatchCoreConfig), 장부(git_commit,
# record_test_access), 픽셀 지표(aupro, pixel_auroc, pro_histograms), 분할 선택(select, pool_folds 등), VisA
# 범주 목록. torch와 patchcore는 run_category 안에서 늦게 import한다.
import numpy as np

from . import paths
from .cache import ImageCache
from .configs import PatchCoreConfig
from .ledger import git_commit, record_test_access
from .metrics import aupro, pixel_auroc, pro_histograms
from .splits import PARTS, ManifestRow, pool_folds, read_manifest, select
from .visa import CATEGORIES

# PRO_BINS: AUPRO 부트스트랩용 히스토그램 구간 수(run_patchcore 등과 같은 값). BACKBONES: --backbone 선택지.
# dinov2_vitb14는 고를 수 있지만 4단계 격자 다섯 설정에는 들어 있지 않다.
PRO_BINS = 2000
BACKBONES = ("wrn50", "dinov2_vits14", "dinov2_vitb14")


# 비율 폴더 이름(예: r0.01). float의 repr을 쓰므로 같은 비율은 항상 같은 이름이 되고, _is_ratio_folder가 같은
# 규칙으로 이름을 되짚어 확인한다.
def ratio_name(ratio: float) -> str:
    """Folder name of a ratio: `r0.1`, `r0.01`, `r0.001`."""
    return f"r{float(ratio)!r}"


# 전체 패치 수와 비율로 뱅크 행 수를 계산한다. patchcore.build_bank와 같은 반올림이어야 '10% 뱅크의 앞 1% 행 =
# 1%로 직접 만든 뱅크'가 성립한다. export와 run_export_eval도 이 함수로 같은 길이를 잘라 쓴다.
def ratio_rows(n_features: int, ratio: float) -> int:
    """Bank rows for `ratio` of `n_features` patch features: the same rounding as `patchcore.build_bank`."""
    return n_features if ratio >= 1 else max(1, round(ratio * n_features))


# 설정 이름(예: wrn50-256). 출력 폴더와 장부의 config에 쓴다.
def setting_name(backbone: str, size: int) -> str:
    return f"{backbone}-{size}"


# 폴더 이름이 이 모듈이 쓰는 비율 폴더 형식(r + 0~1 사이 비율의 repr)인지 확인한다. clear_outputs가 다른
# 폴더를 지우지 않게 하는 안전장치다.
def _is_ratio_folder(name: str) -> bool:
    try:
        ratio = float(name[1:])
    except ValueError:
        return False
    return 0 < ratio <= 1 and ratio_name(ratio) == name


# 폴더가 비었을 때만 지운다. 이 모듈이 쓰지 않은 파일이 남아 있으면 폴더를 그대로 둔다.
def _remove_if_empty(folder: Path) -> None:
    try:
        folder.rmdir()
    except OSError:
        pass  # something that is not a grid output is in there: keep the folder


# 출력 폴더에 이전 실행이 남긴 결과를 지운다. 이전(또는 중간에 끊긴) 실행의 다른 비율·범주 결과가 남아 있으면
# compare.load_patchcore가 완성된 실행처럼 읽어 버리기 때문이다(docstring). 이 모듈이 쓰는 이름의 파일만
# 지우고, 최상위 run.json을 먼저 지워 '완성' 표시부터 없앤다.
# 호출 위치: main(--overwrite이거나 새 실행일 때, 장부 기록 전).
def clear_outputs(out_dir: Path) -> None:
    """Delete what earlier calls wrote to `out_dir`: run.json, the ratio folders' runs, the saved banks.

    A call owns its output folder: results of other ratios or categories left by an earlier (or an
    interrupted) call would otherwise still load as a finished run through `compare.load_patchcore`.
    Only files with the names this module writes are deleted. The top-level run.json goes first, so the
    folder stops counting as finished before anything else changes.
    """
    (out_dir / "run.json").unlink(missing_ok=True)
    for folder in sorted(p for p in out_dir.glob("r*") if p.is_dir() and _is_ratio_folder(p.name)):
        for path in [folder / "run.json", *sorted(folder.glob("*.npz"))]:
            path.unlink(missing_ok=True)
        _remove_if_empty(folder)
    bank_dir = out_dir / "banks"
    if bank_dir.is_dir():
        for pattern in ("*.json", "*_full.npy", "*_minus_fold_*.npy"):
            for path in sorted(bank_dir.glob(pattern)):
                path.unlink(missing_ok=True)
        _remove_if_empty(bank_dir)


# 범주 하나를 처리한다: 정상 풀 특징을 한 번 모으고, 가장 큰 비율로 전체 풀 뱅크와 겹을 뺀 뱅크들을 만든 뒤,
# 비율마다 평가 세트 점수·교차 적합 점수·픽셀 지표를 계산해 <비율 폴더>/<범주>.npz로 저장한다. 출력: 비율 →
# 요약 dict.
# 입력의 cfg.coreset_ratio는 ratios 가운데 가장 큰 값이어야 한다(작은 비율 뱅크를 앞부분 자르기로 얻기 때문).
def run_category(
    cfg: PatchCoreConfig,
    ratios: list[float],
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache: ImageCache,
    extractor,
    device: str,
    allow_test: bool,
    out_dir: Path,
    save_banks: bool = False,
) -> dict[float, dict]:
    """Build the banks at `cfg.coreset_ratio` (the largest ratio) and score every ratio in `ratios`."""
    import torch

    from .patchcore import build_bank, collect_features, score_images

    if max(ratios) != cfg.coreset_ratio:
        raise ValueError("cfg.coreset_ratio must be the largest ratio")

    # 구성의 배치 크기·재가중 k·맵 σ를 고정해 둔 채점 도우미(patchcore.score_images).
    def score(bank, images):
        return score_images(
            extractor,
            bank,
            images,
            batch_size=cfg.batch_size,
            device=device,
            reweight_k=cfg.reweight_k,
            sigma=cfg.sigma,
        )

    # 분할 선택: pool_normal은 뱅크 재료(dev는 겹 1~4, test는 겹 0~4), eval_normal/eval_defect는 평가
    # 세트(dev는 겹 0 정상 + dev 결함, test는 봉인 테스트). 봉인 테스트는 allow_test가 있어야 select가
    # 돌려준다.
    pool = select(manifest, protocol=protocol, part="pool_normal", category=category)
    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    eval_rows = eval_normal + eval_defect
    labels = np.array([0] * len(eval_normal) + [1] * len(eval_defect), dtype=np.int8)

    # GPU 최대 메모리(max_memory_allocated)를 범주마다 다시 재서 run.json에 남긴다. VRAM을 넘기면 공유
    # 메모리로 넘어가 20배쯤 느려지기 때문이다.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pool_images = cache.images(pool)
    feats, (grid_h, grid_w) = collect_features(
        extractor, pool_images, batch_size=cfg.batch_size, device=device
    )
    dim = feats.shape[1]
    per_image = grid_h * grid_w
    # 특징을 [이미지 수, 패치 수, D]로 보아 겹 단위로 이미지를 고를 수 있게 한다. folds는 풀 이미지마다 겹
    # 번호다.
    feats = feats.view(len(pool), per_image, dim)
    folds = np.array([r.fold for r in pool], dtype=np.int64)

    # 코어셋 선택은 여기서만 한다: 전체 풀로 한 번, 겹마다 '그 겹을 뺀 풀'로 한 번. 각 뱅크가 고른 원래 패치
    # 수 n을 같이 저장해 두어야, 나중에 비율별 행 수(ratio_rows(n, ratio))를 build_bank와 같게 계산할 수 있다.
    # The selections: one for the full pool, one per fold without that fold. N = patches they chose from.
    full_bank = build_bank(feats.reshape(-1, dim), cfg.coreset_ratio, seed=cfg.seed, device=device)
    n_full = len(pool) * per_image
    fold_banks: dict[int, tuple[torch.Tensor, int]] = {}
    for fold in pool_folds(protocol):
        keep = torch.from_numpy(folds != fold)
        bank = build_bank(feats[keep].reshape(-1, dim), cfg.coreset_ratio, seed=cfg.seed, device=device)
        fold_banks[fold] = (bank, int(keep.sum()) * per_image)
    # 큰 특징 텐서는 뱅크를 다 만든 뒤 바로 놓아 메모리를 아낀다.
    del feats
    banks_s = time.perf_counter() - t0

    # --save-banks면 가장 큰 비율의 전체 뱅크와 겹을 뺀 뱅크들, 그리고 각 뱅크의 원래 패치 수를 banks/에
    # 저장한다. 4단계에서는 이 뱅크로 아티팩트를 만들고(export) CPU 파이프라인으로 다시
    # 채점한다(run_export_eval).
    if save_banks:
        bank_dir = out_dir / "banks"
        bank_dir.mkdir(parents=True, exist_ok=True)
        np.save(bank_dir / f"{category}_full.npy", full_bank.numpy())
        sources = {"full": n_full}
        for fold, (bank, n) in fold_banks.items():
            np.save(bank_dir / f"{category}_minus_fold_{fold}.npy", bank.numpy())
            sources[f"minus_fold_{fold}"] = n
        with open(bank_dir / f"{category}.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump({"largest_ratio": cfg.coreset_ratio, "n_features": sources}, f, indent=2)
            f.write("\n")

    eval_images = cache.images(eval_rows)
    masks = cache.masks(eval_rows).astype(bool)
    infos: dict[float, dict] = {}
    # 비율마다 채점만 다시 한다. 평가 세트는 전체 풀 뱅크의 앞 rows_full행으로 채점한다.
    for ratio in sorted(ratios, reverse=True):
        t0 = time.perf_counter()
        rows_full = ratio_rows(n_full, ratio)
        res = score(full_bank[:rows_full], eval_images)
        # 교차 적합(out-of-fold) 점수: 풀 이미지 각각을 '그 이미지의 겹을 뺀 뱅크'로 채점한다. 자기 자신이
        # 뱅크에 들어간 점수로 임계값을 잡으면 정상의 86%가 불량으로 나오므로(README, 재대입), 임계값은 이
        # 점수로 정한다. 모든 풀 이미지가 채워졌는지 NaN으로 확인한다.
        oof = np.full(len(pool), np.nan, dtype=np.float32)
        for fold, (bank, n) in fold_banks.items():
            held = np.flatnonzero(folds == fold)
            oof[held] = score(bank[: ratio_rows(n, ratio)], pool_images[held]).image_scores
        if np.isnan(oof).any():
            raise RuntimeError("cross-fitting did not cover every pool image")

        # 평가 세트의 점수 맵(256 격자)으로 픽셀 AUROC, AUPRO와 부트스트랩용 PRO 히스토그램을 계산한다.
        maps = res.maps.astype(np.float32)
        hist = pro_histograms(maps, masks, bins=PRO_BINS)
        ratio_dir = out_dir / ratio_name(ratio)
        ratio_dir.mkdir(parents=True, exist_ok=True)
        # 범주별 결과 파일: 평가 이미지 경로·라벨·결함 유형, 평가 점수, 풀 이미지 경로·겹·교차 적합 점수, 픽셀
        # 지표, PRO 히스토그램, 뱅크 행 수. run_patchcore와 같은 키라 같은 분석 코드가 읽는다. 맵 자체는
        # 저장하지 않는다.
        np.savez(
            ratio_dir / f"{category}.npz",
            eval_images=np.array([r.image for r in eval_rows]),
            eval_labels=labels,
            eval_defect_types=np.array([r.defect_types for r in eval_rows]),
            eval_score_full=res.image_scores,
            pool_images=np.array([r.image for r in pool]),
            pool_folds=folds,
            pool_score_oof=oof,
            pixel_auroc=np.float64(pixel_auroc(maps, masks)),
            aupro=np.float64(aupro(maps, masks)),
            pro_edges=hist.edges,
            pro_normal=hist.normal,
            pro_components=hist.components,
            pro_component_image=hist.component_image,
            bank_rows=np.int64(rows_full),
        )
        # run.json에 들어갈 범주 요약: 표본 수, 패치 격자·특징 차원(아티팩트 meta에 쓰임), 뱅크 행 수,
        # 채점·뱅크 시간, 최대 VRAM.
        infos[ratio] = {
            "category": category,
            "pool": len(pool),
            "eval_normal": len(eval_normal),
            "eval_defect": len(eval_defect),
            "grid": [grid_h, grid_w],
            "dim": int(dim),
            "bank_rows": {"full": rows_full},
            "score_s": round(time.perf_counter() - t0, 2),
            "banks_s": round(banks_s, 2),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else 0.0,
        }
    return infos


# 비율 폴더 run.json의 config: 설정 이름에 비율을 붙인 이름(예: wrn50-256-r0.01)과 채점 설정.
# export.build_artifacts가 이 값으로 아티팩트 meta를 채운다.
def _ratio_config(cfg: PatchCoreConfig, ratio: float) -> dict:
    return {
        "name": f"{setting_name(cfg.backbone, cfg.img_size)}-{ratio_name(ratio)}",
        "backbone": cfg.backbone,
        "img_size": cfg.img_size,
        "coreset_ratio": ratio,
        "reweight_k": cfg.reweight_k,
        "sigma": cfg.sigma,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
    }


# 명령줄 진입점. 예: `python -m defect_inspect.run_grid --backbone wrn50 --size 256 --protocol dev
# --save-banks`(README 4단계). test 절차는 --allow-test와 --stage가 있어야 하고 장부에 한 줄 남는다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--size", type=int, required=True, help="input size (a multiple of 14 for DINOv2)")
    parser.add_argument("--protocol", choices=["dev", "test"], required=True)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.1, 0.01, 0.001])
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument(
        "--out", type=Path, default=None, help="default: outputs/grid-<backbone>-<size>-<protocol>"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-banks", action="store_true", help="keep the full and the minus-fold banks")
    parser.add_argument("--allow-test", action="store_true", help="read the sealed test set (logged)")
    parser.add_argument(
        "--stage", default="", help="stage label for the test ledger (required with --allow-test)"
    )
    parser.add_argument("--note", default="")
    parser.add_argument(
        "--overwrite", action="store_true", help="redo a finished run (its outputs are deleted first)"
    )
    args = parser.parse_args(argv)

    from .backbones import make_extractor

    # 인자 점검: 비율은 (0, 1], DINOv2는 패치 크기 14의 배수 입력만(그래서 256·384 대신 252·392를 쓴다,
    # experiments.md 4단계), test 절차는 --allow-test와 --stage 필수. 이미 완성된 실행은 --overwrite 없이
    # 덮어쓰지 않는다.
    ratios = sorted(set(args.ratios), reverse=True)
    if any(not 0 < r <= 1 for r in ratios):
        parser.error("ratios must be in (0, 1]")
    if args.backbone.startswith("dinov2") and args.size % 14:
        parser.error("DINOv2 input sizes must be multiples of 14")
    if args.protocol == "test" and (not args.allow_test or not args.stage):
        parser.error("--protocol test needs --allow-test and --stage")
    setting = setting_name(args.backbone, args.size)
    out_dir = args.out or paths.OUTPUTS / f"grid-{setting}-{args.protocol}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")

    # 테스트 이미지를 건드리지 않고 실패할 수 있는 것(범주 중복, 매니페스트에 없는 범주, 캐시 없음, 백본
    # 생성)을 모두 장부 기록 전에 확인한다. 실패한 실행이 '봉인 테스트를 읽었다'는 기록만 남기는 일을 막는다.
    # Everything that can fail without touching a test image comes before the ledger line.
    manifest = read_manifest(paths.VISA_MANIFEST)
    for category in args.categories:
        if args.categories.count(category) > 1:
            parser.error(f"--categories lists {category} twice")
        empty = [
            part
            for part in PARTS  # manifest rows only: no image is read here
            if not select(
                manifest, protocol=args.protocol, part=part, category=category, allow_test=args.allow_test
            )
        ]
        if empty:
            parser.error(f"unknown or empty category {category!r}: the manifest has no {', '.join(empty)}")
    try:
        cache = ImageCache(paths.CACHE, args.size)
    except FileNotFoundError:
        command = f"python -m defect_inspect.cache --size {args.size}"
        parser.error(f"no {args.size} px image cache: build it with `{command}`")
    # 격자용 구성: 가장 큰 비율을 coreset_ratio로 두고, 나머지(재가중 k = 9, σ = 4, 배치 32, seed 0)는
    # PatchCoreConfig 기본값(P0와 같은 문헌 기본값)을 쓴다.
    cfg = PatchCoreConfig(name=setting, backbone=args.backbone, img_size=args.size, coreset_ratio=ratios[0])
    extractor = make_extractor(cfg.backbone, img_size=cfg.img_size).to(args.device).eval()
    # 커밋 해시는 출력에 쓰기 전에 한 번만 구한다(ledger.git_commit docstring 권장). 이전 결과를 지운 뒤,
    # test면 테스트 이미지를 읽기 직전에 장부에 남긴다.
    commit = git_commit()
    out_dir.mkdir(parents=True, exist_ok=True)
    clear_outputs(out_dir)
    if args.protocol == "test":
        record_test_access(
            paths.TEST_LEDGER, stage=args.stage, config=f"grid-{setting}", note=args.note, commit=commit
        )

    per_ratio: dict[float, list[dict]] = {r: [] for r in ratios}
    started = time.perf_counter()
    for category in args.categories:
        t0 = time.perf_counter()
        infos = run_category(
            cfg,
            ratios,
            args.protocol,
            category,
            manifest,
            cache,
            extractor,
            args.device,
            args.allow_test,
            out_dir,
            save_banks=args.save_banks,
        )
        for ratio, info in infos.items():
            per_ratio[ratio].append(info)
        line = {"category": category, "seconds": round(time.perf_counter() - t0, 1)}
        line["banks_s"] = infos[ratios[0]]["banks_s"]
        print(json.dumps(line), flush=True)

    # JSON을 UTF-8·LF로 쓰는 도우미. 비율 폴더 run.json을 먼저 쓰고 최상위 run.json(완성 표시)을 마지막에
    # 써서, 중간에 끊기면 완성된 실행으로 보이지 않게 한다.
    def dump(path: Path, payload: dict) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    for ratio in ratios:
        dump(
            out_dir / ratio_name(ratio) / "run.json",
            {
                "config": _ratio_config(cfg, ratio),
                "protocol": args.protocol,
                "commit": commit,
                "device": args.device,
                "categories": per_ratio[ratio],
            },
        )
    dump(
        out_dir / "run.json",
        {
            "setting": setting,
            "ratios": ratios,
            "protocol": args.protocol,
            "commit": commit,
            "device": args.device,
            "categories": list(args.categories),
            "save_banks": bool(args.save_banks),
            "total_s": round(time.perf_counter() - started, 1),
        },
    )


if __name__ == "__main__":
    main()
