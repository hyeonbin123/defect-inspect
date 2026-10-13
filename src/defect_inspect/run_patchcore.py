"""Build PatchCore memory banks for one protocol and score every set the analysis needs.

Per category this writes `<out>/<category>.npz` (image scores under the full bank, the hold-out bank,
resubstitution and cross-fitting, plus pixel metrics and PRO histograms) and `<out>/<category>_maps.npy`
(score maps of the evaluation set under the full bank). Tables are made by `defect_inspect.analyze`.
"""

# [파일 역할] PatchCore 실행기(python -m defect_inspect.run_patchcore). 구성 하나(p0, d-s, p0-c 등)와
# 절차(dev/test)로 범주마다 메모리 뱅크를 만들고, 1단계 임계값 보정 비교에 필요한 점수 네 가지와 픽셀 지표를
# 한 번에 저장한다.
# [왜 필요한가] 1단계 질문 2("정상 이미지로 고정한 임계값은 목표 오검출률을 지키나")는 보정 방식 세 가지를
# 같은 이미지로 비교한다(docs/experiments.md 1단계 표). 재대입(뱅크를 만든 이미지의 점수), 홀드아웃(겹 하나를
# 뺀 뱅크와 그 겹의 점수), 교차 적합(겹마다 그 겹을 뺀 뱅크로 매긴 점수 모음)에 필요한 뱅크와 점수를 여기서
# 모두 만들어 두고, 표와 판정은 analyze.py가 저장된 점수만 읽어 낸다.
# [흐름] 범주마다: 정상 풀 특징 추출 → 전체 뱅크 → 재대입 점수 → 겹마다 그 겹을 뺀 뱅크로 교차 적합 점수(그중
# 첫 겹의 뱅크가 홀드아웃 뱅크) → 평가 세트를 전체 뱅크와 홀드아웃 뱅크로 채점 → 픽셀 지표 → .npz/.npy 저장.
# 끝나면 run.json으로 실행 전체를 표시한다.
# [연결] analyze.py(1단계 표), compare.load_patchcore(2단계 비교), run_perturb(--save-bank로 남긴 뱅크로 교란
# 이미지 채점), figures(실패 사례 그림의 점수 맵)가 이 출력을 읽는다.

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

# torch와 patchcore 모듈은 모듈 수준에서 가져오지 않고 run_category 안에서 가져온다(torch 없이 도는 CI에서도
# import 가능하게).
import numpy as np

# 내부 모듈: 경로, 리사이즈 캐시, 등록된 구성(get_config), 봉인 테스트 장부, 픽셀 지표, 분할 선택(정상 풀의
# 겹과 홀드아웃 겹), 범주 목록.
from . import paths
from .cache import ImageCache
from .configs import PatchCoreConfig, get_config
from .ledger import git_commit, record_test_access
from .metrics import aupro, pixel_auroc, pro_histograms
from .splits import ManifestRow, holdout_fold, pool_folds, read_manifest, select
from .visa import CATEGORIES

# AUPRO 부트스트랩용 히스토그램 칸 수(pro_histograms에 넘김). 다른 실행기(run_supervised 등)와 같은 값이다.
PRO_BINS = 2000


# 범주 하나에 대해 뱅크를 만들고 점수를 저장한다. 반환은 run.json에 들어갈 범주 요약(장수, 격자, 뱅크 행 수,
# VRAM 최대치, 단계별 시간)이다.
# 호출 위치: main, tests/test_run_patchcore.py.
def run_category(
    cfg: PatchCoreConfig,
    protocol: str,
    category: str,
    manifest: list[ManifestRow],
    cache: ImageCache,
    extractor,
    device: str,
    allow_test: bool,
    out_dir: Path,
    save_bank: bool = False,
) -> dict:
    import torch

    from .patchcore import build_bank, collect_features, score_images

    # 구성의 재가중 이웃 수(reweight_k)와 맵 가우시안 σ를 고정해 채점하는 지역 함수. 아래에서 뱅크만 바꿔 여러
    # 번 부른다.
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

    # 정상 풀은 dev면 겹 1~4, test면 겹 0~4다. 평가 세트는 dev면 겹 0 정상 + dev 결함, test면 봉인 테스트이고
    # allow_test가 없으면 여기서 SealedTestError로 멈춘다.
    pool = select(manifest, protocol=protocol, part="pool_normal", category=category)
    eval_normal = select(
        manifest, protocol=protocol, part="eval_normal", category=category, allow_test=allow_test
    )
    eval_defect = select(
        manifest, protocol=protocol, part="eval_defect", category=category, allow_test=allow_test
    )
    eval_rows = eval_normal + eval_defect
    labels = np.array([0] * len(eval_normal) + [1] * len(eval_defect), dtype=np.int8)

    # VRAM 최대 사용량을 범주마다 따로 재려고 카운터를 초기화한다. 단계별 시간도 timing에 모아 run.json에
    # 남긴다.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timing: dict[str, float] = {}

    t0 = time.perf_counter()
    pool_images = cache.images(pool)
    # 정상 풀 특징을 한 번만 뽑아 [이미지, 패치, 차원]으로 둔다. 겹별 뱅크는 이 특징에서 이미지 단위로 골라
    # 만들므로 특징 추출을 반복하지 않는다.
    feats, (grid_h, grid_w) = collect_features(
        extractor, pool_images, batch_size=cfg.batch_size, device=device
    )
    dim = feats.shape[1]
    feats = feats.view(len(pool), grid_h * grid_w, dim)
    folds = np.array([r.fold for r in pool], dtype=np.int64)
    timing["features_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # 전체 뱅크: 정상 풀 전부의 패치에서 코어셋(P0은 10%)을 고른다. 서빙과 교차 적합·재대입 평가에 쓰는
    # "배포할 모델"이다.
    full_bank = build_bank(feats.reshape(-1, dim), cfg.coreset_ratio, seed=cfg.seed, device=device)
    timing["full_bank_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    # 재대입 점수: 뱅크를 만든 바로 그 이미지를 그 뱅크로 채점한다. 뱅크가 이 이미지들의 패치에서 골라졌으므로
    # 점수가 낮게 나오고, 이 점수로 정한 임계값은 낮게 잡힌다(H1이 재려는 치우침).
    resub = score(full_bank, pool_images).image_scores
    timing["resub_score_s"] = time.perf_counter() - t0

    # Cross-fitting: every pool image is scored by a bank that never saw its fold.
    t0 = time.perf_counter()
    # 교차 적합 점수(out-of-fold)를 NaN으로 채워 시작하고 겹마다 채운다. 끝난 뒤 NaN이 남았는지로 모든
    # 이미지가 채점됐는지 확인한다.
    oof = np.full(len(pool), np.nan, dtype=np.float32)
    holdout_bank = None
    bank_rows = {"full": int(full_bank.shape[0])}
    for fold in pool_folds(protocol):
        # 이 겹을 뺀 이미지들의 패치로 뱅크를 만들고(같은 코어셋 비율·시드), 뺀 겹의 이미지만 그 뱅크로
        # 채점한다.
        keep = torch.from_numpy(folds != fold)
        bank = build_bank(feats[keep].reshape(-1, dim), cfg.coreset_ratio, seed=cfg.seed, device=device)
        held = np.flatnonzero(folds == fold)
        oof[held] = score(bank, pool_images[held]).image_scores
        bank_rows[f"minus_fold_{fold}"] = int(bank.shape[0])
        # 풀의 첫 겹(test 절차 겹 0, dev 절차 겹 1)을 뺀 뱅크는 홀드아웃 방식의 배포 모델이 되므로 따로
        # 보관한다. 홀드아웃의 보정 점수는 analyze.py가 oof에서 그 겹만 골라 쓴다.
        if fold == holdout_fold(protocol):
            holdout_bank = bank
    timing["crossfit_s"] = time.perf_counter() - t0
    if holdout_bank is None or np.isnan(oof).any():
        raise RuntimeError("cross-fitting did not cover every pool image")
    # 정상 풀 특징과 이미지는 더 쓰지 않으므로 평가 세트를 읽기 전에 풀어 메모리를 아낀다.
    del feats, pool_images

    t0 = time.perf_counter()
    eval_images = cache.images(eval_rows)
    masks = cache.masks(eval_rows).astype(bool)
    # 평가 세트를 두 모델로 채점한다. 전체 뱅크 점수는 재대입·교차 적합 방식의 평가에, 홀드아웃 뱅크 점수는
    # 홀드아웃 방식의 평가에 쓴다(analyze._strategy_inputs).
    res_full = score(full_bank, eval_images)
    res_holdout = score(holdout_bank, eval_images)
    timing["eval_score_s"] = time.perf_counter() - t0
    timing["eval_images"] = len(eval_rows)

    t0 = time.perf_counter()
    # 픽셀 지표(픽셀 AUROC, AUPRO, 부트스트랩용 PRO 히스토그램)는 전체 뱅크의 점수 맵으로만 계산한다.
    maps = res_full.maps.astype(np.float32)
    px_auroc = pixel_auroc(maps, masks)
    au_pro = aupro(maps, masks)
    hist = pro_histograms(maps, masks, bins=PRO_BINS)
    timing["pixel_metrics_s"] = time.perf_counter() - t0

    peak_mb = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    # 범주 결과 파일. 평가 이미지 경로·라벨·결함 유형, 평가 점수 두 가지(full, holdout), 정상 풀의
    # 경로·겹·재대입 점수·교차 적합 점수, 픽셀 지표와 PRO 히스토그램을 담는다. 이것만으로 analyze.py가 세 보정
    # 방식의 표를 모두 다시 만들 수 있다.
    np.savez(
        out_dir / f"{category}.npz",
        eval_images=np.array([r.image for r in eval_rows]),
        eval_labels=labels,
        eval_defect_types=np.array([r.defect_types for r in eval_rows]),
        eval_score_full=res_full.image_scores,
        eval_score_holdout=res_holdout.image_scores,
        pool_images=np.array([r.image for r in pool]),
        pool_folds=folds,
        pool_score_resub=resub,
        pool_score_oof=oof,
        pixel_auroc=np.float64(px_auroc),
        aupro=np.float64(au_pro),
        pro_edges=hist.edges,
        pro_normal=hist.normal,
        pro_components=hist.components,
        pro_component_image=hist.component_image,
    )
    np.save(out_dir / f"{category}_maps.npy", res_full.maps)
    if save_bank:
        # The full bank in selection order (fp16): later stages score new conditions against it.
        np.save(out_dir / f"{category}_bank.npy", full_bank.numpy())
    return {
        "category": category,
        "pool": len(pool),
        "eval_normal": len(eval_normal),
        "eval_defect": len(eval_defect),
        "grid": [grid_h, grid_w],
        "dim": int(dim),
        "bank_rows": bank_rows,
        "peak_vram_mb": round(peak_mb, 1),
        "timing": {k: round(v, 2) for k, v in timing.items()},
    }


# 명령행 진입점. --config로 등록된 구성을 고르고, test 절차는 --allow-test와 --stage를 함께 요구한다. 이미
# 끝난 실행(run.json 있음)은 --overwrite 없이는 다시 돌리지 않는다.
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="p0")
    parser.add_argument("--protocol", choices=["dev", "test"], default="dev")
    parser.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    parser.add_argument("--out", type=Path, default=None, help="default: outputs/<config>-<protocol>")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-test", action="store_true", help="read the sealed test set (logged)")
    parser.add_argument(
        "--stage", default="", help="stage label for the test ledger (required with --allow-test)"
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-bank", action="store_true", help="keep each category's full memory bank")
    args = parser.parse_args(argv)

    from .backbones import make_extractor

    cfg = get_config(args.config)
    if args.protocol == "test":
        if not args.allow_test or not args.stage:
            parser.error("--protocol test needs --allow-test and --stage")
    out_dir = args.out or paths.OUTPUTS / f"{cfg.name}-{args.protocol}"
    if (out_dir / "run.json").exists() and not args.overwrite:
        parser.error(f"{out_dir} already holds a finished run; pass --overwrite to redo it")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(paths.VISA_MANIFEST)
    cache = ImageCache(paths.CACHE, cfg.img_size)
    # 결과에 남길 커밋 해시를 실행 전에 한 번 구한다. test 절차면 봉인 테스트를 읽기 전에
    # 장부(reports/test_ledger.jsonl)에 단계·구성·커밋을 기록한다.
    commit = git_commit()
    if args.protocol == "test":
        record_test_access(
            paths.TEST_LEDGER, stage=args.stage, config=cfg.name, note=args.note, commit=commit
        )

    # 구성의 백본(WideResNet-50 또는 DINOv2)과 입력 크기로 특징 추출기를 만든다. centre=True(p0-c, d-s-c)면
    # 이미지마다 패치 특징의 평균을 빼는 7단계 E1 변형이다.
    extractor = make_extractor(cfg.backbone, img_size=cfg.img_size, centre=cfg.centre)
    extractor = extractor.to(args.device).eval()
    summaries = []
    started = time.perf_counter()
    for category in args.categories:
        t0 = time.perf_counter()
        info = run_category(
            cfg,
            args.protocol,
            category,
            manifest,
            cache,
            extractor,
            args.device,
            args.allow_test,
            out_dir,
            save_bank=args.save_bank,
        )
        info["total_s"] = round(time.perf_counter() - t0, 1)
        summaries.append(info)
        print(json.dumps(info, ensure_ascii=False), flush=True)

    # run.json은 모든 범주가 끝난 뒤 마지막에 쓴다. 이 파일이 있으면 끝난 실행이고, compare.load_patchcore와
    # analyze가 구성·커밋·장치를 여기서 읽는다.
    run = {
        "config": asdict(cfg),
        "protocol": args.protocol,
        "commit": commit,
        # Coreset selection differs between devices at float rounding level,
        # so banks that are compared must be built on the same device.
        "device": args.device,
        "categories": summaries,
        "total_s": round(time.perf_counter() - started, 1),
    }
    with open(out_dir / "run.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
