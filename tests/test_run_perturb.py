# [파일 역할] 3-A단계(VisA 합성 교란) 실행 스크립트 `defect_inspect.run_perturb`를 검증하는 테스트 묶음이다.
# 끝난 실행(PatchCore 뱅크 또는 Dinomaly 모델)의 평가 세트를 교란 15조건 + 교란 없음으로 다시 채점하는 과정,
# 교란 없음 점수가 원래 실행과 같아야 한다는 가드(clean guard), 원본 실행 점검, 봉인과 장부 기록 순서를
# 확인한다.
# [왜 필요한가] 3-A는 다시 학습하거나 보정하지 않고 원래 실행의 뱅크·모델과 임계값을 그대로
# 쓴다(docs/experiments.md 3-A). 그래서 '지금 채점하는 것이 그 임계값을 정한 바로 그 뱅크·모델·이미지인가'가
# 결과의 전제이고, 이를 깨는 경우(다른 뱅크, 다른 이미지 목록, 다른 구성, 다른 모델 파일)를 이미지를 읽기 전에
# 막는지가 이 파일의 중심이다.
# [주요 구성] 가짜 데이터·원본 실행 만들기 → 범주 하나의 채점(PatchCore) → clean guard와 원본 점수 검사 →
# CLI(PatchCore: 장부 순서, 원본 점검, 재실행, 두 범주, 원본 덮어쓰기 거부) → Dinomaly 대역 모델 → 7단계용
# 조건 부분집합과 중심화 구성.
# [설계 메모] 원본 실행은 테스트 안에서 run_patchcore.run_category로 실제로 만들어, 두 실행 사이의 파일 형식이
# 맞는지도 함께 확인한다.

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

# PatchCore 채점과 Dinomaly 대역 모델이 torch를 쓰므로 torch가 없는 CI(.github/workflows/ci.yml)에서는 파일
# 전체를 건너뛴다.
torch = pytest.importorskip("torch")

# run_patchcore는 원본 실행을 만들려고, analyze_perturb는 결과를 분석 모듈이 그대로 읽는지 보려고, backbones는
# CLI가 만드는 백본을 가짜로 바꾸려고 가져온다. conditions·patchcore.score_images는 교란·채점 기대값을 직접
# 계산하는 데, sha256_file은 Dinomaly 모델 파일의 해시를 원본 기록과 맞추는 데, splits의 함수는 가짜
# 매니페스트를 쓰고 평가 행을 고르는 데 쓴다.
from defect_inspect import analyze_perturb, backbones, run_patchcore, run_perturb  # noqa: E402
from defect_inspect.conditions import apply_condition, condition_names  # noqa: E402
from defect_inspect.configs import PatchCoreConfig  # noqa: E402
from defect_inspect.download import sha256_file  # noqa: E402
from defect_inspect.patchcore import score_images  # noqa: E402
from defect_inspect.splits import (  # noqa: E402
    SEALED_ROLES,
    ManifestRow,
    SealedTestError,
    select,
    write_manifest,
)

# 가짜 데이터의 기준값. NPZ_KEYS는 교란 실행이 범주마다 쓰는 npz 키 전부다(맵·픽셀 지표는 3-A 규칙상 재지
# 않으므로 없다). DEV_IMAGES는 dev 평가 세트(겹 0 정상 8장 = 순번 0, 5, ..., 35과 dev 결함 6장), TEST_IMAGES는
# 봉인 테스트 정상 10장과 결함 4장이다.
SIZE = 32
NPZ_KEYS = {"eval_images", "eval_labels", "conditions", "scores", "cal_score"}
DEV_IMAGES = [f"toy/n{i}.JPG" for i in range(0, 40, 5)] + [f"toy/d{i}.JPG" for i in range(6)]
TEST_IMAGES = [f"toy/tn{i}.JPG" for i in range(10)] + [f"toy/td{i}.JPG" for i in range(4)]


# 실제 백본 대신 쓰는 작은 특징 추출기: 4x4 칸마다 평균 색(3차원).
class GridMean(torch.nn.Module):
    """백본 대신 쓰는 작은 추출기: 4x4 격자의 칸마다 평균색을 낸다."""

    name = "gridmean"
    dim = 3

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.float(), SIZE // 4).permute(0, 2, 3, 1)


# 들어온 배치 크기를 기록하는 GridMean. 채점이 구성의 batch_size대로 나눠 도는지(기본값을 몰래 쓰지 않는지)
# 확인하는 데 쓴다.
class Recording(GridMean):
    """들어온 배치마다 크기를 적어 두는 GridMean."""

    def __init__(self):
        super().__init__()
        self.batches: list[int] = []

    def forward(self, x):
        self.batches.append(int(x.shape[0]))
        return super().forward(x)


# ImageCache 대역. 결함 이미지는 밝은 사각형을 가진다. requested에 요청된 이미지 경로를 쌓아 무엇을 읽었는지
# 확인하고, ledger 경로가 주어지면 봉인 이미지를 내주기 전에 장부에 줄이 이미 있는지 assert로 확인한다(장부
# 기록이 읽기보다 먼저라는 규칙).
class FakeCache:
    """ImageCache 대신 쓰는 메모리 안 가짜: 결함 이미지에만 정상에 없는 밝은 사각형이 있다."""

    size = SIZE

    def __init__(self, manifest, ledger=None):
        rng = np.random.default_rng(0)
        self._images, self._masks, self._role = {}, {}, {}
        self._ledger = ledger
        self.requested: list[str] = []
        for row in manifest:
            img = rng.integers(100, 140, (SIZE, SIZE, 3), dtype=np.uint8)
            mask = np.zeros((256, 256), dtype=np.uint8)
            if row.label == "anomaly":
                img[8:16, 8:16] = 255
                mask[64:128, 64:128] = 1
            self._images[row.image] = img
            self._masks[row.image] = mask
            self._role[row.image] = row.role

    def _note(self, rows):
        if self._ledger is not None and any(self._role[r.image] in SEALED_ROLES for r in rows):
            # 봉인 이미지 읽기는 이미 장부에 기록돼 있어야 한다.
            assert self._ledger.exists() and self._ledger.read_text(encoding="utf-8").strip()
        self.requested += [r.image for r in rows]

    def images(self, rows):
        self._note(rows)
        return np.stack([self._images[r.image] for r in rows])

    def masks(self, rows):
        self._note(rows)
        return np.stack([self._masks[r.image] for r in rows])


# 범주마다 정상 풀 40장(겹 = 순번 % 5), dev 결함 6장, 봉인 테스트 정상 10장·결함 4장인 가짜 매니페스트.
def _manifest(categories=("toy",)):
    rows = []
    for c in categories:
        for i in range(40):
            rows.append(ManifestRow(f"{c}/n{i}.JPG", "", c, "normal", "pool_normal", i % 5, ""))
        for i in range(6):
            rows.append(ManifestRow(f"{c}/d{i}.JPG", f"{c}/d{i}.png", c, "anomaly", "dev_defect", -1, "hole"))
        for i in range(10):
            rows.append(ManifestRow(f"{c}/tn{i}.JPG", "", c, "normal", "test_normal", -1, ""))
        for i in range(4):
            mask = f"{c}/td{i}.png"
            rows.append(ManifestRow(f"{c}/td{i}.JPG", mask, c, "anomaly", "test_defect", -1, "hole"))
    return rows


# 등록 구성과 같은 이름(p0)을 붙여, 방법을 이름으로 찾는 CLI가 이 구성을 쓸 수 있게 한다. 채점 옵션은
# 하나도 `score_images`의 기본값(9, 4.0, 32)으로 두지 않았다: 구성을 무시한 실행은 이 점수를 재현하지 못한다.
CFG = PatchCoreConfig(
    name="p0", backbone="gridmean", img_size=SIZE, coreset_ratio=0.5, reweight_k=3, sigma=1.0, batch_size=8
)


# 교란 실행의 원본이 될 끝난 PatchCore 실행을 실제 run_patchcore.run_category로 만든다(뱅크 저장 포함).
# run.json에는 원본 커밋을 'src1234'로 적어, 교란 실행이 원본 커밋을 따로 기록하는지 확인할 수 있게 한다.
def _make_source(src, protocol, categories=("toy",)):
    """교란 실행이 기대하는 모양 그대로, 메모리 뱅크까지 저장한 끝난 `run_patchcore` 실행."""
    manifest = _manifest(categories)
    src.mkdir(parents=True)
    cache = FakeCache(manifest)
    infos = [
        run_patchcore.run_category(
            CFG, protocol, c, manifest, cache, GridMean(), "cpu", protocol == "test", src, True
        )
        for c in categories
    ]
    meta = {
        "config": asdict(CFG),
        "protocol": protocol,
        "commit": "src1234",
        "device": "cpu",
        "categories": infos,
    }
    (src / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    return manifest


# 원본 실행이 저장한 뱅크로 run_perturb.patchcore_scorer를 만든다.
def _scorer(src):
    bank = torch.from_numpy(np.load(src / "toy_bank.npy"))
    return run_perturb.patchcore_scorer(CFG, GridMean(), bank, "cpu")


# 범주 toy 하나를 run_category로 교란 채점한다. score를 바꿔 넣으면 clean guard가 어긋난 점수를 잡는지 시험할
# 수 있다.
def _perturb(src, out, protocol="dev", allow_test=False, cache=None, score=None):
    manifest = _manifest()
    out.mkdir(parents=True, exist_ok=True)
    return run_perturb.run_category(
        protocol,
        "toy",
        manifest,
        cache or FakeCache(manifest),
        score or _scorer(src),
        run_perturb.load_source(src, "toy", "p0"),
        allow_test,
        out,
    )


# ---------------------------------------------------------------- 범주 하나, PatchCore


# 교란 실행의 기본 계약을 확인한다. 조건 16개가 등록 순서(clean 먼저)로 채점되고, npz에는 정해진 키만 있으며,
# 교란 없음 점수는 원본의 전체 뱅크 점수와 비트 단위로 같다(같은 뱅크·이미지·배치를 CPU에서 돌리므로). 보정
# 점수는 원본의 교차 적합 점수(pool_score_oof)를 그대로 넘긴다(3-A는 원래 임계값을 그대로 쓴다).
# 각 조건의 점수가 '저장 뱅크로 그 조건의 이미지를 채점한 값'과 같은지 직접 계산해 비교하고, score_images의
# 기본 채점 옵션(재가중 k, σ)으로는 다른 점수가 나와 구성이 실제로 반영됐음을 확인한다.
def test_every_condition_is_scored_in_the_registered_order(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    manifest = _make_source(src, "dev")
    info = _perturb(src, out)
    assert (info["category"], info["eval_normal"], info["eval_defect"], info["cal"]) == ("toy", 8, 6, 32)
    assert info["clean_max_rel_diff"] == 0.0  # CPU에서 같은 뱅크·이미지·배치: 완전히 같다
    assert set(info["timing"]) == {"perturb_s", "score_s"}
    assert sorted(p.name for p in out.iterdir()) == ["toy.npz"]  # 맵도 픽셀 지표도 없다

    names = condition_names()
    with np.load(out / "toy.npz") as z, np.load(src / "toy.npz") as source:
        assert set(z.files) == NPZ_KEYS
        assert z["conditions"].tolist() == names and names[0] == "clean" and len(names) == 16
        assert z["eval_images"].tolist() == source["eval_images"].tolist() == DEV_IMAGES
        assert z["eval_labels"].dtype == np.int8 and z["eval_labels"].tolist() == [0] * 8 + [1] * 6
        assert z["scores"].shape == (16, 14) and z["scores"].dtype == np.float32
        assert np.array_equal(z["scores"][0], source["eval_score_full"])
        # PatchCore 실행의 임계값은 정상 풀의 교차 적합 점수에서 나온다.
        assert z["cal_score"].dtype == np.float32
        assert np.array_equal(z["cal_score"], source["pool_score_oof"])
        scores = z["scores"]

    # 모든 행은 저장된 뱅크로 그 조건의 평가 이미지를 채점한 값이고, 그 밖의 것이 아니다.
    rows = select(manifest, protocol="dev", part="eval_normal", category="toy")
    rows += select(manifest, protocol="dev", part="eval_defect", category="toy")
    images = FakeCache(manifest).images(rows)
    bank = torch.from_numpy(np.load(src / "toy_bank.npy"))
    for k, name in enumerate(names):
        direct = score_images(
            GridMean(),
            bank,
            apply_condition(images, name),
            batch_size=CFG.batch_size,
            device="cpu",
            reweight_k=CFG.reweight_k,
            sigma=CFG.sigma,
        )
        assert np.array_equal(scores[k], direct.image_scores), name
    # 어두운 이미지는 평소 밝기의 패치 뱅크에서 멀다: 모든 정상이 교란 없는 정상보다 높게 채점된다.
    assert scores[names.index("brightness-3")][:8].min() > scores[0][:8].max()
    # 위 비교는 구성에 따라 달라진다: 기본 재가중으로는 다른 점수가 나온다.
    default_k = score_images(GridMean(), bank, images, batch_size=CFG.batch_size, device="cpu")
    assert not np.allclose(default_k.image_scores, scores[0], rtol=1e-3)


# patchcore_scorer가 구성의 batch_size·reweight_k·sigma를 score_images에 넘기는지 스파이로 확인한다. 두
# 구성(배치 8과 5)에서 실제 배치 크기 기록과 넘긴 옵션, 직접 계산한 점수가 모두 맞아야 한다.
def test_the_scorer_takes_batch_size_and_scoring_options_from_the_config(tmp_path, monkeypatch):
    from defect_inspect import patchcore

    src = tmp_path / "src"
    manifest = _make_source(src, "dev")
    images = FakeCache(manifest).images(manifest[:14])
    bank = torch.from_numpy(np.load(src / "toy_bank.npy"))
    passed = []

    def spy(extractor, bank, images, **options):
        passed.append(options)
        return score_images(extractor, bank, images, **options)

    monkeypatch.setattr(patchcore, "score_images", spy)
    for cfg, batches in (
        (CFG, [8, 6]),
        (PatchCoreConfig("x", "gridmean", SIZE, 0.5, reweight_k=5, sigma=2.0, batch_size=5), [5, 5, 4]),
    ):
        extractor = Recording()
        passed.clear()
        scores = run_perturb.patchcore_scorer(cfg, extractor, bank, "cpu")(images)
        assert extractor.batches == batches  # 이미지가 실제로 cfg.batch_size 묶음으로 지나갔다
        assert passed == [
            {"batch_size": cfg.batch_size, "device": "cpu", "reweight_k": cfg.reweight_k, "sigma": cfg.sigma}
        ]
        direct = score_images(
            GridMean(), bank, images, batch_size=cfg.batch_size, device="cpu", reweight_k=cfg.reweight_k
        )
        assert scores.dtype == np.float32 and np.array_equal(scores, direct.image_scores)


# 저장 뱅크를 바꿔치기하면(값에 0.5를 더함) 교란 없음 점수가 원본과 달라 CleanScoreMismatch로 멈춰야 한다. 첫
# 조건(clean)에서 바로 멈추므로 채점은 한 번(14장)뿐이고 npz도 남지 않는다. 원본 임계값이 속하지 않는 뱅크로
# 교란 결과를 내는 일을 막는 가드다.
def test_a_tampered_bank_fails_the_clean_guard(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    _make_source(src, "dev")
    bank = np.load(src / "toy_bank.npy")
    np.save(src / "toy_bank.npy", (bank + np.float16(0.5)).astype(np.float16))
    calls = []
    real = _scorer(src)

    def counting(images):
        calls.append(len(images))
        return real(images)

    with pytest.raises(run_perturb.CleanScoreMismatch, match="clean scores differ"):
        _perturb(src, out, score=counting)
    assert calls == [14]  # clean 조건에서 멈춘다
    assert not (out / "toy.npz").exists()


# clean guard의 기준이 상대 차이 CLEAN_RTOL(1e-3)인지 확인한다. 5e-4만큼 어긋나면 통과하면서 그 차이가
# 기록되고, 2e-3이면 멈춘다.
def test_the_clean_guard_uses_the_relative_limit(tmp_path):
    src = tmp_path / "src"
    _make_source(src, "dev")
    real = _scorer(src)
    info = _perturb(src, tmp_path / "ok", score=lambda images: real(images) * np.float32(1 + 5e-4))
    assert 4e-4 < info["clean_max_rel_diff"] < 6e-4
    with pytest.raises(run_perturb.CleanScoreMismatch):
        _perturb(src, tmp_path / "bad", score=lambda images: real(images) * np.float32(1 + 2e-3))


# 이미지 14장 가운데 한 장(처음, 중간, 마지막)만 어긋나도 clean guard가 잡아야 한다. 평균이 아니라 모든
# 이미지의 최대 상대 차이로 비교하는지 확인한다.
@pytest.mark.parametrize("odd", [0, 5, 13])
def test_the_clean_guard_looks_at_every_image(tmp_path, odd):
    src = tmp_path / "src"
    _make_source(src, "dev")
    real = _scorer(src)

    def one_off(images):
        scores = real(images).copy()
        scores[odd] *= np.float32(1 + 2e-3)  # 14장 중 어디에 있든 한 장
        return scores

    with pytest.raises(run_perturb.CleanScoreMismatch):
        _perturb(src, tmp_path / "out", score=one_off)
    assert not (tmp_path / "out" / "toy.npz").exists()


# max_rel_diff의 경계: 빈 배열은 0, 0과 0은 0, 기준이 0인데 점수가 0이 아니면 아주 큰 값(분모를 0 대신 가장
# 작은 양수로 둠), 모양이 다르면 ValueError.
def test_max_rel_diff():
    assert run_perturb.max_rel_diff(np.array([1.0, 2.0]), np.array([1.0, 2.5])) == pytest.approx(0.2)
    assert run_perturb.max_rel_diff(np.zeros(3), np.zeros(3)) == 0.0
    assert run_perturb.max_rel_diff(np.zeros(0), np.zeros(0)) == 0.0
    assert run_perturb.max_rel_diff(np.array([1e-6]), np.array([0.0])) > 1.0
    with pytest.raises(ValueError):
        run_perturb.max_rel_diff(np.zeros(3), np.zeros(4))


# 원본의 평가 이미지 목록이 매니페스트의 평가 세트와 다르면(여기서는 순서를 뒤집음) 이미지를 한 장도 읽기 전에
# ValueError로 멈춰야 한다. 점수와 이미지의 짝이 어긋난 채 비교하는 일을 막는다.
def test_a_source_that_evaluated_other_images_is_rejected(tmp_path):
    src = tmp_path / "src"
    manifest = _make_source(src, "dev")
    source = run_perturb.load_source(src, "toy", "p0")
    swapped = run_perturb.Source(source.eval_images[::-1].copy(), source.eval_score, source.cal_score)
    cache = FakeCache(manifest)
    with pytest.raises(ValueError, match="did not evaluate"):
        run_perturb.run_category("dev", "toy", manifest, cache, _scorer(src), swapped, False, tmp_path)
    assert cache.requested == []


# 채점 결과가 유한하지 않거나(NaN) 장수가 다르면 각각 FloatingPointError, ValueError로 멈춘다.
def test_bad_scores_are_rejected(tmp_path):
    src = tmp_path / "src"
    _make_source(src, "dev")
    with pytest.raises(FloatingPointError):
        _perturb(src, tmp_path / "nan", score=lambda images: np.full(len(images), np.nan, np.float32))
    with pytest.raises(ValueError, match="expected 14 scores"):
        _perturb(src, tmp_path / "short", score=lambda images: np.zeros(3, np.float32))


# test 절차는 allow_test 없이 SealedTestError로 멈추고 이미지를 읽지 않는다. 허락하면 봉인 테스트 14장만 읽고,
# 보정 점수는 원본 test 실행의 교차 적합 점수 40장(정상 풀 전체)이다.
def test_the_sealed_test_set_needs_permission(tmp_path):
    src = tmp_path / "src"
    manifest = _make_source(src, "test")
    cache = FakeCache(manifest)
    with pytest.raises(SealedTestError):
        _perturb(src, tmp_path / "out", protocol="test", cache=cache)
    assert cache.requested == [] and not (tmp_path / "out" / "toy.npz").exists()

    info = _perturb(src, tmp_path / "out", protocol="test", allow_test=True, cache=cache)
    assert (info["eval_normal"], info["eval_defect"], info["cal"]) == (10, 4, 40)
    assert cache.requested == TEST_IMAGES
    with np.load(tmp_path / "out" / "toy.npz") as z:
        assert z["eval_images"].tolist() == TEST_IMAGES
        assert z["eval_labels"].tolist() == [0] * 10 + [1] * 4
        assert z["scores"].shape == (16, 14) and z["cal_score"].shape == (40,)


# PatchCore 실행을 Dinomaly(dm) 원본으로 읽으려 하면 키 이름(eval_score)이 없다는 KeyError가 난다. 방법과 원본
# 형식이 섞이는 실수를 이름으로 알려 준다.
def test_load_source_names_a_run_of_the_wrong_kind(tmp_path):
    src = tmp_path / "src"
    _make_source(src, "dev")
    with pytest.raises(KeyError, match="eval_score"):
        run_perturb.load_source(src, "toy", "dm")  # PatchCore 실행은 공통 형식이 아니다


# npz의 배열 일부를 바꾸거나(None이면 지움) 다시 저장한다. 원본 실행을 일부러 망가뜨리는 데 쓴다.
def _rewrite(path, **changes):
    """npz 파일의 배열을 바꾼다(None이면 그 배열을 지운다)."""
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    arrays.update(changes)
    np.savez(path, **{k: v for k, v in arrays.items() if v is not None})


# load_source가 파일만 보고 알 수 있는 문제를 모두 거절하는지 확인한다: 평가 점수 장수가 이미지 수와 다름,
# 평가 점수나 보정 점수에 NaN·inf, 보정 점수가 1차원이 아님.
def test_load_source_rejects_scores_that_cannot_be_compared(tmp_path):
    src = tmp_path / "src"
    _make_source(src, "dev")
    good = run_perturb.load_source(src, "toy", "p0")
    assert good.eval_score.shape == (14,) and good.cal_score.shape == (32,)

    _rewrite(src / "toy.npz", eval_score_full=good.eval_score[:-1])
    with pytest.raises(ValueError, match="14 evaluation images"):
        run_perturb.load_source(src, "toy", "p0")
    bad = good.eval_score.copy()
    bad[3] = np.nan
    _rewrite(src / "toy.npz", eval_score_full=bad)
    with pytest.raises(ValueError, match="not finite"):
        run_perturb.load_source(src, "toy", "p0")
    cal = good.cal_score.copy()
    cal[0] = np.inf
    _rewrite(src / "toy.npz", eval_score_full=good.eval_score, pool_score_oof=cal)
    with pytest.raises(ValueError, match="not finite"):
        run_perturb.load_source(src, "toy", "p0")
    _rewrite(src / "toy.npz", pool_score_oof=good.cal_score.reshape(4, 8))
    with pytest.raises(ValueError, match="pool_score_oof"):
        run_perturb.load_source(src, "toy", "p0")


# ---------------------------------------------------------------- 명령줄(CLI), PatchCore


# CLI(main) 테스트용 준비. 매니페스트를 임시 CSV로 쓰고 paths의 매니페스트·장부·출력·캐시 경로를 임시 폴더로
# 돌린다. 캐시·구성·백본 생성은 크기와 이름을 확인하는 가짜로 바꾸고, run_dinomaly를 불러오면 실패하게 해
# PatchCore 방법에서 anomalib 쪽 import가 일어나지 않는지도 본다.
def _project(tmp_path, monkeypatch, categories=("toy",)):
    """가짜 매니페스트·가짜 캐시·가짜 백본과 tmp_path 아래 장부에 연결한 CLI 환경."""
    manifest = _manifest(categories)
    ledger = tmp_path / "ledger.jsonl"
    cache = FakeCache(manifest, ledger=ledger)
    outputs = tmp_path / "outputs"
    extractor = Recording()

    def fake_cache(root, size):
        assert size == SIZE
        return cache

    def fake_extractor(name, *, img_size, pretrained=True, centre=False):
        assert (name, img_size, centre) == ("gridmean", SIZE, CFG.centre)
        return extractor

    def fake_config(name):
        assert name == "p0"
        return CFG

    def no_dinomaly():
        raise AssertionError("run_dinomaly must only be imported for the dm method")

    write_manifest(manifest, tmp_path / "visa.csv")
    monkeypatch.setattr(run_perturb.paths, "VISA_MANIFEST", tmp_path / "visa.csv")
    monkeypatch.setattr(run_perturb.paths, "TEST_LEDGER", ledger)
    monkeypatch.setattr(run_perturb.paths, "OUTPUTS", outputs)
    monkeypatch.setattr(run_perturb.paths, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(run_perturb, "ImageCache", fake_cache)
    monkeypatch.setattr(run_perturb, "get_config", fake_config)
    monkeypatch.setattr(run_perturb, "git_commit", lambda: "abc1234")
    monkeypatch.setattr(run_perturb, "_dinomaly", no_dinomaly)
    monkeypatch.setattr(backbones, "make_extractor", fake_extractor)
    return SimpleNamespace(
        cache=cache, ledger=ledger, outputs=outputs, manifest=manifest, extractor=extractor
    )


# 범주 하나(toy)인 CLI 준비.
@pytest.fixture
def toy_project(tmp_path, monkeypatch):
    return _project(tmp_path, monkeypatch)


@pytest.fixture
def two_project(tmp_path, monkeypatch):
    """범주 둘로 같은 환경을 만든다. 첫 범주만이 아니라 모든 범주에서 성립해야 하는 것을 볼 때 쓴다."""
    return _project(tmp_path, monkeypatch, TWO)


# CLI 인자 묶음: 두 범주 이름, dev·test 절차의 기본 인자, 봉인 허락 인자(--allow-test, --stage 3).
TWO = ("toy", "toz")
DEV = ["--method", "p0", "--protocol", "dev", "--device", "cpu", "--categories", "toy"]
TEST = ["--method", "p0", "--protocol", "test", "--device", "cpu", "--categories", "toy"]
ALLOW = ["--allow-test", "--stage", "3"]


# 장부 줄도, 이미지 읽기도, 출력 폴더도 없으면 참. '실행이 시작하기 전에 거절됐다'를 한 번에 확인하는
# 도우미다.
def _untouched(project, out_name):
    """장부 줄, 이미지 읽기, 출력 폴더가 모두 없다: 실행이 시작하기 전에 멈췄다."""
    return (
        not project.ledger.exists()
        and project.cache.requested == []
        and not (project.outputs / out_name).exists()
    )


# dev 절차 CLI 전체 흐름: run.json에 방법·절차·커밋·장치, 원본 경로와 원본 커밋, 조건 목록, 구성이 기록되고(줄
# 끝은 LF), dev 평가 세트만 읽으며 장부는 쓰지 않는다. 배치는 조건마다 [8, 6]이다.
# 이어서 analyze_perturb가 결과를 읽어, 30% 어둡게(brightness-3)에서 AUROC는 그대로인데 정상 전부가 임계값을
# 넘는 경우(H9가 말하는 고장)를 찾아내는지 본다. 끝난 실행은 --overwrite 없이 덮어쓰지 않는다.
def test_cli_dev_run_and_its_analysis(toy_project, capsys):
    outputs = toy_project.outputs
    _make_source(outputs / "p0-dev", "dev")
    run_perturb.main(DEV)
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [p["category"] for p in printed] == ["toy"]

    out = outputs / "perturb-p0-dev"
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert (run["method"], run["protocol"], run["commit"], run["device"]) == ("p0", "dev", "abc1234", "cpu")
    assert run["source"] == str(outputs / "p0-dev") and run["source_commit"] == "src1234"
    assert run["conditions"] == condition_names() and run["config"] == asdict(CFG)
    assert [c["category"] for c in run["categories"]] == ["toy"]
    assert run["categories"][0]["seconds"] >= 0 and run["categories"][0]["clean_max_rel_diff"] == 0.0
    assert b"\r" not in (out / "run.json").read_bytes()
    assert toy_project.cache.requested == DEV_IMAGES  # dev 평가 세트 말고는 읽지 않았다
    assert not toy_project.ledger.exists()
    assert sorted(p.name for p in out.iterdir()) == ["run.json", "toy.npz"]
    # 이미지 14장을 구성의 배치 크기로, 조건마다 한 번씩.
    assert toy_project.extractor.batches == [8, 6] * 16

    # 분석은 실행을 그대로 읽는다. 어둡게 하면 순위는 그대로(결함은 여전히 두드러진다)인데 모든 정상이
    # 임계값을 넘는다: H9가 다루는 경우다.
    loaded = analyze_perturb.load(out)
    table = analyze_perturb.condition_table(loaded, n_boot=50)
    assert [row["condition"] for row in table] == condition_names()
    dark = table[condition_names().index("brightness-3")]
    assert table[0]["macro_image_auroc"] == 1.0 and dark["d_auroc"] == 0.0
    assert dark["fpr"] == 1.0 and table[0]["fpr"] < 0.5
    assert "brightness-3" in analyze_perturb.h9(table)["conditions"]

    # 끝난 실행은 실수로 덮어쓰지 않는다.
    with pytest.raises(SystemExit):
        run_perturb.main(DEV)
    run_perturb.main(DEV + ["--overwrite"])
    assert (out / "run.json").exists()


# 원본에 뱅크 파일이 없으면 '--save-bank로 원본을 다시 만들라'는 명령을 알려 주며 멈추고, 장부·읽기·출력
# 폴더가 하나도 생기지 않아야 한다.
def test_cli_missing_bank_names_the_fix(toy_project, capsys):
    outputs = toy_project.outputs
    _make_source(outputs / "p0-test", "test")
    (outputs / "p0-test" / "toy_bank.npy").unlink()
    with pytest.raises(SystemExit):
        run_perturb.main(TEST + ["--allow-test", "--stage", "3"])
    err = capsys.readouterr().err
    assert "run_patchcore --config p0 --protocol test --save-bank" in err and "toy" in err
    # 기록도 읽기도 없었고, 남은 것도 없다.
    assert not toy_project.ledger.exists() and toy_project.cache.requested == []
    assert not (outputs / "perturb-p0-test").exists()


# test 절차는 --allow-test와 --stage가 둘 다 필요하다. ImageCache를 None으로 바꿔 두어, 거절 전에 캐시를
# 열려고 하면 테스트가 실패하게 한다. --method와 --protocol도 필수 인자다.
def test_cli_refuses_the_test_protocol_without_flags(toy_project, monkeypatch):
    _make_source(toy_project.outputs / "p0-test", "test")
    monkeypatch.setattr(run_perturb, "ImageCache", None)  # 거절 전에는 아무것도 열면 안 된다
    for extra in ([], ["--allow-test"], ["--stage", "3"]):
        with pytest.raises(SystemExit):
            run_perturb.main(TEST + extra)
    assert not toy_project.ledger.exists() and toy_project.cache.requested == []
    assert not (toy_project.outputs / "perturb-p0-test").exists()
    with pytest.raises(SystemExit):
        run_perturb.main(["--method", "p0"])  # 절차를 밝혀야 한다
    with pytest.raises(SystemExit):
        run_perturb.main(["--protocol", "dev"])  # 방법도 마찬가지


# test 절차가 장부에 줄을 하나 쓰고(단계·구성 이름 perturb-p0·커밋·메모), 그 뒤에 봉인 이미지를 읽는지
# 확인한다. 순서는 FakeCache가 봉인 이미지를 내줄 때 장부 줄이 있는지 assert하는 것으로 검사된다.
def test_cli_test_protocol_records_the_read_first(toy_project):
    outputs = toy_project.outputs
    _make_source(outputs / "p0-test", "test")
    # FakeCache는 봉인 이미지를 내주기 전에 장부 줄이 있는지 단언한다.
    run_perturb.main(TEST + ["--allow-test", "--stage", "3", "--note", "toy"])
    lines = toy_project.ledger.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert (entry["stage"], entry["config"], entry["commit"], entry["note"]) == (
        "3",
        "perturb-p0",
        "abc1234",
        "toy",
    )
    assert toy_project.cache.requested == TEST_IMAGES
    run = json.loads((outputs / "perturb-p0-test" / "run.json").read_text(encoding="utf-8"))
    assert run["protocol"] == "test" and run["categories"][0]["cal"] == 40
    with np.load(outputs / "perturb-p0-test" / "toy.npz") as z:
        assert z["eval_labels"].tolist() == [0] * 10 + [1] * 4 and z["scores"].shape == (16, 14)


# 원본 실행 점검이 장부 기록과 이미지 읽기보다 먼저 일어나는지 경우별로 확인한다: 원본 없음, dev 원본을 test
# 임계값으로 쓰려 함, 등록 구성과 다른 구성(코어셋 0.25), run.json 기록과 행 수가 다른 뱅크(다른 실행이 남긴
# 파일), 모르는 범주, PatchCore 방법에 --model. 어느 경우도 장부 줄이나 읽기가 남지 않아야 한다.
def test_cli_checks_the_source_before_anything_is_read(toy_project, capsys):
    outputs = toy_project.outputs
    allow = ["--allow-test", "--stage", "3"]
    # 원본 실행이 아예 없다.
    with pytest.raises(SystemExit):
        run_perturb.main(TEST + allow)
    assert "holds no finished run" in capsys.readouterr().err

    # dev 실행의 임계값을 test 절차에 빌려줄 수 없다.
    _make_source(outputs / "p0-dev", "dev")
    with pytest.raises(SystemExit):
        run_perturb.main(TEST + allow + ["--source", str(outputs / "p0-dev")])
    assert "'dev' run" in capsys.readouterr().err

    # 등록 구성과 다른 구성으로 만든 원본.
    _make_source(outputs / "p0-test", "test")
    meta = json.loads((outputs / "p0-test" / "run.json").read_text(encoding="utf-8"))
    meta["config"]["coreset_ratio"] = 0.25
    (outputs / "p0-test" / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit):
        run_perturb.main(TEST + allow)
    assert "registered config" in capsys.readouterr().err

    # 원본 실행이 기록한 것과 행 수가 다른 메모리 뱅크(다른 실행이 남긴 파일).
    meta["config"] = asdict(CFG)
    (outputs / "p0-test" / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    bank = np.load(outputs / "p0-test" / "toy_bank.npy")
    np.save(outputs / "p0-test" / "toy_bank.npy", bank[: len(bank) // 2])
    with pytest.raises(SystemExit):
        run_perturb.main(TEST + allow)
    assert "written by another run" in capsys.readouterr().err

    # 모르는 범주, 그리고 모델 파일이 없는 방법에 준 모델 파일.
    with pytest.raises(SystemExit):
        run_perturb.main(DEV[:-1] + ["nothing"])
    with pytest.raises(SystemExit):
        run_perturb.main(DEV + ["--model", str(outputs / "model.pt")])
    assert not toy_project.ledger.exists() and toy_project.cache.requested == []


# 행 수는 같지만 값이 다른 뱅크(1.5배)는 파일 점검을 통과하므로 clean guard가 잡아야 한다. 멈춘 실행은
# run.json을 남기지 않아 끝난 실행으로 보이지 않는다.
def test_cli_stops_at_a_bank_that_does_not_match(toy_project):
    outputs = toy_project.outputs
    _make_source(outputs / "p0-dev", "dev")
    bank = np.load(outputs / "p0-dev" / "toy_bank.npy")
    np.save(outputs / "p0-dev" / "toy_bank.npy", (bank * np.float16(1.5)).astype(np.float16))
    with pytest.raises(run_perturb.CleanScoreMismatch):
        run_perturb.main(DEV)
    assert not (outputs / "perturb-p0-dev" / "run.json").exists()  # 끝난 실행이 아니다


# --overwrite 재실행이 실패하는 두 경우를 구분한다. 시작 전에 거절되면(원본 뱅크 없음) 이전에 끝난 run.json이
# 그대로 남는다. 시작한 뒤 clean guard에서 멈추면 폴더에 재실행의 일부가 섞였으므로 run.json을 지워,
# 분석(analyze_perturb.load)이 그 폴더를 끝난 실행으로 읽지 못하게 한다.
def test_cli_a_redo_that_fails_leaves_no_finished_run(toy_project, capsys):
    outputs = toy_project.outputs
    src, out = outputs / "p0-dev", outputs / "perturb-p0-dev"
    _make_source(src, "dev")
    run_perturb.main(DEV)
    finished = (out / "run.json").read_bytes()

    # 시작 전에 거절됨(원본 뱅크가 사라짐): 끝난 실행은 그대로 남는다.
    bank = np.load(src / "toy_bank.npy")
    (src / "toy_bank.npy").unlink()
    with pytest.raises(SystemExit):
        run_perturb.main(DEV + ["--overwrite"])
    assert "--save-bank" in capsys.readouterr().err
    assert (out / "run.json").read_bytes() == finished

    # 시작한 뒤 가드에서 멈춤(행 수가 같은 다른 뱅크): 폴더에 재실행의 일부가 섞였으므로 더는 끝난
    # 실행으로 보이면 안 된다.
    np.save(src / "toy_bank.npy", (bank * np.float16(1.5)).astype(np.float16))
    with pytest.raises(run_perturb.CleanScoreMismatch):
        run_perturb.main(DEV + ["--overwrite"])
    assert not (out / "run.json").exists()
    with pytest.raises(FileNotFoundError):
        analyze_perturb.load(out)


# 범주가 둘일 때 첫 범주만이 아니라 모든 범주의 원본을 이미지 없이 먼저 점검하는지 확인한다. 두 번째 범주의
# npz만 망가뜨려도(보정 점수 없음, 이미지 목록 뒤바뀜, 점수 장수 틀림, NaN) 장부 줄 없이 멈춰야 하고, 첫
# 범주도 읽지 않아야 한다. 그렇지 않으면 첫 범주에서 봉인 테스트를 읽은 뒤 두 번째에서 멈추는 일이 생긴다.
def test_cli_checks_every_category_before_the_test_read(two_project, capsys):
    outputs = two_project.outputs
    src = outputs / "p0-test"
    _make_source(src, "test", TWO)
    args = TEST + ["toz"] + ALLOW
    with np.load(src / "toz.npz") as z:
        good = {k: z[k] for k in z.files}

    # 아래는 모두 '두 번째' 범주에서만 틀리고 이미지 없이 알아챌 수 있다: 일어나지 않을 읽기에 장부 줄을
    # 쓰면 안 되고, 첫 범주도 읽으면 안 된다.
    broken = {
        "has no ['pool_score_oof']": {"pool_score_oof": None},
        "did not evaluate": {"eval_images": good["eval_images"][::-1].copy()},
        "14 evaluation images": {"eval_score_full": good["eval_score_full"][:-1]},
        "not finite": {"eval_score_full": np.full(14, np.nan, dtype=np.float32)},
    }
    for message, change in broken.items():
        np.savez(src / "toz.npz", **good)
        _rewrite(src / "toz.npz", **change)
        with pytest.raises(SystemExit):
            run_perturb.main(args)
        err = capsys.readouterr().err
        assert message in err and "toz" in err, message
        assert _untouched(two_project, "perturb-p0-test"), message

    # 첫 범주에서 틀려도 마찬가지다.
    np.savez(src / "toz.npz", **good)
    _rewrite(src / "toy.npz", pool_score_oof=None)
    with pytest.raises(SystemExit):
        run_perturb.main(args)
    assert "toy.npz" in capsys.readouterr().err and _untouched(two_project, "perturb-p0-test")


# 두 범주를 한 번에 돌리면 장부 줄은 실행당 하나이고, 범주마다 자기 뱅크·점수·보정 점수를 쓴다(교란 없음
# 점수가 각자의 원본과 같고, 두 범주의 원본 점수는 서로 다르다).
def test_cli_two_categories_use_their_own_bank_and_scores(two_project):
    outputs = two_project.outputs
    src, out = outputs / "p0-test", outputs / "perturb-p0-test"
    _make_source(src, "test", TWO)
    run_perturb.main(TEST + ["toz"] + ALLOW)
    assert len(two_project.ledger.read_text(encoding="utf-8").splitlines()) == 1  # 실행당 한 줄
    assert two_project.cache.requested == TEST_IMAGES + [p.replace("toy/", "toz/") for p in TEST_IMAGES]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert [c["category"] for c in run["categories"]] == list(TWO)
    assert [c["clean_max_rel_diff"] for c in run["categories"]] == [0.0, 0.0]
    for c in TWO:
        with np.load(out / f"{c}.npz") as z, np.load(src / f"{c}.npz") as source:
            assert np.array_equal(z["scores"][0], source["eval_score_full"])
            assert np.array_equal(z["cal_score"], source["pool_score_oof"])
    with np.load(src / "toy.npz") as a, np.load(src / "toz.npz") as b:
        assert not np.array_equal(a["eval_score_full"], b["eval_score_full"])  # 두 범주는 다르다


# --out이 원본 실행 폴더를 가리키면(그대로, '..'을 거친 경로, 끝에 '/'가 붙은 경로, --source로 지정한 같은
# 폴더) --overwrite가 있어도 거절해야 한다. 두 실행 모두 run.json과 <범주>.npz를 쓰므로 그대로 두면 원본이
# 파일 단위로 덮어써진다. 원본 파일이 바이트 단위로 그대로인지 확인한다.
def test_cli_refuses_to_write_into_the_source_run(toy_project, capsys):
    outputs = toy_project.outputs
    src = outputs / "p0-dev"
    _make_source(src, "dev")
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    for out in (str(src), str(src / ".." / "p0-dev"), str(src) + "/"):
        for extra in ([], ["--overwrite"]):
            with pytest.raises(SystemExit):
                run_perturb.main(DEV + ["--out", out] + extra)
            assert "source run" in capsys.readouterr().err
    # 원본 실행은 하나도 바뀌거나 지워지지 않았고, 읽은 것도 없다.
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before
    assert toy_project.cache.requested == []
    with np.load(src / "toy.npz") as z:
        assert "eval_score_full" in z.files and "pool_score_oof" in z.files

    # --source로 지정한 같은 폴더도 마찬가지다.
    with pytest.raises(SystemExit):
        run_perturb.main(DEV + ["--source", str(src), "--out", str(src), "--overwrite"])
    assert "source run" in capsys.readouterr().err
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before


# ---------------------------------------------------------------- Dinomaly, 대역 모델로


# 학습된 Dinomaly 대신 쓰는 대역 모델: 학습 파라미터 gain 하나뿐이다. gain 값을 바꿔 '다른 모델 파일'을 흉내
# 낸다.
class TinyDm(torch.nn.Module):
    """학습된 모델 대신 쓰는 대역: 밝기 점수에 곱하는 학습 파라미터 gain 하나."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(()))


# run_perturb가 run_dinomaly에서 쓰는 것(IMG_SIZE, build_model, load_model, predict)만 흉내 낸 가짜 모듈.
# anomalib 없이 Dinomaly 경로를 시험하려는 것이다. calls에 호출을 기록해 모델 로드와 예측 인자(배치 크기,
# 장치, amp)를 확인한다. with_loader=False면 load_model이 없는 옛 모듈 형태를 흉내 낸다.
def _dm_module(calls, with_loader=True):
    """`run_perturb`가 `run_dinomaly`에서 쓰는 것만, anomalib 없이 흉내 낸다."""

    def build_model(encoder_name="tiny"):
        calls.append(("build", encoder_name))
        return TinyDm()

    def load_model(path):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        model = TinyDm()
        model.load_state_dict(saved["state"])
        calls.append(("load", str(path)))
        return model.eval(), {"encoder": saved["encoder"], "steps": saved["steps"]}

    def predict(model, images, *, batch_size=32, device="cuda", amp=True, map_size=256):
        calls.append(("predict", len(images), batch_size, device, amp))
        x = torch.from_numpy(images).float().flatten(1)
        scores = float(model.gain.detach()) * (x.amax(dim=1) + x.mean(dim=1)) / 255.0
        return SimpleNamespace(image_scores=scores.numpy().astype(np.float32), maps=None)

    module = SimpleNamespace(IMG_SIZE=SIZE, build_model=build_model, predict=predict)
    if with_loader:
        module.load_model = load_model
    return module


# run_dinomaly train이 저장하는 형식({encoder, steps, state})으로 대역 모델 파일을 쓴다. state_key를 바꾸면
# 학습 파라미터 이름이 맞지 않는 파일을 만든다.
def _save_dm(path, gain, state_key="gain"):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": "tiny", "steps": 3, "state": {state_key: torch.tensor(float(gain))}}, path)


# 2단계 공통 형식(eval_images, eval_labels, eval_score, cal_score)의 끝난 Dinomaly 평가를 대역 모델로 만든다.
# run_dinomaly와 같이, test 절차에서는 학습에 쓰지 않은 겹 0 정상의 점수를 보정 점수(홀드아웃)로 두고, dev
# 절차에서는 겹 0 정상이 곧 평가 정상이라 보정 점수가 비어 있다. model_path를 주면 그 파일의 sha256을
# run.json에 기록한다.
def _make_dm_source(src, protocol, module, gain, *, model_path=None, batch_size=5, amp=False):
    """대역 모델로 채점한, 2단계 공통 형식의 끝난 Dinomaly 평가."""
    manifest = _manifest()
    cache = FakeCache(manifest)
    model = TinyDm()
    with torch.no_grad():
        model.gain.fill_(gain)
    allow = protocol == "test"
    rows = select(manifest, protocol=protocol, part="eval_normal", category="toy", allow_test=allow)
    rows += select(manifest, protocol=protocol, part="eval_defect", category="toy", allow_test=allow)
    cal_rows = select(manifest, protocol="dev", part="eval_normal", category="toy") if allow else []
    cal = np.empty(0, dtype=np.float32)
    if cal_rows:
        cal = module.predict(model, cache.images(cal_rows), batch_size=batch_size).image_scores
    src.mkdir(parents=True)
    np.savez(
        src / "toy.npz",
        eval_images=np.array([r.image for r in rows]),
        eval_labels=np.array([int(r.label == "anomaly") for r in rows], dtype=np.int8),
        eval_score=module.predict(model, cache.images(rows), batch_size=batch_size).image_scores,
        cal_score=cal,
    )
    config = {"name": "dm", "batch_size": batch_size, "amp": amp}
    if model_path is not None:
        config["model_sha256"] = sha256_file(model_path)
    meta = {
        "method": "dinomaly",
        "protocol": protocol,
        "commit": "src1234",
        "device": "cpu",
        "config": config,
        "categories": [{"category": "toy"}],
    }
    (src / "run.json").write_text(json.dumps(meta), encoding="utf-8")


# Dinomaly CLI 준비: toy_project에 가짜 run_dinomaly 모듈을 끼우고, 기본 모델 경로(outputs/dm/model.pt)에 gain
# 2.0 모델을 저장한다.
@pytest.fixture
def dm_project(toy_project, monkeypatch):
    calls = []
    module = _dm_module(calls)
    monkeypatch.setattr(run_perturb, "_dinomaly", lambda: module)
    toy_project.calls = calls
    toy_project.module = module
    toy_project.model = toy_project.outputs / "dm" / "model.pt"
    _save_dm(toy_project.model, 2.0)
    return toy_project


DM_DEV = ["--method", "dm", "--protocol", "dev", "--device", "cpu", "--categories", "toy"]
DM_TEST = ["--method", "dm", "--protocol", "test", "--device", "cpu", "--categories", "toy"]


# Dinomaly dev 실행: 기본 모델 파일을 읽고, 조건마다 원본과 같은 배치 크기(5)·정밀도(amp 끔)로 한 번씩
# 예측한다(교란 없음 점수를 원본과 비교할 수 있게). 보정 점수가 비어 있으므로 분석 표에는 AUROC만 있고
# 오검출률 칸은 None이다. run.json의 구성에는 모델 경로와 sha256이 남는다.
def test_cli_dinomaly_dev_has_no_calibration_scores(dm_project):
    outputs = dm_project.outputs
    _make_dm_source(outputs / "dm-dev", "dev", dm_project.module, 2.0, model_path=dm_project.model)
    dm_project.calls.clear()
    run_perturb.main(DM_DEV)

    # 기본 모델 파일을 읽은 뒤, 조건마다 원본 실행의 배치 크기와 정밀도로 한 번씩 예측한다(교란 없음
    # 점수를 원본과 비교할 수 있게).
    assert dm_project.calls[0] == ("load", str(dm_project.model))
    assert dm_project.calls[1:] == [("predict", 14, 5, "cpu", False)] * 16
    out = outputs / "perturb-dm-dev"
    with np.load(out / "toy.npz") as z, np.load(outputs / "dm-dev" / "toy.npz") as source:
        assert set(z.files) == NPZ_KEYS
        assert z["conditions"].tolist() == condition_names()
        assert np.array_equal(z["scores"][0], source["eval_score"])
        assert z["cal_score"].shape == (0,) and z["cal_score"].dtype == np.float32
        assert z["eval_labels"].tolist() == [0] * 8 + [1] * 6
        assert not np.array_equal(z["scores"][condition_names().index("gamma-3")], z["scores"][0])
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["method"] == "dm" and run["source"] == str(outputs / "dm-dev")
    assert run["config"] == {
        "name": "dm",
        "img_size": SIZE,
        "model": str(dm_project.model),
        "model_sha256": sha256_file(dm_project.model),
        "batch_size": 5,
        "amp": False,
    }
    assert not dm_project.ledger.exists()
    # 보정 점수가 없으면 표에는 정확도만 있다.
    table = analyze_perturb.condition_table(analyze_perturb.load(out), n_boot=20)
    assert table[1]["fpr"] is None and table[1]["d_fpr"] is None and table[1]["d_auroc"] is not None


# Dinomaly test 실행: --model로 다른 경로의 모델을 쓰고, 장부에 perturb-dm 줄을 남기며, 원본의 홀드아웃 보정
# 점수(겹 0 정상 8장)를 다시 채점하지 않고 그대로 넘긴다(읽은 이미지는 봉인 테스트 14장뿐).
def test_cli_dinomaly_test_passes_the_holdout_scores_on(dm_project):
    outputs = dm_project.outputs
    model = outputs / "elsewhere" / "model.pt"
    _save_dm(model, 3.0)
    _make_dm_source(outputs / "dm-test", "test", dm_project.module, 3.0, model_path=model)
    dm_project.calls.clear()
    run_perturb.main(DM_TEST + ["--allow-test", "--stage", "3", "--model", str(model)])
    assert dm_project.calls[0] == ("load", str(model))
    entry = json.loads(dm_project.ledger.read_text(encoding="utf-8").splitlines()[0])
    assert (entry["stage"], entry["config"]) == ("3", "perturb-dm")
    assert dm_project.cache.requested == TEST_IMAGES  # 홀드아웃 정상은 다시 채점하지 않는다
    with (
        np.load(outputs / "perturb-dm-test" / "toy.npz") as z,
        np.load(outputs / "dm-test" / "toy.npz") as source,
    ):
        assert z["cal_score"].shape == (8,) and np.array_equal(z["cal_score"], source["cal_score"])
        assert np.array_equal(z["scores"][0], source["eval_score"])
        assert z["eval_images"].tolist() == TEST_IMAGES


# 원본이 기록한 모델 해시와 다른 모델 파일, 없는 모델 파일은 장부·읽기 전에 거절한다. 해시 기록이 없는
# 원본이면 파일 점검으로는 모르므로 clean guard가 다른 모델(gain 2.0 대 3.0)을 잡아야 하고, 멈춘 실행은
# run.json을 남기지 않는다.
def test_cli_dinomaly_refuses_another_model(dm_project, capsys):
    outputs = dm_project.outputs
    allow = ["--allow-test", "--stage", "3"]
    # 원본 실행이 자기 모델 해시를 기록했다: 다른 파일은 아무것도 읽기 전에 거절한다.
    other = outputs / "other" / "model.pt"
    _save_dm(other, 3.0)
    _make_dm_source(outputs / "dm-test", "test", dm_project.module, 3.0, model_path=other)
    with pytest.raises(SystemExit):
        run_perturb.main(DM_TEST + allow)
    assert "sha256 differs" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run_perturb.main(DM_TEST + allow + ["--model", str(outputs / "nowhere.pt")])
    assert "does not exist" in capsys.readouterr().err
    assert not dm_project.ledger.exists() and dm_project.cache.requested == []

    # 해시 기록이 없어도 교란 없음 점수가 잘못된 모델을 드러낸다.
    _make_dm_source(outputs / "dm-dev", "dev", dm_project.module, 3.0)
    with pytest.raises(run_perturb.CleanScoreMismatch):
        run_perturb.main(DM_DEV)
    assert not (outputs / "perturb-dm-dev" / "run.json").exists()


# run_dinomaly에 load_model이 없을 때의 대체 경로: 저장 파일의 encoder 이름으로 모델을 만들고 학습 파라미터만
# 불러와 eval 모드로 둔다. 파일의 파라미터 이름이 모델의 학습 파라미터와 다르면 ValueError로 멈춘다.
def test_load_dinomaly_without_a_loader_follows_the_model_file_layout(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_perturb, "_dinomaly", lambda: _dm_module(calls, with_loader=False))
    _save_dm(tmp_path / "model.pt", 4.0)
    model = run_perturb.load_dinomaly(tmp_path / "model.pt")
    assert calls == [("build", "tiny")] and float(model.gain.detach()) == 4.0 and not model.training

    _save_dm(tmp_path / "odd.pt", 4.0, state_key="bias")
    with pytest.raises(ValueError, match="trainable parameters"):
        run_perturb.load_dinomaly(tmp_path / "odd.pt")


# ---------------------------------------------------------------- 7단계: 조건 부분집합, 중심화 구성


# 7단계 E1 dev 점검(교란 없음, 밝기 0.7, 감마 1.5만 채점)을 위한 조건 부분집합 기능. 요청 순서와 무관하게 등록
# 순서를 지키고, clean이 없으면(원본 점검을 할 수 없으므로) 거절하며, 모르는 조건 이름도 거절한다. 중심화 구성
# p0-c, d-s-c가 PatchCore 방법 목록에 있는지도 확인한다.
def test_select_conditions_keeps_the_registered_order_and_needs_clean():
    assert run_perturb.select_conditions() == condition_names()
    picked = run_perturb.select_conditions(["gamma-3", "clean", "brightness-3"])
    assert picked == ["clean", "brightness-3", "gamma-3"]
    with pytest.raises(ValueError, match="clean"):
        run_perturb.select_conditions(["brightness-3"])
    with pytest.raises(ValueError, match="unknown"):
        run_perturb.select_conditions(["clean", "fog-1"])
    assert {"p0-c", "d-s-c"} <= set(run_perturb.PATCHCORE_METHODS)


# CLI의 --conditions가 세 조건만 등록 순서로 채점하고(배치 [8, 6] × 3), clean이 빠진 요청은 아무것도 읽거나
# 지우기 전에 거절해 이전 결과(run.json)가 그대로 남는지 확인한다.
def test_cli_scores_a_subset_of_the_conditions(toy_project):
    outputs = toy_project.outputs
    _make_source(outputs / "p0-dev", "dev")
    run_perturb.main(DEV + ["--conditions", "gamma-3", "clean", "brightness-3"])
    out = outputs / "perturb-p0-dev"
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["conditions"] == ["clean", "brightness-3", "gamma-3"]
    with np.load(out / "toy.npz") as z:
        assert z["conditions"].tolist() == run["conditions"] and z["scores"].shape == (3, 14)
    assert toy_project.extractor.batches == [8, 6] * 3
    # clean 조건이 없으면 아무것도 읽지 않는다: 원본 실행을 점검하는 것이 clean이기 때문이다.
    with pytest.raises(SystemExit):
        run_perturb.main(DEV + ["--overwrite", "--conditions", "brightness-3"])
    assert (out / "run.json").exists()


# 7단계 전에 만든 원본 run.json에는 centre 필드가 없다. configs.same_config는 이를 centre=False로 보므로 그런
# 원본도 받아들여야 하고, centre=True로 기록된 원본은 등록 구성(p0)이 아니므로 거절해야 한다.
def test_cli_accepts_a_source_recorded_before_the_centre_field(toy_project):
    outputs = toy_project.outputs
    _make_source(outputs / "p0-dev", "dev")
    meta = json.loads((outputs / "p0-dev" / "run.json").read_text(encoding="utf-8"))
    del meta["config"]["centre"]
    (outputs / "p0-dev" / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    run_perturb.main(DEV + ["--conditions", "clean"])
    # 중심화를 켜고 만든 원본은 이 구성이 아니다.
    meta["config"]["centre"] = True
    (outputs / "p0-dev" / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit):
        run_perturb.main(DEV + ["--overwrite", "--conditions", "clean"])
