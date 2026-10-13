"""Fixed method configurations. Values are pre-registered in docs/experiments.md."""

# [파일 역할] PatchCore 계열 검사기의 구성(백본, 입력 크기, 코어셋 비율 등)을 이름 하나(p0, d-s, ...)로 고정해
# 두는 표다.
# [왜 필요한가] 이 프로젝트는 비지도 방법의 구성을 문헌 기본값으로 미리 정하고 docs/experiments.md에 적은 뒤에
# 잰다(결함 라벨 0장으로 정의). 문서의 값과 코드의 값이 어긋나지 않도록 구성을 한곳에 상수로 두고, 실행 명령은
# `--config p0`처럼 이름만 받게 했다.
# [연결] run_patchcore, run_m2ad, run_m2ad_enrol, run_perturb가 get_config로 구성을 읽고, run_perturb는
# same_config로 원본 실행의 구성을 확인한다. run_grid는 PatchCoreConfig 형식만 쓰고, figures는 CONFIGS에서
# 입력 크기를 읽는다. Dinomaly 구성은 이 파일이 아니라 run_dinomaly.py의 CONFIGS에 따로 있다.

# dataclass(frozen=True)로 구성을 불변 객체로 만들고, asdict는 same_config에서 실행 기록(run.json)의 dict와
# 비교할 때 쓴다.
from dataclasses import asdict, dataclass


# PatchCore 구성 하나. frozen이라 실행 중에 등록된 값을 실수로 바꿀 수 없다.
# - name: 구성 이름(실행 폴더·장부에 남는다). backbone: "wrn50", "dinov2_vits14", "dinov2_vitb14" 중 하나.
#   img_size: 정사각형 입력 한 변(px).
# - coreset_ratio: 메모리 뱅크에 남길 패치 비율(0.1 = 10%, 1단계 P0의 문헌 기본값).
# - reweight_k: 이미지 점수의 PatchCore 재가중에 쓰는 최근접 뱅크 패치의 이웃 수(9, 1단계 규칙).
# - sigma: 점수 맵을 흐리는 가우시안 σ(4, 1단계 규칙). batch_size: 특징 추출 배치(0단계 실측의 배치 32). seed:
#   코어셋의 무작위 투영과 시작점 시드.
@dataclass(frozen=True)
class PatchCoreConfig:
    name: str
    backbone: str
    img_size: int
    coreset_ratio: float
    reweight_k: int = 9
    sigma: float = 4.0
    batch_size: int = 32
    seed: int = 0
    # Stage 7 (E1): subtract each image's mean patch feature from its patch features, bank and query alike.
    centre: bool = False


# 등록된 구성 표. 키가 명령의 `--config`·`--method` 값이 된다. 2단계의 "d-b"(ViT-B PatchCore)는 "예산이 남을
# 때만"으로 등록했고 실제로는 돌리지 않았다(docs/experiments.md 2단계 실행 기록). p0-c, d-s-c는 7단계
# E1(이미지별 특징 중심화)로, 원래 구성에서 centre만 켠 것이다.
CONFIGS: dict[str, PatchCoreConfig] = {
    # Stage 1 baseline: literature defaults, no defect labels.
    "p0": PatchCoreConfig(name="p0", backbone="wrn50", img_size=256, coreset_ratio=0.1),
    # Stage 2: the same PatchCore on DINOv2 patch tokens (32x32 grid at 448 px).
    "d-s": PatchCoreConfig(name="d-s", backbone="dinov2_vits14", img_size=448, coreset_ratio=0.1),
    "d-b": PatchCoreConfig(name="d-b", backbone="dinov2_vitb14", img_size=448, coreset_ratio=0.1),
    # Stage 7 (E1): p0 and d-s with per-image feature centring, nothing else changed.
    "p0-c": PatchCoreConfig(name="p0-c", backbone="wrn50", img_size=256, coreset_ratio=0.1, centre=True),
    "d-s-c": PatchCoreConfig(
        name="d-s-c", backbone="dinov2_vits14", img_size=448, coreset_ratio=0.1, centre=True
    ),
}


# 이름으로 등록된 구성을 돌려준다. 없는 이름이면 KeyError 대신 알려진 이름 목록을 담은 ValueError를 낸다(`from
# None`으로 KeyError 연쇄는 감춘다). 호출 위치: run_patchcore, run_m2ad, run_m2ad_enrol, run_perturb의 main.
def get_config(name: str) -> PatchCoreConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        raise ValueError(f"unknown config {name!r}; known: {sorted(CONFIGS)}") from None


# 기록된 구성(dict)이 cfg와 같은지 본다. run_perturb가 교란 채점의 원본 실행(뱅크를 저장한 run_patchcore
# 실행)이 지금 요청한 구성으로 만들어졌는지 확인하는 데 쓴다. 예를 들어 p0 뱅크로 p0-c를 채점하는 실수를
# 막는다.
# 7단계 전에 만든 run.json에는 centre 필드가 없다. 그때는 중심화가 없었으므로 기본값 False를 먼저 깔고 기록된
# 값으로 덮어써, 옛 기록도 같은 구성으로 인정한다.
def same_config(recorded: dict, cfg: PatchCoreConfig) -> bool:
    """Whether a run's recorded config is `cfg`. Runs made before stage 7 have no `centre` (it was off)."""
    # 기록이 없거나 형식이 다르면(dict가 아니면) 같은 구성으로 보지 않는다.
    if not isinstance(recorded, dict):
        return False
    return {"centre": False, **recorded} == asdict(cfg)
