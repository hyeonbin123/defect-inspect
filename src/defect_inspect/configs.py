"""Fixed method configurations. Values are pre-registered in docs/experiments.md."""

from dataclasses import asdict, dataclass


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


def get_config(name: str) -> PatchCoreConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        raise ValueError(f"unknown config {name!r}; known: {sorted(CONFIGS)}") from None


def same_config(recorded: dict, cfg: PatchCoreConfig) -> bool:
    """Whether a run's recorded config is `cfg`. Runs made before stage 7 have no `centre` (it was off)."""
    if not isinstance(recorded, dict):
        return False
    return {"centre": False, **recorded} == asdict(cfg)
