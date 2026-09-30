"""Fixed method configurations. Values are pre-registered in docs/experiments.md."""

from dataclasses import dataclass


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


CONFIGS: dict[str, PatchCoreConfig] = {
    # Stage 1 baseline: literature defaults, no defect labels.
    "p0": PatchCoreConfig(name="p0", backbone="wrn50", img_size=256, coreset_ratio=0.1),
}


def get_config(name: str) -> PatchCoreConfig:
    try:
        return CONFIGS[name]
    except KeyError:
        raise ValueError(f"unknown config {name!r}; known: {sorted(CONFIGS)}") from None
